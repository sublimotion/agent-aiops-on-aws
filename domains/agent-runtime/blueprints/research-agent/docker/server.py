"""
FastAPI HTTP server wrapping the research agent for AgentCore Runtime.

Endpoints (AgentCore Runtime HTTP protocol — port 8080):
  GET  /ping            -- AgentCore health check (returns {"status":"Healthy",...})
  POST /invocations     -- AgentCore Runtime HTTP invocation (MCP JSON-RPC 2.0)

Endpoints (legacy / direct callers):
  GET  /health          -- ALB / ECS health check
  POST /invoke          -- Run a research query; streams SSE events (direct callers)
  POST /               -- MCP JSON-RPC 2.0 handler
  POST /mcp            -- MCP JSON-RPC 2.0 handler (alias)

MCP tools exposed:
  research(query, session_id?) -> str

MCP protocol version: 2024-11-05
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback as _traceback
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure the package root is importable when running from /app
sys.path.insert(0, os.path.dirname(__file__))

from research_agent.agent import run_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("server")


def _load_secret(env_var: str, secret_name: str, json_key: str | None = None) -> None:
    """Load a single secret from Secrets Manager into os.environ if not already set."""
    if os.environ.get(env_var):
        return
    try:
        import boto3
        sm = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        resp = sm.get_secret_value(SecretId=secret_name)
        secret = resp.get("SecretString", "")
        try:
            import json as _json
            parsed = _json.loads(secret)
            if isinstance(parsed, dict) and json_key:
                secret = parsed.get(json_key, secret)
        except Exception:
            pass
        if secret:
            os.environ[env_var] = secret
            logger.info("Loaded %s from Secrets Manager (%s)", env_var, secret_name)
    except Exception as exc:
        logger.warning("Could not load %s from Secrets Manager: %s", env_var, exc)


def _load_secrets() -> None:
    """Load all API keys from AWS Secrets Manager into os.environ at startup."""
    _load_secret(
        "TAVILY_API_KEY",
        os.environ.get("TAVILY_API_KEY_SECRET", "research-agent/tavily-api-key"),
        json_key="TAVILY_API_KEY",
    )
    _load_secret(
        "BRAVE_API_KEY",
        os.environ.get("BRAVE_API_KEY_SECRET", "research-agent/brave-api-key"),
        json_key="BRAVE_API_KEY",
    )


_load_secrets()


def _upload_outputs_to_s3(session_id: str) -> list[str]:
    """Upload all files from FILES_BASE to S3 after a research run. Returns uploaded S3 keys."""
    bucket = os.environ.get("S3_OUTPUT_BUCKET", "")
    if not bucket:
        logger.warning("S3_OUTPUT_BUCKET not set — skipping output upload")
        return []
    files_base = Path(os.environ.get("FILES_BASE", "/app/files"))
    uploaded: list[str] = []
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        for path in sorted(files_base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(files_base)
            key = f"sessions/{session_id}/{rel}"
            s3.upload_file(str(path), bucket, key)
            uploaded.append(key)
            logger.info("Uploaded s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning("S3 upload failed: %s", exc)
    return uploaded


app = FastAPI(title="Research Agent", version="1.0.0")


# ---------------------------------------------------------------------------
# Streaming NDJSON generator for tools/call
# ---------------------------------------------------------------------------


async def _stream_tools_call(req_id, query: str, session_id: str):
    """
    Async generator that yields NDJSON bytes.
    Progress lines come first; the final MCP tools/call result is the last line.
    """

    def progress(label: str) -> bytes:
        return (json.dumps({"type": "progress", "label": label}) + "\n").encode()

    yield progress(f"Research pipeline starting — query: {query[:80]}")

    output_parts: list[str] = []
    error_result: dict | None = None

    async def _collect() -> None:
        nonlocal error_result
        try:
            async for event in run_query(query, session_id):
                etype = event.get("type", "")
                content = event.get("content", "")
                logger.info("event type=%s content_len=%d", etype, len(content))
                if etype == "text" and content:
                    output_parts.append(content)
        except Exception as exc:
            tb = _traceback.format_exc()
            logger.error("run_query failed: %s\n%s", exc, tb)
            error_result = _mcp_error(req_id, -32603, f"Agent error: {exc}")

    task = asyncio.create_task(_collect())

    # Yield keep-alive progress ticks every 30 s while the pipeline runs.
    # AgentCore flushes bytes immediately, preventing idle-connection timeouts.
    # Hard ceiling: cancel after 25 minutes so the connection doesn't hang forever.
    MAX_PIPELINE_SECONDS = 1500
    loop = asyncio.get_event_loop()
    deadline = loop.time() + MAX_PIPELINE_SECONDS
    while not task.done():
        if loop.time() >= deadline:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            yield (json.dumps(_mcp_error(req_id, -32603,
                "Research pipeline timed out after 25 minutes. "
                "Check S3 for any partial output.")) + "\n").encode()
            return
        yield progress("Research in progress...")
        await asyncio.sleep(30)

    await task  # propagate any unexpected exceptions

    if error_result is not None:
        yield (json.dumps(error_result) + "\n").encode()
        return

    full_text = "".join(output_parts) or "(no output)"

    yield progress("Uploading research outputs to S3...")
    uploaded = _upload_outputs_to_s3(session_id)
    if uploaded:
        bucket = os.environ.get("S3_OUTPUT_BUCKET", "")
        report_keys = [k for k in uploaded if "/reports/" in k]
        s3_note = f"\n\n---\n**Research outputs uploaded to S3** (`s3://{bucket}`):\n"
        if report_keys:
            s3_note += "PDF report(s):\n" + "\n".join(f"  - `{k}`" for k in report_keys) + "\n"
        s3_note += f"All files: `sessions/{session_id}/`"
        full_text += s3_note

    final = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": full_text}],
            "isError": False,
        },
    }
    yield (json.dumps(final) + "\n").encode()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    query: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------

MCP_TOOL_DEF = {
    "name": "research",
    "description": (
        "Run a multi-agent research pipeline on a given topic or question. "
        "Spawns sub-agents (Researcher, Data Analyst, Report Writer) and "
        "returns a comprehensive markdown/PDF research report."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research question or topic to investigate.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session identifier for continuity across calls.",
            },
        },
        "required": ["query"],
    },
}


def _mcp_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _mcp_result(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


async def _handle_mcp(body: dict) -> tuple[dict | None, bool]:
    """
    Process one MCP JSON-RPC message.

    Returns (response_dict, is_notification).
    Notifications return (None, True) — no response body should be sent.
    """
    method = body.get("method", "")
    req_id = body.get("id")  # None for notifications
    params = body.get("params", {})

    logger.info("MCP %s id=%s", method, req_id)

    # Notifications (no id, no response expected)
    if method == "notifications/initialized" or (method == "initialized" and req_id is None):
        return None, True

    if method == "initialize":
        return _mcp_result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "research-agent", "version": "1.0.0"},
        }), False

    if method == "tools/list":
        return _mcp_result(req_id, {"tools": [MCP_TOOL_DEF]}), False

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name != "research":
            return _mcp_error(req_id, -32601, f"Unknown tool: {tool_name!r}"), False

        query = args.get("query", "").strip()
        if not query:
            return _mcp_error(req_id, -32602, "argument 'query' must not be empty"), False

        session_id = args.get("session_id") or f"mcp-{uuid.uuid4().hex[:8]}"
        logger.info("tools/call research query=%r session=%s", query[:80], session_id)

        output_parts: list[str] = []
        try:
            async for event in run_query(query, session_id):
                etype = event.get("type", "")
                content = event.get("content", "")
                logger.info("event type=%s content_len=%d", etype, len(content))
                if etype == "text" and content:
                    output_parts.append(content)
        except Exception as exc:
            tb = _traceback.format_exc()
            logger.error("run_query failed: %s\n%s", exc, tb)
            return _mcp_error(req_id, -32603, f"Agent error: {exc}"), False

        full_text = "".join(output_parts) or "(no output)"

        # Upload research outputs to S3 and append location to response
        uploaded = _upload_outputs_to_s3(session_id)
        if uploaded:
            bucket = os.environ.get("S3_OUTPUT_BUCKET", "")
            report_keys = [k for k in uploaded if "/reports/" in k]
            s3_note = f"\n\n---\n**Research outputs uploaded to S3** (`s3://{bucket}`):\n"
            if report_keys:
                s3_note += "PDF report(s):\n" + "\n".join(f"  - `{k}`" for k in report_keys) + "\n"
            s3_note += f"All files: `sessions/{session_id}/`"
            full_text += s3_note

        return _mcp_result(req_id, {
            "content": [{"type": "text", "text": full_text}],
            "isError": False,
        }), False

    if method == "ping":
        return _mcp_result(req_id, {}), False

    if method == "debug/env":
        """Return env diagnostics (no credentials exposed)."""
        diag: dict = {
            "CLAUDE_CODE_USE_BEDROCK": os.environ.get("CLAUDE_CODE_USE_BEDROCK", "NOT SET"),
            "AWS_REGION": os.environ.get("AWS_REGION", "NOT SET"),
            "TAVILY_API_KEY_SET": bool(os.environ.get("TAVILY_API_KEY")),
            "BRAVE_API_KEY_SET": bool(os.environ.get("BRAVE_API_KEY")),
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "set" if os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") else "not set",
            "AWS_ACCESS_KEY_ID_SET": bool(os.environ.get("AWS_ACCESS_KEY_ID")),
        }
        # Check web identity credentials env vars
        diag["AWS_WEB_IDENTITY_TOKEN_FILE"] = "set" if os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE") else "not set"
        diag["AWS_ROLE_ARN"] = os.environ.get("AWS_ROLE_ARN", "not set")[:80]
        # Quick Bedrock call
        try:
            import boto3
            brt = boto3.client("bedrock-runtime", region_name="us-east-1")
            resp = brt.invoke_model(
                modelId="us.anthropic.claude-opus-4-6-v1",
                body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5, "messages": [{"role": "user", "content": "Hi"}]}),
                contentType="application/json",
                accept="application/json",
            )
            body = json.loads(resp["body"].read())
            diag["bedrock_direct"] = "ok: " + str(body.get("content", [{}])[0].get("text", ""))[:50]
        except Exception as e:
            diag["bedrock_direct_error"] = str(e)[:200]

        # Check ClaudeAgentOptions schema
        try:
            from claude_agent_sdk import ClaudeAgentOptions
            import inspect
            sig = inspect.signature(ClaudeAgentOptions)
            diag["options_params"] = list(sig.parameters.keys())[:30]
        except Exception as e:
            diag["options_params_error"] = str(e)[:100]

        try:
            import importlib.util, pathlib
            spec = importlib.util.find_spec("claude_agent_sdk")
            pkg_dir = pathlib.Path(spec.origin).parent
            bundled = pkg_dir / "_bundled" / "claude"
            diag["pkg_dir"] = str(pkg_dir)
            diag["bundled_exists"] = bundled.exists()
            diag["bundled_executable"] = bundled.exists() and os.access(str(bundled), os.X_OK)
            if bundled.exists():
                result = subprocess.run([str(bundled), "--version"], capture_output=True, text=True, timeout=5)
                diag["cli_rc"] = result.returncode
                diag["cli_stdout"] = result.stdout.strip()[:200]
                diag["cli_stderr"] = result.stderr.strip()[:300]
                # Run with print flag to see what the SDK actually invokes
                result2 = subprocess.run(
                    [str(bundled), "--print", "--dangerously-skip-permissions", "-p", "hi",
                     "--model", "us.anthropic.claude-opus-4-6-v1"],
                    capture_output=True, text=True, timeout=10,
                    env={**os.environ, "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
                )
                diag["cli_print_rc"] = result2.returncode
                diag["cli_print_stdout"] = result2.stdout.strip()[:500]
                diag["cli_print_stderr"] = result2.stderr.strip()[:500]
        except Exception as e:
            diag["cli_error"] = str(e)[:300]
        try:
            import boto3
            sts = boto3.client("sts", region_name="us-east-1")
            identity = sts.get_caller_identity()
            diag["aws_identity"] = identity.get("Arn", "")
        except Exception as e:
            diag["aws_identity_error"] = str(e)[:200]
        return _mcp_result(req_id, diag), False

    if method == "debug/test-sdk":
        """Run a minimal SDK query to see if Bedrock auth works."""
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
            stderr_lines: list[str] = []

            def capture_stderr(line: str) -> None:
                stderr_lines.append(line.rstrip())
                logger.error("[test-sdk stderr] %s", line.rstrip())

            # Explicitly fetch AWS credentials from boto3 and pass to subprocess
            import boto3
            session = boto3.Session()
            creds = session.get_credentials()
            frozen = creds.get_frozen_credentials() if creds else None
            extra_env: dict = {}
            if frozen:
                extra_env["AWS_ACCESS_KEY_ID"] = frozen.access_key
                extra_env["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
                if frozen.token:
                    extra_env["AWS_SESSION_TOKEN"] = frozen.token

            test_options = ClaudeAgentOptions(
                model=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"),
                permission_mode="bypassPermissions",
                system_prompt="You are a test assistant. Answer in one word.",
                allowed_tools=[],
                stderr=capture_stderr,
                env={
                    **os.environ,
                    **extra_env,
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "AWS_REGION": "us-east-1",
                },
            )
            result_text: list[str] = []
            async with ClaudeSDKClient(options=test_options) as client:
                await client.query(prompt="Say: HELLO")
                async for msg in client.receive_response():
                    if hasattr(msg, "content"):
                        for block in msg.content:
                            if hasattr(block, "text"):
                                result_text.append(block.text)
            return _mcp_result(req_id, {
                "status": "ok",
                "response": result_text[:3],
                "stderr": stderr_lines[:10],
            }), False
        except Exception as exc:
            tb = _traceback.format_exc()
            return _mcp_result(req_id, {
                "status": "error",
                "error": str(exc)[:500],
                "traceback": tb[:1000],
            }), False

    # Unknown method
    return _mcp_error(req_id, -32601, f"Method not found: {method!r}"), False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/test")
async def test_env() -> dict:
    """Diagnostic: report env vars and SDK import status (no credentials exposed)."""
    import os, subprocess
    env_report = {
        "CLAUDE_CODE_USE_BEDROCK": os.environ.get("CLAUDE_CODE_USE_BEDROCK", "NOT SET"),
        "AWS_REGION": os.environ.get("AWS_REGION", "NOT SET"),
        "TAVILY_API_KEY_SET": "yes" if os.environ.get("TAVILY_API_KEY") else "no",
        "BRAVE_API_KEY_SET": "yes" if os.environ.get("BRAVE_API_KEY") else "no",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "set" if os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") else "not set",
    }
    try:
        from claude_agent_sdk import ClaudeSDKClient  # noqa: F401
        env_report["sdk_import"] = "ok"
    except Exception as exc:
        env_report["sdk_import"] = f"FAILED: {exc}"
    try:
        import claude_agent_sdk._internal.transport.subprocess_cli as t
        cli_path = t._find_cli()
        env_report["cli_path"] = cli_path
        # Just check if binary exists and is executable
        result = subprocess.run([cli_path, "--version"], capture_output=True, text=True, timeout=5)
        env_report["cli_version_rc"] = result.returncode
        env_report["cli_version_out"] = result.stdout.strip()[:100]
        env_report["cli_version_err"] = result.stderr.strip()[:200]
    except Exception as exc:
        env_report["cli_check"] = f"FAILED: {exc}"
    return env_report


@app.get("/ping")
async def ping() -> dict:
    """AgentCore Runtime health check — returns Healthy when ready for work."""
    return {"status": "Healthy", "time_of_last_update": int(time.time())}


@app.get("/health")
async def health() -> dict:
    """ALB / ECS health check."""
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(body: InvokeRequest) -> EventSourceResponse:
    """Stream research agent events as SSE (direct / non-MCP callers)."""

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    logger.info("invoke: query=%r session_id=%r", body.query[:80], body.session_id)

    async def event_generator() -> AsyncIterator[dict]:
        try:
            async for event in run_query(body.query, body.session_id):
                yield {"data": json.dumps(event)}
        except Exception as exc:
            logger.exception("run_query failed: %s", exc)
            yield {"data": json.dumps({"type": "error", "content": str(exc)})}

    return EventSourceResponse(event_generator())


@app.post("/")
async def mcp_root(request: Request) -> Response:
    """
    MCP JSON-RPC 2.0 handler for AgentCore Runtime.

    AgentCore Runtime forwards MCP requests to the container's root path.
    Supports: initialize, tools/list, tools/call, ping, notifications/initialized.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_mcp_error(None, -32700, "Parse error: invalid JSON"),
        )

    response, is_notification = await _handle_mcp(body)

    if is_notification:
        return Response(status_code=204)

    return JSONResponse(content=response)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    """
    AgentCore Runtime HTTP protocol invocation endpoint.

    With serverProtocol=HTTP, AgentCore Runtime forwards the raw payload to
    POST /invocations on port 8080. Payload is MCP JSON-RPC 2.0.

    tools/call is handled with NDJSON StreamingResponse so that AgentCore
    keeps the connection alive while the multi-agent pipeline runs (10-15 min).
    All other methods (initialize, tools/list, ping, notifications/*)
    are handled synchronously via _handle_mcp.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_mcp_error(None, -32700, "Parse error: invalid JSON"),
        )

    method = body.get("method", "")
    logger.info("POST /invocations method=%s", method)

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        req_id = body.get("id")

        if tool_name != "research":
            return JSONResponse(content=_mcp_error(req_id, -32601, f"Unknown tool: {tool_name!r}"))

        query = args.get("query", "").strip()
        if not query:
            return JSONResponse(content=_mcp_error(req_id, -32602, "argument 'query' must not be empty"))

        session_id = args.get("session_id") or f"mcp-{uuid.uuid4().hex[:8]}"
        logger.info("tools/call research query=%r session=%s", query[:80], session_id)

        return StreamingResponse(
            _stream_tools_call(req_id, query, session_id),
            media_type="application/x-ndjson",
        )

    # All other methods — synchronous path (fast, no streaming needed)
    response, is_notification = await _handle_mcp(body)

    if is_notification:
        return Response(status_code=204)

    return JSONResponse(content=response)


@app.post("/mcp")
async def mcp_path(request: Request) -> Response:
    """MCP JSON-RPC 2.0 handler — some MCP clients POST to /mcp."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_mcp_error(None, -32700, "Parse error: invalid JSON"),
        )

    response, is_notification = await _handle_mcp(body)

    if is_notification:
        return Response(status_code=204)

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
