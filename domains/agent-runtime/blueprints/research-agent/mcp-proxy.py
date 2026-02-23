#!/usr/bin/env python3
"""
MCP stdio proxy for research-agent on Amazon Bedrock AgentCore Runtime.

Handles initialize and tools/list locally (fast, no AWS call) so Claude
Desktop doesn't time out during startup.

Async pattern (solves Claude Desktop's hard ~4-minute tool-call timeout):
  research(query)          — starts job in background thread, returns job_id
                             immediately (sub-second response)
  research_status(job_id)  — returns status/result; call every ~2 min until done

The background thread calls invoke_agent_runtime and stores the result in
_jobs (in-memory dict keyed by job_id). Results survive for the lifetime of
the proxy process (typically until Claude Desktop is restarted).
"""

import json
import os
import sys
import time as _time
import threading
import uuid
import boto3
from botocore.config import Config

# Runtime ARN — override via RUNTIME_ARN env var if runtime is recreated
RUNTIME_ARN = os.environ.get(
    "RUNTIME_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:615299764834:runtime/research_agent-fyUZrR80VG"
)
QUALIFIER = "research_agent_endpoint"
REGION = "us-east-1"
S3_BUCKET = os.environ.get(
    "S3_OUTPUT_BUCKET",
    "research-agent-output-20260221145504871200000003",
)

RESEARCH_TOOL = {
    "name": "research",
    "description": (
        "Start a multi-agent research pipeline on a given topic or question. "
        "Returns a job_id immediately — the pipeline runs in the background "
        "(10-15 minutes). Use research_status(job_id) to poll for the result."
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
                "description": "Optional session identifier. Also used as job_id.",
            },
        },
        "required": ["query"],
    },
}

STATUS_TOOL = {
    "name": "research_status",
    "description": (
        "Check the status of a research job started by research(). "
        "Returns 'running', 'done' (with full report), or 'error'. "
        "Poll every 2-3 minutes after calling research()."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The job_id returned by research().",
            },
        },
        "required": ["job_id"],
    },
}

# In-memory job store: job_id -> {status, result?, error?, query}
_jobs: dict = {}
_jobs_lock = threading.Lock()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-agentcore",
            region_name=REGION,
            config=Config(read_timeout=1200, connect_timeout=10,
                          retries={"max_attempts": 1, "mode": "standard"}),
        )
    return _client


def _run_job(job_id: str, query: str) -> None:
    """Background thread: call AgentCore, store result in _jobs."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "research",
            "arguments": {"query": query, "session_id": job_id},
        },
    }
    try:
        resp = get_client().invoke_agent_runtime(
            agentRuntimeArn=RUNTIME_ARN,
            qualifier=QUALIFIER,
            payload=json.dumps(payload).encode(),
            contentType="application/json",
            accept="application/x-ndjson",
        )
        body = resp.get("response", b"")
        final_result = None
        MAX_JOB_SECONDS = 1800  # 30-minute wall-clock cap
        job_start = _time.monotonic()
        if hasattr(body, "iter_lines"):
            for raw in body.iter_lines():
                if _time.monotonic() - job_start > MAX_JOB_SECONDS:
                    raise TimeoutError(f"AgentCore call exceeded {MAX_JOB_SECONDS}s wall-clock limit")
                if isinstance(raw, bytes):
                    raw = raw.decode()
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if "jsonrpc" in obj:
                    final_result = obj
        else:
            raw = body.read() if hasattr(body, "read") else body
            if raw:
                final_result = json.loads(raw)

        if final_result and "result" in final_result:
            text = final_result["result"]["content"][0]["text"]
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = text
        elif final_result and "error" in final_result:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = final_result["error"].get("message", "unknown error")
        else:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "no result returned from agent"

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


def handle(msg: dict) -> dict | None:
    method = msg.get("method", "")
    req_id = msg.get("id")

    # Notifications — never respond
    if req_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "research-agent", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [RESEARCH_TOOL, STATUS_TOOL]},
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        # ------------------------------------------------------------------ #
        # research() — start job, return immediately
        # ------------------------------------------------------------------ #
        if tool_name == "research":
            query = args.get("query", "").strip()
            if not query:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": "argument 'query' must not be empty"},
                }
            job_id = args.get("session_id") or f"job-{uuid.uuid4().hex[:8]}"
            with _jobs_lock:
                _jobs[job_id] = {"status": "running", "query": query}
            t = threading.Thread(target=_run_job, args=(job_id, query), daemon=True)
            t.start()
            msg_text = (
                f"Research job started.\n\n"
                f"**job_id**: `{job_id}`\n"
                f"**query**: {query}\n\n"
                f"The pipeline takes 10-15 minutes (Lead agent + Researchers + Data Analyst + Report Writer).\n"
                f"Call `research_status(\"{job_id}\")` in about 12 minutes to retrieve the report.\n"
                f"Results are also uploaded to `s3://{S3_BUCKET}/sessions/{job_id}/` when complete."
            )
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": msg_text}],
                    "isError": False,
                },
            }

        # ------------------------------------------------------------------ #
        # research_status() — check job state
        # ------------------------------------------------------------------ #
        if tool_name == "research_status":
            job_id = args.get("job_id", "").strip()
            if not job_id:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": "argument 'job_id' must not be empty"},
                }
            with _jobs_lock:
                job = _jobs.get(job_id)

            if job is None:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": (
                            f"No job found with id `{job_id}`.\n"
                            "Jobs are stored in memory and are lost if the proxy process restarts.\n"
                            f"Check S3 directly: `s3://{S3_BUCKET}/sessions/{job_id}/`"
                        )}],
                        "isError": False,
                    },
                }

            status = job["status"]
            if status == "running":
                text = (
                    f"Job `{job_id}` is still running.\n"
                    f"Query: {job['query']}\n\n"
                    "The pipeline typically takes 10-15 minutes total. Check again in a few minutes."
                )
            elif status == "done":
                text = job["result"]
            else:  # error
                text = f"Job `{job_id}` failed:\n\n{job.get('error', 'unknown error')}"

            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": (status == "error"),
                },
            }

        # Unknown tool
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name!r}"},
        }

    # Unknown method
    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method!r}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        try:
            result = handle(msg)
        except Exception as exc:
            req_id = msg.get("id")
            if req_id is None:
                continue
            result = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }

        if result is not None:
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
