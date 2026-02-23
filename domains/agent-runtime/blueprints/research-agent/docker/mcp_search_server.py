#!/usr/bin/env python3
"""
MCP stdio server — web_search tool.

Search backend priority:
  1. Tavily  (TAVILY_API_KEY set)  — primary, AI-native, reliable
  2. Brave   (BRAVE_API_KEY set)   — fallback, direct REST call

Used by the researcher sub-agent via the project-level MCP config
at /app/.claude/settings.json (server name: "search").

Tool exposed: web_search(query, max_results=5)
  → returns plain-text list of results for the researcher to read.
"""

import gzip
import json
import os
import sys
import urllib.parse
import urllib.request


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient  # noqa: PLC0415
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _brave_search(query: str, max_results: int) -> list[dict]:
    url = (
        "https://api.search.brave.com/res/v1/web/search"
        f"?q={urllib.parse.quote(query)}&count={max_results}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": os.environ["BRAVE_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        data = json.loads(raw)
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("description", ""),
        }
        for item in data.get("web", {}).get("results", [])
    ]


def _do_search(query: str, max_results: int = 5) -> str:
    errors: list[str] = []
    results: list[dict] | None = None
    source = ""

    if os.environ.get("TAVILY_API_KEY"):
        try:
            results = _tavily_search(query, max_results)
            source = "Tavily"
        except Exception as exc:
            errors.append(f"Tavily: {exc}")

    if results is None and os.environ.get("BRAVE_API_KEY"):
        try:
            results = _brave_search(query, max_results)
            source = "Brave"
        except Exception as exc:
            errors.append(f"Brave: {exc}")

    if not results:
        err_detail = "; ".join(errors) if errors else "no API keys configured"
        return f"Search failed for query {query!r}. Errors: {err_detail}"

    lines = [f"Search results for: {query!r}  (via {source})\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n"
            f"   URL: {r['url']}\n"
            f"   {r['content'][:400]}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP protocol (JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------

TOOL_DEF = {
    "name": "web_search",
    "description": (
        "Search the web for information on a given query. "
        "Returns a list of relevant results with titles, URLs, and snippets."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def _handle(msg: dict) -> dict | None:
    method = msg.get("method", "")
    req_id = msg.get("id")

    if req_id is None:
        return None  # notification — no response

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "search", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [TOOL_DEF]},
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/call":
        params = msg.get("params", {})
        if params.get("name") != "web_search":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {params.get('name')!r}"},
            }
        args = params.get("arguments", {})
        query = args.get("query", "").strip()
        if not query:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "argument 'query' must not be empty"},
            }
        max_results = int(args.get("max_results", 5))
        text = _do_search(query, max_results)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
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
            result = _handle(msg)
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
