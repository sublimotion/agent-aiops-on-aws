#!/usr/bin/env python3
"""
Functional coding evaluation for Qwen3-Next on SGLang.

Validates end-to-end coding agent feasibility: can the model read code,
identify bugs, propose fixes, and verify them via tool calls?

Inspired by SERA (Soft-Verified Efficient Repository Agents, Dettmers et al.)
SVG approach: generate a fix, then verify the model can reproduce it from
a PR description (soft verification via line-level patch recall).

Two evaluation modes:
  1. Direct Task Completion — give model a bug + codebase, measure if it
     produces a correct fix through multi-turn tool use.
  2. SVG Reproduction — take the model's fix from (1), create a PR
     description, ask the model to reproduce the fix. Measures consistency
     and instruction-following fidelity.

Tasks are self-contained Python modules with known bugs and test suites.
No external repos needed — everything runs in /tmp workspaces.

Usage:
  python3 functional-eval.py --endpoint http://localhost:30000 --model qwen3-next
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-Contained Coding Tasks
# ---------------------------------------------------------------------------
# Each task has: source files (with a bug), test file, bug description,
# and expected fix pattern for SVG verification.


TASKS = [
    {
        "id": "parse_date_none",
        "title": "Fix parse_date returning None",
        "description": "The parse_date function always returns None instead of parsing the date string.",
        "files": {
            "utils.py": textwrap.dedent("""\
                from datetime import datetime

                def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime | None:
                    \"\"\"Parse a date string into a datetime object.\"\"\"
                    return None  # TODO: implement
            """),
            "test_utils.py": textwrap.dedent("""\
                from utils import parse_date
                from datetime import datetime

                def test_parse_date_basic():
                    result = parse_date("2024-01-15")
                    assert result == datetime(2024, 1, 15)

                def test_parse_date_custom_format():
                    result = parse_date("15/01/2024", fmt="%d/%m/%Y")
                    assert result == datetime(2024, 1, 15)

                def test_parse_date_invalid():
                    result = parse_date("not-a-date")
                    assert result is None
            """),
        },
        "bug_prompt": "The tests in test_utils.py are failing because parse_date always returns None. Fix the implementation in utils.py.",
        "fix_patterns": ["datetime.strptime", "strptime"],
        "test_cmd": "python3 -m pytest test_utils.py -v",
    },
    {
        "id": "off_by_one_pagination",
        "title": "Fix off-by-one in pagination",
        "description": "Pagination returns one extra item per page due to an off-by-one error in slice bounds.",
        "files": {
            "paginator.py": textwrap.dedent("""\
                from typing import TypeVar, Sequence

                T = TypeVar("T")

                def paginate(items: Sequence[T], page: int, page_size: int = 10) -> list[T]:
                    \"\"\"Return a single page of items. Pages are 1-indexed.\"\"\"
                    if page < 1:
                        raise ValueError("Page must be >= 1")
                    start = (page - 1) * page_size
                    end = start + page_size + 1  # BUG: off by one
                    return list(items[start:end])

                def total_pages(total_items: int, page_size: int = 10) -> int:
                    \"\"\"Return total number of pages.\"\"\"
                    return (total_items + page_size - 1) // page_size
            """),
            "test_paginator.py": textwrap.dedent("""\
                from paginator import paginate, total_pages

                def test_first_page():
                    items = list(range(25))
                    result = paginate(items, page=1, page_size=10)
                    assert len(result) == 10
                    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                def test_second_page():
                    items = list(range(25))
                    result = paginate(items, page=2, page_size=10)
                    assert len(result) == 10
                    assert result == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

                def test_last_page():
                    items = list(range(25))
                    result = paginate(items, page=3, page_size=10)
                    assert len(result) == 5

                def test_total_pages():
                    assert total_pages(25, 10) == 3
                    assert total_pages(30, 10) == 3
                    assert total_pages(0, 10) == 0
            """),
        },
        "bug_prompt": "The paginator returns too many items per page. Tests are failing with length mismatches. Find and fix the bug in paginator.py.",
        "fix_patterns": ["page_size + 1", "start + page_size"],
        "test_cmd": "python3 -m pytest test_paginator.py -v",
    },
    {
        "id": "missing_auth_check",
        "title": "Add expiration check to JWT auth",
        "description": "The authenticate function decodes JWT tokens but doesn't check expiration.",
        "files": {
            "auth.py": textwrap.dedent("""\
                import json
                import base64
                import time

                SECRET = "test-secret-key"

                def _b64decode(s: str) -> dict:
                    padded = s + "=" * (4 - len(s) % 4)
                    return json.loads(base64.urlsafe_b64decode(padded))

                def authenticate(token: str) -> dict | None:
                    \"\"\"Decode and validate a JWT-like token. Returns payload or None.\"\"\"
                    try:
                        parts = token.split(".")
                        if len(parts) != 3:
                            return None
                        payload = _b64decode(parts[1])
                        # BUG: no expiration check
                        return payload
                    except Exception:
                        return None

                def create_token(user_id: str, expires_in: int = 3600) -> str:
                    \"\"\"Create a simple JWT-like token.\"\"\"
                    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
                    payload = base64.urlsafe_b64encode(json.dumps({
                        "user_id": user_id,
                        "exp": int(time.time()) + expires_in,
                    }).encode()).decode().rstrip("=")
                    return f"{header}.{payload}.signature"
            """),
            "test_auth.py": textwrap.dedent("""\
                import time
                from auth import authenticate, create_token

                def test_valid_token():
                    token = create_token("user-123", expires_in=3600)
                    result = authenticate(token)
                    assert result is not None
                    assert result["user_id"] == "user-123"

                def test_expired_token():
                    token = create_token("user-123", expires_in=-1)
                    result = authenticate(token)
                    assert result is None, "Expired tokens should be rejected"

                def test_malformed_token():
                    result = authenticate("not.a.valid.token.at.all")
                    assert result is None

                def test_missing_parts():
                    result = authenticate("only-one-part")
                    assert result is None
            """),
        },
        "bug_prompt": "test_expired_token is failing — the authenticate function accepts expired JWT tokens. Add an expiration check to auth.py.",
        "fix_patterns": ["exp", "time.time()", "expired"],
        "test_cmd": "python3 -m pytest test_auth.py -v",
    },
    {
        "id": "race_condition_counter",
        "title": "Fix thread-unsafe counter",
        "description": "A shared counter is incremented without locking, causing race conditions.",
        "files": {
            "counter.py": textwrap.dedent("""\
                import threading

                class ThreadSafeCounter:
                    \"\"\"A counter that should be safe to use from multiple threads.\"\"\"

                    def __init__(self, initial: int = 0):
                        self._value = initial
                        # BUG: no lock

                    def increment(self, amount: int = 1) -> int:
                        current = self._value
                        self._value = current + amount  # race condition
                        return self._value

                    def decrement(self, amount: int = 1) -> int:
                        current = self._value
                        self._value = current - amount
                        return self._value

                    @property
                    def value(self) -> int:
                        return self._value
            """),
            "test_counter.py": textwrap.dedent("""\
                import threading
                from counter import ThreadSafeCounter

                def test_single_thread():
                    c = ThreadSafeCounter()
                    for _ in range(100):
                        c.increment()
                    assert c.value == 100

                def test_concurrent_increment():
                    c = ThreadSafeCounter()
                    threads = []
                    for _ in range(10):
                        t = threading.Thread(target=lambda: [c.increment() for _ in range(1000)])
                        threads.append(t)
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    assert c.value == 10000, f"Expected 10000, got {c.value}"

                def test_concurrent_mixed():
                    c = ThreadSafeCounter(initial=5000)
                    inc_threads = [threading.Thread(target=lambda: [c.increment() for _ in range(500)]) for _ in range(5)]
                    dec_threads = [threading.Thread(target=lambda: [c.decrement() for _ in range(500)]) for _ in range(5)]
                    for t in inc_threads + dec_threads:
                        t.start()
                    for t in inc_threads + dec_threads:
                        t.join()
                    assert c.value == 5000
            """),
        },
        "bug_prompt": "The ThreadSafeCounter is failing concurrent tests — it's not actually thread-safe. Add proper locking to counter.py.",
        "fix_patterns": ["Lock", "lock", "threading.Lock"],
        "test_cmd": "python3 -m pytest test_counter.py -v",
    },
    {
        "id": "csv_encoding_error",
        "title": "Fix CSV export encoding handling",
        "description": "CSV export crashes on non-ASCII characters because it opens the file in ASCII mode.",
        "files": {
            "exporter.py": textwrap.dedent("""\
                import csv
                from typing import Sequence

                def export_csv(
                    rows: Sequence[dict],
                    filepath: str,
                    columns: list[str] | None = None,
                ) -> int:
                    \"\"\"Export rows to CSV. Returns number of rows written.\"\"\"
                    if not rows:
                        return 0
                    cols = columns or list(rows[0].keys())
                    with open(filepath, "w", encoding="ascii") as f:  # BUG: ascii encoding
                        writer = csv.DictWriter(f, fieldnames=cols)
                        writer.writeheader()
                        count = 0
                        for row in rows:
                            writer.writerow({k: row.get(k, "") for k in cols})
                            count += 1
                    return count
            """),
            "test_exporter.py": textwrap.dedent("""\
                import os
                import tempfile
                from exporter import export_csv

                def test_basic_export():
                    rows = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                        path = f.name
                    try:
                        count = export_csv(rows, path)
                        assert count == 2
                        with open(path) as f:
                            content = f.read()
                        assert "Alice" in content
                    finally:
                        os.unlink(path)

                def test_unicode_export():
                    rows = [
                        {"name": "Müller", "city": "München"},
                        {"name": "田中", "city": "東京"},
                        {"name": "José", "city": "São Paulo"},
                    ]
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                        path = f.name
                    try:
                        count = export_csv(rows, path)
                        assert count == 3
                        with open(path, encoding="utf-8") as f:
                            content = f.read()
                        assert "Müller" in content
                        assert "田中" in content
                    finally:
                        os.unlink(path)

                def test_empty_rows():
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                        path = f.name
                    try:
                        count = export_csv([], path)
                        assert count == 0
                    finally:
                        os.unlink(path)
            """),
        },
        "bug_prompt": "The CSV exporter crashes with UnicodeEncodeError on international characters. Fix the encoding in exporter.py.",
        "fix_patterns": ["utf-8", "utf8"],
        "test_cmd": "python3 -m pytest test_exporter.py -v",
    },
]


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    role: str  # "model" or "tool_exec"
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    content: str = ""
    latency_ms: float = 0.0


@dataclass
class TaskResult:
    task_id: str
    title: str
    # Phase 1: Direct task completion
    completed: bool = False
    tests_pass: bool = False
    turns_used: int = 0
    max_turns: int = 15
    patch_content: str = ""  # the model's edit
    trajectory: list[TurnRecord] = field(default_factory=list)
    error: str | None = None
    total_latency_ms: float = 0.0
    # Phase 2: SVG reproduction
    svg_attempted: bool = False
    svg_reproduced: bool = False
    svg_recall: float = 0.0  # line-level recall (SERA metric)
    svg_patch: str = ""


# ---------------------------------------------------------------------------
# Tool Execution (Real)
# ---------------------------------------------------------------------------


def exec_tool(tool_name: str, tool_args: dict, workspace: str) -> str:
    """Execute a tool call against the real workspace filesystem."""
    try:
        if tool_name == "read_file":
            path = tool_args.get("path", "")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isfile(full):
                return f"Error: file not found: {path}"
            with open(full, encoding="utf-8") as f:
                return f.read()

        elif tool_name == "write_file":
            path = tool_args.get("path", "")
            content = tool_args.get("content", "")
            full = os.path.join(workspace, path.lstrip("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {path} ({len(content)} chars)"

        elif tool_name == "run_command":
            cmd = tool_args.get("command", "")
            timeout = min(tool_args.get("timeout", 30), 60)
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=workspace,
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\nSTDERR:\n{proc.stderr}"
            if proc.returncode != 0:
                output += f"\n(exit code {proc.returncode})"
            return output[:4000]  # truncate

        elif tool_name == "list_files":
            path = tool_args.get("path", ".")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isdir(full):
                return f"Error: directory not found: {path}"
            entries = sorted(os.listdir(full))
            return "\n".join(entries)

        else:
            return f"Unknown tool: {tool_name}"

    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (max 60)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: workspace root)"},
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a coding agent. You have access to tools for reading files, writing files, listing directories, and running shell commands.

Your workflow:
1. List files to understand the workspace
2. Read the relevant source and test files
3. Identify the bug
4. Fix the bug by writing the corrected file
5. Run the tests to verify your fix

Be precise. Write the complete corrected file content when using write_file. Do not explain excessively — focus on fixing the code and verifying with tests."""


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls_from_msg(msg: dict) -> list[dict]:
    """Extract tool calls from standard array or XML <tool_call> tags in content."""
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        results = []
        for i, tc in enumerate(tool_calls):
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            results.append({
                "id": tc.get("id", f"call_{i}"),
                "function": {"name": fn["name"], "arguments": args},
            })
        return results

    # Fallback: parse <tool_call> XML from content (SGLang qwen3_coder parser)
    content = msg.get("content", "") or ""
    matches = _TOOL_CALL_RE.findall(content)
    results = []
    for i, m in enumerate(matches):
        try:
            obj = json.loads(m)
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            results.append({
                "id": f"call_{i}",
                "function": {"name": name, "arguments": arguments},
            })
        except (json.JSONDecodeError, TypeError):
            continue
    return results


async def run_agent_loop(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    task: dict,
    workspace: str,
    max_turns: int = 15,
) -> TaskResult:
    """Run the model in an agent loop against a real workspace."""
    result = TaskResult(
        task_id=task["id"],
        title=task["title"],
        max_turns=max_turns,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Bug report: {task['bug_prompt']}\n\n"
            f"The workspace contains the relevant files. "
            f"Fix the bug and run the tests to verify: `{task['test_cmd']}`"
        )},
    ]

    start = time.monotonic()

    for turn in range(max_turns):
        result.turns_used = turn + 1

        try:
            payload = {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 2048,
                "temperature": 0.0,
            }

            req_start = time.monotonic()
            async with session.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    result.error = f"HTTP {resp.status}: {text[:300]}"
                    break
                data = await resp.json()

            req_ms = (time.monotonic() - req_start) * 1000
            msg = data["choices"][0]["message"]

            # Extract tool calls from standard array or XML <tool_call> tags
            parsed_tool_calls = extract_tool_calls_from_msg(msg)

            # If model responds with text (no tool call) — it may be done or stuck
            if not parsed_tool_calls:
                content = msg.get("content", "")
                result.trajectory.append(TurnRecord(
                    role="model", content=content[:500], latency_ms=req_ms
                ))
                messages.append({"role": "assistant", "content": content})

                # Check if tests pass now (model might have finished)
                test_out = exec_tool("run_command", {"command": task["test_cmd"]}, workspace)
                if "passed" in test_out and "failed" not in test_out.lower().split("passed")[0]:
                    result.tests_pass = True
                    result.completed = True
                break

            # Build assistant message with tool_calls for conversation history
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], dict)
                            else tc["function"]["arguments"],
                        },
                    }
                    for tc in parsed_tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in parsed_tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}

                # Execute tool for real
                tool_output = exec_tool(name, args, workspace)

                result.trajectory.append(TurnRecord(
                    role="tool_exec",
                    tool_name=name,
                    tool_args=args,
                    tool_result=tool_output[:500],
                    latency_ms=req_ms / len(parsed_tool_calls),
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

                # Track patches
                if name == "write_file":
                    result.patch_content += f"--- {args.get('path', '?')} ---\n{args.get('content', '')}\n\n"

            # After tool execution, check if the last tool call was run_command with tests
            last_tc = parsed_tool_calls[-1]
            last_name = last_tc["function"]["name"]
            if last_name == "run_command":
                last_args = last_tc["function"]["arguments"] if isinstance(last_tc["function"]["arguments"], dict) else {}
                cmd = last_args.get("command", "")
                if "pytest" in cmd or "test" in cmd:
                    test_out = exec_tool("run_command", {"command": task["test_cmd"]}, workspace)
                    if "passed" in test_out and "failed" not in test_out.lower().split("passed")[0]:
                        result.tests_pass = True
                        result.completed = True

        except Exception as e:
            result.error = str(e)[:300]
            break

    # Final test verification regardless of model's last action
    if not result.tests_pass:
        test_out = exec_tool("run_command", {"command": task["test_cmd"]}, workspace)
        if "passed" in test_out and "failed" not in test_out.lower().split("passed")[0]:
            result.tests_pass = True
            result.completed = True

    result.total_latency_ms = (time.monotonic() - start) * 1000
    return result


# ---------------------------------------------------------------------------
# SVG Reproduction Phase
# ---------------------------------------------------------------------------


async def run_svg_reproduction(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    task: dict,
    original_result: TaskResult,
    max_turns: int = 15,
) -> TaskResult:
    """
    SERA-style soft verification: given a PR description derived from the
    model's first fix, ask the model to reproduce the fix from scratch.
    Compare patches via line-level recall.
    """
    if not original_result.completed or not original_result.patch_content:
        original_result.svg_attempted = False
        return original_result

    # Create PR description from the original patch
    pr_description = (
        f"## PR: {task['title']}\n\n"
        f"### Problem\n{task['description']}\n\n"
        f"### Changes\n"
        f"Fixed the bug in the relevant source file. "
        f"All tests now pass.\n\n"
        f"### Files Changed\n"
    )
    # Extract file names from patch
    for line in original_result.patch_content.split("\n"):
        if line.startswith("--- "):
            pr_description += f"- `{line[4:].strip()}`\n"

    # Fresh workspace
    workspace = tempfile.mkdtemp(prefix=f"sera_svg_{task['id']}_")
    try:
        for fname, content in task["files"].items():
            fpath = os.path.join(workspace, fname)
            with open(fpath, "w") as f:
                f.write(content)

        svg_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here is a pull request description for a bug fix:\n\n{pr_description}\n\n"
                f"Reproduce this fix in the workspace. Read the files, apply the same fix, "
                f"and verify tests pass: `{task['test_cmd']}`"
            )},
        ]

        # Run agent loop for reproduction
        svg_result = TaskResult(task_id=task["id"] + "_svg", title=task["title"] + " (SVG)")
        payload_messages = svg_messages

        start = time.monotonic()
        for turn in range(max_turns):
            svg_result.turns_used = turn + 1
            try:
                payload = {
                    "model": model,
                    "messages": payload_messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 2048,
                    "temperature": 0.0,
                }
                async with session.post(
                    f"{endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()

                msg = data["choices"][0]["message"]
                if not msg.get("tool_calls"):
                    payload_messages.append({"role": "assistant", "content": msg.get("content", "")})
                    break

                payload_messages.append(msg)
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    name = fn["name"]
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    tool_output = exec_tool(name, args, workspace)
                    payload_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_output})

                    if name == "write_file":
                        svg_result.patch_content += f"--- {args.get('path', '?')} ---\n{args.get('content', '')}\n\n"

            except Exception:
                break

        svg_result.total_latency_ms = (time.monotonic() - start) * 1000

        # Check tests pass
        test_out = exec_tool("run_command", {"command": task["test_cmd"]}, workspace)
        svg_result.tests_pass = "passed" in test_out and "failed" not in test_out.lower().split("passed")[0]
        svg_result.completed = svg_result.tests_pass

        # Compute line-level recall (SERA SVG metric)
        original_lines = set(
            line.strip()
            for line in original_result.patch_content.split("\n")
            if line.strip() and not line.startswith("---")
        )
        svg_lines = set(
            line.strip()
            for line in svg_result.patch_content.split("\n")
            if line.strip() and not line.startswith("---")
        )

        if original_lines:
            recall = len(svg_lines & original_lines) / len(original_lines)
        else:
            recall = 0.0

        # Update original result with SVG data
        original_result.svg_attempted = True
        original_result.svg_reproduced = svg_result.tests_pass
        original_result.svg_recall = recall
        original_result.svg_patch = svg_result.patch_content

    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return original_result


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------


async def run_evaluation(
    endpoint: str,
    model: str,
    output_dir: str,
    max_concurrent: int = 2,
    skip_svg: bool = False,
):
    """Run functional evaluation across all tasks."""
    os.makedirs(output_dir, exist_ok=True)
    results: list[TaskResult] = []

    log.info(f"Running {len(TASKS)} functional coding tasks against {endpoint}")
    log.info(f"SVG reproduction: {'enabled' if not skip_svg else 'disabled'}")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_task(task: dict) -> TaskResult:
        async with semaphore:
            workspace = tempfile.mkdtemp(prefix=f"sera_eval_{task['id']}_")
            try:
                # Set up workspace
                for fname, content in task["files"].items():
                    fpath = os.path.join(workspace, fname)
                    with open(fpath, "w") as f:
                        f.write(content)

                log.info(f"[{task['id']}] Starting: {task['title']}")

                async with aiohttp.ClientSession() as session:
                    # Phase 1: Direct task completion
                    result = await run_agent_loop(session, endpoint, model, task, workspace)
                    log.info(
                        f"[{task['id']}] Phase 1: "
                        f"{'PASS' if result.tests_pass else 'FAIL'} "
                        f"({result.turns_used} turns, {result.total_latency_ms:.0f}ms)"
                    )

                    # Phase 2: SVG reproduction
                    if not skip_svg and result.completed:
                        result = await run_svg_reproduction(session, endpoint, model, task, result)
                        log.info(
                            f"[{task['id']}] Phase 2 (SVG): "
                            f"{'PASS' if result.svg_reproduced else 'FAIL'} "
                            f"(recall={result.svg_recall:.2f})"
                        )

                return result
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

    tasks = [run_task(t) for t in TASKS]
    results = await asyncio.gather(*tasks)

    # --- Summary ---
    completed = sum(1 for r in results if r.completed)
    tests_pass = sum(1 for r in results if r.tests_pass)
    svg_attempted = sum(1 for r in results if r.svg_attempted)
    svg_reproduced = sum(1 for r in results if r.svg_reproduced)
    avg_turns = sum(r.turns_used for r in results) / len(results) if results else 0
    avg_latency = sum(r.total_latency_ms for r in results) / len(results) if results else 0

    summary = {
        "total_tasks": len(results),
        "completed": completed,
        "tests_pass": tests_pass,
        "completion_rate": round(completed / len(results), 4) if results else 0,
        "test_pass_rate": round(tests_pass / len(results), 4) if results else 0,
        "svg_attempted": svg_attempted,
        "svg_reproduced": svg_reproduced,
        "svg_reproduction_rate": round(svg_reproduced / svg_attempted, 4) if svg_attempted else 0,
        "avg_svg_recall": round(
            sum(r.svg_recall for r in results if r.svg_attempted) / svg_attempted, 4
        ) if svg_attempted else 0,
        "avg_turns": round(avg_turns, 1),
        "avg_latency_ms": round(avg_latency, 0),
        "model": model,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write results
    with open(os.path.join(output_dir, "functional_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    detailed = []
    for r in results:
        d = {
            "task_id": r.task_id,
            "title": r.title,
            "completed": r.completed,
            "tests_pass": r.tests_pass,
            "turns_used": r.turns_used,
            "total_latency_ms": round(r.total_latency_ms, 1),
            "error": r.error,
            "svg_attempted": r.svg_attempted,
            "svg_reproduced": r.svg_reproduced,
            "svg_recall": round(r.svg_recall, 4),
            "trajectory": [
                {
                    "role": t.role,
                    "tool": t.tool_name,
                    "args_keys": list(t.tool_args.keys()),
                    "result_preview": t.tool_result[:200],
                    "latency_ms": round(t.latency_ms, 1),
                }
                for t in r.trajectory
            ],
        }
        detailed.append(d)

    with open(os.path.join(output_dir, "functional_detailed.json"), "w") as f:
        json.dump(detailed, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"  S5: Functional Coding Evaluation Results")
    print(f"{'='*60}")
    print(f"  Model:    {model}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Tasks:    {len(results)}")
    print()
    print(f"  {'Task':<30} {'Complete':>10} {'Tests':>8} {'Turns':>7} {'SVG':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*7} {'-'*6}")
    for r in results:
        svg_str = f"{r.svg_recall:.0%}" if r.svg_attempted else "n/a"
        print(
            f"  {r.title:<30} "
            f"{'PASS' if r.completed else 'FAIL':>10} "
            f"{'PASS' if r.tests_pass else 'FAIL':>8} "
            f"{r.turns_used:>7} "
            f"{svg_str:>6}"
        )
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*7} {'-'*6}")
    print(f"  {'TOTALS':<30} {completed}/{len(results):>7} {tests_pass}/{len(results):>5}")
    print()
    print(f"  Task completion rate:   {summary['completion_rate']*100:.0f}%")
    print(f"  Test pass rate:         {summary['test_pass_rate']*100:.0f}%")
    if svg_attempted:
        print(f"  SVG reproduction rate:  {summary['svg_reproduction_rate']*100:.0f}%")
        print(f"  SVG avg line recall:    {summary['avg_svg_recall']*100:.0f}%")
    print(f"  Avg turns per task:     {summary['avg_turns']}")
    print(f"  Avg latency per task:   {summary['avg_latency_ms']:.0f}ms")
    print()

    # Feasibility verdict
    pass_rate = summary["test_pass_rate"]
    if pass_rate >= 0.8:
        verdict = "STRONG — model reliably fixes bugs through multi-turn tool use"
    elif pass_rate >= 0.6:
        verdict = "VIABLE — model can fix most bugs; retry strategy recommended for production"
    elif pass_rate >= 0.4:
        verdict = "MARGINAL — model struggles with some bug patterns; swarm-only viability"
    else:
        verdict = "NOT VIABLE — model cannot reliably fix bugs through tool use"

    print(f"  Verdict: {verdict}")
    print(f"{'='*60}")
    print(f"  Results: {output_dir}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Functional coding evaluation (SERA-inspired)")
    parser.add_argument("--endpoint", required=True, help="SGLang endpoint URL")
    parser.add_argument("--model", default="qwen3-next", help="Model name")
    parser.add_argument("--output-dir", default="./results/s5_functional", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent tasks")
    parser.add_argument("--skip-svg", action="store_true", help="Skip SVG reproduction phase")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            endpoint=args.endpoint,
            model=args.model,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            skip_svg=args.skip_svg,
        )
    )


if __name__ == "__main__":
    main()
