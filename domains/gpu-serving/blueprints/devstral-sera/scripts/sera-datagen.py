#!/usr/bin/env python3
"""
SERA SVG Data Generation Pipeline for Devstral Small 2.

Generates training pairs using the Soft-Verified Generation (SVG) approach:
  1. GENERATE: Model fixes a bug using multi-turn tool use
  2. VERIFY: Run the repo's test suite against the fix
  3. DESCRIBE: Convert the fix to a PR description
  4. REPRODUCE: Model generates fix from PR description alone
  5. SCORE: Compute line-level patch recall between (1) and (4)
  6. FILTER: Keep pairs where verify=PASS AND recall >= threshold
  7. FORMAT: Save as chat-format training examples

Input: SWE-bench dataset (issues with repo, base commit, test patch, test command)
Output: Filtered training pairs in JSONL format

Usage:
  # Generate from SWE-bench Lite
  python3 sera-datagen.py --endpoint http://localhost:9000 --dataset swebench-lite \
      --output-dir /mnt/nvme/sera-data --max-issues 300

  # Generate from a custom issue list
  python3 sera-datagen.py --endpoint http://localhost:9000 --issues issues.jsonl \
      --output-dir /mnt/nvme/sera-data
"""

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """A SWE-bench issue to process."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str  # patch to apply to set up test expectations
    test_cmd: str    # command to run tests
    hints: str = ""


@dataclass
class SVGResult:
    """Result of the full SVG pipeline for one issue."""
    instance_id: str
    # Phase 1: Generate fix
    fix_generated: bool = False
    fix_patch: str = ""
    fix_messages: list = field(default_factory=list)
    fix_turns: int = 0
    fix_latency_ms: float = 0
    # Phase 2: Verify
    tests_pass: bool = False
    test_output: str = ""
    # Phase 3: Describe
    pr_description: str = ""
    # Phase 4: Reproduce
    repro_generated: bool = False
    repro_patch: str = ""
    repro_messages: list = field(default_factory=list)
    repro_turns: int = 0
    # Phase 5: Score
    line_recall: float = 0.0
    # Phase 6: Filter
    accepted: bool = False
    reject_reason: str = ""
    # Meta
    error: Optional[str] = None
    total_latency_ms: float = 0


# ---------------------------------------------------------------------------
# Tool Definitions (same as functional-eval)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns numbered lines. Use start_line/end_line to read a specific range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "start_line": {"type": "integer", "description": "First line to read (1-based, optional)"},
                    "end_line": {"type": "integer", "description": "Last line to read (inclusive, optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file. old_text must match exactly (including whitespace). For new files, use write_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Text to replace it with"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or completely overwrite an existing file. For modifying existing files, prefer edit_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "content": {"type": "string", "description": "Complete file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the repository root and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory (non-recursive)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: repo root)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a pattern in files using grep. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: repo root)"},
                    "file_glob": {"type": "string", "description": "File glob pattern (e.g. '*.py')"},
                },
                "required": ["pattern"],
            },
        },
    },
]

FIX_SYSTEM_PROMPT = """\
You are a coding agent working on an open-source repository. You have tools \
to read files, edit files, write files, list directories, search code, and run shell commands.

Your task: fix the reported issue by modifying the source code. Then run the \
tests to verify your fix works.

Workflow:
1. Read the issue description carefully
2. Explore the relevant source files to understand the codebase
3. Identify the root cause
4. Apply the fix using edit_file (preferred for existing files) or write_file (for new files)
5. Run the test command to verify

Use edit_file to make targeted changes — provide the exact text to replace \
and the new text. This is more reliable than rewriting entire files. \
Focus on fixing the code correctly — do not add unrelated changes."""

REPRO_SYSTEM_PROMPT = """\
You are a coding agent. Given a PR description that describes a bug fix, \
reproduce the fix by modifying the source code. The PR description tells you \
what was changed and why.

Use edit_file to make targeted changes — provide the exact text to replace \
and the new text. Apply the described changes, then run the tests to verify."""

DESCRIBE_SYSTEM_PROMPT = """\
You are a technical writer. Given a diff (patch) that fixes a bug, write a \
concise PR description that another developer could use to reproduce the fix \
from scratch. Include:

1. What the bug was
2. Which files were modified
3. What changes were made and why
4. How to verify the fix

Do NOT include the actual diff or code — describe the changes in natural language."""


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------

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


def exec_tool(tool_name: str, tool_args: dict, workspace: str) -> str:
    """Execute a tool call against the workspace filesystem."""
    try:
        if tool_name == "read_file":
            path = tool_args.get("path", "")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isfile(full):
                return f"Error: file not found: {path}"
            with open(full, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = tool_args.get("start_line", 1)
            end = tool_args.get("end_line", len(lines))
            start = max(1, int(start)) if start else 1
            end = min(len(lines), int(end)) if end else len(lines)
            selected = lines[start - 1:end]
            numbered = [f"{start + i:5d} | {line}" for i, line in enumerate(selected)]
            content = "".join(numbered)
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated, file too large)"
            return content

        elif tool_name == "edit_file":
            path = tool_args.get("path", "")
            old_text = tool_args.get("old_text", "")
            new_text = tool_args.get("new_text", "")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isfile(full):
                return f"Error: file not found: {path}"
            with open(full, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if old_text not in content:
                return f"Error: old_text not found in {path}. Make sure it matches exactly."
            count = content.count(old_text)
            if count > 1:
                return f"Error: old_text found {count} times in {path}. Provide more context to match uniquely."
            content = content.replace(old_text, new_text, 1)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File edited: {path} (replaced {len(old_text)} chars with {len(new_text)} chars)"

        elif tool_name == "write_file":
            path = tool_args.get("path", "")
            content = tool_args.get("content", "")
            full = os.path.join(workspace, path.lstrip("/"))
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {path} ({len(content)} chars)"

        elif tool_name == "run_command":
            cmd = tool_args.get("command", "")
            timeout = min(tool_args.get("timeout", 60), 120)
            # Activate virtualenv if it exists
            venv_activate = os.path.join(workspace, ".venv", "bin", "activate")
            if os.path.exists(venv_activate):
                cmd = f"source {venv_activate} && {cmd}"
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=workspace, executable="/bin/bash",
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\nSTDERR:\n{proc.stderr}"
            if proc.returncode != 0:
                output += f"\n(exit code {proc.returncode})"
            return output[:8000]

        elif tool_name == "list_files":
            path = tool_args.get("path", ".")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isdir(full):
                return f"Error: directory not found: {path}"
            entries = sorted(os.listdir(full))
            return "\n".join(entries[:200])

        elif tool_name == "search_files":
            pattern = tool_args.get("pattern", "")
            path = tool_args.get("path", ".")
            file_glob = tool_args.get("file_glob", "")
            full = os.path.join(workspace, path.lstrip("/"))
            cmd = f"grep -rn --include='{file_glob}' '{pattern}' '{full}'" if file_glob else f"grep -rn '{pattern}' '{full}'"
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=workspace,
            )
            output = proc.stdout
            if not output:
                return "No matches found."
            # Make paths relative
            output = output.replace(workspace + "/", "")
            lines = output.split("\n")
            if len(lines) > 50:
                output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return output[:8000]

        else:
            return f"Unknown tool: {tool_name}"

    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


async def run_agent_loop(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_message: str,
    workspace: str,
    max_turns: int = 40,
    test_cmd: Optional[str] = None,
) -> tuple[bool, str, list[dict], int, float]:
    """
    Run the model in an agent loop.

    Returns: (tests_pass, patch_diff, messages, turns, latency_ms)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start = time.monotonic()
    tests_pass = False
    turns = 0

    for turn in range(max_turns):
        turns = turn + 1

        try:
            payload = {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 4096,
                "temperature": 0.0,
            }

            async with session.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                data = await resp.json()

            msg = data["choices"][0]["message"]
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            parsed_tool_calls = extract_tool_calls_from_msg(msg)

            log.info(f"  Turn {turns}: finish={finish_reason}, tool_calls={len(parsed_tool_calls)}, content_len={len(msg.get('content', '') or '')}")

            if not parsed_tool_calls:
                # Model finished — text response, no more tool calls
                content = msg.get("content", "")
                messages.append({"role": "assistant", "content": content})
                log.info(f"  Model text (first 300): {(content or '')[:300]}")
                break

            # Build assistant message with tool_calls
            # Preserve content if present (some models return both text + tool_calls)
            msg_content = msg.get("content") if msg.get("content") else None
            assistant_msg = {
                "role": "assistant",
                "content": msg_content,
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
                tool_output = exec_tool(name, args, workspace)
                log.info(f"    Tool: {name}({json.dumps(args)[:100]}) -> {len(tool_output)} chars")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

        except Exception as e:
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            break

    # Run tests if test_cmd provided
    if test_cmd:
        test_output = exec_tool("run_command", {"command": test_cmd, "timeout": 120}, workspace)
        tests_pass = _check_tests_pass(test_output)
    else:
        test_output = ""

    latency_ms = (time.monotonic() - start) * 1000

    # Generate diff
    patch_diff = _get_git_diff(workspace)

    return tests_pass, patch_diff, messages, turns, latency_ms


def _check_tests_pass(output: str) -> bool:
    """Check if test output indicates all tests passed (pytest or Django runner)."""
    output_lower = output.lower()

    # Pytest format: "X passed" with no "failed" or "error" before it
    if "passed" in output_lower:
        before_passed = output_lower.split("passed")[0]
        if "failed" not in before_passed and "error" not in before_passed:
            return True

    # Django test runner format: "OK" at end, or "Ran X tests" with "OK"
    if "\nok" in output_lower or output_lower.rstrip().endswith("ok"):
        if "fail" not in output_lower and "error" not in output_lower:
            return True

    # Django: "Ran N test(s)" + "OK"
    if "ran " in output_lower and " test" in output_lower:
        lines = output.strip().split("\n")
        last_lines = " ".join(lines[-3:]).lower()
        if "ok" in last_lines and "fail" not in last_lines:
            return True

    return False


def _get_git_diff(workspace: str) -> str:
    """Get git diff of workspace changes."""
    try:
        proc = subprocess.run(
            "git diff", shell=True, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        return proc.stdout[:50000]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Workspace Management
# ---------------------------------------------------------------------------


def setup_workspace(issue: Issue, base_dir: str) -> str:
    """
    Set up a workspace for an issue: clone repo, checkout base commit, apply test patch.
    Returns the workspace path.
    """
    workspace = os.path.join(base_dir, issue.instance_id)
    repo_cache = os.path.join(base_dir, "_repo_cache", issue.repo.replace("/", "__"))

    # Clone or use cached repo
    if not os.path.isdir(repo_cache):
        os.makedirs(os.path.dirname(repo_cache), exist_ok=True)
        log.info(f"[{issue.instance_id}] Cloning {issue.repo}...")
        proc = subprocess.run(
            f"git clone --depth 1000 https://github.com/{issue.repo}.git {repo_cache}",
            shell=True, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {proc.stderr[:500]}")

    # Copy repo to workspace
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    shutil.copytree(repo_cache, workspace, symlinks=True)

    # Checkout base commit
    proc = subprocess.run(
        f"git checkout -f {issue.base_commit}",
        shell=True, capture_output=True, text=True,
        timeout=30, cwd=workspace,
    )
    if proc.returncode != 0:
        # Fetch more history and retry
        subprocess.run(
            f"git fetch --unshallow origin && git checkout -f {issue.base_commit}",
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=workspace,
        )

    # Apply test patch (sets up test expectations) and commit so it won't show in git diff
    if issue.test_patch:
        proc = subprocess.run(
            f"git apply --allow-empty",
            shell=True, input=issue.test_patch, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        if proc.returncode != 0:
            log.warning(f"[{issue.instance_id}] Test patch apply failed: {proc.stderr[:200]}")
        else:
            subprocess.run(
                "git add -A && git commit -m 'apply test patch' --allow-empty",
                shell=True, capture_output=True, text=True,
                timeout=10, cwd=workspace,
            )

    # Install repo dependencies
    _install_repo_deps(workspace, issue)

    return workspace


# Known repo setup commands (SWE-bench repos that need special setup)
_REPO_SETUP = {
    "django/django": "pip install -e . -q",
    "astropy/astropy": "pip install -e '.[test]' -q",
    "sympy/sympy": "pip install -e . -q",
    "scikit-learn/scikit-learn": "pip install -e . -q",
    "matplotlib/matplotlib": "pip install -e . -q",
    "sphinx-doc/sphinx": "pip install -e '.[test]' -q",
    "pytest-dev/pytest": "pip install -e . -q",
    "pallets/flask": "pip install -e '.[dev]' -q",
    "psf/requests": "pip install -e . -q",
    "pydata/xarray": "pip install -e . -q",
    "pylint-dev/pylint": "pip install -e . -q",
    "mwaskom/seaborn": "pip install -e '.[dev]' -q",
}


def _install_repo_deps(workspace: str, issue: Issue):
    """Install repository dependencies in an isolated virtualenv."""
    repo = issue.repo
    setup_cmd = _REPO_SETUP.get(repo)

    if not setup_cmd:
        # Generic fallback: try pip install -e . if setup.py or pyproject.toml exists
        if os.path.exists(os.path.join(workspace, "setup.py")) or \
           os.path.exists(os.path.join(workspace, "pyproject.toml")):
            setup_cmd = "pip install -e . -q 2>/dev/null"
        else:
            return

    venv_dir = os.path.join(workspace, ".venv")

    # Use python3.11 if available (SWE-bench repos need 3.10+)
    python_bin = "python3.11" if shutil.which("python3.11") else "python3"

    # Create virtualenv and symlink python -> python3 (model often uses 'python' not 'python3')
    log.info(f"[{issue.instance_id}] Creating virtualenv ({python_bin}) + installing deps...")
    venv_setup = (
        f"{python_bin} -m venv {venv_dir} && "
        f"ln -sf python3 {venv_dir}/bin/python 2>/dev/null; "
        f"source {venv_dir}/bin/activate && "
        f"{setup_cmd}"
    )
    proc = subprocess.run(
        venv_setup, shell=True, capture_output=True, text=True,
        timeout=300, cwd=workspace, executable="/bin/bash",
    )
    if proc.returncode != 0:
        log.warning(f"[{issue.instance_id}] Dep install failed (exit {proc.returncode}): {proc.stderr[-200:]}")


# ---------------------------------------------------------------------------
# SVG Pipeline Stages
# ---------------------------------------------------------------------------


async def stage_generate(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    issue: Issue,
    workspace: str,
) -> tuple[bool, str, list[dict], int, float]:
    """Stage 1: Model generates a fix for the issue."""
    user_msg = f"""\
Issue: {issue.problem_statement}

Repository: {issue.repo}
The repository is checked out at the relevant commit. Fix the issue and run tests to verify:
  {issue.test_cmd}"""

    if issue.hints:
        user_msg += f"\n\nHints:\n{issue.hints}"

    return await run_agent_loop(
        session, endpoint, model,
        system_prompt=FIX_SYSTEM_PROMPT,
        user_message=user_msg,
        workspace=workspace,
        max_turns=40,
        test_cmd=issue.test_cmd,
    )


async def stage_describe(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    patch_diff: str,
    issue: Issue,
) -> str:
    """Stage 3: Convert a patch to a PR description."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Repository: {issue.repo}\n"
                f"Issue: {issue.problem_statement[:2000]}\n\n"
                f"Diff:\n```\n{patch_diff[:10000]}\n```"
            )},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Describe failed: HTTP {resp.status}: {text[:300]}")
        data = await resp.json()

    return data["choices"][0]["message"].get("content", "")


async def stage_reproduce(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    pr_description: str,
    issue: Issue,
    workspace: str,
) -> tuple[bool, str, list[dict], int, float]:
    """Stage 4: Model reproduces the fix from PR description alone."""
    user_msg = f"""\
PR Description:
{pr_description}

Repository: {issue.repo}
Apply the changes described in the PR, then verify with:
  {issue.test_cmd}"""

    return await run_agent_loop(
        session, endpoint, model,
        system_prompt=REPRO_SYSTEM_PROMPT,
        user_message=user_msg,
        workspace=workspace,
        max_turns=15,
        test_cmd=issue.test_cmd,
    )


def stage_score(original_patch: str, repro_patch: str) -> float:
    """Stage 5: Compute line-level recall between original and reproduced patches."""
    if not original_patch or not repro_patch:
        return 0.0

    original_lines = set(
        line.strip()
        for line in original_patch.split("\n")
        if line.strip()
        and not line.startswith("---")
        and not line.startswith("+++")
        and not line.startswith("@@")
        and not line.startswith("diff ")
        and not line.startswith("index ")
    )
    repro_lines = set(
        line.strip()
        for line in repro_patch.split("\n")
        if line.strip()
        and not line.startswith("---")
        and not line.startswith("+++")
        and not line.startswith("@@")
        and not line.startswith("diff ")
        and not line.startswith("index ")
    )

    if not original_lines:
        return 0.0
    return len(repro_lines & original_lines) / len(original_lines)


# ---------------------------------------------------------------------------
# Full SVG Pipeline
# ---------------------------------------------------------------------------


async def process_issue(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    issue: Issue,
    base_dir: str,
    recall_threshold: float = 0.8,
) -> SVGResult:
    """Run the full SVG pipeline for a single issue."""
    result = SVGResult(instance_id=issue.instance_id)
    start = time.monotonic()

    try:
        # Setup workspace
        workspace = setup_workspace(issue, base_dir)

        # Stage 1: Generate fix
        log.info(f"[{issue.instance_id}] Stage 1: Generating fix...")
        tests_pass, patch_diff, messages, turns, latency = await stage_generate(
            session, endpoint, model, issue, workspace,
        )
        result.fix_generated = bool(patch_diff)
        result.fix_patch = patch_diff
        result.fix_messages = messages
        result.fix_turns = turns
        result.fix_latency_ms = latency

        # Stage 2: Verify
        result.tests_pass = tests_pass
        if not tests_pass:
            result.reject_reason = "fix_tests_fail"
            log.info(f"[{issue.instance_id}] Stage 2: Tests FAIL — skipping SVG")
            return result

        if not patch_diff:
            result.reject_reason = "no_patch"
            log.info(f"[{issue.instance_id}] Stage 2: No diff — skipping SVG")
            return result

        log.info(f"[{issue.instance_id}] Stage 2: Tests PASS ({turns} turns, {latency:.0f}ms)")

        # Stage 3: Describe
        log.info(f"[{issue.instance_id}] Stage 3: Generating PR description...")
        pr_description = await stage_describe(
            session, endpoint, model, patch_diff, issue,
        )
        result.pr_description = pr_description

        if not pr_description or len(pr_description) < 50:
            result.reject_reason = "bad_description"
            log.info(f"[{issue.instance_id}] Stage 3: Description too short — skipping")
            return result

        # Reset workspace for reproduction
        shutil.rmtree(workspace, ignore_errors=True)
        workspace = setup_workspace(issue, base_dir)

        # Stage 4: Reproduce from PR description
        log.info(f"[{issue.instance_id}] Stage 4: Reproducing from PR description...")
        repro_pass, repro_patch, repro_msgs, repro_turns, _ = await stage_reproduce(
            session, endpoint, model, pr_description, issue, workspace,
        )
        result.repro_generated = bool(repro_patch)
        result.repro_patch = repro_patch
        result.repro_messages = repro_msgs
        result.repro_turns = repro_turns

        # Stage 5: Score
        result.line_recall = stage_score(patch_diff, repro_patch)
        log.info(
            f"[{issue.instance_id}] Stage 5: recall={result.line_recall:.2f}, "
            f"repro_tests={'PASS' if repro_pass else 'FAIL'}"
        )

        # Stage 6: Filter
        if result.line_recall >= recall_threshold:
            result.accepted = True
            log.info(f"[{issue.instance_id}] ACCEPTED (recall={result.line_recall:.2f})")
        else:
            result.reject_reason = f"low_recall_{result.line_recall:.2f}"
            log.info(f"[{issue.instance_id}] REJECTED (recall={result.line_recall:.2f} < {recall_threshold})")

    except Exception as e:
        result.error = str(e)[:500]
        log.error(f"[{issue.instance_id}] ERROR: {e}")

    finally:
        result.total_latency_ms = (time.monotonic() - start) * 1000
        # Clean up workspace (keep repo cache)
        workspace = os.path.join(base_dir, issue.instance_id)
        if os.path.exists(workspace):
            shutil.rmtree(workspace, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def _build_test_cmd(row: dict) -> str:
    """Build a test command from SWE-bench FAIL_TO_PASS field."""
    repo = row["repo"]
    fail_to_pass = row.get("FAIL_TO_PASS", "")

    # Parse FAIL_TO_PASS — it's a JSON string list of test identifiers
    if isinstance(fail_to_pass, str):
        try:
            tests = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            tests = [fail_to_pass] if fail_to_pass else []
    else:
        tests = fail_to_pass or []

    if not tests:
        return "python -m pytest"

    # Django uses its own test runner
    if "django" in repo:
        # Extract test module from "test_name (module.path.Class)"
        test_modules = set()
        for t in tests:
            if "(" in t:
                module_path = t.split("(")[1].rstrip(")")
                # Take the top-level test module (e.g., test_utils.tests → test_utils)
                parts = module_path.split(".")
                if len(parts) >= 1:
                    test_modules.add(parts[0])
            else:
                test_modules.add(t.split(".")[0])
        modules_str = " ".join(sorted(test_modules))
        return f"python3 tests/runtests.py {modules_str}"

    # Pytest-based repos: extract test file paths
    test_paths = set()
    for t in tests:
        if "::" in t:
            test_paths.add(t.split("::")[0])
        elif "(" in t:
            # "test_func (module.path)" format
            module_path = t.split("(")[1].rstrip(")")
            path = module_path.replace(".", "/") + ".py"
            test_paths.add(path)
        else:
            test_paths.add(t)

    if test_paths:
        paths_str = " ".join(sorted(test_paths))
        return f"python3 -m pytest {paths_str} -x"

    return "python3 -m pytest"


def load_swebench_dataset(name: str, max_issues: int = 0, skip_issues: int = 0) -> list[Issue]:
    """Load SWE-bench dataset. Requires 'datasets' library."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        sys.exit(1)

    dataset_map = {
        "swebench-lite": "princeton-nlp/SWE-bench_Lite",
        "swebench-verified": "princeton-nlp/SWE-bench_Verified",
        "swebench": "princeton-nlp/SWE-bench",
    }

    ds_name = dataset_map.get(name, name)
    log.info(f"Loading dataset: {ds_name}")
    ds = load_dataset(ds_name, split="test")

    issues = []
    for row in ds:
        issue = Issue(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            test_patch=row.get("test_patch", ""),
            test_cmd=_build_test_cmd(row),
            hints=row.get("hints_text", ""),
        )
        issues.append(issue)

    if max_issues > 0:
        issues = issues[:max_issues]

    log.info(f"Loaded {len(issues)} issues from {ds_name}")
    return issues


def load_custom_issues(path: str) -> list[Issue]:
    """Load issues from a JSONL file."""
    issues = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            issues.append(Issue(**d))
    log.info(f"Loaded {len(issues)} issues from {path}")
    return issues


# ---------------------------------------------------------------------------
# Training Data Formatting (Stage 7)
# ---------------------------------------------------------------------------


def format_training_example(result: SVGResult) -> Optional[dict]:
    """
    Convert an accepted SVG result into a chat-format training example.

    The training example uses the reproduction conversation (Stage 4) as the
    target — the model learns to fix code given a PR description.
    """
    if not result.accepted:
        return None

    # Use the reproduction messages as training data
    # Filter out system messages and clean up for training
    messages = []
    for msg in result.repro_messages:
        role = msg.get("role")
        if role == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            entry = {"role": "assistant"}
            if msg.get("content"):
                entry["content"] = msg["content"]
            if msg.get("tool_calls"):
                entry["content"] = None
                entry["tool_calls"] = msg["tool_calls"]
            messages.append(entry)
        elif role == "tool":
            messages.append({
                "role": "tool",
                "content": msg["content"][:4000],  # truncate tool results for training
                "tool_call_id": msg.get("tool_call_id", "call_0"),
            })

    if len(messages) < 3:
        return None

    return {
        "instance_id": result.instance_id,
        "messages": messages,
        "metadata": {
            "line_recall": result.line_recall,
            "fix_turns": result.fix_turns,
            "repro_turns": result.repro_turns,
        },
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    endpoint: str,
    model: str,
    issues: list[Issue],
    output_dir: str,
    max_concurrent: int = 4,
    recall_threshold: float = 0.8,
    workspace_dir: Optional[str] = None,
):
    """Run the full SVG pipeline across all issues."""
    os.makedirs(output_dir, exist_ok=True)
    base_dir = workspace_dir or tempfile.mkdtemp(prefix="sera_")

    log.info(f"Processing {len(issues)} issues, concurrency={max_concurrent}")
    log.info(f"Endpoint: {endpoint}, Model: {model}")
    log.info(f"Workspace: {base_dir}")
    log.info(f"Recall threshold: {recall_threshold}")

    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[SVGResult] = []

    async def process_with_limit(issue: Issue) -> SVGResult:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await process_issue(
                    session, endpoint, model, issue, base_dir, recall_threshold,
                )

    tasks = [process_with_limit(issue) for issue in issues]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.error(f"[{issues[i].instance_id}] Unhandled exception: {r}")
            final_results.append(SVGResult(
                instance_id=issues[i].instance_id,
                error=str(r)[:500],
            ))
        else:
            final_results.append(r)

    # Write results
    accepted = [r for r in final_results if r.accepted]
    failed = [r for r in final_results if not r.accepted]

    # Raw results (all issues)
    results_path = os.path.join(output_dir, "svg_results.jsonl")
    with open(results_path, "w") as f:
        for r in final_results:
            d = {
                "instance_id": r.instance_id,
                "fix_generated": r.fix_generated,
                "tests_pass": r.tests_pass,
                "line_recall": round(r.line_recall, 4),
                "accepted": r.accepted,
                "reject_reason": r.reject_reason,
                "fix_turns": r.fix_turns,
                "repro_turns": r.repro_turns,
                "total_latency_ms": round(r.total_latency_ms, 0),
                "error": r.error,
            }
            f.write(json.dumps(d) + "\n")

    # Training data (accepted only)
    train_path = os.path.join(output_dir, "train.jsonl")
    training_count = 0
    with open(train_path, "w") as f:
        for r in accepted:
            example = format_training_example(r)
            if example:
                f.write(json.dumps(example) + "\n")
                training_count += 1

    # Summary
    summary = {
        "total_issues": len(final_results),
        "fix_generated": sum(1 for r in final_results if r.fix_generated),
        "tests_pass": sum(1 for r in final_results if r.tests_pass),
        "accepted": len(accepted),
        "training_examples": training_count,
        "avg_recall": round(
            sum(r.line_recall for r in final_results if r.tests_pass)
            / max(1, sum(1 for r in final_results if r.tests_pass)),
            4,
        ),
        "avg_fix_turns": round(
            sum(r.fix_turns for r in final_results) / max(1, len(final_results)),
            1,
        ),
        "avg_latency_ms": round(
            sum(r.total_latency_ms for r in final_results) / max(1, len(final_results)),
            0,
        ),
        "errors": sum(1 for r in final_results if r.error),
        "endpoint": endpoint,
        "model": model,
        "recall_threshold": recall_threshold,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"  SERA SVG Data Generation Results")
    print(f"{'='*60}")
    print(f"  Issues processed:    {summary['total_issues']}")
    print(f"  Fixes generated:     {summary['fix_generated']}")
    print(f"  Tests passed:        {summary['tests_pass']}")
    print(f"  SVG accepted:        {summary['accepted']} (recall >= {recall_threshold})")
    print(f"  Training examples:   {summary['training_examples']}")
    print(f"  Avg line recall:     {summary['avg_recall']:.1%}")
    print(f"  Avg fix turns:       {summary['avg_fix_turns']}")
    print(f"  Avg latency:         {summary['avg_latency_ms']:.0f}ms")
    print(f"  Errors:              {summary['errors']}")
    print(f"{'='*60}")
    print(f"  Results: {results_path}")
    print(f"  Training: {train_path}")
    print()

    # Rejection breakdown
    reject_reasons = {}
    for r in failed:
        reason = r.reject_reason or ("error" if r.error else "unknown")
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    if reject_reasons:
        print("  Rejection reasons:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        print()

    return summary


def main():
    parser = argparse.ArgumentParser(description="SERA SVG Data Generation Pipeline")
    parser.add_argument("--endpoint", required=True, help="vLLM/SGLang endpoint URL")
    parser.add_argument("--model", default="devstral-small-2", help="Model name")
    parser.add_argument("--dataset", default="swebench-lite",
                        help="Dataset: swebench-lite, swebench-verified, swebench")
    parser.add_argument("--issues", help="Custom issues JSONL file (overrides --dataset)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--workspace-dir", help="Workspace directory for repo checkouts")
    parser.add_argument("--max-issues", type=int, default=0, help="Max issues to process (0=all)")
    parser.add_argument("--skip-issues", type=int, default=0, help="Skip first N issues")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent issues")
    parser.add_argument("--recall-threshold", type=float, default=0.8, help="Min SVG recall to accept")
    args = parser.parse_args()

    if args.issues:
        issues = load_custom_issues(args.issues)
    else:
        issues = load_swebench_dataset(args.dataset, max_issues=0)
        if args.skip_issues > 0:
            issues = issues[args.skip_issues:]
        if args.max_issues > 0:
            issues = issues[:args.max_issues]

    asyncio.run(run_pipeline(
        endpoint=args.endpoint,
        model=args.model,
        issues=issues,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        recall_threshold=args.recall_threshold,
        workspace_dir=args.workspace_dir,
    ))


if __name__ == "__main__":
    main()
