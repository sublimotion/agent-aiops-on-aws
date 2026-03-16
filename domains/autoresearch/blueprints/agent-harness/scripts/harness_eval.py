#!/usr/bin/env python3
"""
Phase 1: Turn Degradation Analysis for Agent Harness Experiment.

Runs the SERA agent loop with configurable turn budgets and per-turn
instrumentation. Supports restart-with-summary and context compaction strategies.

Forks the agent loop from sera-datagen.py — only the evaluation path
(Stage 1: Generate + Stage 2: Verify). No SVG pipeline.

Usage:
  # Run a single config
  python3 harness_eval.py --endpoint http://localhost:9000 \
      --config A --max-turns 10 --output results/phase1_A.jsonl

  # Run all Phase 1 configs
  python3 harness_eval.py --endpoint http://localhost:9000 --run-all

  # List the 50-issue subset (seed 42)
  python3 harness_eval.py --list-subset
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import sys
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
# Configuration
# ---------------------------------------------------------------------------

PHASE1_CONFIGS = {
    "A": {"max_turns": 10, "strategy": "strict"},
    "B": {"max_turns": 15, "strategy": "strict"},
    "C": {"max_turns": 20, "strategy": "strict"},
    "D": {"max_turns": 30, "strategy": "strict"},
    "E": {"max_turns": 30, "strategy": "restart", "restart_at": 15},
    "F": {"max_turns": 30, "strategy": "compact", "compact_at": 15},
}

SUBSET_SEED = 42
SUBSET_SIZE = 50

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    test_cmd: str
    hints: str = ""


@dataclass
class TurnMetric:
    """Per-turn instrumentation data."""
    turn_number: int
    action_type: str  # read/edit/write/run/list/search/text
    tool_name: str
    edit_applied: Optional[bool] = None  # for edit_file: did it apply?
    context_tokens: int = 0  # approximate token count in messages
    repeated_action: bool = False  # identical to a previous turn?
    latency_ms: float = 0


@dataclass
class EvalResult:
    """Result for one issue under one config."""
    instance_id: str
    config: str
    max_turns: int
    strategy: str
    # Outcome
    fix_generated: bool = False
    tests_pass: bool = False
    turns_used: int = 0
    total_latency_ms: float = 0
    # Per-turn data
    turn_metrics: list = field(default_factory=list)
    # Error
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool Definitions (from sera-datagen.py)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns numbered lines. Use start_line/end_line for ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "start_line": {"type": "integer", "description": "First line (1-based)"},
                    "end_line": {"type": "integer", "description": "Last line (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file. old_text must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {"type": "string", "description": "Exact text to find"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create/overwrite a file. For modifying existing files, prefer edit_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the repo root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
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
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a pattern in files using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search"},
                    "file_glob": {"type": "string", "description": "File glob (e.g. '*.py')"},
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

import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------
# Tool Execution (from sera-datagen.py)
# ---------------------------------------------------------------------------


def extract_tool_calls_from_msg(msg: dict) -> list[dict]:
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
    try:
        if tool_name == "read_file":
            path = tool_args.get("path", "")
            full = os.path.join(workspace, path.lstrip("/"))
            if not os.path.isfile(full):
                return f"Error: file not found: {path}"
            with open(full, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = max(1, int(tool_args.get("start_line", 1) or 1))
            end = min(len(lines), int(tool_args.get("end_line", len(lines)) or len(lines)))
            selected = lines[start - 1:end]
            numbered = [f"{start + i:5d} | {line}" for i, line in enumerate(selected)]
            content = "".join(numbered)
            return content[:50000] if len(content) <= 50000 else content[:50000] + "\n... (truncated)"

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
                return f"Error: old_text found {count} times in {path}. Provide more context."
            content = content.replace(old_text, new_text, 1)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File edited: {path}"

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
            return "\n".join(sorted(os.listdir(full))[:200])

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


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = sum(len(json.dumps(m)) for m in messages)
    return total_chars // 4


def _check_tests_pass(output: str) -> bool:
    output_lower = output.lower()
    if "passed" in output_lower:
        before_passed = output_lower.split("passed")[0]
        if "failed" not in before_passed and "error" not in before_passed:
            return True
    if "\nok" in output_lower or output_lower.rstrip().endswith("ok"):
        if "fail" not in output_lower and "error" not in output_lower:
            return True
    if "ran " in output_lower and " test" in output_lower:
        lines = output.strip().split("\n")
        last_lines = " ".join(lines[-3:]).lower()
        if "ok" in last_lines and "fail" not in last_lines:
            return True
    return False


def _get_git_diff(workspace: str) -> str:
    try:
        proc = subprocess.run(
            "git diff", shell=True, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        return proc.stdout[:50000]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Context Compaction
# ---------------------------------------------------------------------------


def compact_context(messages: list[dict]) -> list[dict]:
    """
    Compact message history: keep system + user prompt, summarize tool interactions.
    Used by Config F at the compaction turn.
    """
    if len(messages) < 4:
        return messages

    # Keep system and initial user message
    compacted = [messages[0], messages[1]]

    # Summarize tool interactions
    edits_made = []
    files_read = []
    commands_run = []
    errors = []

    for msg in messages[2:]:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if "File edited:" in content:
                edits_made.append(content)
            elif "Error:" in content:
                errors.append(content[:200])
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name == "read_file":
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    files_read.append(args.get("path", "unknown"))
                elif name == "run_command":
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    commands_run.append(args.get("command", "")[:100])

    summary = "Context summary from previous turns:\n"
    if files_read:
        summary += f"- Files read: {', '.join(files_read[:20])}\n"
    if edits_made:
        summary += f"- Edits made: {len(edits_made)}\n"
        for e in edits_made[:5]:
            summary += f"  - {e}\n"
    if commands_run:
        summary += f"- Commands run: {len(commands_run)}\n"
        for c in commands_run[:5]:
            summary += f"  - {c}\n"
    if errors:
        summary += f"- Errors encountered: {len(errors)}\n"
        for e in errors[:5]:
            summary += f"  - {e}\n"
    summary += "\nContinue working on the fix. Review what you've tried and try a different approach if previous attempts failed."

    compacted.append({"role": "user", "content": summary})
    return compacted


# ---------------------------------------------------------------------------
# Instrumented Agent Loop
# ---------------------------------------------------------------------------


async def run_instrumented_loop(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    issue: Issue,
    workspace: str,
    config: dict,
    config_name: str,
) -> EvalResult:
    """
    Run the agent loop with per-turn instrumentation.
    Supports strict cutoff, restart-with-summary, and context compaction.
    """
    max_turns = config["max_turns"]
    strategy = config["strategy"]
    restart_at = config.get("restart_at", 0)
    compact_at = config.get("compact_at", 0)

    result = EvalResult(
        instance_id=issue.instance_id,
        config=config_name,
        max_turns=max_turns,
        strategy=strategy,
    )

    user_msg = (
        f"Issue: {issue.problem_statement}\n\n"
        f"Repository: {issue.repo}\n"
        f"The repository is checked out at the relevant commit. Fix the issue and run tests to verify:\n"
        f"  {issue.test_cmd}"
    )
    if issue.hints:
        user_msg += f"\n\nHints:\n{issue.hints}"

    messages = [
        {"role": "system", "content": FIX_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    start = time.monotonic()
    previous_actions = []  # track (tool_name, args_hash) for repetition detection
    turn = 0

    while turn < max_turns:
        turn += 1

        # Strategy: restart with summary at restart_at
        if strategy == "restart" and turn == restart_at + 1:
            # Reset workspace to clean state
            if os.path.isdir(workspace):
                subprocess.run(
                    "git checkout -f HEAD", shell=True, capture_output=True,
                    timeout=10, cwd=workspace,
                )
            # Build summary of what was tried
            summary = compact_context(messages)
            messages = [
                messages[0],  # system prompt
                messages[1],  # original user message
                {"role": "user", "content": (
                    summary[-1]["content"] if len(summary) > 2 else
                    "Previous attempt did not succeed. Try a different approach."
                )},
            ]
            log.info(f"  [{issue.instance_id}] Turn {turn}: RESTART — context reset with summary")

        # Strategy: compact context at compact_at
        if strategy == "compact" and turn == compact_at + 1:
            messages = compact_context(messages)
            log.info(f"  [{issue.instance_id}] Turn {turn}: COMPACT — context compacted")

        turn_start = time.monotonic()

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
            parsed_tool_calls = extract_tool_calls_from_msg(msg)

            if not parsed_tool_calls:
                # Model finished
                content = msg.get("content", "")
                messages.append({"role": "assistant", "content": content})

                metric = TurnMetric(
                    turn_number=turn,
                    action_type="text",
                    tool_name="none",
                    context_tokens=_estimate_tokens(messages),
                    latency_ms=(time.monotonic() - turn_start) * 1000,
                )
                result.turn_metrics.append(asdict(metric))
                break

            # Process tool calls
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

                # Repetition detection
                action_sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                is_repeated = action_sig in previous_actions
                previous_actions.append(action_sig)

                # Execute tool
                tool_output = exec_tool(name, args, workspace)
                edit_applied = None
                if name == "edit_file":
                    edit_applied = "Error:" not in tool_output

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

                # Record per-turn metric
                action_type = {
                    "read_file": "read", "edit_file": "edit", "write_file": "write",
                    "run_command": "run", "list_files": "list", "search_files": "search",
                }.get(name, "unknown")

                metric = TurnMetric(
                    turn_number=turn,
                    action_type=action_type,
                    tool_name=name,
                    edit_applied=edit_applied,
                    context_tokens=_estimate_tokens(messages),
                    repeated_action=is_repeated,
                    latency_ms=(time.monotonic() - turn_start) * 1000,
                )
                result.turn_metrics.append(asdict(metric))

        except Exception as e:
            log.error(f"  [{issue.instance_id}] Turn {turn} error: {e}")
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            break

    result.turns_used = turn

    # Run tests
    test_output = exec_tool("run_command", {"command": issue.test_cmd, "timeout": 120}, workspace)
    result.tests_pass = _check_tests_pass(test_output)
    result.fix_generated = bool(_get_git_diff(workspace))
    result.total_latency_ms = (time.monotonic() - start) * 1000

    status = "PASS" if result.tests_pass else "FAIL"
    log.info(
        f"[{issue.instance_id}] Config {config_name}: {status} "
        f"({result.turns_used} turns, {result.total_latency_ms:.0f}ms)"
    )

    return result


# ---------------------------------------------------------------------------
# Workspace Management (from sera-datagen.py)
# ---------------------------------------------------------------------------

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


def setup_workspace(issue: Issue, base_dir: str) -> str:
    workspace = os.path.join(base_dir, issue.instance_id)
    repo_cache = os.path.join(base_dir, "_repo_cache", issue.repo.replace("/", "__"))

    if not os.path.isdir(repo_cache):
        os.makedirs(os.path.dirname(repo_cache), exist_ok=True)
        log.info(f"[{issue.instance_id}] Cloning {issue.repo}...")
        proc = subprocess.run(
            f"git clone --depth 1000 https://github.com/{issue.repo}.git {repo_cache}",
            shell=True, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {proc.stderr[:500]}")

    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    shutil.copytree(repo_cache, workspace, symlinks=True)

    proc = subprocess.run(
        f"git checkout -f {issue.base_commit}",
        shell=True, capture_output=True, text=True,
        timeout=30, cwd=workspace,
    )
    if proc.returncode != 0:
        subprocess.run(
            f"git fetch --unshallow origin && git checkout -f {issue.base_commit}",
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=workspace,
        )

    if issue.test_patch:
        proc = subprocess.run(
            "git apply --allow-empty",
            shell=True, input=issue.test_patch, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        if proc.returncode == 0:
            subprocess.run(
                "git add -A && git commit -m 'apply test patch' --allow-empty",
                shell=True, capture_output=True, text=True,
                timeout=10, cwd=workspace,
            )

    # Install deps
    setup_cmd = _REPO_SETUP.get(issue.repo)
    if not setup_cmd:
        if os.path.exists(os.path.join(workspace, "setup.py")) or \
           os.path.exists(os.path.join(workspace, "pyproject.toml")):
            setup_cmd = "pip install -e . -q 2>/dev/null"
    if setup_cmd:
        python_bin = "python3.11" if shutil.which("python3.11") else "python3"
        venv_dir = os.path.join(workspace, ".venv")
        venv_setup = (
            f"{python_bin} -m venv {venv_dir} && "
            f"ln -sf python3 {venv_dir}/bin/python 2>/dev/null; "
            f"source {venv_dir}/bin/activate && "
            f"{setup_cmd}"
        )
        log.info(f"[{issue.instance_id}] Installing deps...")
        subprocess.run(
            venv_setup, shell=True, capture_output=True, text=True,
            timeout=300, cwd=workspace, executable="/bin/bash",
        )

    return workspace


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def _build_test_cmd(row: dict) -> str:
    repo = row["repo"]
    fail_to_pass = row.get("FAIL_TO_PASS", "")
    if isinstance(fail_to_pass, str):
        try:
            tests = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            tests = [fail_to_pass] if fail_to_pass else []
    else:
        tests = fail_to_pass or []

    if not tests:
        return "python -m pytest"

    if "django" in repo:
        test_modules = set()
        for t in tests:
            if "(" in t:
                module_path = t.split("(")[1].rstrip(")")
                parts = module_path.split(".")
                if parts:
                    test_modules.add(parts[0])
            else:
                test_modules.add(t.split(".")[0])
        return f"python3 tests/runtests.py {' '.join(sorted(test_modules))}"

    test_paths = set()
    for t in tests:
        if "::" in t:
            test_paths.add(t.split("::")[0])
        elif "(" in t:
            module_path = t.split("(")[1].rstrip(")")
            test_paths.add(module_path.replace(".", "/") + ".py")
        else:
            test_paths.add(t)
    if test_paths:
        return f"python3 -m pytest {' '.join(sorted(test_paths))} -x"
    return "python3 -m pytest"


def load_subset(seed: int = SUBSET_SEED, size: int = SUBSET_SIZE) -> list[Issue]:
    """Load a deterministic subset of SWE-bench Lite."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    all_issues = []
    for row in ds:
        all_issues.append(Issue(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            test_patch=row.get("test_patch", ""),
            test_cmd=_build_test_cmd(row),
            hints=row.get("hints_text", ""),
        ))

    # Deterministic sample for reproducibility
    rng = random.Random(seed)
    # Stratified by repo to ensure diversity
    by_repo = {}
    for issue in all_issues:
        by_repo.setdefault(issue.repo, []).append(issue)

    selected = []
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)

    # Round-robin across repos until we hit subset size
    idx = {r: 0 for r in repos}
    while len(selected) < size:
        for repo in repos:
            issues = by_repo[repo]
            if idx[repo] < len(issues) and len(selected) < size:
                selected.append(issues[idx[repo]])
                idx[repo] += 1

    log.info(f"Selected {len(selected)} issues across {len(set(i.repo for i in selected))} repos")
    return selected


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


async def run_config(
    endpoint: str,
    model: str,
    config_name: str,
    config: dict,
    issues: list[Issue],
    workspace_dir: str,
    output_path: str,
    concurrency: int = 2,
):
    """Run one Phase 1 config against all issues."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def process_one(issue: Issue):
        async with sem:
            try:
                workspace = setup_workspace(issue, workspace_dir)
                connector = aiohttp.TCPConnector(limit=10)
                async with aiohttp.ClientSession(connector=connector) as session:
                    result = await run_instrumented_loop(
                        session, endpoint, model, issue, workspace, config, config_name,
                    )
                    results.append(result)
            except Exception as e:
                log.error(f"[{issue.instance_id}] Fatal error: {e}")
                results.append(EvalResult(
                    instance_id=issue.instance_id,
                    config=config_name,
                    max_turns=config["max_turns"],
                    strategy=config["strategy"],
                    error=str(e)[:500],
                ))
            finally:
                workspace = os.path.join(workspace_dir, issue.instance_id)
                if os.path.exists(workspace):
                    shutil.rmtree(workspace, ignore_errors=True)

    log.info(f"=== Config {config_name}: max_turns={config['max_turns']}, strategy={config['strategy']} ===")
    tasks = [process_one(issue) for issue in issues]
    await asyncio.gather(*tasks)

    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Summary
    n_pass = sum(1 for r in results if r.tests_pass)
    n_fix = sum(1 for r in results if r.fix_generated)
    avg_turns = sum(r.turns_used for r in results) / max(len(results), 1)
    n_repeated = sum(
        1 for r in results
        for tm in r.turn_metrics
        if tm.get("repeated_action")
    )

    log.info(
        f"=== Config {config_name} DONE ===\n"
        f"  Pass rate: {n_pass}/{len(results)} ({100*n_pass/max(len(results),1):.1f}%)\n"
        f"  Fix generated: {n_fix}/{len(results)}\n"
        f"  Avg turns: {avg_turns:.1f}\n"
        f"  Repeated actions: {n_repeated}\n"
        f"  Output: {output_path}"
    )

    return {
        "config": config_name,
        "pass_rate": n_pass / max(len(results), 1),
        "passes": n_pass,
        "fixes": n_fix,
        "total": len(results),
        "avg_turns": avg_turns,
        "repeated_actions": n_repeated,
    }


async def main():
    parser = argparse.ArgumentParser(description="Phase 1: Turn Degradation Analysis")
    parser.add_argument("--endpoint", default="http://localhost:9000", help="vLLM endpoint")
    parser.add_argument("--model", default="devstral-small-2", help="Model name")
    parser.add_argument("--config", help="Config to run (A-F)")
    parser.add_argument("--run-all", action="store_true", help="Run all Phase 1 configs")
    parser.add_argument("--list-subset", action="store_true", help="List the 50-issue subset")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--workspace-dir", default="/mnt/nvme/sera-workspaces", help="Workspace dir")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent issues")
    args = parser.parse_args()

    if args.list_subset:
        issues = load_subset()
        for issue in issues:
            print(f"{issue.instance_id}\t{issue.repo}")
        return

    issues = load_subset()

    if args.run_all:
        all_summaries = []
        for name, config in sorted(PHASE1_CONFIGS.items()):
            output = os.path.join(args.output_dir, f"phase1_{name}.jsonl")
            summary = await run_config(
                args.endpoint, args.model, name, config, issues,
                args.workspace_dir, output, args.concurrency,
            )
            all_summaries.append(summary)

        # Print final comparison
        print("\n=== PHASE 1 RESULTS ===")
        for s in all_summaries:
            print(f"  Config {s['config']}: pass_rate={100*s['pass_rate']:.1f}%, "
                  f"avg_turns={s['avg_turns']:.1f}, repeated={s['repeated_actions']}")
        baseline = next((s for s in all_summaries if s["config"] == "D"), None)
        if baseline:
            best = max(all_summaries, key=lambda s: s["pass_rate"])
            print(f"\nBaseline (D): {100*baseline['pass_rate']:.1f}%")
            print(f"Best: Config {best['config']} ({100*best['pass_rate']:.1f}%)")
        print("=== END ===")
        return

    if args.config:
        if args.config not in PHASE1_CONFIGS:
            print(f"Unknown config: {args.config}. Choose from: {list(PHASE1_CONFIGS.keys())}")
            sys.exit(1)
        config = PHASE1_CONFIGS[args.config]
        output = os.path.join(args.output_dir, f"phase1_{args.config}.jsonl")
        await run_config(
            args.endpoint, args.model, args.config, config, issues,
            args.workspace_dir, output, args.concurrency,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
