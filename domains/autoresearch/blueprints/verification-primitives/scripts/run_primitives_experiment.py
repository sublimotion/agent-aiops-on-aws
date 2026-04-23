#!/usr/bin/env python3
"""
Verification Primitives Experiment Runner (Phase 1 + Phase 2).

Runs the SERA agent loop with configurable verification primitives as
additional tools. Measures whether agents use verification tools voluntarily
and whether usage improves pass rate.

Phase 1: Baseline (no verification tools)
Phase 2: 5 experiment cells (A-E) with different tool configurations

Experiment cells:
  control  - Standard tools only (baseline)
  A        - Standard + generate_tests_confirmatory
  B        - Standard + generate_tests_adversarial
  C        - Standard + run_tests (existing repo tests only)
  D        - Standard + adversarial_review
  E        - All verification primitives available

Usage:
  # Run control baseline
  python3 run_primitives_experiment.py --cell control --output results/phase2_control.jsonl

  # Run cell B (adversarial test generation)
  python3 run_primitives_experiment.py --cell B --output results/phase2_B.jsonl

  # Run all cells
  python3 run_primitives_experiment.py --run-all

  # Smoke test on 5 issues
  python3 run_primitives_experiment.py --cell B --smoke-test --output results/smoke_B.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Import shared infrastructure from agent-harness
HARNESS_DIR = Path(__file__).resolve().parents[2] / "agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_DIR))
from harness_eval import Issue, load_subset, setup_workspace

# Import verification primitives
TOOLS_DIR = Path(__file__).resolve().parent.parent / "skills" / "verification-primitives" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

CELLS = {
    "control": {
        "description": "Standard tools only (baseline)",
        "tools": [],
        "system_suffix": "",
    },
    "A": {
        "description": "Standard + confirmatory test generation",
        "tools": ["generate_tests_confirmatory"],
        "system_suffix": (
            "\n\nYou have an additional tool: generate_tests. "
            "Call it to generate tests that verify your fix works correctly. "
            "You can also run the generated tests with run_tests."
        ),
    },
    "B": {
        "description": "Standard + adversarial test generation",
        "tools": ["generate_tests_adversarial"],
        "system_suffix": (
            "\n\nYou have an additional tool: generate_tests. "
            "Call it to generate adversarial tests designed to break your fix. "
            "These tests target edge cases and assumptions your patch might miss. "
            "You can also run the generated tests with run_tests."
        ),
    },
    "C": {
        "description": "Standard + run_tests (repo tests only)",
        "tools": ["run_tests"],
        "system_suffix": (
            "\n\nYou have an additional tool: run_tests. "
            "Call it to run the existing test suite against your patched code."
        ),
    },
    "D": {
        "description": "Standard + adversarial review",
        "tools": ["adversarial_review"],
        "system_suffix": (
            "\n\nYou have an additional tool: adversarial_review. "
            "Call it to get an adversarial code review of your patch. "
            "An expert reviewer will try to find bugs in your work."
        ),
    },
    "E": {
        "description": "All verification primitives",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have three verification tools available:\n"
            "1. generate_tests - Generate adversarial tests designed to break your fix\n"
            "2. run_tests - Run generated tests against your patched code\n"
            "3. adversarial_review - Get an adversarial code review of your patch\n"
            "Use these before submitting your final patch."
        ),
    },
    # --- Phase 2b: Guided Composition ---
    "B_nudge": {
        "description": "Adversarial tests + nudge in system prompt",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have verification tools available:\n"
            "- generate_tests: Generate adversarial tests designed to break your fix\n"
            "- run_tests: Run generated tests against your patched code\n"
            "- adversarial_review: Get an adversarial code review of your patch\n\n"
            "BEST PRACTICE: Before submitting your final patch, generate adversarial tests "
            "and run them. Patches that pass adversarial tests are more likely to be correct. "
            "Developers who skip verification have a 60% higher failure rate."
        ),
    },
    "B_checkpoint": {
        "description": "All tools + checkpoint injection at 70% budget",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have verification tools available:\n"
            "- generate_tests: Generate adversarial tests designed to break your fix\n"
            "- run_tests: Run generated tests against your patched code\n"
            "- adversarial_review: Get an adversarial code review of your patch"
        ),
        "checkpoint_at_pct": 0.7,  # Inject reminder at 70% of turn budget
        "checkpoint_msg": (
            "CHECKPOINT: You have {turns_left} turns remaining. "
            "Before submitting, run adversarial tests on your patch: "
            "call generate_tests, then run_tests. If tests fail, fix the issues."
        ),
    },
    "B_tdd": {
        "description": "All tools + TDD-first guidance",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have verification tools available:\n"
            "- generate_tests: Generate adversarial tests designed to break your fix\n"
            "- run_tests: Run generated tests against your patched code\n"
            "- adversarial_review: Get an adversarial code review of your patch\n\n"
            "WORKFLOW: Follow this process:\n"
            "1. Read the bug report and explore the code\n"
            "2. BEFORE writing your fix, use run_command to run the existing test suite "
            "to reproduce the bug\n"
            "3. Write your fix\n"
            "4. Generate adversarial tests targeting edge cases of your fix\n"
            "5. Run all tests (both existing and generated)\n"
            "6. Get an adversarial review before finishing\n"
            "7. Only finish when tests pass and review is positive"
        ),
    },
    "B_early50": {
        "description": "All tools + single checkpoint at 50% budget",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have verification tools available:\n"
            "- generate_tests: Validate your fix by generating edge-case tests\n"
            "- run_tests: Run generated tests against your patched code\n"
            "- adversarial_review: Get a thorough code review of your patch"
        ),
        "checkpoints": [
            {
                "pct": 0.5,
                "msg": (
                    "CHECKPOINT: You have {turns_left} turns remaining. "
                    "You should have your edit ready by now. "
                    "Run generate_tests and run_tests to validate your patch, "
                    "then use adversarial_review for a final check."
                ),
            },
        ],
    },
    "B_twostage": {
        "description": "All tools + edit checkpoint at 40% + verify checkpoint at 55%",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou have verification tools available:\n"
            "- generate_tests: Validate your fix by generating edge-case tests\n"
            "- run_tests: Run generated tests against your patched code\n"
            "- adversarial_review: Get a thorough code review of your patch"
        ),
        "checkpoints": [
            {
                "pct": 0.4,
                "msg": (
                    "TIME CHECK: You have {turns_left} turns remaining. "
                    "You've explored enough — make your edit now so you have "
                    "time to verify it before submitting."
                ),
            },
            {
                "pct": 0.55,
                "msg": (
                    "VERIFY NOW: You have {turns_left} turns remaining. "
                    "Run generate_tests and run_tests on your patch. "
                    "Then call adversarial_review before finishing. "
                    "If tests fail, you still have time to fix and re-verify."
                ),
            },
        ],
    },
    "B_mandatory": {
        "description": "All tools + mandatory verification (strongest guidance)",
        "tools": ["generate_tests_adversarial", "run_tests", "adversarial_review"],
        "system_suffix": (
            "\n\nYou MUST use these verification tools before finishing:\n"
            "- generate_tests: Generate adversarial tests designed to break your fix\n"
            "- run_tests: Run the generated tests against your patched code\n"
            "- adversarial_review: Get an adversarial code review\n\n"
            "REQUIRED STEPS (in this order):\n"
            "1. Explore code and fix the bug\n"
            "2. Call generate_tests to create adversarial tests\n"
            "3. Call run_tests to execute them\n"
            "4. If any tests fail, fix your code and repeat steps 2-3\n"
            "5. Call adversarial_review for final verification\n"
            "6. Only finish when all tests pass AND review verdict is likely_correct\n\n"
            "DO NOT finish without calling these tools. Skipping verification is not allowed."
        ),
    },
}

# Standard agent system prompt (from agent-harness)
BASE_SYSTEM_PROMPT = """You are a software engineer debugging an issue in a Python repository.

Your task:
1. Read the problem statement carefully
2. Explore the repository to understand the codebase
3. Identify the root cause
4. Edit the source files to fix the bug
5. Verify your fix

Use the tools provided to read files, search code, and make edits.
Be methodical: understand before you edit."""

MAX_TURNS = 30
MODEL = "haiku"  # Default model for agent loop
VERIFIER_MODEL = "haiku"  # Model for verification primitives


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ToolInvocation:
    """Log entry for a verification tool call."""
    turn_number: int
    tool_name: str
    mode: str  # e.g., "adversarial", "confirmatory", ""
    input_summary: str  # truncated input for logging
    output_summary: str  # truncated output for logging
    latency_ms: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ExperimentResult:
    """Result for one issue under one cell."""
    instance_id: str
    cell: str
    cell_description: str
    # Agent outcome
    fix_generated: bool = False
    patch_diff: str = ""
    turns_used: int = 0
    total_tokens: int = 0
    agent_cost_usd: float = 0.0
    # Verification tool usage
    tool_invocations: list = field(default_factory=list)
    num_tool_invocations: int = 0
    tools_used: list = field(default_factory=list)
    first_tool_turn: int = -1  # -1 if never used
    # Generated test quality
    tests_generated: int = 0
    tests_compiled: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    # Review results
    review_verdict: str = ""
    review_score: float = -1.0
    # Timing
    elapsed_s: float = 0.0
    # Verification cost
    verification_cost_usd: float = 0.0
    # Error
    error: str = ""


# ---------------------------------------------------------------------------
# Verification Tool Wrappers (called when agent invokes them)
# ---------------------------------------------------------------------------

def handle_generate_tests(
    problem_statement: str,
    diff: str,
    mode: str,
    workspace: str,
) -> dict:
    """Handle agent's call to generate_tests. Returns test code + metadata."""
    from generate_tests import generate_tests
    result = generate_tests(
        problem_statement=problem_statement,
        diff=diff,
        mode=mode,
        model=VERIFIER_MODEL,
    )
    return result


def handle_run_tests(
    test_code: str,
    workspace: str,
) -> dict:
    """Handle agent's call to run_tests."""
    from run_tests import run_tests
    return run_tests(test_code=test_code, workspace=workspace)


def handle_adversarial_review(
    problem_statement: str,
    diff: str,
    test_results: str = "",
) -> dict:
    """Handle agent's call to adversarial_review."""
    from adversarial_review import adversarial_review
    return adversarial_review(
        problem_statement=problem_statement,
        diff=diff,
        model=VERIFIER_MODEL,
        test_results=test_results,
    )


# ---------------------------------------------------------------------------
# Agent Loop with Verification Primitives
# ---------------------------------------------------------------------------

async def run_agent_with_primitives(
    issue: Issue,
    workspace: str,
    cell_name: str,
    cell_config: dict,
    endpoint: str = "",
    model: str = "",
) -> ExperimentResult:
    """
    Run the agent loop on one issue with verification primitives available.

    For this experiment we use the Anthropic API directly (not vLLM).
    The agent is a simple tool-use loop using Claude Haiku/Sonnet.
    """
    import boto3

    result = ExperimentResult(
        instance_id=issue.instance_id,
        cell=cell_name,
        cell_description=cell_config["description"],
    )

    start = time.monotonic()

    # Build system prompt with cell-specific suffix
    system_prompt = BASE_SYSTEM_PROMPT + cell_config["system_suffix"]

    # Build tool definitions
    tools = _build_standard_tools()
    verification_tools = _build_verification_tools(cell_config["tools"])
    all_tools = tools + verification_tools

    # Agent conversation
    messages = [
        {"role": "user", "content": f"## Problem Statement\n\n{issue.problem_statement[:6000]}\n\n## Repository\n\nThe repository is checked out at `{workspace}`. Explore the code and fix the bug."},
    ]

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    model_id = f"us.anthropic.claude-{MODEL}-4-5-20251001-v1:0" if MODEL == "haiku" else f"us.anthropic.claude-{MODEL}-4-6"

    last_generated_tests = ""  # Track last generated test code for chaining

    # Support single checkpoint (legacy) or multiple checkpoints
    checkpoints_fired = set()
    checkpoints = cell_config.get("checkpoints", [])
    if not checkpoints and cell_config.get("checkpoint_at_pct"):
        checkpoints = [{"pct": cell_config["checkpoint_at_pct"], "msg": cell_config["checkpoint_msg"]}]

    for turn in range(MAX_TURNS):
        # Fire any checkpoints whose turn threshold has been reached
        for ci, cp in enumerate(checkpoints):
            if ci not in checkpoints_fired and turn >= int(MAX_TURNS * cp["pct"]):
                msg = cp["msg"].format(turns_left=MAX_TURNS - turn)
                messages.append({"role": "user", "content": [{"type": "text", "text": msg}]})
                checkpoints_fired.add(ci)

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages,
                "tools": all_tools,
            }

            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            resp = json.loads(response["body"].read())
            usage = resp.get("usage", {})
            result.total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            # Estimate agent cost
            pricing = {"haiku": (0.80, 4.00), "sonnet": (3.00, 15.00)}
            ip, op = pricing.get(MODEL, (0.80, 4.00))
            result.agent_cost_usd += (usage.get("input_tokens", 0) * ip + usage.get("output_tokens", 0) * op) / 1_000_000

            # Process response
            content = resp.get("content", [])
            stop_reason = resp.get("stop_reason", "")

            # Add assistant message
            messages.append({"role": "assistant", "content": content})

            if stop_reason == "end_turn":
                # Agent is done
                result.turns_used = turn + 1
                break

            if stop_reason != "tool_use":
                result.turns_used = turn + 1
                break

            # Process tool calls
            tool_results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue

                tool_name = block["name"]
                tool_input = block.get("input", {})
                tool_id = block["id"]

                tool_output = _execute_tool(
                    tool_name, tool_input, workspace, issue,
                    cell_config, result, turn, last_generated_tests,
                )

                # Track generated tests for chaining
                if tool_name == "generate_tests" and "test_code" in tool_output:
                    tc = tool_output.get("test_code", "")
                    if tc:
                        last_generated_tests = tc

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(tool_output)[:8000],
                })

            messages.append({"role": "user", "content": tool_results})

        except Exception as e:
            result.error = str(e)[:500]
            log.error(f"[{issue.instance_id}] Turn {turn} error: {e}")
            break

    # Capture final diff
    try:
        proc = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, timeout=10, cwd=workspace,
        )
        result.patch_diff = proc.stdout
        result.fix_generated = len(proc.stdout.strip()) > 0
    except Exception:
        pass

    result.turns_used = result.turns_used or MAX_TURNS
    result.elapsed_s = time.monotonic() - start
    result.num_tool_invocations = len(result.tool_invocations)
    result.tools_used = list(set(
        inv.tool_name for inv in result.tool_invocations
        if inv.tool_name in ("generate_tests", "run_tests", "adversarial_review")
    ))

    return result


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    workspace: str,
    issue: Issue,
    cell_config: dict,
    result: ExperimentResult,
    turn: int,
    last_generated_tests: str,
) -> dict:
    """Execute a tool call and return the result."""

    # Standard tools
    if tool_name == "read_file":
        return _tool_read_file(tool_input, workspace)
    elif tool_name == "edit_file":
        return _tool_edit_file(tool_input, workspace)
    elif tool_name == "write_file":
        return _tool_write_file(tool_input, workspace)
    elif tool_name == "run_command":
        return _tool_run_command(tool_input, workspace)
    elif tool_name == "list_files":
        return _tool_list_files(tool_input, workspace)
    elif tool_name == "search_files":
        return _tool_search_files(tool_input, workspace)

    # Verification primitives
    elif tool_name == "generate_tests":
        mode = tool_input.get("mode", "adversarial")
        # Force mode based on cell config
        if "generate_tests_confirmatory" in cell_config["tools"] and "generate_tests_adversarial" not in cell_config["tools"]:
            mode = "confirmatory"
        elif "generate_tests_adversarial" in cell_config["tools"]:
            mode = "adversarial"

        # Get current diff
        try:
            proc = subprocess.run(["git", "diff"], capture_output=True, text=True, timeout=10, cwd=workspace)
            diff = proc.stdout
        except Exception:
            diff = ""

        gen_result = handle_generate_tests(
            problem_statement=issue.problem_statement,
            diff=diff,
            mode=mode,
            workspace=workspace,
        )

        inv = ToolInvocation(
            turn_number=turn,
            tool_name="generate_tests",
            mode=mode,
            input_summary=f"diff_len={len(diff)}",
            output_summary=f"tests_len={len(gen_result.get('test_code', ''))}",
            latency_ms=gen_result.get("latency_ms", 0),
            cost_usd=gen_result.get("cost_usd", 0.0),
            success=gen_result.get("error") is None,
            error=gen_result.get("error", "") or "",
        )
        result.tool_invocations.append(inv)
        result.verification_cost_usd += gen_result.get("cost_usd", 0.0)

        if result.first_tool_turn < 0:
            result.first_tool_turn = turn

        if gen_result.get("test_code"):
            result.tests_generated += 1

        return {
            "test_code": gen_result.get("test_code", ""),
            "mode": mode,
            "note": "Tests generated. Call run_tests to execute them.",
        }

    elif tool_name == "run_tests":
        test_code = tool_input.get("test_code", "") or last_generated_tests
        if not test_code:
            return {"error": "No test code provided. Generate tests first."}

        run_result = handle_run_tests(test_code=test_code, workspace=workspace)

        inv = ToolInvocation(
            turn_number=turn,
            tool_name="run_tests",
            mode="",
            input_summary=f"test_code_len={len(test_code)}",
            output_summary=f"passed={run_result.get('passed', 0)}/{run_result.get('total', 0)}",
            latency_ms=run_result.get("elapsed_ms", 0),
            success=run_result.get("error") is None,
            error=run_result.get("error", "") or "",
        )
        result.tool_invocations.append(inv)

        if result.first_tool_turn < 0:
            result.first_tool_turn = turn

        if not run_result.get("compile_error"):
            result.tests_compiled += 1
        result.tests_run += run_result.get("total", 0)
        result.tests_passed += run_result.get("passed", 0)
        result.tests_failed += run_result.get("failed", 0)

        return {
            "passed": run_result.get("passed", 0),
            "failed": run_result.get("failed", 0),
            "total": run_result.get("total", 0),
            "test_results": run_result.get("test_results", []),
            "stdout": run_result.get("stdout", "")[-2000:],
            "error": run_result.get("error"),
        }

    elif tool_name == "adversarial_review":
        try:
            proc = subprocess.run(["git", "diff"], capture_output=True, text=True, timeout=10, cwd=workspace)
            diff = proc.stdout
        except Exception:
            diff = ""

        test_results = tool_input.get("test_results", "")

        review_result = handle_adversarial_review(
            problem_statement=issue.problem_statement,
            diff=diff,
            test_results=test_results,
        )

        inv = ToolInvocation(
            turn_number=turn,
            tool_name="adversarial_review",
            mode="v009",
            input_summary=f"diff_len={len(diff)}",
            output_summary=f"verdict={review_result.get('verdict', 'unknown')}",
            latency_ms=review_result.get("latency_ms", 0),
            cost_usd=review_result.get("cost_usd", 0.0),
            success=review_result.get("parse_success", False),
            error=review_result.get("error", "") or "",
        )
        result.tool_invocations.append(inv)
        result.verification_cost_usd += review_result.get("cost_usd", 0.0)

        if result.first_tool_turn < 0:
            result.first_tool_turn = turn

        result.review_verdict = review_result.get("verdict", "")
        result.review_score = review_result.get("overall_score", -1.0) or -1.0

        return {
            "verdict": review_result.get("verdict"),
            "overall_score": review_result.get("overall_score"),
            "attack_result": review_result.get("attack_result"),
            "reasoning": review_result.get("reasoning"),
        }

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Standard Tool Implementations
# ---------------------------------------------------------------------------

def _tool_read_file(inp: dict, workspace: str) -> dict:
    path = inp.get("path", "")
    full = os.path.join(workspace, path)
    try:
        content = Path(full).read_text()
        return {"content": content[:20000]}
    except Exception as e:
        return {"error": str(e)}


def _tool_edit_file(inp: dict, workspace: str) -> dict:
    path = inp.get("path", "")
    old = inp.get("old_string", "")
    new = inp.get("new_string", "")
    full = os.path.join(workspace, path)
    try:
        content = Path(full).read_text()
        if old not in content:
            return {"error": f"old_string not found in {path}"}
        updated = content.replace(old, new, 1)
        Path(full).write_text(updated)
        return {"success": True, "path": path}
    except Exception as e:
        return {"error": str(e)}


def _tool_write_file(inp: dict, workspace: str) -> dict:
    path = inp.get("path", "")
    content = inp.get("content", "")
    full = os.path.join(workspace, path)
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        Path(full).write_text(content)
        return {"success": True, "path": path}
    except Exception as e:
        return {"error": str(e)}


def _tool_run_command(inp: dict, workspace: str) -> dict:
    cmd = inp.get("command", "")
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=60, cwd=workspace,
        )
        return {
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (60s)"}
    except Exception as e:
        return {"error": str(e)}


def _tool_list_files(inp: dict, workspace: str) -> dict:
    path = inp.get("path", ".")
    full = os.path.join(workspace, path)
    try:
        entries = sorted(os.listdir(full))[:200]
        return {"files": entries}
    except Exception as e:
        return {"error": str(e)}


def _tool_search_files(inp: dict, workspace: str) -> dict:
    pattern = inp.get("pattern", "")
    path = inp.get("path", ".")
    full = os.path.join(workspace, path)
    try:
        proc = subprocess.run(
            ["grep", "-rn", "--include=*.py", "-l", pattern, full],
            capture_output=True, text=True, timeout=30,
        )
        files = proc.stdout.strip().split("\n")[:50]
        # Make paths relative to workspace
        rel = [f.replace(workspace + "/", "") for f in files if f]
        return {"matches": rel}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool Definitions for Bedrock API
# ---------------------------------------------------------------------------

def _build_standard_tools() -> list:
    """Standard agent tools (same as agent-harness)."""
    return [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "edit_file",
            "description": "Edit a file by replacing old_string with new_string. The old_string must match exactly.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "old_string": {"type": "string", "description": "Exact string to find and replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file, creating it if needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "run_command",
            "description": "Run a shell command in the repository workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "list_files",
            "description": "List files in a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to directory (default: repo root)"},
                },
                "required": [],
            },
        },
        {
            "name": "search_files",
            "description": "Search for a pattern across Python files in the repository.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search for"},
                    "path": {"type": "string", "description": "Subdirectory to search in (default: repo root)"},
                },
                "required": ["pattern"],
            },
        },
    ]


def _build_verification_tools(enabled: list) -> list:
    """Build tool definitions for enabled verification primitives."""
    tools = []

    if any("generate_tests" in t for t in enabled):
        tools.append({
            "name": "generate_tests",
            "description": (
                "Validate your fix by generating edge-case tests. "
                "Catches issues before submission and increases the chance "
                "your patch passes. Returns test code you can run with run_tests."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["confirmatory", "adversarial"],
                        "description": "Test generation mode. 'adversarial' generates thorough edge-case tests for higher confidence.",
                    },
                },
                "required": [],
            },
        })

    if "run_tests" in enabled or any("generate_tests" in t for t in enabled):
        tools.append({
            "name": "run_tests",
            "description": (
                "Run test code against the current state of the repository. "
                "If you just generated tests, they will be run automatically. "
                "Returns pass/fail results per test."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_code": {
                        "type": "string",
                        "description": "Python test code to run. If empty, runs the last generated tests.",
                    },
                },
                "required": [],
            },
        })

    if "adversarial_review" in enabled:
        tools.append({
            "name": "adversarial_review",
            "description": (
                "Get a thorough code review of your patch before submission. "
                "An expert reviewer checks for correctness, edge cases, and completeness. "
                "Returns a verdict: likely_correct, uncertain, or likely_incorrect."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_results": {
                        "type": "string",
                        "description": "Optional: paste test results for additional context.",
                    },
                },
                "required": [],
            },
        })

    return tools


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_one_issue_sync(
    issue: Issue,
    idx: int,
    total: int,
    cell_name: str,
    cell_config: dict,
    workspace_dir: str,
) -> ExperimentResult:
    """Run a single issue synchronously (called from thread pool)."""
    log.info(f"[{idx+1}/{total}] {issue.instance_id} (cell={cell_name})")
    try:
        workspace = setup_workspace(issue, workspace_dir)
        # run_agent_with_primitives is async but uses sync boto3 calls,
        # so we run it in a fresh event loop within this thread
        exp_result = asyncio.run(run_agent_with_primitives(
            issue=issue,
            workspace=workspace,
            cell_name=cell_name,
            cell_config=cell_config,
        ))
        status = "FIX" if exp_result.fix_generated else "NOEDIT"
        tools_used = ",".join(exp_result.tools_used) or "none"
        log.info(f"  [{idx+1}] {issue.instance_id} -> {status} | turns={exp_result.turns_used} | tools={tools_used} | cost=${exp_result.agent_cost_usd + exp_result.verification_cost_usd:.3f}")
        return exp_result
    except Exception as e:
        log.error(f"  [{idx+1}] {issue.instance_id} ERROR: {e}")
        return ExperimentResult(
            instance_id=issue.instance_id,
            cell=cell_name,
            cell_description=cell_config["description"],
            error=str(e)[:500],
        )
    finally:
        # Clean up workspace to avoid disk exhaustion
        ws = os.path.join(workspace_dir, issue.instance_id)
        if os.path.exists(ws):
            shutil.rmtree(ws, ignore_errors=True)


async def _run_one_issue(
    issue: Issue,
    idx: int,
    total: int,
    cell_name: str,
    cell_config: dict,
    workspace_dir: str,
    semaphore: asyncio.Semaphore,
) -> ExperimentResult:
    """Run a single issue with semaphore-limited concurrency via thread pool."""
    async with semaphore:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            _run_one_issue_sync,
            issue, idx, total, cell_name, cell_config, workspace_dir,
        )


def _load_completed(output_path: str) -> set[str]:
    """Load instance IDs already completed (for resume)."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(r["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


async def run_cell(
    cell_name: str,
    issues: list[Issue],
    workspace_dir: str,
    output_path: str,
    concurrency: int = 1,
):
    """Run one experiment cell across all issues with optional concurrency."""
    cell_config = CELLS[cell_name]

    # Resume: skip already-completed issues
    completed = _load_completed(output_path)
    if completed:
        log.info(f"Resuming: {len(completed)} issues already done, skipping them")
        remaining = [iss for iss in issues if iss.instance_id not in completed]
    else:
        remaining = issues

    log.info(f"=== Cell {cell_name}: {cell_config['description']} ===")
    log.info(f"Issues: {len(remaining)} remaining (of {len(issues)} total), Concurrency: {concurrency}, Tools: {cell_config['tools']}")

    if concurrency <= 1:
        # Sequential (original behavior)
        results = []
        for i, issue in enumerate(remaining):
            result = await _run_one_issue(
                issue, len(completed) + i, len(issues),
                cell_name, cell_config, workspace_dir,
                asyncio.Semaphore(1),
            )
            results.append(result)
            # Append incrementally to file
            _append_result(result, output_path)
    else:
        # Concurrent with semaphore
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        for i, issue in enumerate(remaining):
            task = asyncio.create_task(_run_one_issue(
                issue, len(completed) + i, len(issues),
                cell_name, cell_config, workspace_dir,
                semaphore,
            ))
            tasks.append(task)

        # Collect results as they complete and save incrementally
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            _append_result(result, output_path)
            if len(results) % 10 == 0:
                done = len(completed) + len(results)
                fixes = sum(1 for r in results if r.fix_generated)
                log.info(f"  Progress: {done}/{len(issues)} done, {fixes} fixes so far")

    # Final summary (load all results including resumed ones)
    all_results = []
    with open(output_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Create dummy ExperimentResult-like objects for summary
    log.info(f"\n{'='*60}")
    log.info(f"Cell {cell_name} Summary ({len(all_results)} issues)")
    n = len(all_results)
    fixes = sum(1 for r in all_results if r.get("fix_generated"))
    tool_users = sum(1 for r in all_results if r.get("num_tool_invocations", 0) > 0)
    total_cost = sum(r.get("agent_cost_usd", 0) + r.get("verification_cost_usd", 0) for r in all_results)
    log.info(f"  Fix rate: {fixes}/{n} ({100*fixes/max(n,1):.0f}%)")
    log.info(f"  Tool users: {tool_users}/{n} ({100*tool_users/max(n,1):.0f}%)")
    log.info(f"  Total cost: ${total_cost:.2f}")
    log.info(f"{'='*60}\n")
    return results


def _save_results(results: list[ExperimentResult], output_path: str):
    """Save results to JSONL (full overwrite)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            d = asdict(r)
            d["tool_invocations"] = [asdict(inv) if hasattr(inv, "__dataclass_fields__") else inv for inv in r.tool_invocations]
            f.write(json.dumps(d) + "\n")


def _append_result(result: ExperimentResult, output_path: str):
    """Append a single result to JSONL (thread-safe for concurrent writes)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    d = asdict(result)
    d["tool_invocations"] = [asdict(inv) if hasattr(inv, "__dataclass_fields__") else inv for inv in result.tool_invocations]
    with open(output_path, "a") as f:
        f.write(json.dumps(d) + "\n")


def _print_summary(cell_name: str, results: list[ExperimentResult]):
    """Print summary statistics."""
    n = len(results)
    fixes = sum(1 for r in results if r.fix_generated)
    tool_users = sum(1 for r in results if r.num_tool_invocations > 0)
    total_cost = sum(r.agent_cost_usd + r.verification_cost_usd for r in results)
    avg_turns = sum(r.turns_used for r in results) / max(n, 1)

    log.info(f"\n{'='*60}")
    log.info(f"Cell {cell_name} Summary ({n} issues)")
    log.info(f"  Fix rate: {fixes}/{n} ({100*fixes/max(n,1):.0f}%)")
    log.info(f"  Tool users: {tool_users}/{n} ({100*tool_users/max(n,1):.0f}%)")
    log.info(f"  Avg turns: {avg_turns:.1f}")
    log.info(f"  Total cost: ${total_cost:.2f}")

    if tool_users > 0:
        avg_first_tool = sum(r.first_tool_turn for r in results if r.first_tool_turn >= 0) / tool_users
        log.info(f"  Avg first tool turn: {avg_first_tool:.1f}")

        # Tool usage breakdown
        tool_counts = {}
        for r in results:
            for inv in r.tool_invocations:
                tool_counts[inv.tool_name] = tool_counts.get(inv.tool_name, 0) + 1
        for tool, count in sorted(tool_counts.items()):
            log.info(f"  {tool}: {count} invocations")

    log.info(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verification Primitives Experiment")
    parser.add_argument("--cell", choices=list(CELLS.keys()), help="Experiment cell to run")
    parser.add_argument("--run-all", action="store_true", help="Run all cells")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--workspace-dir", default="/tmp/vp-workspaces")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 issues only")
    parser.add_argument("--n-issues", type=int, default=50, help="Number of issues to load (default 50, max 300 for SWE-bench Lite)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of issues to run concurrently (default 1)")
    parser.add_argument("--model", default="haiku", choices=["haiku", "sonnet"])
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()

    global MODEL, MAX_TURNS
    MODEL = args.model
    MAX_TURNS = args.max_turns

    issues = load_subset(size=args.n_issues)
    if args.smoke_test:
        issues = issues[:5]
        log.info(f"Smoke test mode: {len(issues)} issues")
    else:
        log.info(f"Loaded {len(issues)} issues")

    if args.run_all:
        for cell_name in CELLS:
            output = f"results/phase2_{cell_name}.jsonl"
            asyncio.run(run_cell(cell_name, issues, args.workspace_dir, output, concurrency=args.concurrency))
    elif args.cell:
        output = args.output or f"results/phase2_{args.cell}.jsonl"
        asyncio.run(run_cell(args.cell, issues, args.workspace_dir, output, concurrency=args.concurrency))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
