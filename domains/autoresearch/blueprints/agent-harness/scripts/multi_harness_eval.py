#!/usr/bin/env python3
"""
Phase 2: Multi-Harness Comparison for Agent Harness Experiment.

Runs multiple coding agent harnesses against the same SWE-bench Lite subset
and produces a harness leaderboard.

Each harness is invoked via an adapter script that standardizes the interface.

Usage:
  # Run a single harness
  python3 multi_harness_eval.py --harness sera --endpoint http://localhost:9000

  # Run all harnesses
  python3 multi_harness_eval.py --run-all --endpoint http://localhost:9000

  # Generate comparison report
  python3 multi_harness_eval.py --report results/phase2_*.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Import shared infrastructure from Phase 1
sys.path.insert(0, os.path.dirname(__file__))
from harness_eval import (
    Issue, load_subset, setup_workspace, run_instrumented_loop,
    PHASE1_CONFIGS, _check_tests_pass, _get_git_diff, FIX_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Prompt Variants (T6: Adversarial Self-Critique in Generation)
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "control": "",
    "self-critique": (
        "\n\nAfter writing your fix, review it critically: assume the patch is wrong "
        "and try to find a bug. If you find one, fix it before finishing."
    ),
    "self-critique-strong": (
        "\n\nIMPORTANT: Before you finish, you MUST do a self-review. "
        "Assume your patch is incorrect. Try to construct an input that would make "
        "the patched code fail. If you find a plausible failure, fix the patch. "
        "Only finish when you cannot break your own fix."
    ),
}

# ---------------------------------------------------------------------------
# Harness Definitions
# ---------------------------------------------------------------------------

HARNESSES = {
    "sera": {
        "name": "SERA (baseline)",
        "type": "builtin",  # uses our own agent loop
        "description": "Custom Python agent loop from sera-datagen.py",
    },
    "claude-code": {
        "name": "Claude Code",
        "type": "cli",
        "install_cmd": "npm install -g @anthropic-ai/claude-code",
        "check_cmd": "claude --version",
        "description": "Anthropic CLI agent with auto-compaction",
    },
    "openhands": {
        "name": "OpenHands",
        "type": "cli",
        "install_cmd": "pip install openhands-ai",
        "check_cmd": "python3 -c 'import openhands'",
        "description": "CodeAct paradigm — Jupyter + bash sandbox",
    },
    "swe-agent": {
        "name": "SWE-agent",
        "type": "cli",
        "install_cmd": "pip install sweagent",
        "check_cmd": "sweagent --version 2>/dev/null || python3 -c 'import sweagent'",
        "description": "Purpose-built Agent-Computer Interface for SWE-bench",
    },
    "aider": {
        "name": "Aider",
        "type": "cli",
        "install_cmd": "pip install aider-chat",
        "check_cmd": "aider --version",
        "description": "Edit-focused with repo-map and architect mode",
    },
    "opencode": {
        "name": "OpenCode",
        "type": "cli",
        "install_cmd": "curl -fsSL https://opencode.ai/install | bash",
        "check_cmd": "opencode --version",
        "description": "Terminal coding agent (TypeScript/Bun), 75+ LLM providers",
    },
    "langgraph": {
        "name": "LangGraph ReAct",
        "type": "script",
        "install_cmd": "pip install langgraph langchain-openai",
        "check_cmd": "python3 -c 'import langgraph'",
        "description": "ReAct agent with standard tool use",
    },
    "deepagents": {
        "name": "DeepAgents",
        "type": "script",
        "install_cmd": "pip install deepagents langchain-openai",
        "check_cmd": "python3 -c 'import deepagents'",
        "description": "Batteries-included LangGraph agent with planning, sub-agents, auto-summarization",
    },
    "piagent": {
        "name": "Pi Agent",
        "type": "script",
        "install_cmd": "npm install -g @mariozechner/pi-coding-agent",
        "check_cmd": "pi --version",
        "description": "Base Pi framework — str_replace edits, skills, sub-agents (control for oh-my-pi)",
    },
    "ohmypi": {
        "name": "oh-my-pi",
        "type": "script",
        "install_cmd": "npm install -g @oh-my-pi/pi-coding-agent",
        "check_cmd": "omp --version",
        "description": "Hash-anchored LINE#ID edits, LSP, ast_grep, sub-agents (the 'harness problem' framework)",
    },
    "letta-code": {
        "name": "Letta Code",
        "type": "script",
        "install_cmd": "npm install -g @letta-ai/letta-code && npm install -g @letta-ai/letta-code-sdk",
        "check_cmd": "node -e \"require('@letta-ai/letta-code-sdk')\" 2>/dev/null || letta --version",
        "description": "Memory-first agent with persistent memory, skill learning, sleeptime (self-hosted)",
    },
    "codex": {
        "name": "Codex CLI",
        "type": "cli",
        "install_cmd": "npm install -g @openai/codex",
        "check_cmd": "codex --version",
        "description": "OpenAI coding agent using Responses API, sandboxed shell + file tools",
    },
    "droid": {
        "name": "Factory Droid",
        "type": "cli",
        "install_cmd": "npm install -g droid",
        "check_cmd": "droid --version",
        "description": "Factory AI coding agent with custom vLLM support via generic-chat-completion-api",
    },
}


@dataclass
class HarnessResult:
    """Result for one issue under one harness."""
    instance_id: str
    harness: str
    tests_pass: bool = False
    fix_generated: bool = False
    turns_used: int = 0
    tokens_consumed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0
    error: Optional[str] = None
    # Not serialized to summary JSONL — saved to separate files
    diff: Optional[str] = None
    trajectory: Optional[list] = None


# ---------------------------------------------------------------------------
# Harness Adapters
# ---------------------------------------------------------------------------


def check_harness_installed(harness_id: str) -> bool:
    """Check if a harness is installed and available."""
    harness = HARNESSES[harness_id]
    if harness["type"] == "builtin":
        return True

    check_cmd = harness.get("check_cmd", "")
    if not check_cmd:
        return False

    proc = subprocess.run(
        check_cmd, shell=True, capture_output=True, text=True, timeout=10,
    )
    return proc.returncode == 0


def install_harness(harness_id: str) -> bool:
    """Install a harness. Returns True if successful."""
    harness = HARNESSES[harness_id]
    install_cmd = harness.get("install_cmd", "")
    if not install_cmd:
        return False

    log.info(f"Installing {harness['name']}: {install_cmd}")
    proc = subprocess.run(
        install_cmd, shell=True, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        log.error(f"Install failed: {proc.stderr[:500]}")
        return False
    return True


async def run_sera(
    issue: Issue,
    workspace: str,
    endpoint: str,
    model: str,
    best_turn_config: dict,
    prompt_variant: str = "control",
) -> HarnessResult:
    """Run SERA baseline (our own agent loop with best Phase 1 config)."""
    import aiohttp

    result = HarnessResult(instance_id=issue.instance_id, harness="sera")
    start = time.monotonic()

    try:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Inject prompt variant into config for system prompt modification
            config = dict(best_turn_config)
            config["_prompt_suffix"] = PROMPT_VARIANTS.get(prompt_variant, "")
            eval_result = await run_instrumented_loop(
                session, endpoint, model, issue, workspace,
                config, "sera",
            )
            result.tests_pass = eval_result.tests_pass
            result.fix_generated = eval_result.fix_generated
            result.turns_used = eval_result.turns_used
            result.tokens_consumed = sum(
                tm.get("context_tokens", 0) for tm in eval_result.turn_metrics
            )
    except Exception as e:
        result.error = str(e)[:500]

    result.total_latency_ms = (time.monotonic() - start) * 1000
    return result


def run_cli_harness(
    harness_id: str,
    issue: Issue,
    workspace: str,
    endpoint: str,
    model: str,
) -> HarnessResult:
    """
    Run a CLI-based harness via adapter script.

    Adapter scripts live at scripts/adapters/run_<harness>.sh and must:
    1. Accept: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD as env vars
    2. Output a JSON object on the last line: {"pass": bool, "turns": int, "tokens": int}
    """
    result = HarnessResult(instance_id=issue.instance_id, harness=harness_id)
    start = time.monotonic()

    adapter_script = os.path.join(
        os.path.dirname(__file__), "adapters", f"run_{harness_id.replace('-', '_')}.sh"
    )

    if not os.path.isfile(adapter_script):
        result.error = f"Adapter script not found: {adapter_script}"
        log.warning(f"[{issue.instance_id}] {result.error}")
        return result

    env = os.environ.copy()
    env.update({
        "WORKSPACE": workspace,
        "ENDPOINT": endpoint,
        "MODEL": model,
        "ISSUE_ID": issue.instance_id,
        "PROBLEM_STATEMENT": issue.problem_statement[:10000],
        "TEST_CMD": issue.test_cmd,
        "REPO": issue.repo,
        "PROMPT_VARIANT": os.environ.get("PROMPT_VARIANT", "control"),
    })

    # Tell adapter where to save raw trajectory events
    traj_file = os.path.join(
        os.environ.get("TRAJECTORY_DIR", "/tmp"),
        f"{harness_id}_{issue.instance_id}.jsonl",
    )
    env["TRAJECTORY_FILE"] = traj_file

    try:
        proc = subprocess.run(
            ["bash", adapter_script],
            capture_output=True, text=True,
            timeout=600,  # 10 min per issue
            cwd=workspace,
            env=env,
        )

        # Parse JSON result from last non-empty line
        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        if lines:
            try:
                output = json.loads(lines[-1])
                result.tests_pass = output.get("pass", False)
                result.turns_used = output.get("turns", 0)
                result.tokens_consumed = output.get("tokens", 0)
                result.input_tokens = output.get("input_tokens", 0)
                result.output_tokens = output.get("output_tokens", 0)
                result.fix_generated = output.get("fix_generated", result.tests_pass)
            except json.JSONDecodeError:
                result.error = f"Could not parse adapter output: {lines[-1][:200]}"
        else:
            result.error = "No output from adapter script"

        if proc.returncode != 0 and not result.error:
            result.error = f"Exit code {proc.returncode}: {proc.stderr[:300]}"

    except subprocess.TimeoutExpired:
        result.error = "Adapter timed out (600s)"
    except Exception as e:
        result.error = str(e)[:500]

    result.total_latency_ms = (time.monotonic() - start) * 1000

    # Capture git diff
    diff = _get_git_diff(workspace)
    if diff:
        result.diff = diff
        result.fix_generated = True
    elif not result.fix_generated:
        result.fix_generated = False

    # Load raw trajectory if adapter saved one
    if os.path.isfile(traj_file):
        try:
            with open(traj_file) as tf:
                result.trajectory = [json.loads(l) for l in tf if l.strip()]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


async def run_harness(
    harness_id: str,
    issues: list[Issue],
    endpoint: str,
    model: str,
    workspace_dir: str,
    output_path: str,
    best_turn_config: dict,
    prompt_variant: str = "control",
):
    """Run one harness against all issues."""
    harness = HARNESSES[harness_id]
    log.info(f"=== Harness: {harness['name']} ({harness['description']}) ===")

    # Set trajectory dir so adapters can persist raw events
    traj_raw_dir = os.path.join(os.path.dirname(output_path), "trajectories", "raw")
    os.makedirs(traj_raw_dir, exist_ok=True)
    os.environ["TRAJECTORY_DIR"] = traj_raw_dir

    results = []
    for i, issue in enumerate(issues):
        log.info(f"  [{i+1}/{len(issues)}] {issue.instance_id}")
        workspace = setup_workspace(issue, workspace_dir)

        try:
            if harness["type"] == "builtin":
                result = await run_sera(issue, workspace, endpoint, model, best_turn_config, prompt_variant)
            else:
                os.environ["PROMPT_VARIANT"] = prompt_variant
                result = run_cli_harness(harness_id, issue, workspace, endpoint, model)
            results.append(result)
        except Exception as e:
            results.append(HarnessResult(
                instance_id=issue.instance_id,
                harness=harness_id,
                error=str(e)[:500],
            ))
        finally:
            if os.path.exists(workspace):
                import shutil
                shutil.rmtree(workspace, ignore_errors=True)

    # Write results — diffs and trajectories go to separate files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diff_dir = os.path.join(os.path.dirname(output_path), "diffs", harness_id)
    traj_dir = os.path.join(os.path.dirname(output_path), "trajectories", harness_id)
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for r in results:
            # Save diff to separate file
            if r.diff:
                diff_path = os.path.join(diff_dir, f"{r.instance_id}.diff")
                with open(diff_path, "w") as df:
                    df.write(r.diff)

            # Save trajectory to separate file
            if r.trajectory:
                traj_path = os.path.join(traj_dir, f"{r.instance_id}.jsonl")
                with open(traj_path, "w") as tf:
                    for evt in r.trajectory:
                        tf.write(json.dumps(evt) + "\n")

            # Strip heavy fields from summary JSONL
            d = asdict(r)
            d.pop("diff", None)
            d.pop("trajectory", None)
            f.write(json.dumps(d) + "\n")

    # Summary
    n_pass = sum(1 for r in results if r.tests_pass)
    n_fix = sum(1 for r in results if r.fix_generated)
    n_err = sum(1 for r in results if r.error)
    avg_turns = sum(r.turns_used for r in results) / max(len(results), 1)
    avg_tokens = sum(r.tokens_consumed for r in results) / max(n_pass, 1) if n_pass else 0

    summary = {
        "harness": harness_id,
        "name": harness["name"],
        "pass_rate": n_pass / max(len(results), 1),
        "passes": n_pass,
        "fixes": n_fix,
        "errors": n_err,
        "total": len(results),
        "avg_turns": avg_turns,
        "tokens_per_fix": avg_tokens,
    }

    log.info(
        f"=== {harness['name']} DONE ===\n"
        f"  Pass rate: {n_pass}/{len(results)} ({100*summary['pass_rate']:.1f}%)\n"
        f"  Fix generated: {n_fix}/{len(results)}\n"
        f"  Errors: {n_err}\n"
        f"  Avg turns: {avg_turns:.1f}\n"
        f"  Tokens/fix: {avg_tokens:.0f}\n"
        f"  Output: {output_path}"
    )

    return summary


def generate_report(result_files: list[str]):
    """Generate comparison report from Phase 2 result files."""
    summaries = []

    for filepath in result_files:
        results = []
        with open(filepath) as f:
            for line in f:
                results.append(json.loads(line))

        if not results:
            continue

        harness = results[0].get("harness", "unknown")
        n_pass = sum(1 for r in results if r.get("tests_pass"))
        n_fix = sum(1 for r in results if r.get("fix_generated"))
        n_err = sum(1 for r in results if r.get("error"))
        avg_turns = sum(r.get("turns_used", 0) for r in results) / max(len(results), 1)
        total_tokens = sum(r.get("tokens_consumed", 0) for r in results)
        tokens_per_fix = total_tokens / max(n_pass, 1) if n_pass else 0

        summaries.append({
            "harness": harness,
            "pass_rate": n_pass / max(len(results), 1),
            "passes": n_pass,
            "fixes": n_fix,
            "errors": n_err,
            "total": len(results),
            "avg_turns": avg_turns,
            "tokens_per_fix": tokens_per_fix,
        })

    # Sort by pass rate descending
    summaries.sort(key=lambda s: s["pass_rate"], reverse=True)

    print("\n=== PHASE 2: HARNESS LEADERBOARD ===")
    print(f"{'Rank':<5} {'Harness':<20} {'Pass Rate':<12} {'Fixes':<8} {'Avg Turns':<12} {'Tok/Fix':<10} {'Errors':<8}")
    print("-" * 75)
    for i, s in enumerate(summaries):
        print(
            f"{i+1:<5} {s['harness']:<20} "
            f"{100*s['pass_rate']:>5.1f}%      "
            f"{s['fixes']}/{s['total']:<5} "
            f"{s['avg_turns']:>6.1f}      "
            f"{s['tokens_per_fix']:>8.0f}  "
            f"{s['errors']}"
        )

    if len(summaries) >= 2:
        best = summaries[0]
        sera = next((s for s in summaries if s["harness"] == "sera"), None)
        print(f"\nBest: {best['harness']} ({100*best['pass_rate']:.1f}%)")
        if sera:
            print(f"SERA baseline: {100*sera['pass_rate']:.1f}%")
            delta = best["pass_rate"] - sera["pass_rate"]
            print(f"Best vs SERA: {100*delta:+.1f}pp")

    print("=== END ===\n")


async def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Harness Comparison")
    parser.add_argument("--endpoint", default="http://localhost:9000", help="Model endpoint")
    parser.add_argument("--model", default="devstral-small-2", help="Model name")
    parser.add_argument("--harness", help="Harness to run")
    parser.add_argument("--run-all", action="store_true", help="Run all harnesses")
    parser.add_argument("--report", nargs="+", help="Generate report from result files")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--workspace-dir", default="/mnt/nvme/sera-workspaces", help="Workspace dir")
    parser.add_argument("--best-config", default="D", help="Best Phase 1 config for SERA baseline")
    parser.add_argument("--check-installed", action="store_true", help="Check which harnesses are installed")
    parser.add_argument("--shard", type=str, default=None,
                        help="Run a subset of issues: SHARD_ID/NUM_SHARDS (e.g. 0/4, 1/4)")
    parser.add_argument("--prompt-variant", default="control",
                        choices=list(PROMPT_VARIANTS.keys()),
                        help="Prompt variant for T6 self-critique experiment")
    args = parser.parse_args()

    if args.report:
        generate_report(args.report)
        return

    # Pre-flight smoke test — abort early if endpoint is broken
    if not args.check_installed:
        smoke_script = os.path.join(os.path.dirname(__file__), "smoke_test.py")
        if os.path.isfile(smoke_script):
            log.info("Running pre-flight smoke test...")
            smoke = subprocess.run(
                [sys.executable, smoke_script,
                 "--endpoint", args.endpoint, "--model", args.model,
                 "--workspace-dir", args.workspace_dir],
                timeout=300,
            )
            if smoke.returncode != 0:
                log.error("Smoke test FAILED — aborting eval. Fix the issues above before retrying.")
                sys.exit(1)
            log.info("Smoke test passed.")

    if args.check_installed:
        for hid, h in HARNESSES.items():
            installed = check_harness_installed(hid)
            status = "INSTALLED" if installed else "NOT INSTALLED"
            print(f"  {hid:<15} {status:<15} {h['name']}")
            if not installed and h.get("install_cmd"):
                print(f"                 Install: {h['install_cmd']}")
        return

    # Get best Phase 1 config for SERA
    from harness_eval import PHASE1_CONFIGS
    best_config = PHASE1_CONFIGS.get(args.best_config, PHASE1_CONFIGS["D"])

    issues = load_subset()

    # Shard issues for parallel execution across GPUs
    if args.shard:
        shard_id, num_shards = map(int, args.shard.split("/"))
        issues = [iss for i, iss in enumerate(issues) if i % num_shards == shard_id]
        log.info(f"Shard {shard_id}/{num_shards}: {len(issues)} issues")

    if args.run_all:
        all_summaries = []
        for hid in HARNESSES:
            if not check_harness_installed(hid):
                log.warning(f"Skipping {hid} — not installed")
                if not install_harness(hid):
                    log.error(f"Could not install {hid}, skipping")
                    continue

            output = os.path.join(args.output_dir, f"phase2_{hid}.jsonl")
            summary = await run_harness(
                hid, issues, args.endpoint, args.model,
                args.workspace_dir, output, best_config, args.prompt_variant,
            )
            all_summaries.append(summary)

        # Print leaderboard
        all_summaries.sort(key=lambda s: s["pass_rate"], reverse=True)
        print("\n=== PHASE 2: HARNESS LEADERBOARD ===")
        for i, s in enumerate(all_summaries):
            print(f"  {i+1}. {s['name']}: {100*s['pass_rate']:.1f}% "
                  f"({s['passes']}/{s['total']} pass, {s['avg_turns']:.1f} avg turns)")
        print("=== END ===")
        return

    if args.harness:
        if args.harness not in HARNESSES:
            print(f"Unknown harness: {args.harness}. Choose from: {list(HARNESSES.keys())}")
            sys.exit(1)

        if not check_harness_installed(args.harness):
            if not install_harness(args.harness):
                log.error(f"Could not install {args.harness}")
                sys.exit(1)

        shard_suffix = f"_s{args.shard.split('/')[0]}" if args.shard else ""
        variant_suffix = f"_{args.prompt_variant}" if args.prompt_variant != "control" else ""
        output = os.path.join(args.output_dir, f"phase2_{args.harness}{variant_suffix}{shard_suffix}.jsonl")
        await run_harness(
            args.harness, issues, args.endpoint, args.model,
            args.workspace_dir, output, best_config, args.prompt_variant,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
