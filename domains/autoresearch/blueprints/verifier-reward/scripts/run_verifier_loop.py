#!/usr/bin/env python3
"""
T6b: Verifier-in-the-loop experiment.

Wraps the SERA harness with the v001∩v009 ensemble as a post-generation skill.
For each issue:
  1. SERA × Devstral generates a patch (normal flow)
  2. v001∩v009 ensemble verifies the patch via Claude Haiku
  3. If rejected, reset workspace and retry (up to N attempts)
  4. Save all candidate diffs + final selected diff

This tests whether using the verifier as a skill improves pass rate
compared to single-shot generation.

Usage:
  # Run on g7e (Devstral served locally, verifier calls Bedrock)
  python3 run_verifier_loop.py \
    --endpoint http://localhost:8000 \
    --model devstral-small-2 \
    --max-attempts 3 \
    --output-dir results/t6b_verifier_loop

  # Dry run (skip verification, just generate)
  python3 run_verifier_loop.py \
    --endpoint http://localhost:8000 \
    --model devstral-small-2 \
    --dry-run

Requires:
  - Devstral served on --endpoint (vLLM OpenAI-compatible)
  - AWS credentials for Bedrock (Claude Haiku verifier)
  - pip install aiohttp boto3 datasets
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
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Import SERA harness infrastructure
HARNESS_DIR = Path(__file__).resolve().parent.parent.parent / "agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_DIR))

from harness_eval import (
    Issue, load_subset, setup_workspace, run_instrumented_loop,
    PHASE1_CONFIGS, _get_git_diff,
)

# Import verifier ensemble
VERIFIER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VERIFIER_DIR))
from run_cross_verifier import run_ensemble, load_problems, VERIFIER_MODELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERA_CONFIG = PHASE1_CONFIGS["D"]  # 30 turns, strict — best Phase 1 config
WORKSPACE_BASE = "/mnt/nvme/sera-workspaces"


def reset_workspace(issue: Issue, workspace: str):
    """Hard-reset workspace to base commit (keeps test patch)."""
    subprocess.run(
        "git checkout -f HEAD && git clean -fd",
        shell=True, capture_output=True, timeout=30, cwd=workspace,
    )


async def run_single_attempt(
    session,
    endpoint: str,
    model: str,
    issue: Issue,
    workspace: str,
    attempt: int,
    prompt_suffix: str = "",
) -> dict:
    """Run one SERA generation attempt and capture the diff."""
    config = dict(SERA_CONFIG)
    config["_prompt_suffix"] = prompt_suffix

    start = time.monotonic()
    try:
        eval_result = await run_instrumented_loop(
            session, endpoint, model, issue, workspace,
            config, f"t6b_attempt{attempt}",
        )
        diff = _get_git_diff(workspace)
        return {
            "attempt": attempt,
            "fix_generated": eval_result.fix_generated or bool(diff),
            "tests_pass": eval_result.tests_pass,
            "turns_used": eval_result.turns_used,
            "diff": diff,
            "latency_s": time.monotonic() - start,
            "error": None,
        }
    except Exception as e:
        return {
            "attempt": attempt,
            "fix_generated": False,
            "tests_pass": False,
            "turns_used": 0,
            "diff": "",
            "latency_s": time.monotonic() - start,
            "error": str(e)[:500],
        }


def verify_patch(problem: str, diff: str) -> dict:
    """Run v001∩v009 ensemble on a patch. Returns ensemble result."""
    if not diff or len(diff.strip()) < 10:
        return {"ensemble_pass": False, "reason": "empty_diff", "total_cost_usd": 0}

    try:
        result = run_ensemble(problem, diff, "haiku")
        return result
    except Exception as e:
        return {"ensemble_pass": False, "reason": f"verifier_error: {e}", "total_cost_usd": 0}


async def run_issue_with_verification(
    endpoint: str,
    model: str,
    issue: Issue,
    problems: dict,
    max_attempts: int,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """Run SERA with verifier-in-the-loop for one issue."""
    import aiohttp

    workspace = setup_workspace(issue, WORKSPACE_BASE)
    problem = problems.get(issue.instance_id, "")
    all_attempts = []
    selected = None

    try:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    reset_workspace(issue, workspace)

                log.info(f"  Attempt {attempt}/{max_attempts}...")
                result = await run_single_attempt(
                    session, endpoint, model, issue, workspace, attempt,
                )

                # Save candidate diff
                if result["diff"]:
                    diff_dir = os.path.join(output_dir, "diffs", "candidates")
                    os.makedirs(diff_dir, exist_ok=True)
                    with open(os.path.join(diff_dir, f"{issue.instance_id}_a{attempt}.diff"), "w") as f:
                        f.write(result["diff"])

                # Verify
                if dry_run or not result["fix_generated"]:
                    verification = {"ensemble_pass": False, "reason": "dry_run" if dry_run else "no_fix"}
                else:
                    log.info(f"  Verifying attempt {attempt}...")
                    verification = verify_patch(problem, result["diff"])

                result["verification"] = verification
                all_attempts.append(result)

                if verification.get("ensemble_pass"):
                    log.info(f"  VERIFIED on attempt {attempt}")
                    selected = result
                    break
                else:
                    reason = verification.get("reason", verification.get("v001_verdict", "rejected"))
                    log.info(f"  Rejected: {reason}")

        # If no attempt was verified, use the last one (fallback)
        if selected is None and all_attempts:
            selected = all_attempts[-1]

        # Save selected diff
        if selected and selected.get("diff"):
            diff_dir = os.path.join(output_dir, "diffs", "sera_verifier_loop")
            os.makedirs(diff_dir, exist_ok=True)
            with open(os.path.join(diff_dir, f"{issue.instance_id}.diff"), "w") as f:
                f.write(selected["diff"])

        total_gen_cost = 0  # self-hosted, $0
        total_verify_cost = sum(
            a.get("verification", {}).get("total_cost_usd", 0) for a in all_attempts
        )

        return {
            "instance_id": issue.instance_id,
            "attempts": len(all_attempts),
            "verified": selected is not None and selected.get("verification", {}).get("ensemble_pass", False),
            "selected_attempt": selected["attempt"] if selected else None,
            "fix_generated": selected["fix_generated"] if selected else False,
            "turns_used": sum(a["turns_used"] for a in all_attempts),
            "total_latency_s": sum(a["latency_s"] for a in all_attempts),
            "verify_cost_usd": round(total_verify_cost, 6),
            "attempt_results": [
                {
                    "attempt": a["attempt"],
                    "fix_generated": a["fix_generated"],
                    "verified": a.get("verification", {}).get("ensemble_pass", False),
                    "v001_verdict": a.get("verification", {}).get("v001_verdict", ""),
                    "v009_lc_count": a.get("verification", {}).get("v009_lc_count", 0),
                }
                for a in all_attempts
            ],
            "error": selected.get("error") if selected else "no_attempts",
        }

    finally:
        if os.path.exists(workspace):
            shutil.rmtree(workspace, ignore_errors=True)


async def main():
    parser = argparse.ArgumentParser(description="T6b: Verifier-in-the-loop experiment")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="vLLM endpoint")
    parser.add_argument("--model", default="devstral-small-2", help="Model name")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max generation attempts per issue")
    parser.add_argument("--output-dir", default="results/t6b_verifier_loop", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit to first N issues")
    parser.add_argument("--issue", type=str, help="Run a single issue")
    parser.add_argument("--resume", action="store_true", help="Skip completed issues")
    parser.add_argument("--dry-run", action="store_true", help="Skip verification (generate only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "t6b_results.jsonl")

    # Load data
    log.info("Loading problems and issues...")
    problems = load_problems()
    issues = load_subset()

    if args.issue:
        issues = [i for i in issues if i.instance_id == args.issue]
    if args.limit:
        issues = issues[:args.limit]

    # Resume
    completed = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                row = json.loads(line)
                completed.add(row["instance_id"])
        log.info(f"Resuming: {len(completed)} already done")

    # Run
    n_verified = 0
    n_fix = 0
    total_verify_cost = 0.0

    for idx, issue in enumerate(issues):
        if issue.instance_id in completed:
            continue

        log.info(f"[{idx+1}/{len(issues)}] {issue.instance_id} ({issue.repo})")

        result = await run_issue_with_verification(
            args.endpoint, args.model, issue, problems,
            args.max_attempts, args.output_dir, args.dry_run,
        )

        if result["fix_generated"]:
            n_fix += 1
        if result["verified"]:
            n_verified += 1
        total_verify_cost += result.get("verify_cost_usd", 0)

        status = "VERIFIED" if result["verified"] else ("FIX" if result["fix_generated"] else "FAIL")
        log.info(
            f"  {status} | attempts={result['attempts']} "
            f"turns={result['turns_used']} "
            f"verify_cost=${result.get('verify_cost_usd', 0):.4f}"
        )

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    done = len(issues) - len(completed)
    log.info(f"\n{'='*60}")
    log.info(f"T6b: Verifier-in-the-loop | Model: {args.model} | Max attempts: {args.max_attempts}")
    log.info(f"Issues: {done} | Fixes: {n_fix} | Verified: {n_verified}")
    log.info(f"Verify cost: ${total_verify_cost:.2f}")
    log.info(f"Results: {output_file}")
    log.info(f"Diffs: {args.output_dir}/diffs/sera_verifier_loop/")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
