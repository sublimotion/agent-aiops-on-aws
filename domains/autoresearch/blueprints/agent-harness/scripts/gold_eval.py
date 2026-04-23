#!/usr/bin/env python3
"""
Gold test evaluation for agent harness patches.

For each diff produced by a harness:
1. Clone/checkout the repo at base_commit
2. Apply the agent's diff
3. Apply the gold test_patch from SWE-bench
4. Run FAIL_TO_PASS tests
5. Record pass/fail

Usage:
    python3 gold_eval.py --harness droid --diff-dir results/diffs/droid \
        --output results/eval_droid.jsonl --workspace-dir /mnt/nvme/sera-workspaces

    # Shard across GPUs (no GPU needed, but parallelizes I/O)
    python3 gold_eval.py --harness droid --diff-dir results/diffs/droid \
        --shard 0/4 --output results/eval_droid_s0.jsonl
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from harness_eval import Issue, load_subset, setup_workspace


def apply_patch(workspace: str, patch_content: str, label: str) -> bool:
    """Apply a patch via git apply. Returns True if successful."""
    proc = subprocess.run(
        ["git", "apply", "--allow-empty", "-"],
        input=patch_content, capture_output=True, text=True,
        timeout=30, cwd=workspace,
    )
    if proc.returncode != 0:
        # Try with --3way as fallback
        proc = subprocess.run(
            ["git", "apply", "--allow-empty", "--3way", "-"],
            input=patch_content, capture_output=True, text=True,
            timeout=30, cwd=workspace,
        )
    if proc.returncode != 0:
        log.warning(f"  [{label}] patch apply failed: {proc.stderr[:200]}")
        return False
    return True


def run_tests(workspace: str, test_cmd: str, timeout: int = 180) -> tuple[bool, list[dict]]:
    """Run test command and parse results. Returns (all_pass, details)."""
    try:
        proc = subprocess.run(
            test_cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=workspace,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, [{"test": test_cmd, "pass": False, "output": "TIMEOUT"}]

    # Parse individual test results
    details = []
    passed_all = True

    # Check for pytest-style output
    lines = output.split("\n")
    for line in lines:
        if "PASSED" in line or "FAILED" in line or "ERROR" in line:
            test_pass = "PASSED" in line and "FAILED" not in line
            if not test_pass:
                passed_all = False
            details.append({
                "test": line.strip()[:200],
                "pass": test_pass,
                "output": "",
            })

    # Fallback: check overall output if no individual test lines found
    if not details:
        output_lower = output.lower()
        if "passed" in output_lower and "failed" not in output_lower and "error" not in output_lower:
            passed_all = True
        elif "no tests ran" in output_lower:
            passed_all = False
        else:
            passed_all = False
        details.append({
            "test": test_cmd[:200],
            "pass": passed_all,
            "output": output[-2000:],
        })

    return passed_all, details


def eval_one(issue: Issue, diff_path: str, workspace_dir: str) -> dict:
    """Evaluate one issue's patch against gold tests."""
    result = {
        "instance_id": issue.instance_id,
        "harness": "droid",
        "patch_applied": False,
        "tests_pass": False,
        "test_details": [],
        "elapsed": 0,
    }

    start = time.monotonic()

    # Use a separate eval dir so we don't clobber the harness workspace
    eval_base = os.path.join(workspace_dir, "_eval")
    os.makedirs(eval_base, exist_ok=True)
    eval_workspace = os.path.join(eval_base, issue.instance_id)

    try:
        # Setup clean workspace at base commit with test_patch applied
        workspace = setup_workspace(issue, eval_base)

        # Apply agent's diff
        with open(diff_path) as f:
            agent_patch = f.read()

        if not agent_patch.strip():
            log.warning(f"  Empty diff for {issue.instance_id}")
            result["elapsed"] = time.monotonic() - start
            return result

        applied = apply_patch(workspace, agent_patch, "agent_patch")
        result["patch_applied"] = applied

        if not applied:
            result["elapsed"] = time.monotonic() - start
            return result

        # Run FAIL_TO_PASS tests
        if issue.test_cmd:
            passed, details = run_tests(workspace, issue.test_cmd)
            result["tests_pass"] = passed
            result["test_details"] = details
        else:
            log.warning(f"  No test_cmd for {issue.instance_id}")

    except Exception as e:
        log.error(f"  Error evaluating {issue.instance_id}: {e}")
        result["test_details"] = [{"test": "eval_error", "pass": False, "output": str(e)[:500]}]
    finally:
        # Cleanup eval workspace
        if os.path.exists(eval_workspace):
            import shutil
            shutil.rmtree(eval_workspace, ignore_errors=True)

    result["elapsed"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description="Gold test evaluation")
    parser.add_argument("--harness", required=True)
    parser.add_argument("--diff-dir", required=True, help="Directory with .diff files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--workspace-dir", default="/mnt/nvme/sera-workspaces")
    parser.add_argument("--shard", type=str, default=None, help="SHARD_ID/NUM_SHARDS")
    args = parser.parse_args()

    issues = load_subset()
    issue_map = {iss.instance_id: iss for iss in issues}

    # Find diffs
    diff_dir = Path(args.diff_dir)
    diff_files = sorted(diff_dir.glob("*.diff"))
    log.info(f"Found {len(diff_files)} diffs in {diff_dir}")

    # Shard if requested
    if args.shard:
        shard_id, num_shards = map(int, args.shard.split("/"))
        diff_files = [d for i, d in enumerate(diff_files) if i % num_shards == shard_id]
        log.info(f"Shard {shard_id}/{num_shards}: {len(diff_files)} diffs")

    results = []
    for i, diff_path in enumerate(diff_files):
        instance_id = diff_path.stem  # foo__bar-1234.diff → foo__bar-1234
        issue = issue_map.get(instance_id)
        if not issue:
            log.warning(f"  [{i+1}/{len(diff_files)}] {instance_id}: not in subset, skipping")
            continue

        log.info(f"  [{i+1}/{len(diff_files)}] {instance_id}")
        result = eval_one(issue, str(diff_path), args.workspace_dir)
        result["harness"] = args.harness
        results.append(result)

        status = "PASS" if result["tests_pass"] else ("APPLIED" if result["patch_applied"] else "FAIL")
        log.info(f"    → {status}")

    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    n_applied = sum(1 for r in results if r["patch_applied"])
    n_pass = sum(1 for r in results if r["tests_pass"])
    log.info(f"\n{'='*60}")
    log.info(f"Eval complete: {len(results)} issues")
    log.info(f"Patches applied: {n_applied}/{len(results)}")
    log.info(f"Tests pass: {n_pass}/{len(results)} ({100*n_pass/max(len(results),1):.0f}%)")
    log.info(f"Precision: {n_pass}/{n_applied} ({100*n_pass/max(n_applied,1):.0f}%)")
    log.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
