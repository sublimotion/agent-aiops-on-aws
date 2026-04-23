#!/usr/bin/env python3
"""
Gold evaluation using local workspaces.

Instead of cloning repos fresh, this script:
1. Uses the experiment's workspace dir (repos already cloned + cached)
2. Resets to base_commit + applies test_patch
3. Applies the agent diff
4. Runs FAIL_TO_PASS tests

Usage:
    python3 gold_eval_local.py --cell control --workspace-dir /tmp/vp-full-control \
        --diffs ../results/diffs/control --output ../results/eval_control.jsonl

    python3 gold_eval_local.py --cell B_checkpoint --workspace-dir /tmp/vp-full-checkpoint \
        --diffs ../results/diffs/B_checkpoint --output ../results/eval_B_checkpoint.jsonl
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HARNESS_DIR = Path(__file__).resolve().parents[2] / "agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_DIR))
from harness_eval import Issue, load_subset, setup_workspace


def apply_patch(workspace: str, patch_content: str, label: str) -> bool:
    """Apply a patch via git apply. Returns True if successful."""
    proc = subprocess.run(
        ["git", "apply", "--allow-empty", "-"],
        input=patch_content, capture_output=True, text=True,
        timeout=30, cwd=workspace,
    )
    if proc.returncode != 0:
        # Try with --3way
        proc = subprocess.run(
            ["git", "apply", "--allow-empty", "--3way", "-"],
            input=patch_content, capture_output=True, text=True,
            timeout=30, cwd=workspace,
        )
    if proc.returncode != 0:
        log.warning(f"  [{label}] apply failed: {proc.stderr[:300]}")
        return False
    return True


def run_tests(workspace: str, test_cmd: str, timeout: int = 180) -> tuple[bool, list[dict]]:
    """Run test command and parse results."""
    venv_activate = os.path.join(workspace, ".venv", "bin", "activate")
    if os.path.exists(venv_activate):
        test_cmd = f"source {venv_activate} && {test_cmd}"
    try:
        proc = subprocess.run(
            test_cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=workspace, executable="/bin/bash",
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, [{"test": test_cmd, "pass": False, "output": "TIMEOUT"}]

    # Primary signal: return code
    passed_all = proc.returncode == 0
    details = [{"test": test_cmd[:200], "pass": passed_all, "output": output[-2000:]}]

    return passed_all, details


def eval_one(issue: Issue, diff_content: str, workspace_base: str) -> dict:
    """Evaluate one issue."""
    result = {
        "instance_id": issue.instance_id,
        "patch_applied": False,
        "tests_pass": False,
        "test_details": [],
        "elapsed": 0,
    }
    start = time.monotonic()

    eval_dir = os.path.join(workspace_base, "_gold_eval")
    os.makedirs(eval_dir, exist_ok=True)

    try:
        workspace = setup_workspace(issue, eval_dir)

        if not apply_patch(workspace, diff_content, "agent"):
            result["elapsed"] = time.monotonic() - start
            return result
        result["patch_applied"] = True

        if issue.test_cmd:
            passed, details = run_tests(workspace, issue.test_cmd)
            result["tests_pass"] = passed
            result["test_details"] = details

    except Exception as e:
        log.error(f"  Error: {e}")
        result["test_details"] = [{"test": "error", "pass": False, "output": str(e)[:500]}]
    finally:
        ws = os.path.join(eval_dir, issue.instance_id)
        if os.path.exists(ws):
            shutil.rmtree(ws, ignore_errors=True)

    result["elapsed"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--workspace-dir", required=True, help="Experiment workspace dir (has _repo_cache)")
    parser.add_argument("--diffs", required=True, help="Directory with .diff files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-issues", type=int, default=50, help="Number of issues to load (default 50, max 300)")
    args = parser.parse_args()

    issues = load_subset(size=args.n_issues)
    issue_map = {iss.instance_id: iss for iss in issues}

    diff_dir = Path(args.diffs)
    diff_files = sorted(diff_dir.glob("*.diff"))
    log.info(f"Found {len(diff_files)} diffs in {diff_dir}")

    results = []
    for i, diff_path in enumerate(diff_files):
        instance_id = diff_path.stem
        issue = issue_map.get(instance_id)
        if not issue:
            log.warning(f"  [{i+1}] {instance_id}: not in subset")
            continue

        log.info(f"  [{i+1}/{len(diff_files)}] {instance_id}")
        diff_content = diff_path.read_text()
        result = eval_one(issue, diff_content, args.workspace_dir)
        result["harness"] = args.cell
        results.append(result)

        status = "PASS" if result["tests_pass"] else ("APPLIED" if result["patch_applied"] else "FAIL")
        log.info(f"    -> {status}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_applied = sum(1 for r in results if r["patch_applied"])
    n_pass = sum(1 for r in results if r["tests_pass"])
    log.info(f"\n{'='*60}")
    log.info(f"Gold eval: {len(results)} issues, applied={n_applied}, pass={n_pass} ({100*n_pass/max(len(results),1):.0f}%)")
    log.info(f"Precision: {n_pass}/{n_applied} ({100*n_pass/max(n_applied,1):.0f}%)")


if __name__ == "__main__":
    main()
