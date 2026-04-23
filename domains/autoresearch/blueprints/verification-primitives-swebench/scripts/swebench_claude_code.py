#!/usr/bin/env python3
"""
Orchestrator: Run Claude Code on SWE-bench issues with verification primitives.

For each issue:
  1. Create workspace (git clone at base_commit)
  2. Copy verify/ scripts + CLAUDE.md into workspace
  3. Write problem.txt with issue description
  4. Run Claude Code headless
  5. Extract diff (git diff)
  6. Format as SWE-bench prediction
  7. Collect telemetry

Usage:
    # Run on SWE-bench Lite (full 300)
    python3 swebench_claude_code.py --dataset princeton-nlp/SWE-bench_Lite --output results/predictions_lite.jsonl

    # Run on specific instances (for testing)
    python3 swebench_claude_code.py --dataset princeton-nlp/SWE-bench_Lite \
        --instances django__django-11630 sympy__sympy-20590 \
        --output results/predictions_test.jsonl

    # Resume from where you left off
    python3 swebench_claude_code.py --dataset princeton-nlp/SWE-bench_Lite \
        --output results/predictions_lite.jsonl --resume
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
VERIFY_DIR = SCRIPT_DIR / "verify"
TEMPLATE_DIR = SCRIPT_DIR / "workspace_template"
MODEL_LABEL = "claude-code-verify-primitives"


def load_dataset(dataset_name: str, split: str = "test") -> list[dict]:
    """Load SWE-bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split)
        return list(ds)
    except ImportError:
        log.error("Install datasets: pip install datasets")
        sys.exit(1)


def setup_workspace(issue: dict, workspace_base: str) -> str:
    """Clone repo at base_commit and set up workspace."""
    instance_id = issue["instance_id"]
    repo = issue["repo"]
    base_commit = issue["base_commit"]
    workspace = os.path.join(workspace_base, instance_id)

    if os.path.exists(workspace):
        shutil.rmtree(workspace)

    # Clone repo
    repo_url = f"https://github.com/{repo}.git"
    log.info(f"  Cloning {repo} at {base_commit[:8]}...")
    proc = subprocess.run(
        ["git", "clone", "--quiet", repo_url, workspace],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Clone failed: {proc.stderr[:300]}")

    # Checkout base commit
    subprocess.run(
        ["git", "checkout", "--quiet", base_commit],
        cwd=workspace, capture_output=True, text=True, timeout=30,
    )

    # Copy verify/ scripts
    dest_verify = os.path.join(workspace, "verify")
    shutil.copytree(str(VERIFY_DIR), dest_verify)

    # Copy CLAUDE.md
    shutil.copy2(str(TEMPLATE_DIR / "CLAUDE.md"), os.path.join(workspace, "CLAUDE.md"))

    # Write problem.txt
    problem_text = issue.get("problem_statement", "")
    Path(os.path.join(workspace, "problem.txt")).write_text(problem_text)

    return workspace


def run_claude_code(workspace: str, problem_statement: str, timeout: int = 600) -> dict:
    """Run Claude Code headless on the workspace."""
    prompt = f"""Fix this GitHub issue in the current repository.

The issue description is in problem.txt. Read it first, then explore the codebase, make your fix, and verify it using the tools in verify/.

Issue summary: {problem_statement[:2000]}"""

    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--max-turns", "30",
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout, cwd=workspace,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONINTERACTIVE_HINT": "1"},
        )
        elapsed = time.monotonic() - start

        # Parse JSON output — may be multi-line JSON or streaming
        output = {}
        if proc.stdout.strip():
            try:
                output = json.loads(proc.stdout)
            except json.JSONDecodeError:
                # Try last line (streaming output may have multiple JSON objects)
                for line in reversed(proc.stdout.strip().split("\n")):
                    try:
                        output = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                if not output:
                    output = {"raw_output": proc.stdout[-3000:]}

        return {
            "success": proc.returncode == 0,
            "elapsed_s": round(elapsed, 1),
            "output": output,
            "stderr": proc.stderr[-1000:] if proc.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed_s": timeout,
            "output": {},
            "stderr": f"Timeout after {timeout}s",
        }


def extract_diff(workspace: str) -> str:
    """Extract git diff from workspace."""
    proc = subprocess.run(
        ["git", "diff"],
        capture_output=True, text=True, cwd=workspace, timeout=30,
    )
    return proc.stdout


def collect_telemetry(workspace: str) -> list[dict]:
    """Collect telemetry.jsonl from workspace."""
    telemetry_path = os.path.join(workspace, "verify", "telemetry.jsonl")
    if not os.path.exists(telemetry_path):
        return []
    entries = []
    with open(telemetry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_completed_ids(output_path: str) -> set:
    """Get instance IDs already in the output file (for resume)."""
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["instance_id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Run Claude Code on SWE-bench with verification primitives")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path")
    parser.add_argument("--telemetry-dir", default=None,
                        help="Directory for per-issue telemetry (default: results/telemetry/)")
    parser.add_argument("--workspace-base", default="/tmp/swebench-workspaces",
                        help="Base directory for workspaces")
    parser.add_argument("--instances", nargs="*", help="Specific instance IDs to run")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per issue in seconds (default: 600)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip issues already in output file")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    # Defaults
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    telemetry_dir = args.telemetry_dir or os.path.join(output_dir or "results", "telemetry")
    os.makedirs(telemetry_dir, exist_ok=True)
    os.makedirs(args.workspace_base, exist_ok=True)

    # Check Claude Code is installed
    if not args.dry_run:
        proc = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("Claude Code not found. Install: npm install -g @anthropic-ai/claude-code")
            sys.exit(1)
        log.info(f"Claude Code version: {proc.stdout.strip()}")

    # Load dataset
    log.info(f"Loading dataset {args.dataset} ({args.split})...")
    issues = load_dataset(args.dataset, args.split)
    log.info(f"Loaded {len(issues)} issues")

    # Filter to specific instances if requested
    if args.instances:
        instance_set = set(args.instances)
        issues = [i for i in issues if i["instance_id"] in instance_set]
        log.info(f"Filtered to {len(issues)} specified instances")

    # Resume support
    completed = set()
    if args.resume:
        completed = get_completed_ids(args.output)
        log.info(f"Resuming: {len(completed)} already completed, {len(issues) - len(completed)} remaining")

    # Build work queue (skip completed)
    work = []
    for idx, issue in enumerate(issues):
        if issue["instance_id"] not in completed:
            work.append((idx, issue))

    if args.dry_run:
        for idx, issue in work:
            log.info(f"  [DRY RUN] [{idx+1}/{len(issues)}] {issue['instance_id']}")
        return

    # Thread-safe output writer
    output_lock = threading.Lock()

    def process_issue(idx: int, issue: dict) -> str:
        instance_id = issue["instance_id"]
        log.info(f"[{idx + 1}/{len(issues)}] {instance_id}")

        try:
            workspace = setup_workspace(issue, args.workspace_base)

            log.info(f"  [{instance_id}] Running Claude Code...")
            cc_result = run_claude_code(workspace, issue.get("problem_statement", ""), args.timeout)
            log.info(f"  [{instance_id}] Done in {cc_result['elapsed_s']}s (success={cc_result['success']})")

            cc_log_path = os.path.join(telemetry_dir, f"{instance_id}_claude_code.json")
            with open(cc_log_path, "w") as f:
                json.dump(cc_result, f, indent=2, default=str)

            diff = extract_diff(workspace)
            has_diff = bool(diff.strip())
            log.info(f"  [{instance_id}] Diff: {len(diff)} chars" if has_diff else f"  [{instance_id}] No diff produced")

            telemetry = collect_telemetry(workspace)
            tools_used = list({e["tool"] for e in telemetry})
            log.info(f"  [{instance_id}] Tools: {tools_used or 'none'}")

            if telemetry:
                tel_path = os.path.join(telemetry_dir, f"{instance_id}.jsonl")
                with open(tel_path, "w") as f_tel:
                    for entry in telemetry:
                        f_tel.write(json.dumps(entry) + "\n")

            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": MODEL_LABEL,
                "model_patch": diff,
            }
            with output_lock:
                with open(args.output, "a") as f:
                    f.write(json.dumps(prediction) + "\n")

            status = "PATCH" if has_diff else "NO_DIFF"
            log.info(f"  [{instance_id}] -> {status}")
            return status

        except Exception as e:
            log.error(f"  [{instance_id}] Error: {e}")
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": MODEL_LABEL,
                "model_patch": "",
            }
            with output_lock:
                with open(args.output, "a") as f:
                    f.write(json.dumps(prediction) + "\n")
            return "ERROR"

        finally:
            ws = os.path.join(args.workspace_base, instance_id)
            if os.path.exists(ws):
                shutil.rmtree(ws, ignore_errors=True)

    # Run with concurrency
    log.info(f"Running {len(work)} issues with concurrency={args.concurrency}")
    if args.concurrency == 1:
        for idx, issue in work:
            process_issue(idx, issue)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_issue, idx, issue): issue["instance_id"]
                       for idx, issue in work}
            for future in as_completed(futures):
                iid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    log.error(f"  [{iid}] Unhandled: {e}")

    log.info(f"\nDone. Predictions written to {args.output}")
    log.info(f"Telemetry saved to {telemetry_dir}/")

    if os.path.exists(args.output):
        with open(args.output) as f:
            preds = [json.loads(line) for line in f if line.strip()]
        n_patch = sum(1 for p in preds if p["model_patch"].strip())
        log.info(f"Total: {len(preds)} predictions, {n_patch} with patches ({100*n_patch/max(len(preds),1):.0f}%)")


if __name__ == "__main__":
    main()
