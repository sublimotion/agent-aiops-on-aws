#!/usr/bin/env python3
"""
Gold eval on arbitrary diffs directory. Thin wrapper around gold_eval.py logic.

Usage:
  python3 gold_eval_dir.py --diffs-dir results/diffs/devstral_sera_bon_selected \
    --output results/gold_devstral_sera_bon_selected.jsonl --resume
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Reuse gold_eval's core logic
sys.path.insert(0, str(Path(__file__).parent))
from gold_eval import load_subset, run_gold_eval_docker, DOCKER_IMAGE, REPO_CACHE_VOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--issue", type=str, help="Single issue ID")
    args = parser.parse_args()

    diffs_dir = Path(args.diffs_dir)
    output_file = Path(args.output)

    # Pull image and create cache volume
    log.info(f"Ensuring Docker image {DOCKER_IMAGE}...")
    subprocess.run(["docker", "pull", DOCKER_IMAGE], capture_output=True, timeout=300)
    subprocess.run(["docker", "volume", "create", REPO_CACHE_VOL], capture_output=True)

    log.info("Loading SWE-bench Lite subset...")
    issues = load_subset()

    completed = set()
    if args.resume and output_file.exists():
        for line in output_file.read_text().strip().split("\n"):
            if line:
                try:
                    completed.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info(f"Resuming: {len(completed)} already done")

    diff_files = sorted(diffs_dir.glob("*.diff"))
    if args.issue:
        diff_files = [d for d in diff_files if d.stem == args.issue]
    if args.limit:
        diff_files = diff_files[:args.limit]

    total = len(diff_files)
    passed = 0
    evaluated = 0

    log.info(f"Gold eval: {total} diffs from {diffs_dir}")

    for idx, diff_file in enumerate(diff_files):
        instance_id = diff_file.stem
        if instance_id in completed:
            continue
        if instance_id not in issues:
            log.warning(f"[{idx+1}/{total}] {instance_id} not in subset, skipping")
            continue

        issue = issues[instance_id]
        log.info(f"[{idx+1}/{total}] {instance_id} ({issue.repo})")

        result = run_gold_eval_docker(issue, str(diff_file))
        evaluated += 1
        if result["passed"]:
            passed += 1

        status = "PASS" if result["passed"] else "FAIL"
        log.info(f"  {status} | patch={result['patch_applied']} time={result['elapsed_s']}s")

        row = {"instance_id": instance_id, "model": "devstral_sera_bon", **result}
        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    log.info(f"\nResult: {passed}/{evaluated} passed ({100*passed/max(evaluated,1):.0f}%)")


if __name__ == "__main__":
    main()
