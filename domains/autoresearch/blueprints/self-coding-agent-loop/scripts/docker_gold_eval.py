#!/usr/bin/env python3
"""docker_gold_eval.py — SWE-rebench V2 eval harness wrapper.

Rewritten 2026-05-11 after FM-6.3: our instances are from `nebius/SWE-rebench` (v1, 21K tasks),
and v1/V2 use the SWE-rebench-V2 eval harness (github.com/SWE-rebench/SWE-rebench-V2).
That harness wants per-instance task records (instance_id + repo + image_name + install_config +
FAIL_TO_PASS + PASS_TO_PASS + patch + test_patch) joined with our predicted model_patch.

Input from our pipeline: predictions.jsonl rows with {instance_id, model_patch}
Output: gold_eval_results.jsonl with {instance_id, resolved, error}

Designed to run on m7i.4xlarge (swebench-eval box), which has SWE-rebench-V2 cloned at
/home/ubuntu/SWE-rebench-V2 and swebench-env venv with dependencies installed.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="JSONL with {instance_id, model_patch} per row")
    p.add_argument("--output", required=True,
                   help="JSONL result: {instance_id, resolved, error}")
    p.add_argument("--run-id", required=True)
    p.add_argument("--task-dataset", default="nebius/SWE-rebench",
                   help="HF dataset with task metadata (image_name, install_config, tests)")
    p.add_argument("--task-splits", nargs="+", default=["test"],
                   help="Which splits to load from --task-dataset")
    p.add_argument("--rebench-v2-path", default="/home/ubuntu/SWE-rebench-V2")
    p.add_argument("--python", default="/home/ubuntu/swebench-env/bin/python")
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    preds = Path(args.predictions)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = out.parent / f"rebench_run_{args.run_id}"
    workdir.mkdir(exist_ok=True)

    # Step 1: load all task records from the nebius dataset, keep only rows for our instance_ids
    print(f"[docker_gold_eval] loading {args.task_dataset} tasks…", file=sys.stderr)
    import pandas as pd
    from datasets import load_dataset

    pred_records = [json.loads(line) for line in open(preds)]
    wanted = {r["instance_id"] for r in pred_records}
    print(f"[docker_gold_eval] {len(pred_records)} predictions, {len(wanted)} unique instances", file=sys.stderr)

    tasks = []
    for split in args.task_splits:
        ds = load_dataset(args.task_dataset, split=split)
        for row in ds:
            if row["instance_id"] in wanted:
                tasks.append(dict(row))
    print(f"[docker_gold_eval] joined {len(tasks)} task records from {args.task_dataset}", file=sys.stderr)
    missing = wanted - {t["instance_id"] for t in tasks}
    if missing:
        print(f"[docker_gold_eval] WARN: {len(missing)} instance_ids not in task dataset (examples: {list(missing)[:3]})", file=sys.stderr)

    # Step 2: overlay our model_patch as the `prediction_patch` field eval.py consumes via --patches
    pred_map = {r["instance_id"]: r.get("model_patch", "") for r in pred_records}

    # Step 3: write tasks.jsonl for eval.py --json, and patches override file
    tasks_path = workdir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in tasks:
            # Convert numpy arrays/etc to plain JSON
            def to_plain(v):
                if hasattr(v, "tolist"): return [to_plain(x) for x in v.tolist()]
                if isinstance(v, dict): return {k: to_plain(x) for k, x in v.items()}
                if isinstance(v, list): return [to_plain(x) for x in v]
                return v
            row = {k: to_plain(v) for k, v in t.items() if v is not None}
            f.write(json.dumps(row) + "\n")

    patches_path = workdir / "patches.json"
    with open(patches_path, "w") as f:
        json.dump({iid: {"prediction_patch": patch} for iid, patch in pred_map.items()}, f)

    report_path = workdir / "eval_report.json"

    # Step 4: invoke SWE-rebench-V2 eval.py
    cmd = [
        args.python,
        str(Path(args.rebench_v2_path) / "scripts" / "eval.py"),
        "--json", str(tasks_path),
        "--patches", str(patches_path),
        "--max-workers", str(args.max_workers),
        "--report-json", str(report_path),
    ]
    print(f"[docker_gold_eval] running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=False, cwd=workdir)

    # Step 5: parse eval_report.json → our output format
    if not report_path.exists():
        print(f"[docker_gold_eval] ERROR: no report at {report_path}", file=sys.stderr)
        with open(out, "w"): pass
        return

    with open(report_path) as f:
        report = json.load(f)

    # eval.py report schema (from SWE-rebench-V2 source): per-instance {instance_id, resolved_status, test_pass/fail/error}
    # Adapt liberally — field names may vary
    with open(out, "w") as fout:
        for item in (report.get("results") or report.get("instances") or []):
            iid = item.get("instance_id") or item.get("id")
            if not iid:
                continue
            resolved = 1 if item.get("resolved") or item.get("resolved_status") == "resolved" else 0
            error = 1 if item.get("error") or item.get("status") in ("error", "failed") else 0
            fout.write(json.dumps({"instance_id": iid, "resolved": resolved, "error": error}) + "\n")

    # Also write a summary
    summary_path = Path(str(out) + ".summary.json")
    total = sum(1 for _ in open(out))
    with open(out) as f:
        resolved_count = sum(1 for line in f if json.loads(line)["resolved"] == 1)
    summary = {
        "run_id": args.run_id,
        "n": total,
        "resolved": resolved_count,
        "gold_pass_rate": resolved_count / total if total else 0.0,
        "report_path": str(report_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[docker_gold_eval] {total} evaluated, {resolved_count} resolved ({summary['gold_pass_rate']:.1%})", file=sys.stderr)


if __name__ == "__main__":
    main()
