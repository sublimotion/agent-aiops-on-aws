#!/usr/bin/env python3
"""
Gold evaluation: Run SWE-bench Docker eval on patches from SERA harness results.

Usage:
    python3 run_gold_eval.py --input results/swarm_phase1_qwen3-235b-sft-d_sera.jsonl \
                             --output results/gold_eval_qwen3-235b-sft-d.json

Requires: swebench, docker
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def extract_predictions(input_path: str) -> list[dict]:
    """Extract patches from SERA harness JSONL and format for SWE-bench eval."""
    predictions = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            instance_id = result.get("instance_id", "")
            patch_diff = result.get("patch_diff", "")

            if not patch_diff or not instance_id:
                continue

            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": "sft-eval",
                "model_patch": patch_diff,
            })

    return predictions


def run_swebench_eval(predictions_path: str, output_dir: str, max_workers: int = 4):
    """Run official SWE-bench Docker evaluation."""
    cmd = [
        "python3", "-m", "swebench.harness.run_evaluation",
        "--predictions_path", predictions_path,
        "--swe_bench_tasks", "princeton-nlp/SWE-bench_Lite",
        "--namespace", "swebench",  # Use pre-built Docker images
        "--max_workers", str(max_workers),
        "--run_id", "gold_eval",
        "--output_dir", output_dir,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def summarize_results(output_dir: str, predictions: list[dict]) -> dict:
    """Parse SWE-bench eval results and summarize."""
    # Find the report file
    report_files = list(Path(output_dir).rglob("*.json"))

    results = {
        "total_predictions": len(predictions),
        "resolved": [],
        "failed": [],
    }

    for rf in report_files:
        if "report" in rf.name.lower() or "results" in rf.name.lower():
            with open(rf) as f:
                report = json.load(f)

            # SWE-bench reports resolved instances
            if isinstance(report, dict):
                resolved = report.get("resolved", [])
                if isinstance(resolved, list):
                    results["resolved"] = resolved
                    results["pass_rate"] = len(resolved) / len(predictions) if predictions else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Gold eval: SWE-bench Docker evaluation")
    parser.add_argument("--input", required=True, help="SERA harness JSONL results file")
    parser.add_argument("--output", default="gold_eval_results.json", help="Output summary")
    parser.add_argument("--output-dir", default="./gold_eval_output", help="SWE-bench output dir")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel Docker workers")
    parser.add_argument("--predictions-only", action="store_true", help="Just generate predictions file, don't run eval")
    args = parser.parse_args()

    # Step 1: Extract predictions
    print(f"Extracting predictions from {args.input}...")
    predictions = extract_predictions(args.input)
    print(f"  Found {len(predictions)} patches to evaluate")

    if not predictions:
        print("No patches found. Exiting.")
        return

    # Step 2: Write predictions file
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"  Written {len(predictions)} predictions to {pred_path}")

    if args.predictions_only:
        print("Predictions-only mode. Done.")
        return

    # Step 3: Run evaluation
    print("\nRunning SWE-bench Docker evaluation...")
    rc = run_swebench_eval(pred_path, args.output_dir, args.max_workers)

    if rc != 0:
        print(f"  WARNING: Evaluation exited with code {rc}")

    # Step 4: Summarize
    results = summarize_results(args.output_dir, predictions)
    results["input_file"] = args.input

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== GOLD EVAL RESULTS ===")
    print(f"  Total patches: {results['total_predictions']}")
    print(f"  Resolved: {len(results.get('resolved', []))}")
    if "pass_rate" in results:
        print(f"  Pass rate: {results['pass_rate']:.1%}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
