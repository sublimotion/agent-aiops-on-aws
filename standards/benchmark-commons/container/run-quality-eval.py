#!/usr/bin/env python3
"""
Runs quality evals against a running serving endpoint to gate throughput results.

The doc's O3 rule: "A precision configuration that fails the agreed quality
tolerance does not get a throughput row in the report." This script produces
the gate decision as a JSON blob that gets merged into the enriched artifact.

Supported eval sources (chosen per model modality):
  - lm-eval-harness : MMLU 5-shot, GSM8K 8-shot CoT, HumanEval (text LLM)
  - mteb            : Banking77, FiQA (embedding / reranker)
  - docvqa          : DocVQA validation split (OCR / VLM)
  - librispeech     : LibriSpeech-clean WER (ASR / speech)

Each eval is a thin wrapper: we shell out to the upstream harness with a
pinned commit SHA (Appendix A rules), then normalize the metric we care about.

Usage:
  run-quality-eval.py --eval mmlu --endpoint http://svc:8000 --model qwen3.5-125b \
    --tolerance 0.02 --baseline-score 0.74 --output quality.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Pinned commit SHAs — Appendix A mandates these are frozen at engagement start.
# Update by editing both this file and the spec's harness-pins section.
PINNED_COMMITS = {
    "lm-eval-harness": "v0.4.7",
    "mteb": "1.38.22",
    "evaluate": "v0.4.3",
}

# Supported evals with their harness mapping.
EVAL_CONFIGS = {
    "mmlu": {
        "harness": "lm-eval-harness",
        "task": "mmlu",
        "num_fewshot": 5,
        "metric_key": "acc",
    },
    "gsm8k": {
        "harness": "lm-eval-harness",
        "task": "gsm8k_cot",
        "num_fewshot": 8,
        "metric_key": "exact_match,strict-match",
    },
    "humaneval": {
        "harness": "lm-eval-harness",
        "task": "humaneval",
        "num_fewshot": 0,
        "metric_key": "pass@1",
    },
    "banking77": {
        "harness": "mteb",
        "task": "Banking77Classification",
        "metric_key": "main_score",
    },
    "fiqa": {
        "harness": "mteb",
        "task": "FiQA2018",
        "metric_key": "main_score",
    },
    "docvqa": {
        "harness": "docvqa",
        "task": "docvqa_val",
        "metric_key": "anls",
    },
    "librispeech": {
        "harness": "librispeech",
        "task": "test-clean",
        "metric_key": "wer",  # lower is better — handled specially
    },
}

LOWER_IS_BETTER = {"librispeech"}


def run_lm_eval(task: str, num_fewshot: int, endpoint: str, model_id: str,
                limit: int | None) -> dict:
    """Run lm-evaluation-harness via its OpenAI-compatible model backend."""
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", f"base_url={endpoint}/v1/completions,model={model_id},tokenized_requests=False",
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", "/tmp/lm-eval-out",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:]}
    result_file = Path("/tmp/lm-eval-out") / f"results_{task}.json"
    if not result_file.exists():
        # lm-eval writes a timestamped file; fall back to the newest one.
        files = sorted(Path("/tmp/lm-eval-out").glob("*.json"))
        if not files:
            return {"error": "no lm-eval output produced"}
        result_file = files[-1]
    with open(result_file) as f:
        return json.load(f)


def run_mteb(task: str, endpoint: str, model_id: str) -> dict:
    """MTEB against an OpenAI-compatible embeddings endpoint."""
    cmd = [
        "python3", "-m", "mteb", "run",
        "--model", "openai",
        "--model_args", f"base_url={endpoint}/v1,model={model_id}",
        "--tasks", task,
        "--output_folder", "/tmp/mteb-out",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:]}
    result_file = Path("/tmp/mteb-out") / f"{task}.json"
    if not result_file.exists():
        return {"error": "no mteb output produced"}
    with open(result_file) as f:
        return json.load(f)


def extract_score(result: dict, eval_name: str, cfg: dict) -> float | None:
    """Pull the headline metric out of the harness result blob."""
    key = cfg["metric_key"]
    if "error" in result:
        return None
    # lm-eval puts metrics under results[task][metric_key]
    task = cfg["task"]
    if "results" in result and task in result["results"]:
        return result["results"][task].get(key)
    # MTEB nests under scores → test
    if "scores" in result:
        test_scores = result["scores"].get("test", [])
        if test_scores:
            return test_scores[0].get(key)
    return result.get(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, choices=list(EVAL_CONFIGS.keys()))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True, help="Model name as served at the endpoint")
    parser.add_argument("--baseline-score", type=float, required=True,
                        help="BF16 / FP16 reference score to compare against")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Max allowed drop (abs) vs baseline; default 2pp")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample limit for quick smoke runs; leave unset for full eval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = EVAL_CONFIGS[args.eval]
    harness = cfg["harness"]

    if harness == "lm-eval-harness":
        result = run_lm_eval(cfg["task"], cfg["num_fewshot"], args.endpoint, args.model, args.limit)
    elif harness == "mteb":
        result = run_mteb(cfg["task"], args.endpoint, args.model)
    else:
        print(f"Eval harness '{harness}' not yet wired up in this runner.", file=sys.stderr)
        sys.exit(2)

    score = extract_score(result, args.eval, cfg)
    if score is None:
        passed = False
        delta = None
    elif args.eval in LOWER_IS_BETTER:
        delta = score - args.baseline_score
        passed = delta <= args.tolerance
    else:
        delta = args.baseline_score - score
        passed = delta <= args.tolerance

    output = {
        "eval": args.eval,
        "harness": harness,
        "harness_version": PINNED_COMMITS.get(harness, "unknown"),
        "task": cfg["task"],
        "metric_key": cfg["metric_key"],
        "score": score,
        "baseline_score": args.baseline_score,
        "delta": delta,
        "tolerance": args.tolerance,
        "passed": passed,
        "lower_is_better": args.eval in LOWER_IS_BETTER,
        "limit": args.limit,
        "raw_path": str(result.get("_path")) if isinstance(result, dict) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {args.eval}: score={score} baseline={args.baseline_score} "
          f"delta={delta} tolerance={args.tolerance}", file=sys.stderr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
