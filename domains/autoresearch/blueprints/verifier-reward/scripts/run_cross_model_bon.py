#!/usr/bin/env python3
"""
Iteration 25: Cross-model Best-of-N on Claude patches.

For each issue with patches from multiple Claude models (haiku, sonnet, opus),
runs v001∩v009 ensemble on each candidate and selects the best.

Tests whether the verifier (precision=1.00 on Claude patches) can improve
effective pass rate by picking the best model's patch per issue.

Usage:
  python3 run_cross_model_bon.py
  python3 run_cross_model_bon.py --limit 5
  python3 run_cross_model_bon.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

SCRIPT_DIR = Path(__file__).resolve().parent
BLUEPRINT_DIR = SCRIPT_DIR.parent
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"
RESULTS_DIR = BLUEPRINT_DIR / "results"
V001_RUBRIC = VERSIONS_DIR / "v001_baseline.md"
V009_RUBRIC = VERSIONS_DIR / "v009_adversarial.md"

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
HAIKU_PRICING = (0.80, 4.00)

V009_RUNS = 3
V009_THRESHOLD = 2

MODELS = ["haiku", "sonnet", "opus"]


def call_haiku(prompt: str, temperature: float = 0.0) -> dict:
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.monotonic()
    response = client.invoke_model(
        modelId=HAIKU_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]
    usage = result.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_ms": latency_ms,
    }


def parse_json_output(text: str) -> dict:
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def verify_single(rubric_path: Path, problem: str, diff: str,
                   temperature: float) -> dict:
    rubric = rubric_path.read_text()
    MAX_DIFF = 100000
    diff_trimmed = diff[:MAX_DIFF] if len(diff) > MAX_DIFF else diff

    prompt = f"""{rubric}

## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{diff_trimmed}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        resp = call_haiku(prompt, temperature)
        parsed = parse_json_output(resp["text"])
        cost = (resp["input_tokens"] * HAIKU_PRICING[0] + resp["output_tokens"] * HAIKU_PRICING[1]) / 1_000_000
        return {
            "parsed": parsed,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "latency_ms": resp["latency_ms"],
            "cost_usd": cost,
            "error": None if parsed else "parse_failed",
        }
    except Exception as e:
        return {"parsed": None, "error": str(e)[:500], "cost_usd": 0}


def run_ensemble(problem: str, diff: str) -> dict:
    """Run v001∩v009(2+/3) ensemble."""
    calls = [
        ("v001", V001_RUBRIC, 0.0),
        ("v009_r1", V009_RUBRIC, 0.3),
        ("v009_r2", V009_RUBRIC, 0.3),
        ("v009_r3", V009_RUBRIC, 0.3),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(verify_single, rub, problem, diff, temp): name
            for name, rub, temp in calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()

    v001 = results["v001"]
    v001_verdict = (v001.get("parsed") or {}).get("verdict", "")
    v001_score = (v001.get("parsed") or {}).get("overall_score", 0)
    v001_pass = v001_verdict == "likely_correct"

    v009_names = [n for n in results if n.startswith("v009")]
    v009_lc = sum(
        1 for n in v009_names
        if (results[n].get("parsed") or {}).get("verdict") == "likely_correct"
    )
    v009_pass = v009_lc >= V009_THRESHOLD
    ensemble_pass = v001_pass and v009_pass

    total_cost = sum(r.get("cost_usd", 0) for r in results.values())
    errors = [f"{n}: {r['error']}" for n, r in results.items() if r.get("error")]

    return {
        "ensemble_pass": ensemble_pass,
        "v001_verdict": v001_verdict,
        "v001_score": v001_score if isinstance(v001_score, (int, float)) else 0,
        "v009_lc_count": v009_lc,
        "total_cost_usd": round(total_cost, 6),
        "errors": errors if errors else None,
    }


def load_candidates() -> dict:
    """Load Claude patches grouped by issue.

    Returns: {instance_id: [(model_name, diff_content), ...]}
    """
    candidates = defaultdict(list)
    for model in MODELS:
        diffs_dir = RESULTS_DIR / "diffs" / f"opencode_{model}"
        if not diffs_dir.exists():
            continue
        for f in sorted(diffs_dir.glob("*.diff")):
            instance_id = f.stem
            candidates[instance_id].append((model, f.read_text()))
    return dict(candidates)


def load_gold_labels() -> dict:
    """Load gold labels for all models. Returns {(instance_id, model): passed}."""
    gold = {}
    for model in MODELS:
        gold_file = RESULTS_DIR / f"gold_{model}_opencode.jsonl"
        if not gold_file.exists():
            continue
        with open(gold_file) as f:
            for line in f:
                row = json.loads(line)
                gold[(row["instance_id"], model)] = row.get("passed", False)
    return gold


def load_problems() -> dict:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def main():
    parser = argparse.ArgumentParser(description="Iter 25: Cross-model BoN on Claude patches")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=2,
                        help="Only include issues with >= N model patches")
    args = parser.parse_args()

    output_file = RESULTS_DIR / "t7_cross_model_bon.jsonl"
    summary_file = RESULTS_DIR / "t7_cross_model_bon_summary.json"

    print("Loading candidates...")
    candidates = load_candidates()
    # Filter to issues with min_candidates
    candidates = {k: v for k, v in candidates.items() if len(v) >= args.min_candidates}
    total_cands = sum(len(v) for v in candidates.values())
    print(f"Found {total_cands} candidates across {len(candidates)} issues (>={args.min_candidates} models each)")

    print("Loading gold labels...")
    gold = load_gold_labels()

    print("Loading problem statements...")
    problems = load_problems()

    issues = sorted(candidates.keys())
    if args.limit:
        issues = issues[:args.limit]

    # Resume
    completed = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                row = json.loads(line)
                completed.add(row["instance_id"])
        print(f"Resuming: {len(completed)} done")

    total_cost = 0.0
    selections = []

    for i, instance_id in enumerate(issues):
        if instance_id in completed:
            continue

        problem = problems.get(instance_id, "")
        issue_candidates = candidates[instance_id]

        # Show gold labels for this issue
        gold_str = " ".join(
            f"{m}={'P' if gold.get((instance_id, m), False) else 'F'}"
            for m, _ in issue_candidates
        )
        print(f"\n[{i+1}/{len(issues)}] {instance_id} ({len(issue_candidates)} models) gold=[{gold_str}]")

        candidate_results = []
        for model_name, diff in issue_candidates:
            print(f"  {model_name}...", end=" ", flush=True)
            result = run_ensemble(problem, diff)
            result["model"] = model_name
            candidate_results.append(result)
            total_cost += result["total_cost_usd"]

            status = "PASS" if result["ensemble_pass"] else "FAIL"
            gold_pass = gold.get((instance_id, model_name), False)
            gold_str = "gold=P" if gold_pass else "gold=F"
            print(f"{status} v001={result['v001_verdict']} v009={result['v009_lc_count']}/3 {gold_str} ${result['total_cost_usd']:.4f}")

        # Selection
        passing = [r for r in candidate_results if r["ensemble_pass"]]
        if passing:
            selected = max(passing, key=lambda r: (r["v009_lc_count"], r["v001_score"]))
            selection_method = "ensemble_pass"
        else:
            selected = max(candidate_results, key=lambda r: (r["v009_lc_count"], r["v001_score"]))
            selection_method = "least_rejected"

        selected_gold = gold.get((instance_id, selected["model"]), False)
        label = "TP" if selected["ensemble_pass"] and selected_gold else \
                "FP" if selected["ensemble_pass"] and not selected_gold else \
                "correct_reject" if not selected["ensemble_pass"] and not selected_gold else \
                "missed"

        print(f"  → Selected: {selected['model']} ({selection_method}) {label}")

        row = {
            "instance_id": instance_id,
            "num_candidates": len(issue_candidates),
            "selected_model": selected["model"],
            "selection_method": selection_method,
            "selected_ensemble_pass": selected["ensemble_pass"],
            "selected_gold_pass": selected_gold,
            "selected_v001_verdict": selected["v001_verdict"],
            "selected_v009_lc": selected["v009_lc_count"],
            "all_results": candidate_results,
            "issue_cost_usd": round(sum(r["total_cost_usd"] for r in candidate_results), 6),
        }
        selections.append(row)

        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Analysis
    import random
    n_issues = len(selections)
    bon_passes = sum(1 for s in selections if s["selected_gold_pass"])

    # Oracle
    oracle_passes = sum(
        1 for s in selections
        if any(gold.get((s["instance_id"], r["model"]), False) for r in s["all_results"])
    )

    # Random baseline (1000 trials)
    random.seed(42)
    random_trials = []
    for _ in range(1000):
        p = sum(
            1 for s in selections
            if gold.get((s["instance_id"], random.choice(s["all_results"])["model"]), False)
        )
        random_trials.append(p)
    random_mean = sum(random_trials) / len(random_trials)

    # Individual model baselines
    model_passes = {}
    for model in MODELS:
        model_passes[model] = sum(
            1 for s in selections
            if gold.get((s["instance_id"], model), False)
        )

    # Precision on ensemble passes
    ensemble_pass_issues = [s for s in selections if s["selected_ensemble_pass"]]
    tp = sum(1 for s in ensemble_pass_issues if s["selected_gold_pass"])
    fp = len(ensemble_pass_issues) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Cross-Model Best-of-N Summary ({n_issues} issues)")
    print(f"{'='*60}")
    print(f"Oracle:         {oracle_passes}/{n_issues} ({oracle_passes/n_issues*100:.0f}%)")
    print(f"Random:         {random_mean:.1f}/{n_issues} ({random_mean/n_issues*100:.1f}%)")
    print(f"BoN verifier:   {bon_passes}/{n_issues} ({bon_passes/n_issues*100:.0f}%)")
    for model in MODELS:
        print(f"  {model:12s}:  {model_passes[model]}/{n_issues} ({model_passes[model]/n_issues*100:.0f}%)")
    print(f"\nEnsemble passes: {len(ensemble_pass_issues)}")
    print(f"Precision: {precision:.2f} (TP={tp}, FP={fp})")
    print(f"Total cost: ${total_cost:.4f}")

    # Show ensemble passes
    if ensemble_pass_issues:
        print("\nEnsemble passes:")
        for s in ensemble_pass_issues:
            label = "TP" if s["selected_gold_pass"] else "FP"
            print(f"  {label} {s['instance_id']:40s} model={s['selected_model']} v009={s['selected_v009_lc']}/3")

    # Show missed TPs (gold pass but not selected)
    missed = [s for s in selections if not s["selected_gold_pass"] and
              any(gold.get((s["instance_id"], r["model"]), False) for r in s["all_results"])]
    if missed:
        print(f"\nMissed (gold pass exists but BoN picked wrong):")
        for s in missed:
            passing_models = [r["model"] for r in s["all_results"]
                            if gold.get((s["instance_id"], r["model"]), False)]
            print(f"  {s['instance_id']:40s} selected={s['selected_model']} gold_passes={passing_models}")

    summary = {
        "experiment": "t7_cross_model_bon",
        "n_issues": n_issues,
        "oracle": oracle_passes,
        "random_mean": round(random_mean, 1),
        "bon_passes": bon_passes,
        "model_passes": model_passes,
        "ensemble_pass_count": len(ensemble_pass_issues),
        "precision": round(precision, 4),
        "tp": tp, "fp": fp,
        "total_cost_usd": round(total_cost, 4),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
