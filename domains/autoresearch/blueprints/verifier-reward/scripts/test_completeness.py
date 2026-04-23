#!/usr/bin/env python3
"""
Test v010 completeness rubric on Devstral SERA patches.

Runs multiple configurations:
  A) v010 standalone (temp=0.0)
  B) v010 majority (3x temp=0.3, 2+/3)
  C) v001∩v010 (confirmatory AND completeness)
  D) v001∩v010∩v009 (triple gate)
  E) v010∩v009 (completeness AND adversarial, no confirmatory)

Usage:
  python3 test_completeness.py
  python3 test_completeness.py --issue pallets__flask-4045
  python3 test_completeness.py --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

SCRIPT_DIR = Path(__file__).resolve().parent
BLUEPRINT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BLUEPRINT_DIR / "results"
DIFFS_DIR = RESULTS_DIR / "diffs" / "devstral_sera_verifier_loop"
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"

V001 = VERSIONS_DIR / "v001_baseline.md"
V009 = VERSIONS_DIR / "v009_adversarial.md"
V010 = VERSIONS_DIR / "v010_completeness.md"


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
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]
    usage = result.get("usage", {})
    cost = (usage.get("input_tokens", 0) * 0.80 + usage.get("output_tokens", 0) * 4.00) / 1_000_000
    return {"text": text, "cost_usd": cost, "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0)}


def parse_json_output(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def verify_single(rubric_path: Path, problem: str, diff: str, temp: float) -> dict:
    rubric = rubric_path.read_text()
    prompt = f"""{rubric}

## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        resp = call_haiku(prompt, temp)
        parsed = parse_json_output(resp["text"])
        return {
            "parsed": parsed,
            "cost_usd": resp["cost_usd"],
            "error": None if parsed else "parse_failed",
            "raw_text": resp["text"][:500],
        }
    except Exception as e:
        return {"parsed": None, "error": str(e)[:500], "cost_usd": 0}


def run_all_calls(problem: str, diff: str) -> dict:
    """Run all rubric calls in parallel."""
    calls = [
        ("v001", V001, 0.0),
        ("v009_r1", V009, 0.3),
        ("v009_r2", V009, 0.3),
        ("v009_r3", V009, 0.3),
        ("v010_r0", V010, 0.0),
        ("v010_r1", V010, 0.3),
        ("v010_r2", V010, 0.3),
        ("v010_r3", V010, 0.3),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(verify_single, rub, problem, diff, temp): name
            for name, rub, temp in calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()

    return results


def extract_verdict(result: dict) -> str:
    return (result.get("parsed") or {}).get("verdict", "error")


def compute_configs(results: dict) -> dict:
    """Compute all ensemble configurations from raw results."""
    v001_v = extract_verdict(results["v001"])
    v001_pass = v001_v == "likely_correct"

    v009_lc = sum(1 for n in ["v009_r1", "v009_r2", "v009_r3"]
                  if extract_verdict(results[n]) == "likely_correct")
    v009_pass = v009_lc >= 2

    v010_0_v = extract_verdict(results["v010_r0"])
    v010_0_pass = v010_0_v == "likely_correct"

    v010_lc = sum(1 for n in ["v010_r1", "v010_r2", "v010_r3"]
                  if extract_verdict(results[n]) == "likely_correct")
    v010_maj_pass = v010_lc >= 2

    # Extract v010 details
    v010_parsed = results["v010_r0"].get("parsed") or {}
    missing_changes = v010_parsed.get("missing_changes", [])
    regression_risk = v010_parsed.get("regression_risk", "")
    completeness_score = (v010_parsed.get("scores") or {}).get("completeness", None)

    configs = {
        # A: v010 standalone (deterministic)
        "A_v010_solo": v010_0_pass,
        # B: v010 majority (2+/3 at temp=0.3)
        "B_v010_maj": v010_maj_pass,
        # C: v001 ∩ v010 (confirmatory AND completeness)
        "C_v001_v010": v001_pass and v010_0_pass,
        # D: v001 ∩ v010 ∩ v009 (triple gate)
        "D_triple": v001_pass and v010_0_pass and v009_pass,
        # E: v010 ∩ v009 (completeness AND adversarial)
        "E_v010_v009": v010_0_pass and v009_pass,
        # F: v001 ∩ v009 (original baseline)
        "F_baseline": v001_pass and v009_pass,
        # G: v001 ∩ v010(maj) (confirmatory AND completeness majority)
        "G_v001_v010maj": v001_pass and v010_maj_pass,
    }

    details = {
        "v001_verdict": v001_v,
        "v009_lc": v009_lc,
        "v010_verdict_det": v010_0_v,
        "v010_lc": v010_lc,
        "v010_missing_changes": missing_changes,
        "v010_regression_risk": regression_risk,
        "v010_completeness_score": completeness_score,
    }

    total_cost = sum(r.get("cost_usd", 0) for r in results.values())

    return {**configs, **details, "total_cost_usd": round(total_cost, 6)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--issue", type=str)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load data
    print("Loading SWE-bench data...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    gold = {}
    with open(RESULTS_DIR / "gold_devstral_sera_vloop_opencode.jsonl") as f:
        for line in f:
            row = json.loads(line)
            gold[row["instance_id"]] = row["passed"]

    diff_files = sorted(DIFFS_DIR.glob("*.diff"))
    if args.issue:
        diff_files = [d for d in diff_files if d.stem == args.issue]
    if args.limit:
        diff_files = diff_files[:args.limit]

    output_file = RESULTS_DIR / "test_completeness.jsonl"

    completed = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                completed.add(json.loads(line)["instance_id"])
        print(f"Resuming: {len(completed)} done")

    print(f"Diffs: {len(diff_files)}, Gold passes: {sum(gold.values())}\n")

    # Track per-config confusion matrices
    config_names = ["A_v010_solo", "B_v010_maj", "C_v001_v010",
                    "D_triple", "E_v010_v009", "F_baseline", "G_v001_v010maj"]
    cm = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in config_names}
    total_cost = 0.0

    for idx, diff_file in enumerate(diff_files):
        iid = diff_file.stem
        if iid in completed or iid not in gold:
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        gold_pass = gold[iid]

        print(f"[{idx+1}/{len(diff_files)}] {iid} (gold={'PASS' if gold_pass else 'FAIL'})...", end=" ", flush=True)

        results = run_all_calls(problem, diff_text)
        config_results = compute_configs(results)
        total_cost += config_results["total_cost_usd"]

        # Update confusion matrices
        for c in config_names:
            pred = config_results[c]
            if pred and gold_pass: cm[c]["tp"] += 1
            elif pred and not gold_pass: cm[c]["fp"] += 1
            elif not pred and gold_pass: cm[c]["fn"] += 1
            else: cm[c]["tn"] += 1

        # Print v010 detail
        v010_v = config_results["v010_verdict_det"]
        missing = config_results["v010_missing_changes"]
        risk = config_results["v010_regression_risk"]
        baseline_label = "PASS" if config_results["F_baseline"] else "FAIL"
        v010_label = "PASS" if config_results["A_v010_solo"] else "FAIL"

        print(f"v010={v010_v} baseline={baseline_label} | missing={len(missing)} risk={risk[:30] if risk else 'n/a'} | ${config_results['total_cost_usd']:.3f}")

        if missing:
            for m in missing[:2]:
                print(f"    → {m[:100]}")

        row = {"instance_id": iid, "gold_pass": gold_pass, **config_results}
        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Config':25s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s} {'Prec':>6s} {'Rec':>6s} {'F0.5':>6s} {'Calls':>5s}")
    print(f"{'='*80}")

    calls_per = {"A_v010_solo": 1, "B_v010_maj": 3, "C_v001_v010": 2,
                 "D_triple": 5, "E_v010_v009": 4, "F_baseline": 4, "G_v001_v010maj": 4}

    for c in config_names:
        m = cm[c]
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        print(f"{c:25s} {tp:3d} {fp:3d} {fn:3d} {tn:3d} {prec:6.2f} {rec:6.2f} {f05:6.2f} {calls_per.get(c, '?'):>5}")

    print(f"\nTotal cost: ${total_cost:.2f}")

    # Save summary
    summary = {}
    for c in config_names:
        m = cm[c]
        tp, fp = m["tp"], m["fp"]
        fn = m["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        summary[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": m["tn"],
                       "precision": round(prec, 4), "recall": round(rec, 4), "f05": round(f05, 4)}
    summary["total_cost_usd"] = round(total_cost, 4)

    with open(RESULTS_DIR / "test_completeness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
