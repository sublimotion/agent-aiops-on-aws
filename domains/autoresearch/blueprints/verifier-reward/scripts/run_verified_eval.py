#!/usr/bin/env python3
"""
Large-scale evaluation of v001∩v009 ensemble on SWE-bench Verified (483 patches).

Runs the best config (5 Haiku calls per patch) on Claude 3.5 Sonnet predictions
to get statistically meaningful precision/recall with tight confidence intervals.

Config: v001(t=0.0) + v009(1×t=0.0 + 3×t=0.3), threshold 2+/4 lc for v009.
Ensemble: patch passes only if v001=likely_correct AND v009 has ≥2/4 likely_correct.

Usage:
  python3 run_verified_eval.py                    # full run
  python3 run_verified_eval.py --resume           # resume partial run
  python3 run_verified_eval.py --limit 10         # test on first 10
  python3 run_verified_eval.py --workers 30       # increase parallelism
"""

import argparse
import json
import math
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
DIFFS_DIR = RESULTS_DIR / "diffs" / "swebench_verified_sonnet"
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"
GOLD_FILE = RESULTS_DIR / "gold_swebench_verified_sonnet.jsonl"
OUTPUT_FILE = RESULTS_DIR / "verified_eval_results.jsonl"

V001 = VERSIONS_DIR / "v001_baseline.md"
V009 = VERSIONS_DIR / "v009_adversarial.md"


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
    return {"text": text, "cost_usd": cost, "latency_ms": latency_ms}


def parse_json_output(text: str):
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


def run_verifier(rubric_text: str, problem: str, diff: str, temperature: float) -> dict:
    prompt = f"""{rubric_text}

## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        resp = call_haiku(prompt, temperature)
        parsed = parse_json_output(resp["text"])
        return {
            "verdict": (parsed or {}).get("verdict", "error"),
            "score": (parsed or {}).get("overall_score"),
            "cost_usd": resp["cost_usd"],
            "latency_ms": resp["latency_ms"],
        }
    except Exception as e:
        return {"verdict": "error", "score": None, "cost_usd": 0, "latency_ms": 0, "error": str(e)[:200]}


def evaluate_patch(instance_id: str, problem: str, diff: str,
                   v001_text: str, v009_text: str) -> dict:
    """Run the full 5-call ensemble on one patch."""
    results = {}
    total_cost = 0.0

    # Run all 5 calls in parallel
    calls = [
        ("v001_t0", v001_text, 0.0),
        ("v009_t0", v009_text, 0.0),
        ("v009_t03a", v009_text, 0.3),
        ("v009_t03b", v009_text, 0.3),
        ("v009_t03c", v009_text, 0.3),
    ]

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {}
        for name, rubric, temp in calls:
            f = pool.submit(run_verifier, rubric, problem, diff, temp)
            futures[f] = name

        for f in as_completed(futures):
            name = futures[f]
            try:
                result = f.result()
            except Exception as e:
                result = {"verdict": "error", "score": None, "cost_usd": 0, "error": str(e)[:200]}
            results[name] = result
            total_cost += result.get("cost_usd", 0)

    # Ensemble logic
    v001_pass = results.get("v001_t0", {}).get("verdict") == "likely_correct"
    v009_lc = sum(1 for k, v in results.items() if k.startswith("v009") and v.get("verdict") == "likely_correct")
    v009_pass = v009_lc >= 2  # 2+/4 threshold

    ensemble_pass = v001_pass and v009_pass

    return {
        "instance_id": instance_id,
        "v001_verdict": results.get("v001_t0", {}).get("verdict", "error"),
        "v009_lc_count": v009_lc,
        "v009_pass": v009_pass,
        "ensemble_pass": ensemble_pass,
        "total_cost_usd": round(total_cost, 6),
        "calls": {k: {"verdict": v.get("verdict"), "score": v.get("score")} for k, v in results.items()},
    }


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return (0, 1)
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lo = max(0, (centre - spread) / denom)
    hi = min(1, (centre + spread) / denom)
    return (round(lo, 4), round(hi, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=20, help="Parallel patch evaluations")
    args = parser.parse_args()

    # Load problem statements
    print("Loading SWE-bench Verified dataset...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    # Load gold labels
    gold = {}
    with open(GOLD_FILE) as f:
        for line in f:
            row = json.loads(line)
            gold[row["instance_id"]] = row["passed"]

    # Load rubrics
    v001_text = V001.read_text()
    v009_text = V009.read_text()

    # Get all diff files
    diff_files = sorted(DIFFS_DIR.glob("*.diff"))
    instances = [(df.stem, df) for df in diff_files if df.stem in gold and df.stem in problems]

    if args.limit:
        instances = instances[:args.limit]

    # Resume support
    completed = set()
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed.add(row["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(completed)} already done")

    remaining = [(iid, df) for iid, df in instances if iid not in completed]
    print(f"Patches: {len(instances)} total, {len(remaining)} to evaluate")
    print(f"Gold: {sum(1 for iid, _ in instances if gold.get(iid))} PASS, {sum(1 for iid, _ in instances if not gold.get(iid))} FAIL")
    print(f"Workers: {args.workers}, Est cost: ${len(remaining) * 0.048:.0f}")
    print()

    total_cost = 0.0
    done = 0
    errors = 0

    def process_one(iid, diff_file):
        diff_text = diff_file.read_text()
        problem = problems[iid]
        return evaluate_patch(iid, problem, diff_text, v001_text, v009_text)

    # Open output in append mode
    out_f = open(OUTPUT_FILE, "a")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for iid, df in remaining:
                f = pool.submit(process_one, iid, df)
                futures[f] = iid

            for f in as_completed(futures):
                iid = futures[f]
                done += 1
                try:
                    result = f.result()
                    result["gold_pass"] = gold.get(iid)
                    total_cost += result["total_cost_usd"]

                    # Write result
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()

                    # Progress
                    g = "PASS" if result["gold_pass"] else "FAIL"
                    e = "PASS" if result["ensemble_pass"] else "FAIL"
                    ok = "OK" if (result["ensemble_pass"] == result["gold_pass"]) else "WRONG"
                    short = iid.split("__")[-1]
                    print(f"[{done}/{len(remaining)}] {short:35s} gold={g} pred={e} {ok} | v001={result['v001_verdict'][:3]} v009={result['v009_lc_count']}/4 | ${result['total_cost_usd']:.3f}")

                except Exception as e:
                    errors += 1
                    print(f"[{done}/{len(remaining)}] {iid}: ERROR {e}")
                    out_f.write(json.dumps({"instance_id": iid, "error": str(e)[:500], "gold_pass": gold.get(iid)}) + "\n")
                    out_f.flush()

    finally:
        out_f.close()

    # Final metrics
    print(f"\n{'='*70}")
    print(f"Evaluation complete: {done} patches, {errors} errors, ${total_cost:.2f}")
    print(f"Results: {OUTPUT_FILE}")

    # Compute metrics from all results (including resumed)
    all_results = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    tp = fp = fn = tn = 0
    for r in all_results:
        if "ensemble_pass" not in r:
            continue
        pred = r["ensemble_pass"]
        actual = r.get("gold_pass", False)
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1

    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0

    prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0, 1)
    rec_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0, 1)

    print(f"\nConfusion matrix (n={total}):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"\nPrecision = {prec:.4f}  95% CI [{prec_ci[0]:.4f}, {prec_ci[1]:.4f}]")
    print(f"Recall    = {rec:.4f}  95% CI [{rec_ci[0]:.4f}, {rec_ci[1]:.4f}]")
    print(f"F0.5      = {f05:.4f}")
    print(f"Total cost: ${total_cost:.2f}")

    # Breakdown by gold label
    gold_pass_count = sum(1 for r in all_results if r.get("gold_pass"))
    gold_fail_count = sum(1 for r in all_results if not r.get("gold_pass") and "ensemble_pass" in r)
    print(f"\nGold PASS: {gold_pass_count} (TP={tp}, FN={fn})")
    print(f"Gold FAIL: {gold_fail_count} (FP={fp}, TN={tn})")

    # List FPs if any
    fps = [r for r in all_results if r.get("ensemble_pass") and not r.get("gold_pass")]
    if fps:
        print(f"\nFalse Positives ({len(fps)}):")
        for r in fps:
            print(f"  {r['instance_id']} v001={r.get('v001_verdict','?')[:3]} v009={r.get('v009_lc_count','?')}/4")


if __name__ == "__main__":
    main()
