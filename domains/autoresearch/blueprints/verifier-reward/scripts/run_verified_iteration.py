#!/usr/bin/env python3
"""
Iteration loop on Verified iteration set (250 patches).

Tests rubric variants against the 5 known FPs and a sample of TPs/TNs
to measure precision/recall changes without running the full 250-patch set.

Usage:
  # Quick test on 5 FPs + 10 TPs + 10 TNs (25 patches, ~$1.20)
  python3 run_verified_iteration.py --rubric v015 --quick

  # Full iteration set (250 patches, ~$6)
  python3 run_verified_iteration.py --rubric v015

  # Compare v009 vs v015 on FPs only
  python3 run_verified_iteration.py --rubric v015 --fps-only
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
ITER_GOLD = RESULTS_DIR / "gold_verified_iteration.jsonl"
ITER_IDS = RESULTS_DIR / "verified_iteration_ids.txt"

# Known FPs from T10 (all in iteration set)
KNOWN_FPS = [
    "django__django-12039",
    "django__django-14315",
    "django__django-15103",
    "django__django-16667",
    "pydata__xarray-6992",
]


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
            "completeness": (parsed or {}).get("scores", {}).get("completeness"),
            "has_test_script": (parsed or {}).get("has_test_script"),
            "completeness_check": (parsed or {}).get("completeness_check", ""),
            "reasoning": ((parsed or {}).get("reasoning") or "")[:300],
            "cost_usd": resp["cost_usd"],
        }
    except Exception as e:
        return {"verdict": "error", "cost_usd": 0, "error": str(e)[:200]}


def evaluate_patch(iid, problem, diff, rubric_text, n_runs=4):
    """Run rubric n_runs times: 1×t=0.0 + (n-1)×t=0.3."""
    results = {}
    total_cost = 0.0

    temps = [("r0", 0.0)] + [(f"r{i}", 0.3) for i in range(1, n_runs)]

    with ThreadPoolExecutor(max_workers=n_runs) as pool:
        futures = {}
        for name, temp in temps:
            f = pool.submit(run_verifier, rubric_text, problem, diff, temp)
            futures[f] = name
        for f in as_completed(futures):
            name = futures[f]
            result = f.result()
            results[name] = result
            total_cost += result.get("cost_usd", 0)

    lc_count = sum(1 for v in results.values() if v.get("verdict") == "likely_correct")
    return {
        "instance_id": iid,
        "lc_count": lc_count,
        "n_runs": n_runs,
        "total_cost_usd": round(total_cost, 6),
        "details": {k: {
            "verdict": v.get("verdict"),
            "score": v.get("score"),
            "completeness": v.get("completeness"),
            "has_test_script": v.get("has_test_script"),
            "reasoning": v.get("reasoning", "")[:200],
        } for k, v in results.items()},
    }


def wilson_ci(s, n, z=1.96):
    if n == 0: return (0, 1)
    p = s / n; d = 1 + z**2 / n; c = p + z**2 / (2 * n)
    sp = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (max(0, (c - sp) / d), min(1, (c + sp) / d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric", required=True, help="Rubric version (e.g., v015)")
    parser.add_argument("--quick", action="store_true", help="Test on 5 FPs + 10 TPs + 10 TNs only")
    parser.add_argument("--fps-only", action="store_true", help="Test only on 5 known FPs")
    parser.add_argument("--runs", type=int, default=4, help="Number of runs per patch")
    parser.add_argument("--threshold", type=int, default=2, help="Minimum lc count to pass")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    # Load rubric
    rubric_file = VERSIONS_DIR / f"{args.rubric}*.md"
    candidates = list(VERSIONS_DIR.glob(f"{args.rubric}*.md"))
    if not candidates:
        print(f"No rubric matching {args.rubric}* in {VERSIONS_DIR}")
        sys.exit(1)
    rubric_file = candidates[0]
    rubric_text = rubric_file.read_text()
    print(f"Rubric: {rubric_file.name}")

    # Load problems
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    problems = {r["instance_id"]: r["problem_statement"] for r in ds}

    # Load iteration gold
    gold = {}
    with open(ITER_GOLD) as f:
        for line in f:
            r = json.loads(line)
            gold[r["instance_id"]] = r["passed"]

    # Select patches to test
    if args.fps_only:
        test_ids = KNOWN_FPS
        print(f"Testing {len(test_ids)} known FPs only")
    elif args.quick:
        import random
        rng = random.Random(42)
        tps = [iid for iid, g in gold.items() if g]
        tns = [iid for iid, g in gold.items() if not g and iid not in KNOWN_FPS]
        rng.shuffle(tps)
        rng.shuffle(tns)
        test_ids = KNOWN_FPS + tps[:10] + tns[:10]
        print(f"Quick mode: {len(KNOWN_FPS)} FPs + 10 TPs + 10 TNs = {len(test_ids)} patches")
    else:
        test_ids = list(gold.keys())
        print(f"Full iteration set: {len(test_ids)} patches")

    total_cost = 0.0
    results = []

    def process(iid):
        diff_file = DIFFS_DIR / f"{iid}.diff"
        if not diff_file.exists():
            return None
        return evaluate_patch(iid, problems.get(iid, ""), diff_file.read_text(), rubric_text, args.runs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, iid): iid for iid in test_ids}
        done = 0
        for f in as_completed(futures):
            iid = futures[f]
            done += 1
            result = f.result()
            if result is None:
                continue
            result["gold_pass"] = gold.get(iid)
            result["is_known_fp"] = iid in KNOWN_FPS
            pred = result["lc_count"] >= args.threshold
            result["pred_pass"] = pred
            total_cost += result["total_cost_usd"]
            results.append(result)

            g = "PASS" if result["gold_pass"] else "FAIL"
            p = "PASS" if pred else "FAIL"
            ok = "OK" if (pred == result["gold_pass"]) else "WRONG"
            fp_mark = " [KNOWN-FP]" if result["is_known_fp"] else ""
            short = iid.split("__")[-1]
            r0 = result["details"].get("r0", {})
            print(f"[{done}/{len(test_ids)}] {short:35s} gold={g} pred={p} {ok} | lc={result['lc_count']}/{args.runs} score={r0.get('score','?')} compl={r0.get('completeness','?')}{fp_mark}")

    # Metrics
    tp = fp = fn = tn = 0
    for r in results:
        pred = r["pred_pass"]
        actual = r["gold_pass"]
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0

    # FP analysis
    known_fp_caught = sum(1 for r in results if r["is_known_fp"] and not r["pred_pass"])
    known_fp_leaked = sum(1 for r in results if r["is_known_fp"] and r["pred_pass"])

    print(f"\n{'='*60}")
    print(f"Rubric: {rubric_file.name} | Threshold: {args.threshold}+/{args.runs} | n={len(results)}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.3f} Recall={rec:.3f} F0.5={f05:.3f}")
    print(f"Known FPs: {known_fp_caught}/5 caught, {known_fp_leaked}/5 leaked")
    print(f"Cost: ${total_cost:.2f}")

    # Detail on known FPs
    print(f"\nKnown FP results:")
    for r in results:
        if r["is_known_fp"]:
            r0 = r["details"].get("r0", {})
            print(f"  {r['instance_id']:40s} lc={r['lc_count']}/{args.runs} score={r0.get('score','?')} compl={r0.get('completeness','?')} test_script={r0.get('has_test_script','?')}")
            print(f"    reasoning: {r0.get('reasoning','')[:150]}")

    # Save results
    output_file = RESULTS_DIR / f"verified_iter_{args.rubric}_{args.threshold}of{args.runs}.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
