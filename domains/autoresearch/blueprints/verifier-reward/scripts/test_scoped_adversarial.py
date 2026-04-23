#!/usr/bin/env python3
"""
Iteration 44: Test-scoped adversarial verification.

Two-phase approach:
  Phase 1: Generate test description — ask model what the regression test likely checks
  Phase 2: Run v013 (test-scoped adversarial) with the test description as context

Also runs v001 and v009 for comparison.

Tests on 9 critical patches: 5 FPs + 4 FNs from dev set.

Usage:
  python3 test_scoped_adversarial.py
  python3 test_scoped_adversarial.py --full  # all 49 dev patches
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
DIFFS_DIR = RESULTS_DIR / "diffs" / "opencode_sonnet"
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"
GOLD_FILE = RESULTS_DIR / "gold_sonnet_opencode.jsonl"

V001 = VERSIONS_DIR / "v001_baseline.md"
V009 = VERSIONS_DIR / "v009_adversarial.md"
V013 = VERSIONS_DIR / "v013_testscoped_adversarial.md"

# Critical patches for targeted testing
# FPs: patches that v001∩v009 should reject (gold=FAIL, v001=lc)
# FNs: patches that v001∩v009 misses (gold=PASS, ensemble=reject)
CRITICAL_FPS = [
    "astropy__astropy-14365",
    "astropy__astropy-14995",
    "pallets__flask-4992",
    "pallets__flask-4045",
    "pallets__flask-5063",
]
CRITICAL_FNS = [
    "django__django-10924",
    "django__django-11001",
    "sphinx-doc__sphinx-10325",
    "sphinx-doc__sphinx-11445",
]

TESTGEN_PROMPT = """You are a QA engineer. Given this bug report, describe what a MINIMAL regression test would check. Focus on the SPECIFIC behavior that's broken, not comprehensive testing.

## Bug Report

{problem}

## Response Format

Respond with a SHORT description (2-4 sentences) of what the regression test checks:
- What specific function/method/endpoint is called
- What input is provided
- What output or behavior is expected
- What was wrong before the fix

Do NOT write actual test code. Just describe the test behavior in plain English."""


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


def generate_test_description(problem: str) -> dict:
    """Phase 1: Generate what the regression test likely checks."""
    prompt = TESTGEN_PROMPT.format(problem=problem[:8000])
    resp = call_haiku(prompt, temperature=0.0)
    return {"test_description": resp["text"], "cost_usd": resp["cost_usd"]}


def verify_with_rubric(rubric_path: Path, problem: str, diff: str, temp: float,
                       test_description: str = None) -> dict:
    """Run a single verification call, optionally with test context."""
    rubric = rubric_path.read_text()

    if test_description and rubric_path == V013:
        # v013 gets the test description injected
        prompt = f"""{rubric}

## Bug Report

{problem[:8000]}

## What the Regression Test Checks

{test_description}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate whether this patch will PASS the described test. Respond with ONLY the JSON object."""
    else:
        # v001/v009 get standard prompt
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


def extract_verdict(result: dict) -> str:
    return (result.get("parsed") or {}).get("verdict", "error")


def run_patch(problem: str, diff: str) -> dict:
    """Run full 2-phase verification on one patch."""

    # Phase 1: Generate test description
    testgen = generate_test_description(problem)
    test_desc = testgen["test_description"]

    # Phase 2: Run all rubrics in parallel
    calls = [
        ("v001", V001, 0.0, None),
        ("v009_r0", V009, 0.0, None),
        ("v009_r1", V009, 0.3, None),
        ("v009_r2", V009, 0.3, None),
        ("v009_r3", V009, 0.3, None),
        ("v013_r0", V013, 0.0, test_desc),
        ("v013_r1", V013, 0.3, test_desc),
        ("v013_r2", V013, 0.3, test_desc),
        ("v013_r3", V013, 0.3, test_desc),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {
            pool.submit(verify_with_rubric, rub, problem, diff, temp, td): name
            for name, rub, temp, td in calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()

    # Compute ensemble configurations
    v001_pass = extract_verdict(results["v001"]) == "likely_correct"

    v009_lc = sum(1 for n in ["v009_r0", "v009_r1", "v009_r2", "v009_r3"]
                  if extract_verdict(results[n]) == "likely_correct")
    v009_pass_2of4 = v009_lc >= 2

    v013_lc = sum(1 for n in ["v013_r0", "v013_r1", "v013_r2", "v013_r3"]
                  if extract_verdict(results[n]) == "likely_correct")
    v013_pass_2of4 = v013_lc >= 2

    total_cost = testgen["cost_usd"] + sum(r.get("cost_usd", 0) for r in results.values())

    configs = {
        # A: Current best (v001∩v009 2+/4)
        "A_baseline": v001_pass and v009_pass_2of4,
        # B: v001∩v013 (test-scoped adversarial)
        "B_v001_v013": v001_pass and v013_pass_2of4,
        # C: v001∩v013∩v009 (triple gate — most conservative)
        "C_triple": v001_pass and v013_pass_2of4 and v009_pass_2of4,
        # D: v001 ∩ (v013 OR v009) — rescue rejected patches via test-scoped
        "D_rescue": v001_pass and (v013_pass_2of4 or v009_pass_2of4),
        # E: v013 standalone (test-scoped only)
        "E_v013_solo": v013_pass_2of4,
    }

    return {
        "test_description": test_desc[:300],
        "v001_verdict": extract_verdict(results["v001"]),
        "v009_lc": v009_lc,
        "v013_lc": v013_lc,
        "configs": configs,
        "total_cost_usd": round(total_cost, 6),
        "details": {
            name: {
                "verdict": extract_verdict(r),
                "score": (r.get("parsed") or {}).get("overall_score"),
                "reasoning": ((r.get("parsed") or {}).get("reasoning") or "")[:150],
            }
            for name, r in sorted(results.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 49 dev patches")
    args = parser.parse_args()

    # Load SWE-bench problem statements
    print("Loading SWE-bench data...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    # Load gold labels
    gold = {}
    with open(GOLD_FILE) as f:
        for line in f:
            row = json.loads(line)
            gold[row["instance_id"]] = row["passed"]

    # Select patches
    if args.full:
        diff_files = sorted(DIFFS_DIR.glob("*.diff"))
    else:
        critical = CRITICAL_FPS + CRITICAL_FNS
        diff_files = [DIFFS_DIR / f"{iid}.diff" for iid in critical
                      if (DIFFS_DIR / f"{iid}.diff").exists()]

    print(f"Patches: {len(diff_files)}")

    config_names = ["A_baseline", "B_v001_v013", "C_triple", "D_rescue", "E_v013_solo"]
    cm = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in config_names}
    total_cost = 0.0

    output_file = RESULTS_DIR / "iter44_scoped_adversarial.jsonl"

    for idx, diff_file in enumerate(diff_files):
        iid = diff_file.stem
        if iid not in gold:
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        gold_pass = gold[iid]

        label = "PASS" if gold_pass else "FAIL"
        critical_type = "FP" if iid in CRITICAL_FPS else ("FN" if iid in CRITICAL_FNS else "—")
        print(f"[{idx+1}/{len(diff_files)}] {iid} (gold={label}, {critical_type})...", end=" ", flush=True)

        result = run_patch(problem, diff_text)
        total_cost += result["total_cost_usd"]

        # Update confusion matrices
        for c in config_names:
            pred = result["configs"][c]
            if pred and gold_pass: cm[c]["tp"] += 1
            elif pred and not gold_pass: cm[c]["fp"] += 1
            elif not pred and gold_pass: cm[c]["fn"] += 1
            else: cm[c]["tn"] += 1

        # Print key results
        baseline = "PASS" if result["configs"]["A_baseline"] else "FAIL"
        v013_rescue = "PASS" if result["configs"]["D_rescue"] else "FAIL"
        print(f"v009={result['v009_lc']}/4 v013={result['v013_lc']}/4 baseline={baseline} rescue={v013_rescue} | ${result['total_cost_usd']:.3f}")
        print(f"    test: {result['test_description'][:120]}")

        row = {"instance_id": iid, "gold_pass": gold_pass, "critical_type": critical_type, **result}
        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Config':20s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s} {'Prec':>6s} {'Rec':>6s} {'F0.5':>6s}")
    print(f"{'='*90}")

    for c in config_names:
        m = cm[c]
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        print(f"{c:20s} {tp:3d} {fp:3d} {fn:3d} {tn:3d} {prec:6.2f} {rec:6.2f} {f05:6.2f}")

    print(f"\nTotal cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
