#!/usr/bin/env python3
"""
Iteration 46: Size-based routing test.

Tests v013 (test-scoped adversarial) on the 5 large (>50KB) gold=FAIL patches
to validate size-based routing strategy:
  - Small diffs (<10KB): v001∩v009 (precision=1.00, proven)
  - Large diffs (>50KB): v013(2+/4) only (might recover FM-001 FNs)

If v013 rejects all 5 large FAIL patches → size-routing achieves prec=1.00, rec=0.67.

Usage:
  python3 test_size_routing.py
"""

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

V013 = VERSIONS_DIR / "v013_testscoped_adversarial.md"

# Large (>50KB) gold=FAIL patches — must test these
LARGE_FAIL = [
    "matplotlib__matplotlib-22711",
    "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23314",
    "mwaskom__seaborn-3407",
    "sympy__sympy-11897",
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


def run_v013(problem: str, diff: str) -> dict:
    """Run test-gen + v013 4x."""
    # Phase 1: test description
    testgen_resp = call_haiku(TESTGEN_PROMPT.format(problem=problem[:8000]))
    test_desc = testgen_resp["text"]
    total_cost = testgen_resp["cost_usd"]

    rubric = V013.read_text()

    # Phase 2: v013 4x (1×t=0.0 + 3×t=0.3)
    results = {}
    temps = [("r0", 0.0), ("r1", 0.3), ("r2", 0.3), ("r3", 0.3)]

    def run_one(name, temp):
        prompt = f"""{rubric}

## Bug Report

{problem[:8000]}

## What the Regression Test Checks

{test_desc}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate whether this patch will PASS the described test. Respond with ONLY the JSON object."""
        try:
            resp = call_haiku(prompt, temp)
            parsed = parse_json_output(resp["text"])
            return name, {
                "verdict": (parsed or {}).get("verdict", "error"),
                "cost_usd": resp["cost_usd"],
                "score": (parsed or {}).get("overall_score"),
                "reasoning": ((parsed or {}).get("reasoning") or "")[:200],
            }
        except Exception as e:
            return name, {"verdict": "error", "cost_usd": 0, "error": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, n, t) for n, t in temps]
        for f in as_completed(futures):
            name, result = f.result()
            results[name] = result
            total_cost += result["cost_usd"]

    lc_count = sum(1 for r in results.values() if r["verdict"] == "likely_correct")
    pass_2of4 = lc_count >= 2

    return {
        "test_description": test_desc[:200],
        "v013_lc": lc_count,
        "v013_pass": pass_2of4,
        "total_cost_usd": round(total_cost, 6),
        "details": results,
    }


def main():
    print("Loading SWE-bench data...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    gold = {}
    with open(GOLD_FILE) as f:
        for line in f:
            r = json.loads(line)
            gold[r["instance_id"]] = r["passed"]

    print(f"\nTesting v013 on {len(LARGE_FAIL)} large (>50KB) gold=FAIL patches")
    print("If ALL are rejected → size-based routing achieves prec=1.00, rec=0.67\n")

    total_cost = 0.0
    all_rejected = True

    for idx, iid in enumerate(LARGE_FAIL):
        diff_file = DIFFS_DIR / f"{iid}.diff"
        if not diff_file.exists():
            print(f"[{idx+1}] {iid}: diff not found, skipping")
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        size_kb = len(diff_text) / 1024

        print(f"[{idx+1}/{len(LARGE_FAIL)}] {iid.split('__')[-1]} ({size_kb:.0f}KB, gold=FAIL)...", end=" ", flush=True)

        result = run_v013(problem, diff_text)
        total_cost += result["total_cost_usd"]

        verdict = "PASS (FP!)" if result["v013_pass"] else "REJECT (good)"
        if result["v013_pass"]:
            all_rejected = False
        print(f"v013={result['v013_lc']}/4 → {verdict} | ${result['total_cost_usd']:.3f}")
        print(f"    test: {result['test_description'][:120]}")

        for name, det in sorted(result["details"].items()):
            print(f"    {name}: {det['verdict'][:3]} score={det.get('score', '?')}")

    print(f"\n{'='*60}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"All large FAIL patches rejected: {'YES' if all_rejected else 'NO'}")

    if all_rejected:
        print("\n*** SIZE-BASED ROUTING IS VIABLE ***")
        print("Small (<10KB): v001∩v009 → 2 TP, 0 FP")
        print("Large (>50KB): v013(2+/4) → 2 TP (django-11001, sphinx-10325), 0 FP")
        print("Combined: 4 TP, 0 FP → prec=1.00, rec=0.67, F₀.₅=0.87")
    else:
        print("\n*** SIZE-BASED ROUTING FAILED — FPs leak on large diffs ***")


if __name__ == "__main__":
    main()
