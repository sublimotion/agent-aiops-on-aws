#!/usr/bin/env python3
"""
Iteration 50: Meta-verification of adversarial attacks.

Hypothesis: v009 rejects both FPs and FNs with "incompleteness" attacks.
But FP attacks target behaviors LIKELY TESTED (core functionality), while
FN attacks target behaviors UNLIKELY TESTED (secondary methods like deconstruct()).

Two-phase:
  Phase 1: Run v009 on 9 critical patches, extract full attack reasoning
  Phase 2: Meta-verify each attack — "Is this concern likely caught by a regression test?"
  Decision: If meta-verifier says "unlikely to be tested" → dismiss attack → patch passes

If meta-verification dismisses FN attacks but upholds FP attacks → recall improves.

Usage:
  python3 test_meta_verify.py
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

V009 = VERSIONS_DIR / "v009_adversarial.md"

CRITICAL = [
    "astropy__astropy-14365",
    "astropy__astropy-14995",
    "pallets__flask-4992",
    "pallets__flask-4045",
    "pallets__flask-5063",
    "django__django-10924",
    "django__django-11001",
    "sphinx-doc__sphinx-10325",
    "sphinx-doc__sphinx-11445",
]

META_VERIFY_PROMPT = """You are a QA test analyst. A code reviewer raised a concern about a patch. Your job is to determine whether this concern would be caught by a MINIMAL regression test for the reported bug.

## Bug Report

{problem}

## Reviewer's Concern About the Patch

{attack}

## Question

Would a MINIMAL regression test for this specific bug report likely test the behavior described in the reviewer's concern?

Think about:
- Regression tests typically focus on the SPECIFIC broken behavior described in the bug report
- They usually test ONE code path with ONE input that demonstrates the bug
- They do NOT comprehensively test all related functionality
- They do NOT test migration serialization, backward compatibility, or secondary methods unless the bug report specifically mentions them

Respond with ONLY a JSON object:
```json
{{
  "test_relevance": "likely_tested" | "unlikely_tested" | "uncertain",
  "reasoning": "1-2 sentence explanation",
  "confidence": 0.0-1.0
}}
```"""


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


def run_v009(rubric_text: str, problem: str, diff: str) -> dict:
    """Run v009 once at temp=0.0 to get full attack reasoning."""
    prompt = f"""{rubric_text}

## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    resp = call_haiku(prompt, 0.0)
    parsed = parse_json_output(resp["text"])
    return {
        "verdict": (parsed or {}).get("verdict", "error"),
        "reasoning": (parsed or {}).get("reasoning", ""),
        "overall_score": (parsed or {}).get("overall_score"),
        "cost_usd": resp["cost_usd"],
    }


def meta_verify(problem: str, attack: str) -> dict:
    """Phase 2: Is this attack likely caught by a regression test?"""
    prompt = META_VERIFY_PROMPT.format(problem=problem[:6000], attack=attack[:2000])
    resp = call_haiku(prompt, 0.0)
    parsed = parse_json_output(resp["text"])
    return {
        "test_relevance": (parsed or {}).get("test_relevance", "error"),
        "reasoning": (parsed or {}).get("reasoning", ""),
        "confidence": (parsed or {}).get("confidence", 0),
        "cost_usd": resp["cost_usd"],
    }


def main():
    print("Loading SWE-bench data...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    gold = {}
    with open(GOLD_FILE) as f:
        for line in f:
            row = json.loads(line)
            gold[row["instance_id"]] = row["passed"]

    rubric_text = V009.read_text()
    total_cost = 0.0

    print(f"\nPhase 1: Run v009 on {len(CRITICAL)} critical patches")
    print(f"Phase 2: Meta-verify each attack for test relevance\n")

    results = []

    for idx, iid in enumerate(CRITICAL):
        diff_file = DIFFS_DIR / f"{iid}.diff"
        if not diff_file.exists():
            print(f"[{idx+1}] {iid}: diff not found, skipping")
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        gold_pass = gold.get(iid)
        label = "PASS" if gold_pass else "FAIL"
        short = iid.split("__")[-1]

        print(f"[{idx+1}/{len(CRITICAL)}] {short} (gold={label})...", end=" ", flush=True)

        # Phase 1: Get v009 attack
        v009_result = run_v009(rubric_text, problem, diff_text)
        total_cost += v009_result["cost_usd"]
        attack = v009_result["reasoning"]

        print(f"v009={v009_result['verdict'][:3]}", end=" ", flush=True)

        # Phase 2: Meta-verify the attack
        meta_result = meta_verify(problem, attack)
        total_cost += meta_result["cost_usd"]

        # Decision: if attack is "unlikely_tested" → dismiss → patch passes
        attack_dismissed = meta_result["test_relevance"] == "unlikely_tested"
        meta_pass = attack_dismissed or v009_result["verdict"] == "likely_correct"

        print(f"| meta={meta_result['test_relevance'][:7]} conf={meta_result['confidence']:.1f} | {'DISMISS' if attack_dismissed else 'UPHOLD'}")
        print(f"    v009 attack: {attack[:150]}")
        print(f"    meta reason: {meta_result['reasoning'][:150]}")

        results.append({
            "instance_id": iid,
            "gold_pass": gold_pass,
            "v009_verdict": v009_result["verdict"],
            "v009_attack": attack[:500],
            "meta_relevance": meta_result["test_relevance"],
            "meta_confidence": meta_result["confidence"],
            "meta_reasoning": meta_result["reasoning"][:300],
            "attack_dismissed": attack_dismissed,
            "meta_pass": meta_pass,
        })

    # Confusion matrix
    print(f"\n{'='*70}")
    print(f"Meta-verification results:")
    print(f"{'='*70}")

    tp = fp = fn = tn = 0
    for r in results:
        pred = r["meta_pass"]
        gold_p = r["gold_pass"]
        if pred and gold_p: tp += 1
        elif pred and not gold_p: fp += 1
        elif not pred and gold_p: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.2f} Recall={rec:.2f} F0.5={f05:.2f}")

    print(f"\nPer-patch breakdown:")
    for r in results:
        iid_short = r["instance_id"].split("__")[-1]
        gold_label = "PASS" if r["gold_pass"] else "FAIL"
        meta_label = "DISMISS" if r["attack_dismissed"] else "UPHOLD"
        correct = (r["meta_pass"] == r["gold_pass"])
        mark = "OK" if correct else "WRONG"
        print(f"  {iid_short:30s} gold={gold_label} meta={meta_label} ({r['meta_relevance']}) → {mark}")

    print(f"\nTotal cost: ${total_cost:.2f}")

    # Save results
    output_file = RESULTS_DIR / "iter50_meta_verify.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
