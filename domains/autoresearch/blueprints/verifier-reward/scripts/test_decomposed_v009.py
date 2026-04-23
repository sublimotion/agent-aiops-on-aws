#!/usr/bin/env python3
"""
Iteration 45: Decomposed v009 — break problem into testable claims, run v009 on each.

Hypothesis: v009 rejects FNs because it evaluates the WHOLE problem (finds incompleteness).
If we decompose the problem into individual testable claims and run v009 on each,
a patch that correctly fixes ONE claim will get "likely_correct" for that claim,
even if other claims are unfixed.

Two-phase:
  Phase 1: Decompose problem → list of 2-5 testable claims
  Phase 2: For each claim, run v009 with ONLY that claim as the problem statement
  Decision: PASS if v009 says "likely_correct" for ANY claim (2+/4 per claim)

Tests on 9 critical patches (5 FPs + 4 FNs).
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

DECOMPOSE_PROMPT = """You are a QA engineer. Given this bug report, decompose it into INDIVIDUAL testable claims. Each claim should describe ONE specific behavior that could be tested independently.

## Bug Report

{problem}

## Instructions

List 2-5 testable claims. Each claim should be:
- A single, specific behavior (e.g., "function X returns Y when called with Z")
- Independently verifiable (could be tested without checking other claims)
- Directly mentioned or implied by the bug report

Format: Return a JSON array of strings, each being one testable claim.

```json
["claim 1", "claim 2", ...]
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
    # Try array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def decompose_problem(problem: str) -> dict:
    """Phase 1: Decompose problem into testable claims."""
    prompt = DECOMPOSE_PROMPT.format(problem=problem[:8000])
    resp = call_haiku(prompt, temperature=0.0)
    claims = parse_json_output(resp["text"])
    if isinstance(claims, list):
        return {"claims": claims[:5], "cost_usd": resp["cost_usd"]}
    return {"claims": [problem[:2000]], "cost_usd": resp["cost_usd"]}


def verify_claim(rubric_text: str, claim: str, diff: str, temp: float) -> dict:
    """Run v009 on a single claim + patch."""
    prompt = f"""{rubric_text}

## Problem Statement

{claim}

## Proposed Patch

```diff
{diff[:100000]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        resp = call_haiku(prompt, temp)
        parsed = parse_json_output(resp["text"])
        verdict = (parsed or {}).get("verdict", "error") if parsed else "error"
        return {"verdict": verdict, "cost_usd": resp["cost_usd"], "parsed": parsed}
    except Exception as e:
        return {"verdict": "error", "cost_usd": 0, "error": str(e)[:200]}


def run_patch(problem: str, diff: str) -> dict:
    """Run decomposed v009 on one patch."""
    rubric_text = V009.read_text()

    # Phase 1: Decompose
    decomp = decompose_problem(problem)
    claims = decomp["claims"]
    total_cost = decomp["cost_usd"]

    # Phase 2: For each claim, run v009 4x (1×t=0.0 + 3×t=0.3)
    claim_results = {}
    all_calls = []
    for ci, claim in enumerate(claims):
        for ri, temp in enumerate([0.0, 0.3, 0.3, 0.3]):
            all_calls.append((f"c{ci}_r{ri}", ci, claim, temp))

    results = {}
    with ThreadPoolExecutor(max_workers=min(20, len(all_calls))) as pool:
        futures = {
            pool.submit(verify_claim, rubric_text, claim, diff, temp): name
            for name, ci, claim, temp in all_calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()
            total_cost += results[name]["cost_usd"]

    # Compute per-claim verdicts
    claim_verdicts = {}
    for ci, claim in enumerate(claims):
        lc_count = sum(1 for ri in range(4)
                       if results.get(f"c{ci}_r{ri}", {}).get("verdict") == "likely_correct")
        claim_verdicts[f"claim_{ci}"] = {
            "text": claim[:150],
            "lc_count": lc_count,
            "pass_2of4": lc_count >= 2,
        }

    # Decision: ANY claim passes (2+/4) → patch passes
    any_claim_passes = any(cv["pass_2of4"] for cv in claim_verdicts.values())
    # Also track: ALL claims pass (intersection)
    all_claims_pass = all(cv["pass_2of4"] for cv in claim_verdicts.values())
    # And: how many claims pass
    n_claims_pass = sum(1 for cv in claim_verdicts.values() if cv["pass_2of4"])

    return {
        "claims": claims,
        "claim_verdicts": claim_verdicts,
        "any_claim_passes": any_claim_passes,
        "all_claims_pass": all_claims_pass,
        "n_claims_pass": n_claims_pass,
        "n_claims": len(claims),
        "total_cost_usd": round(total_cost, 6),
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

    diff_files = [DIFFS_DIR / f"{iid}.diff" for iid in CRITICAL
                  if (DIFFS_DIR / f"{iid}.diff").exists()]

    print(f"Patches: {len(diff_files)}\n")

    total_cost = 0.0
    cm_any = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    cm_all = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    output_file = RESULTS_DIR / "iter45_decomposed_v009.jsonl"

    for idx, diff_file in enumerate(diff_files):
        iid = diff_file.stem
        if iid not in gold:
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        gold_pass = gold[iid]
        label = "PASS" if gold_pass else "FAIL"

        print(f"[{idx+1}/{len(diff_files)}] {iid} (gold={label})...", flush=True)

        result = run_patch(problem, diff_text)
        total_cost += result["total_cost_usd"]

        # Update confusion matrices
        for cm_name, pred in [("any", result["any_claim_passes"]), ("all", result["all_claims_pass"])]:
            cm = cm_any if cm_name == "any" else cm_all
            if pred and gold_pass: cm["tp"] += 1
            elif pred and not gold_pass: cm["fp"] += 1
            elif not pred and gold_pass: cm["fn"] += 1
            else: cm["tn"] += 1

        # Print
        status = "ANY_PASS" if result["any_claim_passes"] else "ALL_FAIL"
        print(f"  {result['n_claims']} claims, {result['n_claims_pass']} pass | {status} | ${result['total_cost_usd']:.3f}")
        for k, cv in result["claim_verdicts"].items():
            mark = "✓" if cv["pass_2of4"] else "✗"
            print(f"    {mark} [{cv['lc_count']}/4] {cv['text'][:100]}")

        row = {"instance_id": iid, "gold_pass": gold_pass, **result}
        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary
    print(f"\n{'='*70}")
    for name, cm in [("ANY claim passes", cm_any), ("ALL claims pass", cm_all)]:
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:25s}: TP={tp} FP={fp} FN={fn} TN={tn} | prec={prec:.2f} rec={rec:.2f} F0.5={f05:.2f}")

    print(f"\nTotal cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
