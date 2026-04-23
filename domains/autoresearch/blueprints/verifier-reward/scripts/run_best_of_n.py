#!/usr/bin/env python3
"""
Iteration 24: Best-of-N verifier selection on Devstral SERA candidates.

For each issue with multiple candidate patches (_a1, _a2, _a3 suffixes),
runs v001∩v009 ensemble and selects the best candidate. Selection rules:
  1. If any candidate passes ensemble (v001 lc AND v009 2+/3 lc) → pick it
  2. If multiple pass → pick highest v009_lc_count, then highest v001 score
  3. If none pass → pick highest v009_lc_count (least rejected)

Compares verifier selection against random selection and VL selection.

Usage:
  python3 run_best_of_n.py
  python3 run_best_of_n.py --limit 5  # first 5 issues only
  python3 run_best_of_n.py --resume    # resume interrupted run
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
CANDIDATES_DIR = RESULTS_DIR / "diffs" / "devstral_sera_candidates"
V001_RUBRIC = VERSIONS_DIR / "v001_baseline.md"
V009_RUBRIC = VERSIONS_DIR / "v009_adversarial.md"

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
HAIKU_PRICING = (0.80, 4.00)  # per 1M tokens (input, output)

V009_RUNS = 3
V009_THRESHOLD = 2


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
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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
    """Load candidate diffs grouped by issue.

    Returns: {instance_id: [(attempt_id, diff_content), ...]}
    """
    candidates = defaultdict(list)
    for f in sorted(CANDIDATES_DIR.glob("*_a[0-9]*.diff")):
        # Extract instance_id and attempt from filename like django__django-10914_a1.diff
        match = re.match(r"(.+)_(a\d+)\.diff$", f.name)
        if match:
            instance_id = match.group(1)
            attempt = match.group(2)
            candidates[instance_id].append((attempt, f.read_text()))
    return dict(candidates)


def load_problems() -> dict:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def main():
    parser = argparse.ArgumentParser(description="Iter 24: Best-of-N verifier selection")
    parser.add_argument("--limit", type=int, help="Limit to first N issues")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_file = RESULTS_DIR / "t6c_best_of_n_selection.jsonl"
    summary_file = RESULTS_DIR / "t6c_best_of_n_summary.json"

    # Load data
    print("Loading candidates...")
    candidates = load_candidates()
    print(f"Found {sum(len(v) for v in candidates.values())} candidates across {len(candidates)} issues")

    print("Loading problem statements...")
    problems = load_problems()

    issues = sorted(candidates.keys())
    if args.limit:
        issues = issues[:args.limit]

    # Resume support
    completed = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                row = json.loads(line)
                completed.add(row["instance_id"])
        print(f"Resuming: {len(completed)} already done")

    total_cost = 0.0
    selections = []

    for i, instance_id in enumerate(issues):
        if instance_id in completed:
            continue

        problem = problems.get(instance_id, "")
        issue_candidates = candidates[instance_id]

        print(f"\n[{i+1}/{len(issues)}] {instance_id} ({len(issue_candidates)} candidates)")

        # Run ensemble on each candidate
        candidate_results = []
        for attempt, diff in issue_candidates:
            print(f"  {attempt}...", end=" ", flush=True)
            result = run_ensemble(problem, diff)
            result["attempt"] = attempt
            candidate_results.append(result)
            total_cost += result["total_cost_usd"]

            status = "PASS" if result["ensemble_pass"] else "FAIL"
            print(f"{status} v001={result['v001_verdict']} v009={result['v009_lc_count']}/3 ${result['total_cost_usd']:.4f}")

        # Selection: pick best candidate
        passing = [r for r in candidate_results if r["ensemble_pass"]]
        if passing:
            # Among passing candidates, pick highest v009_lc_count, then v001_score
            selected = max(passing, key=lambda r: (r["v009_lc_count"], r["v001_score"]))
            selection_method = "ensemble_pass"
        else:
            # None passed — pick least rejected (highest v009_lc_count, then v001_score)
            selected = max(candidate_results, key=lambda r: (r["v009_lc_count"], r["v001_score"]))
            selection_method = "least_rejected"

        print(f"  → Selected: {selected['attempt']} ({selection_method})")

        row = {
            "instance_id": instance_id,
            "num_candidates": len(issue_candidates),
            "selected_attempt": selected["attempt"],
            "selection_method": selection_method,
            "selected_ensemble_pass": selected["ensemble_pass"],
            "selected_v001_verdict": selected["v001_verdict"],
            "selected_v009_lc": selected["v009_lc_count"],
            "all_results": candidate_results,
            "issue_cost_usd": round(sum(r["total_cost_usd"] for r in candidate_results), 6),
        }
        selections.append(row)

        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary
    n_issues = len(selections)
    n_ensemble_pass = sum(1 for s in selections if s["selected_ensemble_pass"])
    n_any_pass = sum(1 for s in selections if any(r["ensemble_pass"] for r in s["all_results"]))

    print(f"\n{'='*60}")
    print(f"Best-of-N Selection Summary")
    print(f"{'='*60}")
    print(f"Issues: {n_issues}")
    print(f"Issues with ≥1 passing candidate: {n_any_pass}")
    print(f"Selected candidates that pass ensemble: {n_ensemble_pass}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost/issue: ${total_cost/max(n_issues,1):.4f}")

    # Selection method breakdown
    methods = defaultdict(int)
    for s in selections:
        methods[s["selection_method"]] += 1
    print(f"\nSelection methods: {dict(methods)}")

    # Save selected diffs for gold eval
    selected_dir = RESULTS_DIR / "diffs" / "devstral_sera_bon_selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for s in selections:
        attempt = s["selected_attempt"]
        instance_id = s["instance_id"]
        src = CANDIDATES_DIR / f"{instance_id}_{attempt}.diff"
        dst = selected_dir / f"{instance_id}.diff"
        if src.exists():
            dst.write_text(src.read_text())

    print(f"\nSelected diffs saved to: {selected_dir}")
    print(f"Run gold_eval.py on these diffs to get pass rate.")

    summary = {
        "experiment": "t6c_best_of_n",
        "n_issues": n_issues,
        "n_ensemble_pass": n_ensemble_pass,
        "n_any_pass": n_any_pass,
        "total_cost_usd": round(total_cost, 4),
        "selection_methods": dict(methods),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
