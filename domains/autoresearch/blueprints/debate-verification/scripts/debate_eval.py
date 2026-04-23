#!/usr/bin/env python3
"""
Debate Verification Experiment: 3-agent debate protocol for SWE-bench patch verification.

Runs Advocate + Challenger + Judge debate via Bedrock Haiku 4.5.
Supports all 4 phases: prompt calibration (n=10), pilot (n=50), full (n=483), combination.

Usage:
    # Phase 1: prompt calibration
    python3 debate_eval.py --phase 1 --n 10

    # Phase 2: pilot
    python3 debate_eval.py --phase 2 --n 50

    # Phase 3: full evaluation
    python3 debate_eval.py --phase 3

    # Phase 4: combination analysis (no API calls, analyzes existing results)
    python3 debate_eval.py --phase 4

    # Variants
    python3 debate_eval.py --phase 2 --variant 2round
    python3 debate_eval.py --phase 2 --variant ensemble
    python3 debate_eval.py --phase 2 --variant asymmetric
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
}
PRICING = {  # per million tokens (input, output)
    "haiku": (0.80, 4.00),
    "sonnet": (3.00, 15.00),
}
MAX_CONCURRENT = 10


# --- Bedrock API ---
def call_bedrock_sync(prompt: str, system: str, model_key: str = "haiku",
                      temperature: float = 0.0, max_tokens: int = 2048) -> dict:
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.monotonic()
    response = client.invoke_model(
        modelId=BEDROCK_MODELS[model_key],
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]
    usage = result.get("usage", {})
    ip, op = PRICING[model_key]
    cost = (usage.get("input_tokens", 0) * ip + usage.get("output_tokens", 0) * op) / 1_000_000
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_ms": latency_ms,
        "cost_usd": cost,
    }


async def call_bedrock(prompt: str, system: str, model_key: str = "haiku",
                       temperature: float = 0.0) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_bedrock_sync, prompt, system, model_key, temperature)


def parse_json_output(text: str) -> dict:
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# --- Data Loading ---
def load_swebench(local_cache: str = None) -> dict:
    """Load SWE-bench instances. Uses local JSONL cache if available, else HuggingFace."""
    # Try local cache first
    cache_candidates = [
        local_cache,
        str(RESULTS_DIR / "swebench_lite.jsonl"),
        str(SCRIPT_DIR.parent / "results" / "swebench_lite.jsonl"),
    ]
    for path in cache_candidates:
        if path and os.path.exists(path):
            data = {}
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    data[row["instance_id"]] = row
            print(f"  Loaded from local cache: {path}")
            return data

    # Fall back to HuggingFace
    from datasets import load_dataset
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        print(f"  Using SWE-bench Lite from HuggingFace")
    except Exception:
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        print(f"  Using SWE-bench Verified from HuggingFace")
    return {row["instance_id"]: row for row in ds}


def load_gold_labels(path: str) -> dict:
    """Load gold pass/fail labels. Returns {instance_id: bool}."""
    labels = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels[d["instance_id"]] = d["passed"]
    return labels


def load_predictions(path: str) -> dict:
    """Load model predictions (patches). Returns {instance_id: patch_str}."""
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            preds[d["instance_id"]] = d.get("model_patch", "")
    return preds


def load_v009_results(path: str) -> dict:
    """Load existing v009 verification results for combination analysis."""
    results = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            results[d["instance_id"]] = d
    return results


# --- Debate Protocol ---
async def run_debate(instance_id: str, problem: str, patch: str,
                     variant: str = "base", semaphore: asyncio.Semaphore = None) -> dict:
    """Run a single debate for one instance."""
    advocate_system = (PROMPTS_DIR / "advocate.md").read_text()
    challenger_system = (PROMPTS_DIR / "challenger.md").read_text()
    judge_system = (PROMPTS_DIR / "judge.md").read_text()

    user_context = f"""## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{patch[:100000]}
```"""

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    agent_outputs = {}

    async def _call(role, system, prompt, model="haiku", temp=0.0):
        nonlocal total_cost, total_input_tokens, total_output_tokens
        resp = await call_bedrock(prompt, system, model, temp)
        total_cost += resp["cost_usd"]
        total_input_tokens += resp["input_tokens"]
        total_output_tokens += resp["output_tokens"]
        try:
            parsed = parse_json_output(resp["text"])
        except (ValueError, json.JSONDecodeError):
            parsed = {"error": "parse_failed", "raw": resp["text"][:500]}
        agent_outputs[role] = {"parsed": parsed, "raw": resp["text"][:2000], "latency_ms": resp["latency_ms"]}
        return parsed

    # Round 1: Advocate and Challenger run in parallel (same input, different prompts)
    challenger_model = "sonnet" if variant == "asymmetric" else "haiku"
    advocate_task = _call("advocate", advocate_system, user_context)
    challenger_task = _call("challenger", challenger_system, user_context, model=challenger_model)
    advocate_result, challenger_result = await asyncio.gather(advocate_task, challenger_task)

    # Judge sees both arguments
    judge_prompt = f"""{user_context}

## Advocate's Argument (patch is CORRECT)

{json.dumps(advocate_result, indent=2)[:4000]}

## Challenger's Argument (patch is WRONG)

{json.dumps(challenger_result, indent=2)[:4000]}

Now evaluate the debate and render your verdict."""

    judge_result = await _call("judge", judge_system, judge_prompt)

    # Optional: Round 2 (rebuttal)
    if variant == "2round":
        rebuttal_advocate_prompt = f"""{user_context}

## Challenger's Argument Against Your Position

{json.dumps(challenger_result, indent=2)[:4000]}

The Challenger argues the patch is wrong. Respond to their specific claims. Defend your position or concede if they found a real bug.

Respond with ONLY a JSON object:
```json
{{"rebuttal": "<your response to the Challenger's claims>", "concessions": ["<any points you concede>"], "final_stance": "<correct|concede>", "confidence": <float 0.0-1.0>}}
```"""

        rebuttal_challenger_prompt = f"""{user_context}

## Advocate's Argument For the Patch

{json.dumps(advocate_result, indent=2)[:4000]}

The Advocate argues the patch is correct. Respond to their specific evidence. Strengthen your attack or concede if their evidence is compelling.

Respond with ONLY a JSON object:
```json
{{"rebuttal": "<your response to the Advocate's claims>", "concessions": ["<any points you concede>"], "final_stance": "<broken|concede>", "confidence": <float 0.0-1.0>}}
```"""

        adv_rebuttal_task = _call("advocate_r2", advocate_system, rebuttal_advocate_prompt)
        chal_rebuttal_task = _call("challenger_r2", challenger_system, rebuttal_challenger_prompt, model=challenger_model)
        adv_rebuttal, chal_rebuttal = await asyncio.gather(adv_rebuttal_task, chal_rebuttal_task)

        judge_r2_prompt = f"""{user_context}

## Round 1 - Advocate
{json.dumps(advocate_result, indent=2)[:2000]}

## Round 1 - Challenger
{json.dumps(challenger_result, indent=2)[:2000]}

## Round 2 - Advocate Rebuttal
{json.dumps(adv_rebuttal, indent=2)[:2000]}

## Round 2 - Challenger Rebuttal
{json.dumps(chal_rebuttal, indent=2)[:2000]}

Now evaluate the full 2-round debate and render your final verdict."""

        judge_result = await _call("judge_r2", judge_system, judge_r2_prompt)

    verdict = judge_result.get("verdict", "UNKNOWN").upper()

    return {
        "instance_id": instance_id,
        "variant": variant,
        "verdict": verdict,
        "judge_confidence": judge_result.get("confidence", 0.0),
        "advocate_confidence": advocate_result.get("confidence", 0.0),
        "challenger_assessment": challenger_result.get("overall_assessment", "unknown"),
        "challenger_bugs_found": len(challenger_result.get("bugs_found", [])),
        "decisive_factor": judge_result.get("decisive_factor", ""),
        "agent_outputs": agent_outputs,
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_ensemble_debate(instance_id: str, problem: str, patch: str,
                               semaphore: asyncio.Semaphore) -> dict:
    """Run 3 independent debates, majority vote on verdicts."""
    tasks = [run_debate(instance_id, problem, patch, "base", semaphore) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    verdicts = [r["verdict"] for r in results]
    verdict_counts = Counter(verdicts)
    majority = verdict_counts.most_common(1)[0][0]
    total_cost = sum(r["total_cost_usd"] for r in results)
    return {
        "instance_id": instance_id,
        "variant": "ensemble",
        "verdict": majority,
        "individual_verdicts": verdicts,
        "verdict_counts": dict(verdict_counts),
        "total_cost_usd": round(total_cost, 6),
        "sub_results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- Metrics ---
def compute_metrics(results: list, gold_labels: dict) -> dict:
    """Compute precision, recall, AUC, ECE for debate results."""
    tp = fp = tn = fn = 0
    uncertain = 0
    scored = []

    for r in results:
        iid = r["instance_id"]
        verdict = r["verdict"]
        gold = gold_labels.get(iid)
        if gold is None:
            continue

        if verdict == "UNCERTAIN":
            uncertain += 1
            continue

        predicted_correct = verdict == "CORRECT"
        if predicted_correct and gold:
            tp += 1
        elif predicted_correct and not gold:
            fp += 1
        elif not predicted_correct and gold:
            fn += 1
        else:
            tn += 1

        scored.append({"confidence": r.get("judge_confidence", 0.5), "predicted": predicted_correct, "gold": gold})

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    coverage = total / (total + uncertain) if (total + uncertain) > 0 else 0.0

    # Bootstrap CI for precision and recall
    def bootstrap_ci(metric_fn, data, n_boot=1000, alpha=0.05):
        if not data:
            return (0.0, 0.0)
        rng = random.Random(42)
        vals = []
        for _ in range(n_boot):
            sample = [rng.choice(data) for _ in range(len(data))]
            vals.append(metric_fn(sample))
        vals.sort()
        lo = vals[int(alpha / 2 * n_boot)]
        hi = vals[int((1 - alpha / 2) * n_boot)]
        return (round(lo, 4), round(hi, 4))

    def precision_fn(sample):
        tp_ = sum(1 for s in sample if s["predicted"] and s["gold"])
        fp_ = sum(1 for s in sample if s["predicted"] and not s["gold"])
        return tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0

    def recall_fn(sample):
        tp_ = sum(1 for s in sample if s["predicted"] and s["gold"])
        fn_ = sum(1 for s in sample if not s["predicted"] and s["gold"])
        return tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0

    precision_ci = bootstrap_ci(precision_fn, scored)
    recall_ci = bootstrap_ci(recall_fn, scored)

    # ECE (10 equal-width bins on judge confidence)
    ece = 0.0
    bins = defaultdict(list)
    for s in scored:
        bin_idx = min(int(s["confidence"] * 10), 9)
        bins[bin_idx].append(s)
    for bin_idx, bin_items in bins.items():
        bin_acc = sum(1 for s in bin_items if s["predicted"] == s["gold"]) / len(bin_items)
        bin_conf = sum(s["confidence"] for s in bin_items) / len(bin_items)
        ece += len(bin_items) / len(scored) * abs(bin_acc - bin_conf)

    total_cost = sum(r.get("total_cost_usd", 0) for r in results)

    return {
        "n": len(results),
        "n_scored": total,
        "n_uncertain": uncertain,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "precision_ci": precision_ci,
        "recall": round(recall, 4),
        "recall_ci": recall_ci,
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "ece": round(ece, 4),
        "total_cost_usd": round(total_cost, 4),
    }


def combination_analysis(debate_results: list, gold_labels: dict,
                          v009_results: dict = None, svg_scores: dict = None) -> dict:
    """Phase 4: analyze debate + v009 + SVG combinations."""
    analysis = {}

    # Debate alone
    analysis["debate_alone"] = compute_metrics(debate_results, gold_labels)

    debate_by_id = {r["instance_id"]: r for r in debate_results}

    if v009_results:
        # v009 alone
        v009_converted = []
        for iid, v in v009_results.items():
            verdict_map = {"likely_correct": "CORRECT", "uncertain": "UNCERTAIN", "likely_incorrect": "INCORRECT"}
            v009_verdict = verdict_map.get(v.get("verdict"), "UNCERTAIN")
            v009_converted.append({"instance_id": iid, "verdict": v009_verdict, "judge_confidence": v.get("overall_score", 0.5)})
        analysis["v009_alone"] = compute_metrics(v009_converted, gold_labels)

        # Debate recovers v009 false negatives?
        v009_fn = set()
        for iid, v in v009_results.items():
            gold = gold_labels.get(iid)
            if gold and v.get("verdict") != "likely_correct":
                v009_fn.add(iid)

        debate_recovers = 0
        for iid in v009_fn:
            if iid in debate_by_id and debate_by_id[iid]["verdict"] == "CORRECT":
                debate_recovers += 1

        analysis["v009_false_negatives"] = len(v009_fn)
        analysis["debate_recovers_v009_fn"] = debate_recovers
        analysis["recovery_rate"] = round(debate_recovers / len(v009_fn), 4) if v009_fn else 0.0

        # Combined: accept if either says CORRECT
        combined = []
        all_ids = set(debate_by_id.keys()) | set(v009_results.keys())
        for iid in all_ids:
            debate_correct = debate_by_id.get(iid, {}).get("verdict") == "CORRECT"
            v009_correct = v009_results.get(iid, {}).get("verdict") == "likely_correct"
            verdict = "CORRECT" if (debate_correct or v009_correct) else "INCORRECT"
            combined.append({"instance_id": iid, "verdict": verdict, "judge_confidence": 0.5})
        analysis["combined_debate_v009"] = compute_metrics(combined, gold_labels)

    return analysis


# --- Main Experiment Runner ---
async def run_phase(phase: int, n: int, variant: str, gold_path: str,
                    predictions_path: str, v009_path: str = None, svg_path: str = None):
    print(f"\n{'='*60}")
    print(f"Phase {phase}: variant={variant}, n={n}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading SWE-bench Verified dataset...")
    swebench = load_swebench()
    print(f"  Loaded {len(swebench)} instances from HuggingFace")

    print("Loading gold labels...")
    gold_labels = load_gold_labels(gold_path)
    print(f"  Loaded {len(gold_labels)} gold labels ({sum(gold_labels.values())} passed)")

    predictions = None
    if predictions_path and os.path.exists(predictions_path):
        print("Loading predictions (patches)...")
        predictions = load_predictions(predictions_path)
        print(f"  Loaded {len(predictions)} predictions")

    # Phase 4: combination analysis only
    if phase == 4:
        # Load existing debate results
        debate_results_path = RESULTS_DIR / "phase3_base.jsonl"
        if not debate_results_path.exists():
            debate_results_path = RESULTS_DIR / "phase2_base.jsonl"
        if not debate_results_path.exists():
            print("ERROR: No debate results found for combination analysis")
            sys.exit(1)

        debate_results = []
        with open(debate_results_path) as f:
            for line in f:
                debate_results.append(json.loads(line))
        print(f"Loaded {len(debate_results)} debate results from {debate_results_path.name}")

        v009_results = load_v009_results(v009_path) if v009_path and os.path.exists(v009_path) else None
        svg_scores = None
        if svg_path and os.path.exists(svg_path):
            svg_scores = {}
            with open(svg_path) as f:
                for line in f:
                    d = json.loads(line)
                    svg_scores[d["instance_id"]] = d

        analysis = combination_analysis(debate_results, gold_labels, v009_results, svg_scores)
        report_path = RESULTS_DIR / "phase4_combination.json"
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nCombination analysis saved to {report_path}")
        print_combination_report(analysis)
        return

    # Select instances
    instance_ids = list(gold_labels.keys())

    if phase == 1:
        # Phase 1: 5 correct + 5 incorrect
        passed = [iid for iid, v in gold_labels.items() if v]
        failed = [iid for iid, v in gold_labels.items() if not v]
        random.seed(42)
        selected = random.sample(passed, min(5, len(passed))) + random.sample(failed, min(5, len(failed)))
    elif phase == 2:
        # Phase 2: balanced sample
        passed = [iid for iid, v in gold_labels.items() if v]
        failed = [iid for iid, v in gold_labels.items() if not v]
        random.seed(42)
        half = n // 2
        selected = random.sample(passed, min(half, len(passed))) + random.sample(failed, min(n - half, len(failed)))
    else:
        # Phase 3: all instances
        selected = instance_ids

    # Filter to instances we have data for
    available = [iid for iid in selected if iid in swebench]
    if predictions:
        available = [iid for iid in available if iid in predictions]
    print(f"\nRunning debate on {len(available)} instances...")

    # Check for existing results (resume support)
    output_path = RESULTS_DIR / f"phase{phase}_{variant}.jsonl"
    completed_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                completed_ids.add(d["instance_id"])
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [iid for iid in available if iid not in completed_ids]
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("All instances already completed!")
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        completed = 0
        total = len(remaining)
        batch_start = time.monotonic()

        async def process_one(iid):
            nonlocal completed
            instance = swebench[iid]
            problem = instance["problem_statement"]
            patch = predictions[iid] if predictions and iid in predictions else instance.get("patch", "")

            if variant == "ensemble":
                result = await run_ensemble_debate(iid, problem, patch, semaphore)
            else:
                result = await run_debate(iid, problem, patch, variant, semaphore)

            # Append result
            with open(output_path, "a") as f:
                # Strip raw outputs for storage efficiency
                slim = {k: v for k, v in result.items() if k != "agent_outputs"}
                slim["advocate_raw"] = result.get("agent_outputs", {}).get("advocate", {}).get("raw", "")[:500]
                slim["challenger_raw"] = result.get("agent_outputs", {}).get("challenger", {}).get("raw", "")[:500]
                slim["judge_raw"] = result.get("agent_outputs", {}).get("judge", {}).get("raw", "")[:500]
                f.write(json.dumps(slim) + "\n")

            completed += 1
            elapsed = time.monotonic() - batch_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{total}] {iid}: {result['verdict']} "
                  f"(${result['total_cost_usd']:.4f}) "
                  f"[{rate:.1f}/min, ETA {eta/60:.1f}m]")
            return result

        # Process in batches of MAX_CONCURRENT to avoid semaphore starvation
        all_batch_results = []
        for batch_start_idx in range(0, len(remaining), MAX_CONCURRENT):
            batch = remaining[batch_start_idx:batch_start_idx + MAX_CONCURRENT]
            batch_tasks = [process_one(iid) for iid in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_batch_results.extend(batch_results)

        errors = [r for r in all_batch_results if isinstance(r, Exception)]
        if errors:
            print(f"\n{len(errors)} errors occurred:")
            for e in errors[:5]:
                print(f"  {type(e).__name__}: {e}")

    # Load all results and compute metrics
    all_results = []
    with open(output_path) as f:
        for line in f:
            all_results.append(json.loads(line))

    metrics = compute_metrics(all_results, gold_labels)
    metrics_path = RESULTS_DIR / f"phase{phase}_{variant}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_metrics_report(phase, variant, metrics)

    # Phase 2 gate check
    if phase == 2:
        if metrics["precision"] < 0.70 or metrics["recall"] < 0.15:
            print(f"\n⚠ GATE FAILED: precision={metrics['precision']:.3f} (need >0.70), "
                  f"recall={metrics['recall']:.3f} (need >0.15)")
            print("Review results before proceeding to Phase 3.")
        else:
            print(f"\n✓ GATE PASSED: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")
            print("Ready for Phase 3.")


def print_metrics_report(phase, variant, m):
    print(f"\n{'='*60}")
    print(f"Phase {phase} Results: {variant}")
    print(f"{'='*60}")
    print(f"  N={m['n']}, scored={m['n_scored']}, uncertain={m['n_uncertain']}")
    print(f"  TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")
    print(f"  Precision: {m['precision']:.4f} {m['precision_ci']}")
    print(f"  Recall:    {m['recall']:.4f} {m['recall_ci']}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Coverage:  {m['coverage']:.4f}")
    print(f"  ECE:       {m['ece']:.4f}")
    print(f"  Cost:      ${m['total_cost_usd']:.4f}")


def print_combination_report(analysis):
    print(f"\n{'='*60}")
    print(f"Combination Analysis")
    print(f"{'='*60}")
    for key, val in analysis.items():
        if isinstance(val, dict) and "precision" in val:
            print(f"\n  {key}:")
            print(f"    Precision: {val['precision']:.4f}, Recall: {val['recall']:.4f}, F1: {val['f1']:.4f}")
        elif isinstance(val, (int, float)):
            print(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(description="Debate Verification Experiment")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=None, help="Number of instances (default: phase-dependent)")
    parser.add_argument("--variant", default="base", choices=["base", "2round", "ensemble", "asymmetric"])
    parser.add_argument("--gold", default=None, help="Path to gold labels JSONL")
    parser.add_argument("--predictions", default=None, help="Path to predictions JSONL")
    parser.add_argument("--v009", default=None, help="Path to v009 results JSONL (for Phase 4)")
    parser.add_argument("--svg", default=None, help="Path to SVG scores JSONL (for Phase 4)")
    args = parser.parse_args()

    # Default n per phase
    if args.n is None:
        args.n = {1: 10, 2: 50, 3: 483, 4: 0}[args.phase]

    # Default data paths — try common locations
    home = Path.home()
    if args.gold is None:
        candidates = [
            Path("../verifier-reward/results/gold_swebench_verified_sonnet.jsonl"),
            home / "gold_swebench_verified_sonnet.jsonl",
        ]
        for c in candidates:
            if c.exists():
                args.gold = str(c)
                break
        if args.gold is None:
            print("ERROR: Could not find gold labels. Specify with --gold")
            sys.exit(1)

    if args.predictions is None:
        candidates = [
            Path("../verification-primitives-swebench/results/predictions_lite.jsonl"),
            home / "results/predictions_lite.jsonl",
        ]
        for c in candidates:
            if c.exists():
                args.predictions = str(c)
                break

    asyncio.run(run_phase(
        phase=args.phase,
        n=args.n,
        variant=args.variant,
        gold_path=args.gold,
        predictions_path=args.predictions,
        v009_path=args.v009,
        svg_path=args.svg,
    ))


if __name__ == "__main__":
    main()
