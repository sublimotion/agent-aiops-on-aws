#!/usr/bin/env python3
"""
T4: Cross-verifier transfer experiment.

Tests whether the v001∩v009 ensemble rubrics work with non-Claude models as
the verifier. If precision holds, the rubric is model-agnostic. If it drops,
the breakthrough is Claude-specific.

Models tested:
  - devstral2: Mistral Devstral 2 123B (coding-focused)
  - mistral-large: Mistral Large 3 675B (general-purpose)
  - nova-pro: Amazon Nova Pro (different model family entirely)

Runs on the same dev set (sonnet patches, 49 issues) used in Phase 2.

Usage:
  python3 run_cross_verifier.py --verifier devstral2
  python3 run_cross_verifier.py --verifier mistral-large
  python3 run_cross_verifier.py --verifier nova-pro
  python3 run_cross_verifier.py --verifier all
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

SCRIPT_DIR = Path(__file__).resolve().parent
BLUEPRINT_DIR = SCRIPT_DIR.parent
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"
RESULTS_DIR = BLUEPRINT_DIR / "results"
DIFFS_DIR = RESULTS_DIR / "diffs"
V001_RUBRIC = VERSIONS_DIR / "v001_baseline.md"
V009_RUBRIC = VERSIONS_DIR / "v009_adversarial.md"

# Non-Claude models on Bedrock
VERIFIER_MODELS = {
    "devstral2": {
        "model_id": "mistral.devstral-2-123b",
        "api": "mistral",  # Mistral chat completions format
        "pricing": (0.60, 1.80),  # per 1M tokens (input, output) — estimate
    },
    "mistral-large": {
        "model_id": "mistral.mistral-large-3-675b-instruct",
        "api": "mistral",
        "pricing": (2.00, 6.00),
    },
    "nova-pro": {
        "model_id": "amazon.nova-pro-v1:0",
        "api": "nova",
        "pricing": (0.80, 3.20),
    },
    # Claude baseline for comparison
    "haiku": {
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "api": "anthropic",
        "pricing": (0.80, 4.00),
    },
}

# Ensemble parameters (same as production)
V009_RUNS = 3
V009_THRESHOLD = 2


def call_model(prompt: str, model_key: str, temperature: float = 0.0) -> dict:
    """Call a model via Bedrock, handling different API formats."""
    config = VERIFIER_MODELS[model_key]
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    api = config["api"]
    model_id = config["model_id"]

    if api == "anthropic":
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif api == "mistral":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": temperature,
        }
    elif api == "nova":
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": 2048,
                "temperature": temperature,
            },
        }
    else:
        raise ValueError(f"Unknown API format: {api}")

    start = time.monotonic()
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    result = json.loads(response["body"].read())

    # Extract text and usage based on API format
    if api == "anthropic":
        text = result["content"][0]["text"]
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif api == "mistral":
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    elif api == "nova":
        text = result["output"]["message"]["content"][0]["text"]
        usage = result.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }


def parse_json_output(text: str) -> dict:
    """Extract JSON from model response."""
    import re
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
                   model_key: str, temperature: float) -> dict:
    """Run a single verification call."""
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

    pricing = VERIFIER_MODELS[model_key]["pricing"]
    try:
        resp = call_model(prompt, model_key, temperature)
        parsed = parse_json_output(resp["text"])
        cost = (resp["input_tokens"] * pricing[0] + resp["output_tokens"] * pricing[1]) / 1_000_000
        return {
            "parsed": parsed,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "latency_ms": resp["latency_ms"],
            "cost_usd": cost,
            "error": None if parsed else "parse_failed",
            "raw_text": resp["text"][:300],
        }
    except Exception as e:
        return {"parsed": None, "error": str(e)[:500], "cost_usd": 0}


def run_ensemble(problem: str, diff: str, model_key: str) -> dict:
    """Run the v001∩v009(2+/3) ensemble with a given verifier model."""
    calls = [
        ("v001", V001_RUBRIC, 0.0),
        ("v009_r1", V009_RUBRIC, 0.3),
        ("v009_r2", V009_RUBRIC, 0.3),
        ("v009_r3", V009_RUBRIC, 0.3),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(verify_single, rub, problem, diff, model_key, temp): name
            for name, rub, temp in calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()

    # Extract verdicts
    v001 = results["v001"]
    v001_verdict = (v001.get("parsed") or {}).get("verdict", "")
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
        "ensemble_verdict": "likely_correct" if ensemble_pass else "likely_incorrect",
        "ensemble_pass": ensemble_pass,
        "v001_verdict": v001_verdict,
        "v001_score": (v001.get("parsed") or {}).get("overall_score"),
        "v009_lc_count": v009_lc,
        "v009_verdicts": {
            n: (results[n].get("parsed") or {}).get("verdict", "error")
            for n in sorted(v009_names)
        },
        "total_cost_usd": round(total_cost, 6),
        "wall_latency_ms": max(r.get("latency_ms", 0) for r in results.values()),
        "errors": errors if errors else None,
    }


def load_gold_labels(patch_source: str) -> dict:
    """Load gold labels for a patch source."""
    # Map patch sources to gold label files
    gold_file_map = {
        "sonnet": RESULTS_DIR / "gold_sonnet_opencode.jsonl",
        "haiku": RESULTS_DIR / "gold_haiku_opencode.jsonl",
        "opus": RESULTS_DIR / "gold_opus_opencode.jsonl",
        "devstral_sera": RESULTS_DIR / "gold_devstral_sera.jsonl",
        "devstral_sera_selfcritique": RESULTS_DIR / "gold_devstral_sera_selfcritique.jsonl",
        "devstral_sera_verifier_loop": RESULTS_DIR / "gold_devstral_sera_verifier_loop.jsonl",
    }
    gold_file = gold_file_map.get(patch_source, RESULTS_DIR / f"gold_{patch_source}_opencode.jsonl")
    labels = {}
    with open(gold_file) as f:
        for line in f:
            row = json.loads(line)
            labels[row["instance_id"]] = row.get("passed", False)
    return labels


def load_problems() -> dict:
    """Load problem statements from the dataset."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def main():
    parser = argparse.ArgumentParser(description="T4: Cross-verifier transfer test")
    parser.add_argument("--verifier", required=True,
                        choices=list(VERIFIER_MODELS.keys()) + ["all"],
                        help="Verifier model to test")
    parser.add_argument("--patch-source", default="sonnet",
                        choices=[
                            "sonnet", "haiku", "opus",
                            "devstral_sera", "devstral_sera_selfcritique",
                            "devstral_sera_verifier_loop",
                            "qwen35",
                        ],
                        help="Which model's patches to verify (default: sonnet dev set)")
    parser.add_argument("--limit", type=int, help="Limit to first N patches")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    verifiers = list(VERIFIER_MODELS.keys()) if args.verifier == "all" else [args.verifier]

    # Load data
    print("Loading problem statements...")
    problems = load_problems()
    gold = load_gold_labels(args.patch_source)

    # Map patch sources to diff directories
    if args.patch_source.startswith("devstral_"):
        diffs_dir = DIFFS_DIR / args.patch_source
    else:
        diffs_dir = DIFFS_DIR / f"opencode_{args.patch_source}"
    diff_files = sorted(diffs_dir.glob("*.diff"))
    if args.limit:
        diff_files = diff_files[:args.limit]

    print(f"Found {len(diff_files)} diffs from {args.patch_source}, {sum(gold.values())} gold passes")

    for verifier_key in verifiers:
        print(f"\n{'='*60}")
        print(f"Verifier: {verifier_key} ({VERIFIER_MODELS[verifier_key]['model_id']})")
        print(f"{'='*60}")

        output_file = RESULTS_DIR / f"t4_cross_verifier_{verifier_key}_{args.patch_source}.jsonl"

        # Resume support
        completed = set()
        if args.resume and output_file.exists():
            with open(output_file) as f:
                for line in f:
                    row = json.loads(line)
                    completed.add(row["instance_id"])
            print(f"Resuming: {len(completed)} already done")

        tp = fp = fn = tn = 0
        total_cost = 0.0
        errors = 0

        for i, diff_file in enumerate(diff_files):
            instance_id = diff_file.stem
            if instance_id in completed:
                continue

            diff_content = diff_file.read_text()
            problem = problems.get(instance_id, "")
            gold_pass = gold.get(instance_id, False)

            print(f"  [{i+1}/{len(diff_files)}] {instance_id} (gold={'PASS' if gold_pass else 'FAIL'})...", end=" ", flush=True)

            try:
                result = run_ensemble(problem, diff_content, verifier_key)
            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1
                result = {"ensemble_pass": False, "error": str(e), "total_cost_usd": 0}

            predicted_pass = result.get("ensemble_pass", False)

            # Confusion matrix
            if predicted_pass and gold_pass:
                tp += 1
                label = "TP"
            elif predicted_pass and not gold_pass:
                fp += 1
                label = "FP"
            elif not predicted_pass and gold_pass:
                fn += 1
                label = "FN"
            else:
                tn += 1
                label = "TN"

            total_cost += result.get("total_cost_usd", 0)
            print(f"{label} | v001={result.get('v001_verdict','')} v009_lc={result.get('v009_lc_count',0)}/3 | ${result.get('total_cost_usd',0):.4f}")

            # Save result
            row = {
                "instance_id": instance_id,
                "verifier_model": verifier_key,
                "patch_source": args.patch_source,
                "gold_pass": gold_pass,
                "predicted_pass": predicted_pass,
                **result,
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(row) + "\n")

        # Summary
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0

        print(f"\n--- {verifier_key} Summary ---")
        print(f"Total: {total} | TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F0.5: {f05:.2f}")
        print(f"Cost: ${total_cost:.4f} | Errors: {errors}")
        print(f"Results: {output_file}")

        # Save summary
        summary = {
            "verifier_model": verifier_key,
            "patch_source": args.patch_source,
            "total": total,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f05": round(f05, 4),
            "total_cost_usd": round(total_cost, 4),
            "errors": errors,
        }
        summary_file = RESULTS_DIR / f"t4_summary_{verifier_key}_{args.patch_source}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
