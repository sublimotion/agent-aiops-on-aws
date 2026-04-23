#!/usr/bin/env python3
"""
Run v009 adversarial rubric on all 300 SWE-bench Lite predictions.

For each patch, calls Haiku 4.5 via Bedrock with 4 runs (1×t=0.0 + 3×t=0.3).
Outputs JSONL with v009_verdict, v009_confidence, lc_count, per-run details.

Cost estimate: 300 patches × 4 runs × ~$0.008/run ≈ $9.60
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3

BASE = Path(__file__).resolve().parent.parent
DEBATE = BASE.parent / "debate-verification"
VERIFIER = BASE.parent / "verifier-reward"

RUBRIC_PATH = VERIFIER / "skills" / "patch-verifier" / "versions" / "v009_adversarial.md"
PREDICTIONS_PATH = DEBATE / "results" / "swebench_lite.jsonl"
PATCHES_PATH = BASE.parent / "verification-primitives-swebench" / "results" / "predictions_lite.jsonl"
OUTPUT_PATH = BASE / "results" / "v009_lite_300.jsonl"

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_CONCURRENT = 10
MAX_DIFF_CHARS = 100000
MAX_PROBLEM_CHARS = 8000

# Pricing per 1M tokens
INPUT_PRICE = 0.80
OUTPUT_PRICE = 4.00


def estimate_cost(input_tokens, output_tokens):
    return (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000


def parse_json_output(text):
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


_bedrock_client = None

def get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _bedrock_client


def call_bedrock_sync(prompt, temperature=0.0, retries=3):
    client = get_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(retries):
        try:
            start = time.monotonic()
            response = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            result = json.loads(response["body"].read())
            text = result["content"][0]["text"]
            usage = result.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_ms": latency_ms,
            }
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt * 5
                print(f"    Retry {attempt+1}/{retries} after {wait}s: {str(e)[:100]}", flush=True)
                time.sleep(wait)
            else:
                raise


async def run_v009_one(instance_id, problem_statement, patch_diff, rubric, semaphore):
    """Run v009 with 4 calls (1×t=0.0 + 3×t=0.3) on one patch."""
    async with semaphore:
        prompt = f"""{rubric}

## Problem Statement

{problem_statement[:MAX_PROBLEM_CHARS]}

## Proposed Patch

```diff
{patch_diff[:MAX_DIFF_CHARS]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

        temperatures = [0.0, 0.3, 0.3, 0.3]
        details = {}
        total_cost = 0.0
        total_input = 0
        total_output = 0
        lc_count = 0

        for i, temp in enumerate(temperatures):
            try:
                resp = await asyncio.get_event_loop().run_in_executor(
                    None, lambda t=temp: call_bedrock_sync(prompt, t)
                )
                parsed = parse_json_output(resp["text"])
                cost = estimate_cost(resp["input_tokens"], resp["output_tokens"])
                total_cost += cost
                total_input += resp["input_tokens"]
                total_output += resp["output_tokens"]

                if parsed:
                    verdict = parsed.get("verdict", "uncertain")
                    score = parsed.get("overall_score", 0.5)
                    if verdict == "likely_correct":
                        lc_count += 1
                    details[f"r{i}"] = {
                        "verdict": verdict,
                        "score": score,
                        "completeness": parsed.get("scores", {}).get("completeness"),
                        "logic_correctness": parsed.get("scores", {}).get("logic_correctness"),
                        "confidence": parsed.get("confidence"),
                        "reasoning": parsed.get("reasoning", "")[:200],
                    }
                else:
                    details[f"r{i}"] = {"verdict": "parse_error", "score": 0.0, "raw": resp["text"][:200]}

            except Exception as e:
                details[f"r{i}"] = {"verdict": "error", "score": 0.0, "error": str(e)[:200]}

        # Aggregate: mean score across runs
        scores = [d.get("score", 0.5) for d in details.values() if isinstance(d.get("score"), (int, float))]
        mean_score = sum(scores) / len(scores) if scores else 0.5

        return {
            "instance_id": instance_id,
            "lc_count": lc_count,
            "n_runs": len(temperatures),
            "v009_verdict": "likely_correct" if lc_count >= 3 else ("uncertain" if lc_count >= 1 else "likely_incorrect"),
            "v009_confidence": mean_score,
            "v009_unanimous": lc_count == 4,
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "details": details,
        }


async def main():
    # Load rubric
    rubric = RUBRIC_PATH.read_text()
    print(f"Rubric: {RUBRIC_PATH.name} ({len(rubric)} chars)")

    # Load problem statements
    problems = {}
    with open(PREDICTIONS_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            problems[d["instance_id"]] = d["problem_statement"]
    print(f"Problem statements: {len(problems)}")

    # Load patches
    patches = {}
    with open(PATCHES_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            patches[d["instance_id"]] = d.get("model_patch", "")
    print(f"Patches: {len(patches)}")

    # Check for existing results (resume support)
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                r = json.loads(line.strip())
                done.add(r["instance_id"])
        print(f"Already completed: {len(done)}")

    # Filter to instances we need to run
    instance_ids = sorted(set(problems.keys()) & set(patches.keys()) - done)
    # Skip empty patches
    instance_ids = [iid for iid in instance_ids if patches.get(iid, "").strip()]
    print(f"To run: {len(instance_ids)}")

    if not instance_ids:
        print("Nothing to do!")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total_cost = 0.0
    completed = 0

    # Process in batches
    for batch_start in range(0, len(instance_ids), MAX_CONCURRENT):
        batch = instance_ids[batch_start:batch_start + MAX_CONCURRENT]
        tasks = [
            run_v009_one(iid, problems[iid], patches[iid], rubric, semaphore)
            for iid in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        with open(OUTPUT_PATH, "a") as f:
            for result in results:
                if isinstance(result, Exception):
                    print(f"  ERROR: {result}")
                    continue
                f.write(json.dumps(result) + "\n")
                total_cost += result["total_cost_usd"]
                completed += 1

        print(f"  Batch {batch_start // MAX_CONCURRENT + 1}: "
              f"{completed}/{len(instance_ids)} done, "
              f"${total_cost:.2f} spent", flush=True)

    print(f"\nComplete: {completed} patches, ${total_cost:.2f} total", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
