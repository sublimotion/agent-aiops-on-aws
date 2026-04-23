#!/usr/bin/env python3
"""
Adversarial code review using v009 rubric via Claude Haiku on Bedrock.

Usage:
    python3 verify/adversarial_review.py --diff patch.diff --problem "bug description"
    python3 verify/adversarial_review.py --diff patch.diff --problem problem.txt --test-results results.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
}

V009_RUBRIC = """# v009: Adversarial Bug-Finding Rubric

You are a code reviewer whose job is to FIND BUGS in proposed patches. A developer has submitted a patch claiming to fix a bug. Your goal is to find reasons why this patch might be WRONG.

## Your Approach

1. **Assume the patch is incorrect.** Most patches (88%) that claim to fix a bug actually introduce subtle errors or miss critical edge cases. Start from this prior.

2. **Actively search for these failure patterns:**
   - Does the fix handle ALL cases in the bug report, or just the example case?
   - Could the fix break existing behavior that isn't mentioned in the bug report?
   - Is the fix in the right location? Could the bug manifest differently in other call sites?
   - Does the fix address the ROOT CAUSE, or just mask a symptom?
   - Are there off-by-one errors, missing None checks, incorrect operator precedence, or wrong comparison direction?
   - Does the fix handle the boundary between the changed code and unchanged code correctly?

3. **Only clear the patch if you cannot find any plausible bug.** If you can articulate even one realistic scenario where this patch would fail, it is NOT likely_correct.

## Analysis Steps

For each patch:

**Step 1 -- Identify the claim**: What does this patch claim to fix? What specific behavior change does it make?

**Step 2 -- Attack the claim**: Try to construct a scenario where the patched code still fails OR breaks something else. Consider:
- Inputs not covered by the bug report
- Interactions with other code paths
- Type mismatches, None propagation, mutability issues
- Whether the fix is complete (does it handle all variants of the bug?)

**Step 3 -- Judge**: Did you find a plausible failure scenario?
- YES -> The patch has a bug. Score accordingly.
- MAYBE -> You're uncertain. The fix looks plausible but you can't fully verify.
- NO -> Despite trying, you couldn't break it. The patch appears correct.

## Output Format

Respond with ONLY a JSON object:

```json
{
  "claimed_fix": "<1 sentence: what the patch claims to do>",
  "attack_result": "<2-3 sentences: your attempt to find a bug>",
  "scores": {
    "problem_alignment": <float 0.0-1.0>,
    "logic_correctness": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "scope": <float 0.0-1.0>,
    "test_safety": <float 0.0-1.0>
  },
  "overall_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "verdict": "<likely_correct|uncertain|likely_incorrect>",
  "reasoning": "<2-3 sentence explanation>"
}
```

**Verdict rules:**
- "likely_correct" ONLY if you tried hard to find a bug and failed.
- "uncertain" if you found a plausible but unconfirmed failure scenario.
- "likely_incorrect" if you found a concrete bug or the fix is clearly incomplete.

**Scoring guidance:**
- logic_correctness = 1.0 ONLY if your attack found NO plausible failure
- logic_correctness = 0.5 if attack found a plausible but uncertain failure
- logic_correctness = 0.0 if attack found a concrete bug
- Weight logic_correctness most heavily in overall_score"""


def call_bedrock(prompt: str, model_key: str = "haiku", temperature: float = 0.0) -> dict:
    import boto3
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": temperature,
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
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "latency_ms": latency_ms,
    }


def parse_json_output(text: str) -> dict:
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def log_telemetry(tool: str, inputs: dict, outputs: dict, elapsed_s: float, cost_usd: float):
    telemetry_path = Path(__file__).parent / "telemetry.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "inputs": inputs,
        "outputs": outputs,
        "elapsed_s": round(elapsed_s, 2),
        "cost_usd": round(cost_usd, 6),
    }
    with open(telemetry_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Adversarial code review of a patch")
    parser.add_argument("--diff", required=True, help="Path to diff file or '-' for stdin")
    parser.add_argument("--problem", required=True, help="Problem statement text or path to file")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test-results", help="Optional: path to test results JSON for context")
    args = parser.parse_args()

    # Read problem
    problem = args.problem
    if os.path.isfile(problem):
        problem = Path(problem).read_text()

    # Read diff
    if args.diff == "-":
        diff = sys.stdin.read()
    else:
        diff = Path(args.diff).read_text()

    # Read optional test results
    test_context = ""
    if args.test_results and os.path.isfile(args.test_results):
        test_context = f"\n\n## Test Execution Results\n\n{Path(args.test_results).read_text()[:4000]}"

    prompt = f"""{V009_RUBRIC}

## Problem Statement

{problem[:8000]}

## Proposed Patch

```diff
{diff[:100000]}
```{test_context}

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    start = time.monotonic()
    try:
        response = call_bedrock(prompt, args.model, args.temperature)
        elapsed = time.monotonic() - start

        pricing = {"haiku": (0.80, 4.00), "sonnet": (3.00, 15.00)}
        ip, op = pricing[args.model]
        cost = (response["input_tokens"] * ip + response["output_tokens"] * op) / 1_000_000

        try:
            parsed = parse_json_output(response["text"])
        except ValueError:
            parsed = {"error": "Failed to parse JSON", "raw": response["text"][:500]}

        log_telemetry(
            tool="adversarial_review",
            inputs={"diff_len": len(diff), "problem_len": len(problem), "model": args.model},
            outputs={"verdict": parsed.get("verdict"), "overall_score": parsed.get("overall_score"),
                     "parse_success": "error" not in parsed},
            elapsed_s=elapsed,
            cost_usd=cost,
        )

        print(json.dumps(parsed, indent=2))

    except Exception as e:
        elapsed = time.monotonic() - start
        log_telemetry(
            tool="adversarial_review",
            inputs={"diff_len": len(diff), "problem_len": len(problem), "model": args.model},
            outputs={"error": str(e)[:200]},
            elapsed_s=elapsed,
            cost_usd=0.0,
        )
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
