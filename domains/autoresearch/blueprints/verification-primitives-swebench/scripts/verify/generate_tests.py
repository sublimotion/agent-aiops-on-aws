#!/usr/bin/env python3
"""
Generate adversarial tests for a patch using Claude Haiku via Bedrock.

Usage:
    python3 verify/generate_tests.py --diff patch.diff --problem "bug description"
    python3 verify/generate_tests.py --diff patch.diff --problem problem.txt --output tests.py
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

ADVERSARIAL_PROMPT = """You are an adversarial test engineer whose goal is to BREAK the proposed patch. A developer claims their patch fixes a bug. Your job is to write tests that expose failures in the patch.

## Your Approach

1. **Assume the patch is incomplete or subtly wrong.** Most patches fix the example case but miss edge cases.
2. **Target the boundaries of the fix**: What inputs weren't mentioned in the bug report? What about None, empty strings, large inputs, negative numbers, nested structures?
3. **Test interactions**: Does the fix break any behavior that was working before? Test adjacent functionality.
4. **Test the root cause, not just the symptom**: If the fix patches a specific code path, test other code paths that might have the same underlying bug.

## Instructions

1. Read the problem statement to understand what should be fixed.
2. Read the patch diff to find assumptions, boundary conditions, and potential gaps.
3. Write 5-8 pytest tests designed to FAIL on a buggy patch:
   - At least 2 tests for edge cases NOT mentioned in the bug report
   - At least 1 test for potential regression (existing behavior that might break)
   - At least 1 test targeting boundary conditions of the fix

## Constraints

- Output ONLY valid Python code (a complete pytest file)
- Use standard library + the repo's own imports only
- Each test function should be independent
- Name tests descriptively: `test_edge_case_empty_input`, `test_regression_original_behavior`, etc.
- Include a module docstring explaining your attack strategy

## Input

### Problem Statement

{problem_statement}

### Patch Diff

```diff
{diff}
```

## Output

Write a complete pytest test file with adversarial tests. Output ONLY the Python code, no markdown fences or explanation."""


def call_bedrock(prompt: str, model_key: str = "haiku", temperature: float = 0.3) -> dict:
    import boto3
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
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


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    if lines and (lines[0].startswith("import ") or lines[0].startswith("#") or
                  lines[0].startswith('"""') or lines[0].startswith("def ")):
        return text.strip()
    return text.strip()


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
    parser = argparse.ArgumentParser(description="Generate adversarial tests for a patch")
    parser.add_argument("--diff", required=True, help="Path to diff file or '-' for stdin")
    parser.add_argument("--problem", required=True, help="Problem statement text or path to file")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", help="Output file path (default: stdout)")
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

    prompt = ADVERSARIAL_PROMPT.format(
        problem_statement=problem[:6000],
        diff=diff[:20000],
    )

    start = time.monotonic()
    try:
        response = call_bedrock(prompt, args.model, args.temperature)
        test_code = extract_python(response["text"])
        elapsed = time.monotonic() - start

        pricing = {"haiku": (0.80, 4.00), "sonnet": (3.00, 15.00)}
        ip, op = pricing[args.model]
        cost = (response["input_tokens"] * ip + response["output_tokens"] * op) / 1_000_000

        log_telemetry(
            tool="generate_tests",
            inputs={"diff_len": len(diff), "problem_len": len(problem), "model": args.model},
            outputs={"test_code_len": len(test_code), "input_tokens": response["input_tokens"],
                     "output_tokens": response["output_tokens"]},
            elapsed_s=elapsed,
            cost_usd=cost,
        )

        if args.output:
            Path(args.output).write_text(test_code)
            print(f"Tests written to {args.output} ({len(test_code)} chars, ${cost:.4f})")
        else:
            print(test_code)

    except Exception as e:
        elapsed = time.monotonic() - start
        log_telemetry(
            tool="generate_tests",
            inputs={"diff_len": len(diff), "problem_len": len(problem), "model": args.model},
            outputs={"error": str(e)[:200]},
            elapsed_s=elapsed,
            cost_usd=0.0,
        )
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
