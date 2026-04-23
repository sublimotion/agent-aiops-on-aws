#!/usr/bin/env python3
"""
Core patch verification: calls Claude API with rubric + patch → structured score.

Usage:
  python3 verify_patch.py --rubric versions/v001_baseline.md \
    --problem "bug description..." \
    --diff path/to/file.diff \
    --model haiku \
    --temperature 0.0

  # Or pipe problem statement:
  echo "bug description" | python3 verify_patch.py --rubric versions/v001_baseline.md \
    --diff path/to/file.diff
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Bedrock model IDs
BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "opus": "us.anthropic.claude-opus-4-6-v1",
}

# Approximate tokens per char (conservative)
CHARS_PER_TOKEN = 4


def call_bedrock(prompt: str, model_key: str, temperature: float = 0.0) -> dict:
    """Call Claude via Bedrock and return response with usage metadata."""
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    model_id = BEDROCK_MODELS[model_key]
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    start = time.monotonic()
    response = client.invoke_model(
        modelId=model_id,
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


def parse_verification_output(text: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def estimate_cost(input_tokens: int, output_tokens: int, model_key: str) -> float:
    """Estimate cost in USD based on Bedrock pricing."""
    # Approximate Bedrock pricing per 1M tokens (input/output)
    pricing = {
        "haiku": (0.80, 4.00),
        "sonnet": (3.00, 15.00),
        "opus": (15.00, 75.00),
    }
    input_price, output_price = pricing[model_key]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def verify_patch(
    rubric_path: str,
    problem_statement: str,
    diff_content: str,
    model_key: str = "haiku",
    temperature: float = 0.0,
) -> dict:
    """
    Verify a patch using the given rubric.

    Returns dict with:
      - parsed: the parsed JSON output from the model
      - raw_text: the raw model response
      - input_tokens, output_tokens, latency_ms, cost_usd
      - parse_success: bool
      - error: str or None
    """
    rubric = Path(rubric_path).read_text()

    # Smart diff preparation: use preprocessed diff if available, raise limit
    # Haiku has 200K context — old 12K limit was using only 3% of capacity
    MAX_DIFF_CHARS = 100000  # ~25K tokens, well within 200K context
    MAX_PROBLEM_CHARS = 8000

    diff_for_prompt = diff_content
    if len(diff_content) > MAX_DIFF_CHARS:
        # Try preprocessing to strip cosmetic changes
        try:
            from preprocess_diff import preprocess
            preprocessed = preprocess(diff_content, max_chars=MAX_DIFF_CHARS)
            diff_for_prompt = preprocessed["functional_diff"]
        except Exception:
            diff_for_prompt = diff_content[:MAX_DIFF_CHARS]

    prompt = f"""{rubric}

## Problem Statement

{problem_statement[:MAX_PROBLEM_CHARS]}

## Proposed Patch

```diff
{diff_for_prompt}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    result = {
        "parsed": None,
        "raw_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "cost_usd": 0.0,
        "parse_success": False,
        "error": None,
        "problem_tokens": len(problem_statement) // CHARS_PER_TOKEN,
        "diff_tokens": len(diff_content) // CHARS_PER_TOKEN,
    }

    try:
        response = call_bedrock(prompt, model_key, temperature)
        result["raw_text"] = response["text"]
        result["input_tokens"] = response["input_tokens"]
        result["output_tokens"] = response["output_tokens"]
        result["latency_ms"] = response["latency_ms"]
        result["cost_usd"] = estimate_cost(
            response["input_tokens"], response["output_tokens"], model_key
        )

        parsed = parse_verification_output(response["text"])
        result["parsed"] = parsed
        result["parse_success"] = True

    except Exception as e:
        result["error"] = str(e)[:500]

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify a patch using Claude")
    parser.add_argument("--rubric", required=True, help="Path to rubric version markdown")
    parser.add_argument("--problem", type=str, help="Problem statement text")
    parser.add_argument("--diff", required=True, help="Path to diff file")
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.problem:
        problem = args.problem
    else:
        problem = sys.stdin.read()

    diff_content = Path(args.diff).read_text()

    result = verify_patch(
        rubric_path=args.rubric,
        problem_statement=problem,
        diff_content=diff_content,
        model_key=args.model,
        temperature=args.temperature,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
