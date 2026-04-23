#!/usr/bin/env python3
"""
Verification primitive: generate test cases for a patch.

Two modes:
  - confirmatory: tests that verify the fix works
  - adversarial: tests designed to break the fix

Uses Claude via Bedrock. Returns generated test file content.
"""

import json
import os
import re
import time
from pathlib import Path

BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
}

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def call_bedrock(prompt: str, model_key: str = "haiku", temperature: float = 0.3) -> dict:
    """Call Claude via Bedrock."""
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

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
    """Extract Python code from model response, stripping markdown fences."""
    # Try code block first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, check if the whole response looks like Python
    lines = text.strip().split("\n")
    if lines and (lines[0].startswith("import ") or lines[0].startswith("#") or
                  lines[0].startswith('"""') or lines[0].startswith("def ")):
        return text.strip()

    return text.strip()


def generate_tests(
    problem_statement: str,
    diff: str,
    mode: str = "adversarial",
    model: str = "haiku",
    temperature: float = 0.3,
    source_files: str = "",
) -> dict:
    """
    Generate test cases for a patch.

    Args:
        problem_statement: The bug report / issue description
        diff: The patch diff content
        mode: "confirmatory" or "adversarial"
        model: "haiku" or "sonnet"
        temperature: Sampling temperature
        source_files: Optional source file content for context

    Returns:
        dict with: test_code, mode, model, input_tokens, output_tokens,
                   latency_ms, cost_usd, error
    """
    prompt_file = PROMPTS_DIR / f"{mode}.md"
    if not prompt_file.exists():
        return {"error": f"Unknown mode: {mode}", "test_code": ""}

    prompt_template = prompt_file.read_text()

    source_context = ""
    if source_files:
        source_context = f"\n### Relevant Source Files\n\n```python\n{source_files[:8000]}\n```"

    prompt = prompt_template.format(
        problem_statement=problem_statement[:6000],
        diff=diff[:20000],
        source_context=source_context,
    )

    result = {
        "test_code": "",
        "mode": mode,
        "model": model,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "cost_usd": 0.0,
        "error": None,
    }

    try:
        response = call_bedrock(prompt, model, temperature)
        result["test_code"] = extract_python(response["text"])
        result["input_tokens"] = response["input_tokens"]
        result["output_tokens"] = response["output_tokens"]
        result["latency_ms"] = response["latency_ms"]

        # Cost estimate
        pricing = {"haiku": (0.80, 4.00), "sonnet": (3.00, 15.00)}
        ip, op = pricing[model]
        result["cost_usd"] = (response["input_tokens"] * ip + response["output_tokens"] * op) / 1_000_000

    except Exception as e:
        result["error"] = str(e)[:500]

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate tests for a patch")
    parser.add_argument("--problem", required=True, help="Problem statement text or file")
    parser.add_argument("--diff", required=True, help="Path to diff file")
    parser.add_argument("--mode", choices=["confirmatory", "adversarial"], default="adversarial")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    problem = args.problem
    if os.path.isfile(problem):
        problem = Path(problem).read_text()

    diff = Path(args.diff).read_text()

    result = generate_tests(
        problem_statement=problem,
        diff=diff,
        mode=args.mode,
        model=args.model,
        temperature=args.temperature,
    )

    if args.output:
        if result["test_code"]:
            Path(args.output).write_text(result["test_code"])
            print(json.dumps({k: v for k, v in result.items() if k != "test_code"}, indent=2))
        else:
            print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
