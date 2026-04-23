#!/usr/bin/env python3
"""
Describe-then-verify: two-stage approach for large reformatted diffs (FM-001).

Stage 1 (cheap): From problem statement ONLY, describe the expected fix.
Stage 2 (guided): Search the diff for that specific fix, ignoring cosmetic changes.

This directly attacks FM-001 by giving the model a search target,
rather than asking it to find a needle in a haystack blind.

Usage:
  python3 describe_then_verify.py \
    --problem "bug description" \
    --diff path/to/file.diff \
    --model haiku
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from verify_patch import call_bedrock, parse_verification_output, estimate_cost

STAGE1_PROMPT = """You are a senior developer. Based on this bug report, describe the minimal code change needed to fix it. Be specific about file, function, and what needs to change.

## Bug Report

{problem}

Respond with ONLY a JSON object:
{{"file": "<most likely file path>", "function": "<function or class to modify>", "expected_change": "<1-2 sentence description of the specific code change>", "key_tokens": ["<3-5 unique tokens/identifiers that would appear in the fix>"]}}"""

STAGE2_PREFIX = """You are searching a large diff for a specific bug fix. The diff contains many cosmetic changes (reformatting, quote style, line wrapping) mixed with functional changes.

## Expected Fix
File: {file}
Function: {function}
Change: {change}
Key tokens: {tokens}

## Task
Search this diff for the expected fix. Ignore all cosmetic changes (whitespace, quotes, imports, line wrapping). Report ONLY whether the expected functional fix is present.

## Diff

```diff
"""

STAGE2_SUFFIX = """
```

Respond with ONLY a JSON object:
{"fix_found": <true/false>, "evidence": "<quote the specific diff lines that implement the fix, or explain why not found>", "scores": {"problem_alignment": <float>, "logic_correctness": <float>, "completeness": <float>, "scope": <float>, "test_safety": <float>}, "overall_score": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, "verdict": "<likely_correct|uncertain|likely_incorrect>", "reasoning": "<2-3 sentences>"}"""


def describe_then_verify(
    problem_statement: str,
    diff_content: str,
    model_key: str = "haiku",
    temperature: float = 0.0,
) -> dict:
    """Two-stage verification for large diffs."""
    result = {
        "stage1_cost": 0.0,
        "stage2_cost": 0.0,
        "total_cost": 0.0,
        "expected_fix": None,
        "parsed": None,
        "parse_success": False,
        "error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }

    # Stage 1: Describe expected fix from problem only
    MAX_PROBLEM = 6000
    s1_prompt = STAGE1_PROMPT.format(problem=problem_statement[:MAX_PROBLEM])

    try:
        s1 = call_bedrock(s1_prompt, model_key, temperature)
        result["stage1_cost"] = estimate_cost(
            s1["input_tokens"], s1["output_tokens"], model_key
        )
        expected = parse_verification_output(s1["text"])
        result["expected_fix"] = expected
    except Exception as e:
        result["error"] = f"Stage 1 failed: {str(e)[:300]}"
        return result

    # Stage 2: Search diff for expected fix
    MAX_DIFF = 80000
    s2_prompt = (
        STAGE2_PREFIX.format(
            file=expected.get("file", "unknown"),
            function=expected.get("function", "unknown"),
            change=expected.get("expected_change", "unknown"),
            tokens=", ".join(expected.get("key_tokens", [])),
        )
        + diff_content[:MAX_DIFF]
        + STAGE2_SUFFIX
    )

    try:
        s2 = call_bedrock(s2_prompt, model_key, temperature)
        result["stage2_cost"] = estimate_cost(
            s2["input_tokens"], s2["output_tokens"], model_key
        )
        result["input_tokens"] = s1["input_tokens"] + s2["input_tokens"]
        result["output_tokens"] = s1["output_tokens"] + s2["output_tokens"]
        result["latency_ms"] = s1["latency_ms"] + s2["latency_ms"]

        parsed = parse_verification_output(s2["text"])
        result["parsed"] = parsed
        result["parse_success"] = True
    except Exception as e:
        result["error"] = f"Stage 2 failed: {str(e)[:300]}"

    result["total_cost"] = result["stage1_cost"] + result["stage2_cost"]
    result["cost_usd"] = result["total_cost"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Describe-then-verify for large diffs")
    parser.add_argument("--problem", type=str)
    parser.add_argument("--diff", required=True)
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    problem = args.problem if args.problem else sys.stdin.read()
    diff_content = Path(args.diff).read_text()

    result = describe_then_verify(
        problem_statement=problem,
        diff_content=diff_content,
        model_key=args.model,
        temperature=args.temperature,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
