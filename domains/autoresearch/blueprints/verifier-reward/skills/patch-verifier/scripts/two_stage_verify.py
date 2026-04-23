#!/usr/bin/env python3
"""
Two-stage verification: extract functional changes first, then evaluate.

Stage 1 (cheap): Ask the model to extract only the functional changes from a diff
Stage 2 (normal): Run the standard verification rubric on the extracted functional diff

This directly attacks FM-001 by using the model itself as a diff preprocessor.

Usage:
  python3 two_stage_verify.py \
    --rubric versions/v001_baseline.md \
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

from verify_patch import call_bedrock, parse_verification_output, estimate_cost, CHARS_PER_TOKEN

EXTRACT_PROMPT = """You are a diff analysis expert. Given a unified diff, extract ONLY the functional changes — lines that modify program behavior. Remove all cosmetic changes (quote style, import reordering, whitespace, line wrapping, comment reformatting).

Output a clean diff containing ONLY the functional hunks. If there are no functional changes, output "NO FUNCTIONAL CHANGES".

## Diff

```diff
{diff}
```

Output the functional-only diff now. No explanation, just the diff:"""


def two_stage_verify(
    rubric_path: str,
    problem_statement: str,
    diff_content: str,
    model_key: str = "haiku",
    temperature: float = 0.0,
) -> dict:
    """Two-stage verification: extract → evaluate."""
    result = {
        "stage1_cost": 0.0,
        "stage2_cost": 0.0,
        "total_cost": 0.0,
        "functional_diff": "",
        "parsed": None,
        "parse_success": False,
        "error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }

    # Stage 1: Extract functional changes
    MAX_EXTRACT = 100000
    extract_prompt = EXTRACT_PROMPT.format(diff=diff_content[:MAX_EXTRACT])

    try:
        s1 = call_bedrock(extract_prompt, model_key, temperature)
        result["stage1_cost"] = estimate_cost(s1["input_tokens"], s1["output_tokens"], model_key)
        functional_diff = s1["text"].strip()
        result["functional_diff"] = functional_diff

        if "NO FUNCTIONAL CHANGES" in functional_diff:
            result["parsed"] = {
                "overall_score": 0.0,
                "confidence": 0.9,
                "verdict": "likely_incorrect",
                "reasoning": "No functional changes found in diff",
                "scores": {},
            }
            result["parse_success"] = True
            result["total_cost"] = result["stage1_cost"]
            return result

    except Exception as e:
        result["error"] = f"Stage 1 failed: {str(e)[:300]}"
        return result

    # Stage 2: Verify functional diff
    rubric = Path(rubric_path).read_text()
    MAX_FUNC_DIFF = 50000
    verify_prompt = f"""{rubric}

## Problem Statement

{problem_statement[:8000]}

## Proposed Patch (functional changes only, cosmetic changes removed)

```diff
{functional_diff[:MAX_FUNC_DIFF]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        s2 = call_bedrock(verify_prompt, model_key, temperature)
        result["stage2_cost"] = estimate_cost(s2["input_tokens"], s2["output_tokens"], model_key)
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
    parser = argparse.ArgumentParser(description="Two-stage patch verification")
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--problem", type=str)
    parser.add_argument("--diff", required=True)
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    problem = args.problem if args.problem else sys.stdin.read()
    diff_content = Path(args.diff).read_text()

    result = two_stage_verify(
        rubric_path=args.rubric,
        problem_statement=problem,
        diff_content=diff_content,
        model_key=args.model,
        temperature=args.temperature,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
