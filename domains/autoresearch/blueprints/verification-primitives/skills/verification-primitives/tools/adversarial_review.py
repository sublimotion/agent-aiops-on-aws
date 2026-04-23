#!/usr/bin/env python3
"""
Verification primitive: adversarial code review using v009 rubric.

Single-call wrapper around the v009 adversarial bug-finding rubric.
Unlike the ensemble verifier (4 calls, $0.030/patch), this is a single
call ($0.008) suitable for agent-invoked use.
"""

import json
import os
import sys
import time
from pathlib import Path

# Reuse the core verify_patch infrastructure
VERIFIER_DIR = Path(__file__).resolve().parents[4] / "verifier-reward" / "skills" / "patch-verifier"
V009_RUBRIC = VERIFIER_DIR / "versions" / "v009_adversarial.md"
VERIFY_SCRIPT = VERIFIER_DIR / "scripts"

# Add verifier scripts to path for imports
sys.path.insert(0, str(VERIFY_SCRIPT))


def adversarial_review(
    problem_statement: str,
    diff: str,
    model: str = "haiku",
    temperature: float = 0.0,
    test_results: str = "",
) -> dict:
    """
    Run adversarial code review on a patch.

    Args:
        problem_statement: The bug report / issue description
        diff: The patch diff content
        model: "haiku" or "sonnet"
        temperature: Sampling temperature (0.0 for deterministic)
        test_results: Optional test execution results for additional context

    Returns:
        dict with: verdict, overall_score, confidence, scores, attack_result,
                   reasoning, input_tokens, output_tokens, latency_ms, cost_usd,
                   parse_success, error
    """
    from verify_patch import verify_patch as _verify

    # Augment diff with test results if provided
    diff_for_review = diff
    if test_results:
        diff_for_review = f"{diff}\n\n## Test Execution Results\n\n{test_results[:4000]}"

    result = _verify(
        rubric_path=str(V009_RUBRIC),
        problem_statement=problem_statement,
        diff_content=diff_for_review,
        model_key=model,
        temperature=temperature,
    )

    # Flatten the parsed output for easier consumption
    output = {
        "verdict": None,
        "overall_score": None,
        "confidence": None,
        "scores": None,
        "attack_result": None,
        "claimed_fix": None,
        "reasoning": None,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "latency_ms": result.get("latency_ms", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "parse_success": result.get("parse_success", False),
        "error": result.get("error"),
    }

    if result.get("parsed"):
        p = result["parsed"]
        output["verdict"] = p.get("verdict")
        output["overall_score"] = p.get("overall_score")
        output["confidence"] = p.get("confidence")
        output["scores"] = p.get("scores")
        output["attack_result"] = p.get("attack_result")
        output["claimed_fix"] = p.get("claimed_fix")
        output["reasoning"] = p.get("reasoning")

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial code review")
    parser.add_argument("--problem", required=True, help="Problem statement text or file")
    parser.add_argument("--diff", required=True, help="Path to diff file")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test-results", help="Optional test results file")
    args = parser.parse_args()

    problem = args.problem
    if os.path.isfile(problem):
        problem = Path(problem).read_text()

    diff = Path(args.diff).read_text()

    test_results = ""
    if args.test_results and os.path.isfile(args.test_results):
        test_results = Path(args.test_results).read_text()

    result = adversarial_review(
        problem_statement=problem,
        diff=diff,
        model=args.model,
        temperature=args.temperature,
        test_results=test_results,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
