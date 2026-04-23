#!/usr/bin/env python3
"""
Smoke test for verification primitives.

Tests all 3 tools on a known issue with a known-good and known-bad patch
to verify they work before running the full experiment.

Usage:
    python3 smoke_test.py
    python3 smoke_test.py --tool generate_tests --mode adversarial
    python3 smoke_test.py --tool run_tests
    python3 smoke_test.py --tool adversarial_review
    python3 smoke_test.py --all
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Setup paths
TOOLS_DIR = Path(__file__).resolve().parent.parent / "skills" / "verification-primitives" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

# A known Django issue with a simple, verifiable fix
SAMPLE_PROBLEM = """
The `make_aware` function in `django/utils/timezone.py` raises an
`AmbiguousTimeError` even when `is_dst` is specified. When converting a
naive datetime that falls in a DST transition period, passing `is_dst=True`
or `is_dst=False` should resolve the ambiguity, but Django passes None to
pytz instead of the user's value.
"""

SAMPLE_GOOD_DIFF = """
diff --git a/django/utils/timezone.py b/django/utils/timezone.py
--- a/django/utils/timezone.py
+++ b/django/utils/timezone.py
@@ -250,7 +250,7 @@ def make_aware(value, timezone=None, is_dst=None):
     if hasattr(timezone, 'localize'):
-        return timezone.localize(value, is_dst=None)
+        return timezone.localize(value, is_dst=is_dst)
     else:
         return value.replace(tzinfo=timezone)
"""

SAMPLE_BAD_DIFF = """
diff --git a/django/utils/timezone.py b/django/utils/timezone.py
--- a/django/utils/timezone.py
+++ b/django/utils/timezone.py
@@ -250,7 +250,7 @@ def make_aware(value, timezone=None, is_dst=None):
     if hasattr(timezone, 'localize'):
-        return timezone.localize(value, is_dst=None)
+        return timezone.localize(value)
     else:
         return value.replace(tzinfo=timezone)
"""

SAMPLE_TEST_CODE = """
import pytest

def test_always_passes():
    assert 1 + 1 == 2

def test_always_fails():
    assert 1 + 1 == 3
"""


def smoke_generate_tests(mode="adversarial"):
    """Test the generate_tests tool."""
    from generate_tests import generate_tests

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: generate_tests (mode={mode})")
    print(f"{'='*60}")

    result = generate_tests(
        problem_statement=SAMPLE_PROBLEM,
        diff=SAMPLE_GOOD_DIFF,
        mode=mode,
        model="haiku",
        temperature=0.3,
    )

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return False

    print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Cost: ${result['cost_usd']:.4f}")
    print(f"Test code length: {len(result['test_code'])} chars")
    print(f"\nGenerated tests:\n{result['test_code'][:1000]}")

    # Verify it's valid Python
    try:
        compile(result["test_code"], "generated_test.py", "exec")
        print("\nCompiles OK")
    except SyntaxError as e:
        print(f"\nSYNTAX ERROR: {e}")
        return False

    return True


def smoke_run_tests():
    """Test the run_tests tool."""
    from run_tests import run_tests

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: run_tests")
    print(f"{'='*60}")

    # Use a temp directory as workspace
    with tempfile.TemporaryDirectory() as workspace:
        result = run_tests(
            test_code=SAMPLE_TEST_CODE,
            workspace=workspace,
            timeout=30,
        )

    print(f"Passed: {result['passed']}, Failed: {result['failed']}, Total: {result['total']}")
    print(f"Elapsed: {result['elapsed_ms']}ms")
    if result.get("error"):
        print(f"Error: {result['error']}")

    for tr in result.get("test_results", []):
        print(f"  {tr['name']}: {tr['status']}")

    # We expect 1 pass, 1 fail
    ok = result["passed"] == 1 and result["failed"] == 1
    print(f"\nExpected 1 pass + 1 fail: {'OK' if ok else 'UNEXPECTED'}")
    return ok


def smoke_adversarial_review():
    """Test the adversarial_review tool."""
    from adversarial_review import adversarial_review

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: adversarial_review (good patch)")
    print(f"{'='*60}")

    result = adversarial_review(
        problem_statement=SAMPLE_PROBLEM,
        diff=SAMPLE_GOOD_DIFF,
        model="haiku",
    )

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return False

    print(f"Verdict: {result['verdict']}")
    print(f"Score: {result['overall_score']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Attack: {result.get('attack_result', '')}")
    print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"Cost: ${result['cost_usd']:.4f}")

    # Now test with the bad patch
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: adversarial_review (bad patch)")
    print(f"{'='*60}")

    result_bad = adversarial_review(
        problem_statement=SAMPLE_PROBLEM,
        diff=SAMPLE_BAD_DIFF,
        model="haiku",
    )

    if result_bad.get("error"):
        print(f"ERROR: {result_bad['error']}")
        return False

    print(f"Verdict: {result_bad['verdict']}")
    print(f"Score: {result_bad['overall_score']}")
    print(f"Attack: {result_bad.get('attack_result', '')}")

    # Good patch should score higher than bad
    good_score = result.get("overall_score", 0) or 0
    bad_score = result_bad.get("overall_score", 0) or 0
    discriminates = good_score > bad_score
    print(f"\nGood score ({good_score:.2f}) > Bad score ({bad_score:.2f}): {'YES' if discriminates else 'NO'}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test verification primitives")
    parser.add_argument("--tool", choices=["generate_tests", "run_tests", "adversarial_review"])
    parser.add_argument("--mode", default="adversarial", choices=["confirmatory", "adversarial"])
    parser.add_argument("--all", action="store_true", help="Run all smoke tests")
    args = parser.parse_args()

    results = {}

    if args.all or args.tool == "generate_tests":
        results["generate_tests_adversarial"] = smoke_generate_tests("adversarial")
        results["generate_tests_confirmatory"] = smoke_generate_tests("confirmatory")

    if args.all or args.tool == "run_tests":
        results["run_tests"] = smoke_run_tests()

    if args.all or args.tool == "adversarial_review":
        results["adversarial_review"] = smoke_adversarial_review()

    if not args.tool and not args.all:
        # Default: run all
        results["generate_tests_adversarial"] = smoke_generate_tests("adversarial")
        results["generate_tests_confirmatory"] = smoke_generate_tests("confirmatory")
        results["run_tests"] = smoke_run_tests()
        results["adversarial_review"] = smoke_adversarial_review()

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
