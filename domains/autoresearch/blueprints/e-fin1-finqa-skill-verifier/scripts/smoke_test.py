#!/usr/bin/env python3
"""Stage-0 eval smoke test (P0 carryover gap from verification-primitives).

Validates the exact_match() scorer on hand-picked FinQA dev examples spanning
edge cases BEFORE the n=100 run, plus an environment/schema check. Prior:
verification-primitives lost iterations to gold-eval environment breakage.

Run: python3 smoke_test.py [--data <sampled.json>]
Exits non-zero if any directional check fails.
"""
import argparse
import json
import sys

from finqa_common import exact_match, build_context, call_bedrock


def directional_checks():
    """Known (pred, gold, expected_match) cases covering FinQA quirks."""
    cases = [
        # (pred, gold_exe_ans, should_match, label)
        ("127.40", 127.4, True, "trailing-zero numeric"),
        ("127.4", 127.4, True, "exact numeric"),
        ("93.5%", 0.935, True, "percent-vs-ratio (gold stored as ratio)"),
        ("24.69%", 24.69136, True, "percent-vs-scaled (gold already scaled)"),
        ("688", 688.0, True, "int vs float"),
        ("$1,234.56", 1234.56, True, "dollar+comma stripping"),
        ("(5.0)", -5.0, True, "paren-negative"),
        ("yes", "yes", True, "categorical yes"),
        ("no", "no", True, "categorical no"),
        ("yes", "no", False, "categorical mismatch"),
        ("123.46", 123.45, True, "rounding noise (<1% -> match)"),
        ("125.0", 123.45, False, "1.3% off -> no match"),
        ("200", 100.0, False, "wrong by 2x"),
        ("0.935", 0.935, True, "ratio-vs-ratio direct"),
        ("19.2", 18.6, False, "3% off -> no match"),
    ]
    passed = 0
    failed = []
    for pred, gold, expected, label in cases:
        got = exact_match(pred, gold)
        ok = (got == expected)
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {label}: match({pred!r},{gold!r})={got} expected={expected}")
        if ok:
            passed += 1
        else:
            failed.append(label)
    print(f"\nscorer: {passed}/{len(cases)} directional checks passed")
    return len(failed) == 0


def env_check(data_path):
    """Schema + dataset integrity check."""
    try:
        data = json.load(open(data_path))
    except Exception as e:
        print(f"  [FAIL] cannot load {data_path}: {e}")
        return False
    bad = 0
    for ex in data[:5]:
        qa = ex.get("qa", {})
        if "exe_ans" not in qa or "question" not in qa:
            bad += 1
        ctx = build_context(ex)
        if not ctx:
            bad += 1
    print(f"  [{'OK ' if bad == 0 else 'FAIL'}] schema/context on first 5 examples: {bad} problems")
    return bad == 0


def live_api_check():
    """Confirm Bedrock is reachable (1 cheap call)."""
    try:
        r = call_bedrock("Reply with only the digit: 7", model_key="haiku",
                         temperature=0.0, max_tokens=8)
        ok = "7" in r["text"]
        print(f"  [{'OK ' if ok else 'FAIL'}] Bedrock haiku reachable: "
              f"resp={r['text']!r} in={r['input_tokens']} out={r['output_tokens']}")
        return ok
    except Exception as e:
        print(f"  [FAIL] Bedrock call failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="sampled dev json for schema check (optional)")
    ap.add_argument("--skip-api", action="store_true")
    args = ap.parse_args()

    print("== Stage-0a: scorer directional checks ==")
    ok1 = directional_checks()

    ok2 = True
    if args.data:
        print("\n== Stage-0b: dataset schema/context check ==")
        ok2 = env_check(args.data)

    ok3 = True
    if not args.skip_api:
        print("\n== Stage-0c: live Bedrock check ==")
        ok3 = live_api_check()

    if ok1 and ok2 and ok3:
        print("\nSMOKE TEST PASSED")
        sys.exit(0)
    print("\nSMOKE TEST FAILED — fix before n=100 run")
    sys.exit(1)


if __name__ == "__main__":
    main()
