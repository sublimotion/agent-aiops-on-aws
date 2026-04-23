#!/usr/bin/env python3
"""
Smoke test: verify skill works on 4 known-outcome patches before launching a sweep.

Selects 2 known-PASS and 2 known-FAIL patches from gold eval results,
runs the verifier on each, and asserts directional correctness.

Usage:
  python3 smoke_test.py --rubric versions/v001_baseline.md --model haiku
  python3 smoke_test.py --rubric versions/v003_fewshot.md --model sonnet --patch-source haiku
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
BLUEPRINT_DIR = SKILL_DIR.parent.parent
RESULTS_DIR = BLUEPRINT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))
from verify_patch import verify_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_smoke_patches(patch_source: str) -> dict:
    """Select 2 PASS and 2 FAIL patches with gold labels."""
    gold_file = RESULTS_DIR / f"gold_{patch_source}_opencode.jsonl"
    if not gold_file.exists():
        log.error(f"No gold labels: {gold_file}")
        sys.exit(1)

    passes = []
    fails = []
    for line in gold_file.read_text().strip().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if not row.get("patch_applied"):
            continue
        diff_file = RESULTS_DIR / "diffs" / f"opencode_{patch_source}" / f"{row['instance_id']}.diff"
        if not diff_file.exists():
            continue
        if row["passed"]:
            passes.append(row["instance_id"])
        else:
            fails.append(row["instance_id"])

    if len(passes) < 2:
        log.error(f"Need >= 2 passing patches, found {len(passes)}")
        sys.exit(1)
    if len(fails) < 2:
        log.error(f"Need >= 2 failing patches, found {len(fails)}")
        sys.exit(1)

    return {"pass": passes[:2], "fail": fails[:2]}


def load_problem_statements() -> dict:
    """Load problem statements from dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def main():
    parser = argparse.ArgumentParser(description="Smoke test a rubric version")
    parser.add_argument("--rubric", required=True, help="Path to rubric version markdown")
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku")
    parser.add_argument("--patch-source", default="sonnet")
    args = parser.parse_args()

    log.info(f"Smoke test: {args.rubric} × {args.model} (patches from {args.patch_source})")

    patches = load_smoke_patches(args.patch_source)
    problems = load_problem_statements()

    all_ids = patches["pass"] + patches["fail"]
    expected = {pid: True for pid in patches["pass"]}
    expected.update({pid: False for pid in patches["fail"]})

    results = []
    all_passed = True

    for instance_id in all_ids:
        gold = expected[instance_id]
        gold_str = "PASS" if gold else "FAIL"
        log.info(f"  Testing {instance_id} (gold={gold_str})")

        diff_path = RESULTS_DIR / "diffs" / f"opencode_{args.patch_source}" / f"{instance_id}.diff"
        problem = problems.get(instance_id, "")

        result = verify_patch(
            rubric_path=args.rubric,
            problem_statement=problem,
            diff_content=diff_path.read_text(),
            model_key=args.model,
            temperature=0.0,
        )

        if not result["parse_success"]:
            log.error(f"    PARSE FAILURE: {result['error']}")
            all_passed = False
            results.append({"id": instance_id, "gold": gold, "check": "PARSE_FAIL"})
            continue

        score = result["parsed"]["overall_score"]
        verdict = result["parsed"]["verdict"]
        reasoning = result["parsed"].get("reasoning", "")[:100]

        # Directional checks
        if gold and score < 0.4:
            check = "FAIL"
            log.warning(f"    FALSE NEGATIVE: gold=PASS but score={score:.2f} verdict={verdict}")
            log.warning(f"    Reasoning: {reasoning}")
            all_passed = False
        elif not gold and score > 0.6:
            check = "FAIL"
            log.warning(f"    FALSE POSITIVE: gold=FAIL but score={score:.2f} verdict={verdict}")
            log.warning(f"    Reasoning: {reasoning}")
            all_passed = False
        else:
            check = "OK"
            log.info(f"    OK: score={score:.2f} verdict={verdict}")

        results.append({
            "id": instance_id,
            "gold": gold,
            "score": score,
            "verdict": verdict,
            "check": check,
            "cost": result["cost_usd"],
        })

    # Summary
    total_cost = sum(r.get("cost", 0) for r in results)
    ok_count = sum(1 for r in results if r["check"] == "OK")

    log.info(f"\n{'='*50}")
    log.info(f"Smoke test: {ok_count}/4 checks passed, cost=${total_cost:.4f}")

    if all_passed:
        log.info("SMOKE TEST PASSED — safe to run sweep")
    else:
        log.error("SMOKE TEST FAILED — DO NOT run sweep until issues are resolved")
        log.error("Failure details:")
        for r in results:
            if r["check"] != "OK":
                log.error(f"  {r['id']}: gold={'PASS' if r['gold'] else 'FAIL'} check={r['check']}")

        # Diagnose common issues
        fn_count = sum(1 for r in results if r["check"] == "FAIL" and r["gold"])
        fp_count = sum(1 for r in results if r["check"] == "FAIL" and not r["gold"])

        if fn_count > 0:
            log.error("\nFalse negatives detected. Possible causes:")
            log.error("  - FM-001: Reformatting noise hiding functional changes")
            log.error("  - Rubric too strict on minimality")
            log.error("  - Diff too large for verifier to process")

        if fp_count > 0:
            log.error("\nFalse positives detected. Possible causes:")
            log.error("  - Rubric too lenient")
            log.error("  - Verifier fooled by plausible-looking but wrong fix")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
