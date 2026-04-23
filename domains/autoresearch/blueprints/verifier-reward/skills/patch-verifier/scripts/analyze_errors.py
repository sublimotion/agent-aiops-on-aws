#!/usr/bin/env python3
"""
Post-sweep error analysis: categorize FPs/FNs, discover failure modes,
suggest rubric improvements.

Usage:
  python3 analyze_errors.py results/sweep_phase2b.jsonl
  python3 analyze_errors.py results/sweep_phase2b.jsonl --output results/errors_phase2b.jsonl
  python3 analyze_errors.py results/sweep_phase2b.jsonl --diagnose  # Use Claude to diagnose errors
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BLUEPRINT_DIR = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = BLUEPRINT_DIR / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Known failure modes from program.md
KNOWN_FAILURE_MODES = {
    "FM-001": {
        "name": "Reformatting noise",
        "description": "Large diffs dominated by style changes hiding a small functional fix",
        "heuristic": lambda e: e.get("diff_tokens", 0) > 2000 and e.get("gold_passed") and e.get("overall_score", 1) < 0.4,
    },
    "FM-002": {
        "name": "Truncated diff",
        "description": "Diff missing trailing context or newline",
        "heuristic": lambda e: False,  # Fixed in gold_eval.py, unlikely now
    },
    "FM-003": {
        "name": "Plausible but wrong",
        "description": "Patch looks correct (high score) but fails gold tests — subtle logic error",
        "heuristic": lambda e: not e.get("gold_passed") and e.get("overall_score", 0) >= 0.7 and e.get("confidence", 0) >= 0.7,
    },
}


def load_sweep_events(path: str) -> list[dict]:
    """Load skill_invocation events from sweep JSONL."""
    events = []
    for line in Path(path).read_text().strip().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") == "skill_invocation":
            events.append(row)
    return events


def load_sweep_summaries(path: str) -> list[dict]:
    """Load sweep_complete events."""
    summaries = []
    for line in Path(path).read_text().strip().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") == "sweep_complete":
            summaries.append(row)
    return summaries


def classify_event(event: dict) -> str:
    """Classify an event as TP, FP, FN, TN, or UNKNOWN."""
    gold = event.get("gold_passed")
    if gold is None:
        return "UNKNOWN"

    verdict = event.get("verdict", "")
    predicted_pass = verdict == "likely_correct"

    if predicted_pass and gold:
        return "TP"
    elif predicted_pass and not gold:
        return "FP"
    elif not predicted_pass and gold:
        return "FN"
    else:
        return "TN"


def match_failure_mode(event: dict) -> str:
    """Match an error event to a known failure mode, or flag as NEW."""
    for fm_id, fm in KNOWN_FAILURE_MODES.items():
        if fm["heuristic"](event):
            return fm_id
    return "NEW"


def compute_diff_stats(instance_id: str, patch_source: str) -> dict:
    """Compute basic diff statistics."""
    diff_file = RESULTS_DIR / "diffs" / f"opencode_{patch_source}" / f"{instance_id}.diff"
    if not diff_file.exists():
        return {}

    content = diff_file.read_text()
    lines = content.split("\n")
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    files_changed = sum(1 for l in lines if l.startswith("diff --git"))

    return {
        "total_lines": len(lines),
        "added": added,
        "removed": removed,
        "files_changed": files_changed,
        "churn": added + removed,
    }


def analyze(events: list[dict]) -> dict:
    """Full error analysis of a sweep."""
    # Group by version
    by_version = defaultdict(list)
    for e in events:
        by_version[e.get("skill_version", "unknown")].append(e)

    analysis = {
        "versions": {},
        "errors": [],
        "failure_mode_counts": Counter(),
        "error_concentration": {},
    }

    for version, version_events in by_version.items():
        # Confusion matrix
        classifications = [classify_event(e) for e in version_events]
        cm = Counter(classifications)

        tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        analysis["versions"][version] = {
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "UNKNOWN": cm["UNKNOWN"]},
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "total_errors": fp + fn,
            "error_rate": round((fp + fn) / max(len(version_events), 1), 4),
        }

        # Analyze each error
        for e in version_events:
            cls = classify_event(e)
            if cls not in ("FP", "FN"):
                continue

            patch_source = e.get("patch_source", "unknown")
            instance_id = e.get("instance_id", "unknown")
            fm = match_failure_mode(e)
            analysis["failure_mode_counts"][fm] += 1

            diff_stats = compute_diff_stats(instance_id, patch_source)

            error_entry = {
                "error_id": f"{cls}-{instance_id}-{version}",
                "type": "false_positive" if cls == "FP" else "false_negative",
                "category": fm,
                "category_name": KNOWN_FAILURE_MODES.get(fm, {}).get("name", "Unknown/New"),
                "instance_id": instance_id,
                "skill_version": version,
                "patch_source": patch_source,
                "verifier_score": e.get("overall_score", 0),
                "verifier_confidence": e.get("confidence", 0),
                "verifier_verdict": e.get("verdict", ""),
                "verifier_reasoning": e.get("reasoning", "")[:300],
                "gold_passed": e.get("gold_passed"),
                "diff_stats": diff_stats,
            }
            analysis["errors"].append(error_entry)

    # Error concentration: by repo, by issue, by failure mode
    repo_errors = Counter()
    issue_errors = Counter()
    for err in analysis["errors"]:
        iid = err["instance_id"]
        repo = iid.rsplit("-", 1)[0].replace("__", "/") if "__" in iid else "unknown"
        repo_errors[repo] += 1
        issue_errors[iid] += 1

    analysis["error_concentration"] = {
        "by_repo": dict(repo_errors.most_common(10)),
        "by_issue": dict(issue_errors.most_common(10)),
        "by_failure_mode": dict(analysis["failure_mode_counts"].most_common(10)),
    }

    return analysis


def print_report(analysis: dict):
    """Print human-readable error analysis report."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)

    for version, stats in analysis["versions"].items():
        cm = stats["confusion_matrix"]
        print(f"\n--- {version} ---")
        print(f"  Confusion matrix: TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")
        print(f"  Precision: {stats['precision']:.2%}  Recall: {stats['recall']:.2%}")
        print(f"  Errors: {stats['total_errors']} ({stats['error_rate']:.0%} error rate)")

    total_errors = len(analysis["errors"])
    fn_count = sum(1 for e in analysis["errors"] if e["type"] == "false_negative")
    fp_count = sum(1 for e in analysis["errors"] if e["type"] == "false_positive")

    print(f"\n--- Error Summary ---")
    print(f"  Total errors: {total_errors} (FP={fp_count}, FN={fn_count})")

    print(f"\n--- Failure Mode Distribution ---")
    for fm, count in analysis["error_concentration"]["by_failure_mode"].items():
        fm_name = KNOWN_FAILURE_MODES.get(fm, {}).get("name", "NEW/Unknown")
        pct = 100 * count / max(total_errors, 1)
        print(f"  {fm} ({fm_name}): {count} ({pct:.0f}%)")

    print(f"\n--- Error Concentration by Repo ---")
    for repo, count in analysis["error_concentration"]["by_repo"].items():
        print(f"  {repo}: {count}")

    # Actionable recommendations
    print(f"\n--- Recommendations ---")
    fm_counts = analysis["failure_mode_counts"]
    if fm_counts.get("FM-001", 0) > total_errors * 0.3:
        print("  HIGH: FM-001 (reformatting noise) is >30% of errors.")
        print("    → Create rubric variant with: 'Ignore style-only changes. Focus on functional modifications.'")
        print("    → Consider preprocessing diffs to strip whitespace-only hunks before verification.")

    if fm_counts.get("NEW", 0) > total_errors * 0.5:
        print("  HIGH: >50% of errors are uncategorized (NEW).")
        print("    → Run with --diagnose to use Claude to categorize these errors.")
        print("    → Add new FM-xxx entries to program.md Known Failure Modes table.")

    if fn_count > fp_count * 2:
        print("  MEDIUM: False negatives dominate (verifier too strict).")
        print("    → Consider relaxing rubric, especially minimality and completeness criteria.")

    if fp_count > fn_count * 2:
        print("  MEDIUM: False positives dominate (verifier too lenient).")
        print("    → Consider adding stricter logic_correctness criteria with concrete examples.")


def main():
    parser = argparse.ArgumentParser(description="Post-sweep error analysis")
    parser.add_argument("sweep_file", help="Path to sweep JSONL")
    parser.add_argument("--output", help="Output errors JSONL")
    parser.add_argument("--diagnose", action="store_true", help="Use Claude to diagnose NEW errors")
    args = parser.parse_args()

    events = load_sweep_events(args.sweep_file)
    if not events:
        log.error("No skill_invocation events found")
        sys.exit(1)

    log.info(f"Loaded {len(events)} invocation events")

    analysis = analyze(events)
    print_report(analysis)

    # Print sweep summaries
    summaries = load_sweep_summaries(args.sweep_file)
    if summaries:
        print(f"\n--- Sweep Summaries ---")
        for s in summaries:
            m = s["metrics"]
            c = s["config"]
            print(f"  {c['skill_version']} × {c['verifier_model']} × {c['patch_source']}: "
                  f"precision={m['precision']:.2f} recall={m['recall']:.2f} "
                  f"lift={m['lift_over_random_pp']:+.1f}pp cost=${m['total_cost_usd']:.4f}")

    # Save errors
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json.dumps({"summary": {
                "total_errors": len(analysis["errors"]),
                "false_positives": sum(1 for e in analysis["errors"] if e["type"] == "false_positive"),
                "false_negatives": sum(1 for e in analysis["errors"] if e["type"] == "false_negative"),
                "failure_mode_counts": dict(analysis["failure_mode_counts"]),
                "error_concentration": analysis["error_concentration"],
            }}) + "\n")
            for error in analysis["errors"]:
                f.write(json.dumps(error) + "\n")
        log.info(f"Errors written to {output_path}")


if __name__ == "__main__":
    main()
