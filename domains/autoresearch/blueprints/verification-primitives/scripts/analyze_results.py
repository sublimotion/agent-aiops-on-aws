#!/usr/bin/env python3
"""
Analyze verification primitives experiment results.

Computes per-cell metrics, tool usage patterns, and statistical comparisons.

Usage:
    python3 analyze_results.py results/full_control.jsonl results/full_B_checkpoint.jsonl
    python3 analyze_results.py --with-gold results/eval_control.jsonl results/eval_B_checkpoint.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_cell(results: list[dict], cell_name: str):
    n = len(results)
    if n == 0:
        print(f"  No results for {cell_name}")
        return {}

    fixes = sum(1 for r in results if r.get("fix_generated"))
    tool_users = sum(1 for r in results if r.get("num_tool_invocations", 0) > 0)
    total_cost = sum(r.get("agent_cost_usd", 0) + r.get("verification_cost_usd", 0) for r in results)
    avg_turns = sum(r.get("turns_used", 0) for r in results) / n

    # Tool usage details
    tool_invocations = []
    for r in results:
        for inv in r.get("tool_invocations", []):
            tool_invocations.append(inv)

    tool_counts = {}
    for inv in tool_invocations:
        name = inv.get("tool_name", "unknown")
        tool_counts[name] = tool_counts.get(name, 0) + 1

    first_tool_turns = [r["first_tool_turn"] for r in results if r.get("first_tool_turn", -1) >= 0]
    avg_first_tool = sum(first_tool_turns) / len(first_tool_turns) if first_tool_turns else -1

    # Test generation quality
    tests_generated = sum(r.get("tests_generated", 0) for r in results)
    tests_compiled = sum(r.get("tests_compiled", 0) for r in results)
    tests_run = sum(r.get("tests_run", 0) for r in results)
    tests_passed = sum(r.get("tests_passed", 0) for r in results)
    tests_failed = sum(r.get("tests_failed", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Cell: {cell_name} (n={n})")
    print(f"{'='*60}")
    print(f"  Fix rate:      {fixes}/{n} ({100*fixes/n:.0f}%)")
    print(f"  Tool users:    {tool_users}/{n} ({100*tool_users/n:.0f}%)")
    print(f"  Avg turns:     {avg_turns:.1f}")
    print(f"  Total cost:    ${total_cost:.2f} (${total_cost/n:.3f}/issue)")

    if tool_users > 0:
        print(f"  Avg 1st tool:  turn {avg_first_tool:.1f} ({100*avg_first_tool/30:.0f}% of budget)")
        for tool, count in sorted(tool_counts.items()):
            print(f"  {tool}:     {count} calls")

    if tests_generated > 0:
        compile_rate = 100 * tests_compiled / tests_generated if tests_generated else 0
        print(f"\n  Test generation:")
        print(f"    Suites generated: {tests_generated}")
        print(f"    Suites compiled:  {tests_compiled} ({compile_rate:.0f}%)")
        print(f"    Tests run:        {tests_run}")
        print(f"    Tests passed:     {tests_passed}")
        print(f"    Tests failed:     {tests_failed}")

    return {
        "cell": cell_name,
        "n": n,
        "fix_rate": fixes / n,
        "fixes": fixes,
        "tool_adoption": tool_users / n,
        "tool_users": tool_users,
        "avg_turns": avg_turns,
        "cost_per_issue": total_cost / n,
        "avg_first_tool_turn": avg_first_tool,
        "tool_counts": tool_counts,
    }


def fisher_exact_test(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """Compute Fisher's exact test p-value (two-sided)."""
    try:
        from scipy.stats import fisher_exact
        table = [[a_success, a_total - a_success], [b_success, b_total - b_success]]
        _, p = fisher_exact(table)
        return p
    except ImportError:
        return -1.0  # scipy not available


def compare_cells(stats_a: dict, stats_b: dict):
    """Compare two cells statistically."""
    if not stats_a or not stats_b:
        return

    print(f"\n{'='*60}")
    print(f"Comparison: {stats_a['cell']} vs {stats_b['cell']}")
    print(f"{'='*60}")

    fix_diff = stats_b["fix_rate"] - stats_a["fix_rate"]
    print(f"  Fix rate diff: {fix_diff:+.0%} ({stats_b['cell']} - {stats_a['cell']})")

    p = fisher_exact_test(
        stats_a["fixes"], stats_a["n"],
        stats_b["fixes"], stats_b["n"],
    )
    if p >= 0:
        sig = "significant" if p < 0.05 else "NOT significant"
        print(f"  Fisher exact p={p:.4f} ({sig})")
    else:
        print(f"  (scipy not available for Fisher exact test)")

    tool_diff = stats_b["tool_adoption"] - stats_a["tool_adoption"]
    print(f"  Tool adoption diff: {tool_diff:+.0%}")

    cost_diff = stats_b["cost_per_issue"] - stats_a["cost_per_issue"]
    print(f"  Cost diff: ${cost_diff:+.3f}/issue")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL result files to analyze")
    parser.add_argument("--with-gold", action="store_true", help="Include gold eval results")
    args = parser.parse_args()

    all_stats = []
    for path in args.files:
        name = Path(path).stem
        results = load_results(path)
        stats = analyze_cell(results, name)
        all_stats.append(stats)

    # Compare first cell (assumed control) vs all others
    if len(all_stats) >= 2:
        for stats in all_stats[1:]:
            compare_cells(all_stats[0], stats)


if __name__ == "__main__":
    main()
