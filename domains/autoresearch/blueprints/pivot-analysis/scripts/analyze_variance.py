#!/usr/bin/env python3
"""
Pivot Point Variance Analysis for VP SWE-bench experiment.

Computes:
1. Per-pivot outcome variance (which decisions have highest impact on pass rate)
2. Information gain (mutual information between pivot choice and gold outcome)
3. Timing analysis (when pivots occur relative to action budget)
4. Composition pattern transitions
5. Early-stopping rule derivation

Reads: results/decision_sequences.json (from extract_pivots.py)
Writes: results/pivot_report.md + results/pivot_data.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent


def load_data():
    path = BASE / "results" / "decision_sequences.json"
    with open(path) as f:
        return json.load(f)


def fisher_exact(a, b, c, d):
    """Fisher exact test for 2x2 table [[a,b],[c,d]]."""
    try:
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
        return odds_ratio, p_value
    except Exception:
        return float("nan"), 1.0


def mutual_information(joint_counts):
    """Compute MI I(X;Y) from joint count dict {(x,y): count}."""
    total = sum(joint_counts.values())
    if total == 0:
        return 0.0
    x_counts = defaultdict(int)
    y_counts = defaultdict(int)
    for (x, y), c in joint_counts.items():
        x_counts[x] += c
        y_counts[y] += c
    mi = 0.0
    for (x, y), c in joint_counts.items():
        if c == 0:
            continue
        p_xy = c / total
        p_x = x_counts[x] / total
        p_y = y_counts[y] / total
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def risk_diff_ci(p1, n1, p0, n0):
    """Risk difference with 95% CI."""
    rd = p1 - p0
    if n1 > 0 and n0 > 0 and 0 < p1 < 1 and 0 < p0 < 1:
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    else:
        se = float("nan")
    return rd, rd - 1.96 * se, rd + 1.96 * se


def pivot_2x2(name, a_pass, a_fail, b_pass, b_fail, label_a="A", label_b="B"):
    """Compute all statistics for a 2×2 pivot."""
    n_a = a_pass + a_fail
    n_b = b_pass + b_fail
    rate_a = a_pass / n_a if n_a > 0 else 0
    rate_b = b_pass / n_b if n_b > 0 else 0
    odds, p_val = fisher_exact(a_pass, a_fail, b_pass, b_fail)
    rd, ci_lo, ci_hi = risk_diff_ci(rate_a, n_a, rate_b, n_b)
    joint = {
        (label_a, "pass"): a_pass, (label_a, "fail"): a_fail,
        (label_b, "pass"): b_pass, (label_b, "fail"): b_fail,
    }
    mi = mutual_information(joint)
    return {
        "name": name,
        f"n_{label_a}": n_a, f"n_{label_b}": n_b,
        f"rate_{label_a}": rate_a, f"rate_{label_b}": rate_b,
        "risk_difference": rd, "rd_ci": [ci_lo, ci_hi],
        "odds_ratio": odds, "p_value": p_val,
        "mutual_information": mi,
    }


# ─── Pivot analyses ───────────────────────────────────────────────

def pivot_tool_usage(data):
    """Used any VP verification tool vs didn't."""
    used_p = sum(1 for d in data if d["num_tool_invocations"] > 0 and d["gold_outcome"] == "pass")
    used_f = sum(1 for d in data if d["num_tool_invocations"] > 0 and d["gold_outcome"] == "fail")
    no_p = sum(1 for d in data if d["num_tool_invocations"] == 0 and d["gold_outcome"] == "pass")
    no_f = sum(1 for d in data if d["num_tool_invocations"] == 0 and d["gold_outcome"] == "fail")
    return pivot_2x2("Tool Usage (used vs not)", used_p, used_f, no_p, no_f, "used", "unused")


def pivot_review(data):
    """Ran adversarial_review vs used tools but no review."""
    reviewed = [d for d in data if "adversarial_review" in d.get("tools_used", [])]
    tests_only = [d for d in data if d["num_tool_invocations"] > 0 and "adversarial_review" not in d.get("tools_used", [])]
    r_p = sum(1 for d in reviewed if d["gold_outcome"] == "pass")
    r_f = sum(1 for d in reviewed if d["gold_outcome"] == "fail")
    t_p = sum(1 for d in tests_only if d["gold_outcome"] == "pass")
    t_f = sum(1 for d in tests_only if d["gold_outcome"] == "fail")
    return pivot_2x2("Adversarial Review vs Tests Only", r_p, r_f, t_p, t_f, "reviewed", "tests_only")


def pivot_revision(data):
    """Revised after test failure vs submitted despite failure."""
    rev = [d for d in data if d["revised_after_failure"]]
    sub = [d for d in data if d["submitted_despite_failure"]]
    r_p = sum(1 for d in rev if d["gold_outcome"] == "pass")
    r_f = sum(1 for d in rev if d["gold_outcome"] == "fail")
    s_p = sum(1 for d in sub if d["gold_outcome"] == "pass")
    s_f = sum(1 for d in sub if d["gold_outcome"] == "fail")
    if len(rev) < 2 or len(sub) < 2:
        return {"name": "Revised After Failure vs Submitted Despite", "note": "insufficient data",
                "n_revised": len(rev), "n_submitted": len(sub)}
    return pivot_2x2("Revised After Failure vs Submitted Despite", r_p, r_f, s_p, s_f, "revised", "submitted")


def pivot_early_edit(data):
    """Early first edit (< median % of actions) vs late."""
    edit_data = []
    for d in data:
        s = d.get("session")
        if s and s.get("first_edit_pct") is not None:
            edit_data.append((s["first_edit_pct"], d["gold_outcome"]))

    if len(edit_data) < 20:
        return {"name": "Early vs Late First Edit", "note": "insufficient data", "n": len(edit_data)}

    pcts = [e[0] for e in edit_data]
    median_pct = float(np.median(pcts))

    early_p = sum(1 for p, o in edit_data if p <= median_pct and o == "pass")
    early_f = sum(1 for p, o in edit_data if p <= median_pct and o == "fail")
    late_p = sum(1 for p, o in edit_data if p > median_pct and o == "pass")
    late_f = sum(1 for p, o in edit_data if p > median_pct and o == "fail")

    result = pivot_2x2("Early vs Late First Edit", early_p, early_f, late_p, late_f, "early", "late")
    result["median_pct"] = median_pct
    result["timing_distribution"] = {
        "mean": float(np.mean(pcts)), "std": float(np.std(pcts)),
        "median": median_pct,
        "q25": float(np.percentile(pcts, 25)), "q75": float(np.percentile(pcts, 75)),
    }
    return result


def pivot_patch_size(data):
    """Large patch (> median) vs small patch."""
    patch_data = [(d["patch_len"], d["gold_outcome"]) for d in data if d["fix_generated"] and d["patch_len"] > 0]
    if len(patch_data) < 20:
        return {"name": "Large vs Small Patch", "note": "insufficient data"}

    sizes = [p[0] for p in patch_data]
    median_size = float(np.median(sizes))

    large_p = sum(1 for s, o in patch_data if s > median_size and o == "pass")
    large_f = sum(1 for s, o in patch_data if s > median_size and o == "fail")
    small_p = sum(1 for s, o in patch_data if s <= median_size and o == "pass")
    small_f = sum(1 for s, o in patch_data if s <= median_size and o == "fail")

    result = pivot_2x2("Large vs Small Patch", large_p, large_f, small_p, small_f, "large", "small")
    result["median_patch_len"] = median_size
    return result


def pivot_explore_ratio(data):
    """High explore ratio (> median) vs low."""
    explore_data = []
    for d in data:
        s = d.get("session")
        if s and s.get("total_actions", 0) > 0:
            ratio = s["num_explores"] / s["total_actions"]
            explore_data.append((ratio, d["gold_outcome"]))

    if len(explore_data) < 20:
        return {"name": "High vs Low Explore Ratio", "note": "insufficient data"}

    ratios = [r[0] for r in explore_data]
    median_ratio = float(np.median(ratios))

    high_p = sum(1 for r, o in explore_data if r > median_ratio and o == "pass")
    high_f = sum(1 for r, o in explore_data if r > median_ratio and o == "fail")
    low_p = sum(1 for r, o in explore_data if r <= median_ratio and o == "pass")
    low_f = sum(1 for r, o in explore_data if r <= median_ratio and o == "fail")

    result = pivot_2x2("High vs Low Explore Ratio", high_p, high_f, low_p, low_f, "high_explore", "low_explore")
    result["median_explore_ratio"] = median_ratio
    return result


def pivot_action_count(data):
    """Many actions (> median) vs few."""
    action_data = []
    for d in data:
        s = d.get("session")
        if s and s.get("total_actions", 0) > 0:
            action_data.append((s["total_actions"], d["gold_outcome"]))

    if len(action_data) < 20:
        return {"name": "Many vs Few Actions", "note": "insufficient data"}

    counts = [a[0] for a in action_data]
    median_count = float(np.median(counts))

    many_p = sum(1 for c, o in action_data if c > median_count and o == "pass")
    many_f = sum(1 for c, o in action_data if c > median_count and o == "fail")
    few_p = sum(1 for c, o in action_data if c <= median_count and o == "pass")
    few_f = sum(1 for c, o in action_data if c <= median_count and o == "fail")

    result = pivot_2x2("Many vs Few Actions", many_p, many_f, few_p, few_f, "many", "few")
    result["median_action_count"] = median_count
    return result


# ─── Composition patterns ─────────────────────────────────────────

def analyze_composition_patterns(data):
    """Analyze pass rates by composition pattern."""
    patterns = defaultdict(lambda: {"pass": 0, "fail": 0, "unknown": 0})
    for d in data:
        patterns[d["composition_pattern"]][d["gold_outcome"]] += 1

    results = {}
    for pat, counts in sorted(patterns.items()):
        total = counts["pass"] + counts["fail"]
        results[pat] = {
            "n": total + counts["unknown"],
            "n_evaluated": total,
            "pass": counts["pass"], "fail": counts["fail"],
            "pass_rate": counts["pass"] / total if total > 0 else 0,
        }

    # Pairwise Fisher tests
    pairwise = []
    pat_names = [p for p in sorted(patterns.keys()) if patterns[p]["pass"] + patterns[p]["fail"] >= 5]
    for i, p1 in enumerate(pat_names):
        for p2 in pat_names[i + 1:]:
            c1, c2 = patterns[p1], patterns[p2]
            odds, pv = fisher_exact(c1["pass"], c1["fail"], c2["pass"], c2["fail"])
            pairwise.append({
                "a": p1, "b": p2,
                "rate_a": c1["pass"] / (c1["pass"] + c1["fail"]),
                "rate_b": c2["pass"] / (c2["pass"] + c2["fail"]),
                "odds_ratio": odds, "p_value": pv,
            })

    return {"patterns": results, "pairwise": pairwise}


# ─── Early stopping ───────────────────────────────────────────────

def analyze_early_stopping(data):
    """Derive and backtest early-stopping rules on session action sequences."""
    rules = []

    # Rule A: No edit by X% of action budget
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        no_edit_by = []
        has_edit_by = []
        for d in data:
            s = d.get("session")
            if not s or s.get("total_actions", 0) == 0:
                continue
            if s.get("first_edit_pct") is None or s["first_edit_pct"] > threshold:
                no_edit_by.append(d)
            else:
                has_edit_by.append(d)

        a_pass = sum(1 for d in no_edit_by if d["gold_outcome"] == "pass")
        a_fail = sum(1 for d in no_edit_by if d["gold_outcome"] == "fail")
        c_pass = sum(1 for d in has_edit_by if d["gold_outcome"] == "pass")
        c_fail = sum(1 for d in has_edit_by if d["gold_outcome"] == "fail")

        n_abort = a_pass + a_fail
        n_cont = c_pass + c_fail
        if n_abort > 0 and n_cont > 0:
            rules.append({
                "rule": f"No edit by {threshold:.0%} of actions",
                "n_abort": n_abort, "n_continue": n_cont,
                "abort_pass_rate": a_pass / n_abort,
                "continue_pass_rate": c_pass / n_cont,
                "false_aborts": a_pass,
                "correct_aborts": a_fail,
                "precision_doomed": a_fail / n_abort,
                "recall_doomed": a_fail / (a_fail + c_fail) if (a_fail + c_fail) > 0 else 0,
            })

    # Rule B: No VP tool usage (binary — this is the dominant signal)
    no_tools = [d for d in data if d["num_tool_invocations"] == 0]
    has_tools = [d for d in data if d["num_tool_invocations"] > 0]
    nt_p = sum(1 for d in no_tools if d["gold_outcome"] == "pass")
    nt_f = sum(1 for d in no_tools if d["gold_outcome"] == "fail")
    ht_p = sum(1 for d in has_tools if d["gold_outcome"] == "pass")
    ht_f = sum(1 for d in has_tools if d["gold_outcome"] == "fail")
    n_nt = nt_p + nt_f
    n_ht = ht_p + ht_f
    if n_nt > 0:
        rules.append({
            "rule": "No VP verification tool used (end of session)",
            "n_abort": n_nt, "n_continue": n_ht,
            "abort_pass_rate": nt_p / n_nt,
            "continue_pass_rate": ht_p / n_ht if n_ht > 0 else 0,
            "false_aborts": nt_p,
            "correct_aborts": nt_f,
            "precision_doomed": nt_f / n_nt,
            "recall_doomed": nt_f / (nt_f + ht_f) if (nt_f + ht_f) > 0 else 0,
        })

    # Rule C: Composite — no edit by 40% AND no VP tools
    composite_abort = [d for d in data
                       if d["num_tool_invocations"] == 0
                       and d.get("session") and d["session"].get("first_edit_pct") is not None
                       and d["session"]["first_edit_pct"] > 0.4]
    composite_rest = [d for d in data if d not in composite_abort]
    ca_p = sum(1 for d in composite_abort if d["gold_outcome"] == "pass")
    ca_f = sum(1 for d in composite_abort if d["gold_outcome"] == "fail")
    cr_p = sum(1 for d in composite_rest if d["gold_outcome"] == "pass")
    cr_f = sum(1 for d in composite_rest if d["gold_outcome"] == "fail")
    n_ca = ca_p + ca_f
    n_cr = cr_p + cr_f
    if n_ca > 0:
        rules.append({
            "rule": "No VP tools AND late edit (>40% of actions)",
            "n_abort": n_ca, "n_continue": n_cr,
            "abort_pass_rate": ca_p / n_ca,
            "continue_pass_rate": cr_p / n_cr if n_cr > 0 else 0,
            "false_aborts": ca_p,
            "correct_aborts": ca_f,
            "precision_doomed": ca_f / n_ca,
            "recall_doomed": ca_f / (ca_f + cr_f) if (ca_f + cr_f) > 0 else 0,
        })

    return rules


# ─── Session timing ───────────────────────────────────────────────

def analyze_timing(data):
    """Analyze action timing from session data."""
    edit_pcts = []
    explore_ratios = []
    action_counts = []

    for d in data:
        s = d.get("session")
        if not s or s.get("total_actions", 0) == 0:
            continue
        if s.get("first_edit_pct") is not None:
            edit_pcts.append((s["first_edit_pct"], d["gold_outcome"]))
        explore_ratios.append((s["num_explores"] / s["total_actions"], d["gold_outcome"]))
        action_counts.append((s["total_actions"], d["gold_outcome"]))

    result = {}
    if edit_pcts:
        pcts = [e[0] for e in edit_pcts]
        pass_pcts = [e[0] for e in edit_pcts if e[1] == "pass"]
        fail_pcts = [e[0] for e in edit_pcts if e[1] == "fail"]
        result["first_edit"] = {
            "n": len(pcts),
            "all": {"median": float(np.median(pcts)), "q25": float(np.percentile(pcts, 25)), "q75": float(np.percentile(pcts, 75))},
            "pass": {"n": len(pass_pcts), "median": float(np.median(pass_pcts))} if pass_pcts else None,
            "fail": {"n": len(fail_pcts), "median": float(np.median(fail_pcts))} if fail_pcts else None,
        }

    if action_counts:
        pass_counts = [a[0] for a in action_counts if a[1] == "pass"]
        fail_counts = [a[0] for a in action_counts if a[1] == "fail"]
        result["action_counts"] = {
            "pass": {"n": len(pass_counts), "median": float(np.median(pass_counts)), "mean": float(np.mean(pass_counts))} if pass_counts else None,
            "fail": {"n": len(fail_counts), "median": float(np.median(fail_counts)), "mean": float(np.mean(fail_counts))} if fail_counts else None,
        }

    return result


# ─── Report generation ────────────────────────────────────────────

def generate_report(pivots, patterns, early_stop, timing, data):
    lines = []
    evaluated = [d for d in data if d["gold_outcome"] in ("pass", "fail")]
    n_pass = sum(1 for d in evaluated if d["gold_outcome"] == "pass")

    lines.append("# Pivot Point Analysis Report")
    lines.append(f"\n**Dataset**: VP SWE-bench production eval (n={len(data)})")
    lines.append(f"**Evaluated**: {len(evaluated)} issues with gold labels")
    lines.append(f"**Pass rate**: {n_pass}/{len(evaluated)} = {n_pass / len(evaluated):.1%}")
    lines.append("")

    # ── Pivot ranking ──
    lines.append("## Pivot Ranking by Outcome Variance")
    lines.append("")
    ranked = sorted(
        [p for p in pivots if "mutual_information" in p],
        key=lambda x: x.get("mutual_information", 0), reverse=True,
    )
    lines.append("| Rank | Pivot | MI (bits) | Risk Diff | 95% CI | p-value | n |")
    lines.append("|------|-------|-----------|-----------|--------|---------|---|")
    for i, p in enumerate(ranked, 1):
        rd = p.get("risk_difference", float("nan"))
        ci = p.get("rd_ci", [float("nan"), float("nan")])
        pv = p.get("p_value", 1.0)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        # Compute total n
        n_keys = [k for k in p if k.startswith("n_")]
        total_n = sum(p[k] for k in n_keys if isinstance(p[k], (int, float)))
        ci_str = f"[{ci[0]:+.1%}, {ci[1]:+.1%}]" if not math.isnan(ci[0]) else "N/A"
        lines.append(f"| {i} | {p['name']} | {p.get('mutual_information', 0):.4f} | "
                     f"{rd:+.1%} | {ci_str} | {pv:.2e}{sig} | {total_n} |")
    lines.append("")

    # ── Detailed pivots ──
    for p in ranked:
        lines.append(f"### {p['name']}")
        lines.append("")
        for k, v in p.items():
            if k in ("name", "rd_ci", "timing_distribution"):
                continue
            if isinstance(v, float):
                if "rate" in k or "difference" in k:
                    lines.append(f"- **{k}**: {v:.1%}")
                else:
                    lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        if "timing_distribution" in p:
            t = p["timing_distribution"]
            lines.append(f"- **timing**: median={t['median']:.1%}, IQR=[{t['q25']:.1%}, {t['q75']:.1%}]")
        lines.append("")

    # ── Composition patterns ──
    lines.append("## Composition Pattern Analysis")
    lines.append("")
    lines.append("| Pattern | n | Pass | Fail | Pass Rate |")
    lines.append("|---------|---|------|------|-----------|")
    for pat, info in sorted(patterns["patterns"].items(), key=lambda x: x[1]["pass_rate"], reverse=True):
        lines.append(f"| {pat} | {info['n_evaluated']} | {info['pass']} | {info['fail']} | {info['pass_rate']:.1%} |")
    lines.append("")

    if patterns["pairwise"]:
        lines.append("### Pairwise Significance Tests")
        lines.append("")
        lines.append("| Pattern A | Pattern B | Rate A | Rate B | OR | p-value |")
        lines.append("|-----------|-----------|--------|--------|-----|---------|")
        for t in sorted(patterns["pairwise"], key=lambda x: x["p_value"]):
            sig = "***" if t["p_value"] < 0.001 else "**" if t["p_value"] < 0.01 else "*" if t["p_value"] < 0.05 else ""
            lines.append(f"| {t['a']} | {t['b']} | {t['rate_a']:.1%} | {t['rate_b']:.1%} | "
                         f"{t['odds_ratio']:.2f} | {t['p_value']:.2e}{sig} |")
        lines.append("")

    # ── Early stopping ──
    lines.append("## Early-Stopping Rule Analysis")
    lines.append("")
    lines.append("| Rule | Abort | Continue | Abort Pass% | Continue Pass% | False Aborts | Precision | Recall |")
    lines.append("|------|-------|----------|-------------|----------------|--------------|-----------|--------|")
    for r in early_stop:
        lines.append(f"| {r['rule']} | {r['n_abort']} | {r['n_continue']} | "
                     f"{r['abort_pass_rate']:.1%} | {r['continue_pass_rate']:.1%} | "
                     f"{r['false_aborts']} | {r['precision_doomed']:.1%} | {r['recall_doomed']:.1%} |")
    lines.append("")

    # ── Timing ──
    lines.append("## Timing Analysis")
    lines.append("")
    if "first_edit" in timing:
        t = timing["first_edit"]
        lines.append(f"### First Edit (Explore→Implement transition)")
        lines.append(f"- n={t['n']}, median={t['all']['median']:.1%}, IQR=[{t['all']['q25']:.1%}, {t['all']['q75']:.1%}]")
        if t.get("pass"):
            lines.append(f"- Pass group: median={t['pass']['median']:.1%} (n={t['pass']['n']})")
        if t.get("fail"):
            lines.append(f"- Fail group: median={t['fail']['median']:.1%} (n={t['fail']['n']})")
        lines.append(f"- **Edit checkpoint (40%) vs empirical median ({t['all']['median']:.1%})**: "
                     f"{'aligned (within 10pp)' if abs(t['all']['median'] - 0.40) < 0.10 else 'misaligned — checkpoint is late'}")
        lines.append("")

    if "action_counts" in timing:
        ac = timing["action_counts"]
        lines.append("### Action Budget Usage")
        if ac.get("pass"):
            lines.append(f"- Pass: median={ac['pass']['median']:.0f} actions, mean={ac['pass']['mean']:.0f}")
        if ac.get("fail"):
            lines.append(f"- Fail: median={ac['fail']['median']:.0f} actions, mean={ac['fail']['mean']:.0f}")
        lines.append("")

    # ── Key findings ──
    lines.append("## Key Findings")
    lines.append("")
    if ranked:
        lines.append(f"1. **Highest-variance pivot**: {ranked[0]['name']} "
                     f"(MI={ranked[0].get('mutual_information', 0):.4f} bits, "
                     f"risk diff={ranked[0].get('risk_difference', 0):+.1%}, "
                     f"p={ranked[0].get('p_value', 1):.2e})")
    if len(ranked) > 1:
        lines.append(f"2. **Second pivot**: {ranked[1]['name']} "
                     f"(MI={ranked[1].get('mutual_information', 0):.4f} bits, "
                     f"p={ranked[1].get('p_value', 1):.2e})")
    if len(ranked) > 2:
        lines.append(f"3. **Third pivot**: {ranked[2]['name']} "
                     f"(MI={ranked[2].get('mutual_information', 0):.4f} bits, "
                     f"p={ranked[2].get('p_value', 1):.2e})")

    # Checkpoint validation
    if "first_edit" in timing:
        med = timing["first_edit"]["all"]["median"]
        lines.append(f"\n**Checkpoint validation**: The 40% edit nudge is "
                     f"{'well-placed' if abs(med - 0.40) < 0.10 else 'too late'} — "
                     f"empirical first-edit median is {med:.1%} of action budget.")

    # Best early stopping rule
    if early_stop:
        best = max(early_stop, key=lambda r: r["precision_doomed"] * r["recall_doomed"])
        lines.append(f"\n**Best early-stopping rule**: \"{best['rule']}\" — "
                     f"precision={best['precision_doomed']:.1%}, recall={best['recall_doomed']:.1%}, "
                     f"would abort {best['n_abort']} issues with {best['false_aborts']} false aborts.")

    lines.append("")
    return "\n".join(lines)


def main():
    print("Loading decision sequences...")
    data = load_data()
    evaluated = [d for d in data if d["gold_outcome"] in ("pass", "fail")]
    print(f"  {len(data)} total, {len(evaluated)} evaluated")

    print("\nComputing pivots...")
    pivots = []
    for fn in [pivot_tool_usage, pivot_review, pivot_revision, pivot_early_edit,
               pivot_patch_size, pivot_explore_ratio, pivot_action_count]:
        p = fn(evaluated)
        pivots.append(p)
        if "risk_difference" in p:
            print(f"  {p['name']}: RD={p['risk_difference']:+.1%}, MI={p.get('mutual_information', 0):.4f}, p={p.get('p_value', 1):.2e}")
        else:
            print(f"  {p['name']}: {p.get('note', 'computed')}")

    print("\nComposition patterns...")
    patterns = analyze_composition_patterns(evaluated)
    for pat, info in sorted(patterns["patterns"].items(), key=lambda x: x[1]["pass_rate"], reverse=True):
        print(f"  {pat:25s}: {info['pass_rate']:.1%} ({info['pass']}/{info['n_evaluated']})")

    print("\nEarly-stopping rules...")
    early_stop = analyze_early_stopping(evaluated)
    for r in early_stop:
        print(f"  {r['rule']:50s}: prec={r['precision_doomed']:.1%}, rec={r['recall_doomed']:.1%}, false_aborts={r['false_aborts']}")

    print("\nTiming analysis...")
    timing = analyze_timing(data)
    if "first_edit" in timing:
        print(f"  First edit: median={timing['first_edit']['all']['median']:.1%}")
    if "action_counts" in timing:
        ac = timing["action_counts"]
        if ac.get("pass"):
            print(f"  Pass actions: median={ac['pass']['median']:.0f}")
        if ac.get("fail"):
            print(f"  Fail actions: median={ac['fail']['median']:.0f}")

    # Generate report
    report = generate_report(pivots, patterns, early_stop, timing, data)
    report_path = BASE / "results" / "pivot_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    # Save data
    all_results = {"pivots": pivots, "patterns": patterns, "early_stopping": early_stop, "timing": timing}
    data_path = BASE / "results" / "pivot_data.json"
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Data: {data_path}")


if __name__ == "__main__":
    main()
