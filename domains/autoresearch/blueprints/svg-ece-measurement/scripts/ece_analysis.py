#!/usr/bin/env python3
"""SVG ECE Measurement — Expected Calibration Error analysis.

Joins SVG line_recall scores (Phase 0, n=300) with Docker eval gold labels
(verification-primitives-swebench, 175/300 resolved) and computes calibration metrics.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
import math

# ── Paths ──
BASE = Path(__file__).resolve().parents[5]  # repo root
SVG_PATH = BASE / "domains/autoresearch/blueprints/learned-verifier/data/phase0/svg_results_production_run1.jsonl"
EVAL_V1 = BASE / "domains/autoresearch/blueprints/verification-primitives-swebench/results/eval_report.json"
EVAL_V2 = BASE / "domains/autoresearch/blueprints/verification-primitives-swebench/results/eval_report_errors_v2.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def load_svg_scores() -> dict[str, float]:
    """Load instance_id -> line_recall from SVG results."""
    scores = {}
    with open(SVG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            scores[rec["instance_id"]] = rec["line_recall"]
    return scores


def load_gold_labels() -> dict[str, int]:
    """Load instance_id -> resolved (1/0) from Docker eval reports (v1 + v2)."""
    labels = {}

    with open(EVAL_V1) as f:
        v1 = json.load(f)
    for iid in v1.get("resolved_ids", []):
        labels[iid] = 1
    for iid in v1.get("unresolved_ids", []):
        labels[iid] = 0

    with open(EVAL_V2) as f:
        v2 = json.load(f)
    for iid in v2.get("resolved_ids", []):
        labels[iid] = 1
    for iid in v2.get("unresolved_ids", []):
        labels[iid] = 0

    return labels


@dataclass
class BinStats:
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float
    accuracy: float
    gap: float  # |accuracy - confidence|


def compute_ece(scores: list[float], labels: list[int], n_bins: int = 10,
                strategy: str = "uniform") -> tuple[float, float, list[BinStats]]:
    """Compute ECE and MCE with equal-width or equal-count binning.

    Returns (ECE, MCE, list of BinStats).
    """
    n = len(scores)
    assert n == len(labels)

    if strategy == "uniform":
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
    elif strategy == "quantile":
        sorted_scores = sorted(scores)
        bin_edges = [0.0]
        for i in range(1, n_bins):
            idx = int(i * n / n_bins)
            bin_edges.append(sorted_scores[min(idx, n - 1)])
        bin_edges.append(1.0)
        # Deduplicate edges
        bin_edges = sorted(set(bin_edges))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    actual_bins = len(bin_edges) - 1
    bins: list[BinStats] = []
    ece = 0.0
    mce = 0.0

    for b in range(actual_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Include right edge for last bin
        if b == actual_bins - 1:
            idxs = [i for i in range(n) if lo <= scores[i] <= hi]
        else:
            idxs = [i for i in range(n) if lo <= scores[i] < hi]

        if not idxs:
            bins.append(BinStats(lo, hi, 0, 0.0, 0.0, 0.0))
            continue

        count = len(idxs)
        mean_conf = sum(scores[i] for i in idxs) / count
        acc = sum(labels[i] for i in idxs) / count
        gap = abs(acc - mean_conf)

        bins.append(BinStats(lo, hi, count, mean_conf, acc, gap))
        ece += (count / n) * gap
        mce = max(mce, gap)

    return ece, mce, bins


def brier_score(scores: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    n = len(scores)
    return sum((scores[i] - labels[i]) ** 2 for i in range(n)) / n


def bootstrap_ece(scores: list[float], labels: list[int], n_boots: int = 1000,
                  n_bins: int = 10, strategy: str = "uniform") -> tuple[float, float, float]:
    """Bootstrap 95% CI for ECE."""
    import random
    random.seed(42)
    n = len(scores)
    ece_samples = []
    for _ in range(n_boots):
        idxs = [random.randint(0, n - 1) for _ in range(n)]
        s = [scores[i] for i in idxs]
        l = [labels[i] for i in idxs]
        e, _, _ = compute_ece(s, l, n_bins, strategy)
        ece_samples.append(e)
    ece_samples.sort()
    lo = ece_samples[int(0.025 * n_boots)]
    hi = ece_samples[int(0.975 * n_boots)]
    mean = sum(ece_samples) / n_boots
    return mean, lo, hi


def platt_scaling(scores: list[float], labels: list[int]) -> tuple[list[float], float, float]:
    """Simple logistic regression recalibration (Platt scaling).

    Fits logit(p) = a * score + b, returns recalibrated scores and (a, b).
    """
    # Gradient descent on log loss
    a, b = 1.0, 0.0
    lr = 0.01
    for _ in range(5000):
        grad_a, grad_b = 0.0, 0.0
        for s, y in zip(scores, labels):
            logit = a * s + b
            logit = max(min(logit, 20), -20)  # clip
            p = 1 / (1 + math.exp(-logit))
            err = p - y
            grad_a += err * s
            grad_b += err
        a -= lr * grad_a / len(scores)
        b -= lr * grad_b / len(scores)

    calibrated = []
    for s in scores:
        logit = a * s + b
        logit = max(min(logit, 20), -20)
        calibrated.append(1 / (1 + math.exp(-logit)))
    return calibrated, a, b


def temperature_scaling(scores: list[float], labels: list[int]) -> tuple[list[float], float]:
    """Single-parameter temperature scaling on logit(score).

    Fits logit(p) = logit(score) / T.
    """
    eps = 1e-7
    logits = [math.log(max(s, eps) / max(1 - s, eps)) for s in scores]

    best_t, best_loss = 1.0, float("inf")
    for t_int in range(1, 200):  # grid search T from 0.1 to 20
        t = t_int * 0.1
        loss = 0.0
        for logit, y in zip(logits, labels):
            scaled = logit / t
            scaled = max(min(scaled, 20), -20)
            p = 1 / (1 + math.exp(-scaled))
            loss -= y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        if loss < best_loss:
            best_loss = loss
            best_t = t

    calibrated = []
    for logit in logits:
        scaled = logit / best_t
        scaled = max(min(scaled, 20), -20)
        calibrated.append(1 / (1 + math.exp(-scaled)))
    return calibrated, best_t


def format_bin_table(bins: list[BinStats], title: str) -> str:
    lines = [f"\n### {title}\n"]
    lines.append("| Bin | Count | Confidence | Accuracy | Gap |")
    lines.append("|-----|------:|-----------:|---------:|----:|")
    for b in bins:
        if b.count == 0:
            lines.append(f"| [{b.bin_lower:.2f}, {b.bin_upper:.2f}) | 0 | — | — | — |")
        else:
            lines.append(
                f"| [{b.bin_lower:.2f}, {b.bin_upper:.2f}] | {b.count} | "
                f"{b.mean_confidence:.3f} | {b.accuracy:.3f} | {b.gap:.3f} |"
            )
    return "\n".join(lines)


def format_reliability_ascii(bins: list[BinStats]) -> str:
    """ASCII reliability diagram."""
    lines = ["\n### Reliability Diagram (ASCII)\n", "```"]
    lines.append("Accuracy")
    lines.append("1.0 |" + " " * 40)

    # 10 rows from 1.0 to 0.0
    for row in range(10, -1, -1):
        y_val = row / 10
        row_str = f"{y_val:.1f} |"
        for b in bins:
            if b.count == 0:
                row_str += "   "
            else:
                if abs(b.accuracy - y_val) < 0.05:
                    row_str += " * "
                elif abs(y_val - b.mean_confidence) < 0.05:
                    row_str += " . "
                else:
                    row_str += "   "
        lines.append(row_str)

    lines.append("    +" + "---" * max(len(bins), 10))
    lines.append("     " + "".join(f"{b.mean_confidence:.1f}" + " " for b in bins if b.count > 0))
    lines.append("              Confidence")
    lines.append("  * = accuracy, . = perfect calibration line")
    lines.append("```")
    return "\n".join(lines)


def main():
    print("Loading data...")
    svg_scores = load_svg_scores()
    gold_labels = load_gold_labels()

    # Join by instance_id
    common_ids = sorted(set(svg_scores.keys()) & set(gold_labels.keys()))
    svg_only = set(svg_scores.keys()) - set(gold_labels.keys())
    gold_only = set(gold_labels.keys()) - set(svg_scores.keys())

    print(f"SVG scores: {len(svg_scores)}")
    print(f"Gold labels: {len(gold_labels)}")
    print(f"Joined: {len(common_ids)}")
    if svg_only:
        print(f"  SVG-only (no gold label): {len(svg_only)}")
    if gold_only:
        print(f"  Gold-only (no SVG score): {len(gold_only)}")

    scores = [svg_scores[iid] for iid in common_ids]
    labels = [gold_labels[iid] for iid in common_ids]
    n = len(scores)

    # ── Basic stats ──
    n_resolved = sum(labels)
    n_nonzero_svg = sum(1 for s in scores if s > 0)
    base_rate = n_resolved / n

    report_lines = [
        "# SVG ECE Measurement Report",
        f"\n**Date**: 2026-04-04",
        f"\n## Data Summary",
        f"\n- Joined instances: **{n}**",
        f"- Gold resolved (label=1): **{n_resolved}** ({n_resolved/n*100:.1f}%)",
        f"- SVG score > 0: **{n_nonzero_svg}** ({n_nonzero_svg/n*100:.1f}%)",
        f"- SVG score = 0: **{n - n_nonzero_svg}**",
        f"- SVG-only (no gold label): {len(svg_only)}",
        f"- Gold-only (no SVG score): {len(gold_only)}",
        f"- Base rate (gold pass): **{base_rate:.3f}**",
    ]

    # ── Score distribution ──
    report_lines.append("\n## SVG Score Distribution\n")
    report_lines.append("| Score Range | Count | Gold Pass | Gold Pass Rate |")
    report_lines.append("|-------------|------:|----------:|---------------:|")
    for lo, hi, label in [(0.0, 0.0, "= 0.0"), (0.001, 0.5, "(0, 0.5]"),
                          (0.5, 0.99, "(0.5, 1.0)"), (1.0, 1.0, "= 1.0")]:
        if lo == hi:
            idxs = [i for i in range(n) if scores[i] == lo]
        else:
            idxs = [i for i in range(n) if lo < scores[i] <= hi]
        if idxs:
            cnt = len(idxs)
            gp = sum(labels[i] for i in idxs)
            report_lines.append(f"| {label} | {cnt} | {gp} | {gp/cnt:.3f} |")
        else:
            report_lines.append(f"| {label} | 0 | 0 | — |")

    # ── ECE (equal-width) ──
    print("\nComputing ECE (equal-width, 10 bins)...")
    ece_uw, mce_uw, bins_uw = compute_ece(scores, labels, 10, "uniform")
    ece_mean, ece_lo, ece_hi = bootstrap_ece(scores, labels, 1000, 10, "uniform")

    report_lines.append(f"\n## Calibration Metrics (Equal-Width, 10 bins)")
    report_lines.append(f"\n| Metric | Value |")
    report_lines.append(f"|--------|------:|")
    report_lines.append(f"| **ECE** | **{ece_uw:.4f}** |")
    report_lines.append(f"| ECE 95% CI (bootstrap) | [{ece_lo:.4f}, {ece_hi:.4f}] |")
    report_lines.append(f"| MCE | {mce_uw:.4f} |")
    report_lines.append(f"| Brier Score | {brier_score(scores, labels):.4f} |")
    report_lines.append(format_bin_table(bins_uw, "Equal-Width Bins"))
    report_lines.append(format_reliability_ascii(bins_uw))

    # ── ECE (quantile) ──
    print("Computing ECE (quantile bins)...")
    ece_q, mce_q, bins_q = compute_ece(scores, labels, 10, "quantile")
    ece_mean_q, ece_lo_q, ece_hi_q = bootstrap_ece(scores, labels, 1000, 10, "quantile")

    report_lines.append(f"\n## Calibration Metrics (Quantile Bins)")
    report_lines.append(f"\n| Metric | Value |")
    report_lines.append(f"|--------|------:|")
    report_lines.append(f"| **ECE** | **{ece_q:.4f}** |")
    report_lines.append(f"| ECE 95% CI (bootstrap) | [{ece_lo_q:.4f}, {ece_hi_q:.4f}] |")
    report_lines.append(f"| MCE | {mce_q:.4f} |")
    report_lines.append(format_bin_table(bins_q, "Quantile Bins"))

    # ── RL-Readiness Assessment ──
    report_lines.append("\n## RL-Readiness Assessment\n")
    ece_val = ece_uw
    if ece_val < 0.05:
        assessment = "EXCELLENT — SVG ready for RL reward signal"
    elif ece_val < 0.10:
        assessment = "ACCEPTABLE — SVG usable for RL with temperature scaling"
    elif ece_val < 0.20:
        assessment = "MARGINAL — Use for rejection sampling SFT only, not RL"
    else:
        assessment = "POOR — Recalibrate via Platt scaling before any use"
    report_lines.append(f"- ECE = **{ece_val:.4f}**")
    report_lines.append(f"- Assessment: **{assessment}**")

    # ── Recalibration (always run for comparison) ──
    print("Running Platt scaling...")
    platt_scores, platt_a, platt_b = platt_scaling(scores, labels)
    ece_platt, mce_platt, bins_platt = compute_ece(platt_scores, labels, 10, "uniform")
    ece_platt_mean, ece_platt_lo, ece_platt_hi = bootstrap_ece(platt_scores, labels, 1000, 10, "uniform")

    print("Running temperature scaling...")
    temp_scores, temp_T = temperature_scaling(scores, labels)
    ece_temp, mce_temp, bins_temp = compute_ece(temp_scores, labels, 10, "uniform")
    ece_temp_mean, ece_temp_lo, ece_temp_hi = bootstrap_ece(temp_scores, labels, 1000, 10, "uniform")

    report_lines.append(f"\n## Recalibration Results\n")
    report_lines.append("| Method | ECE | ECE 95% CI | MCE | Brier | Parameters |")
    report_lines.append("|--------|----:|-----------|----:|------:|-----------|")
    report_lines.append(
        f"| Raw | {ece_uw:.4f} | [{ece_lo:.4f}, {ece_hi:.4f}] | {mce_uw:.4f} | "
        f"{brier_score(scores, labels):.4f} | — |"
    )
    report_lines.append(
        f"| Platt scaling | {ece_platt:.4f} | [{ece_platt_lo:.4f}, {ece_platt_hi:.4f}] | "
        f"{mce_platt:.4f} | {brier_score(platt_scores, labels):.4f} | a={platt_a:.3f}, b={platt_b:.3f} |"
    )
    report_lines.append(
        f"| Temperature scaling | {ece_temp:.4f} | [{ece_temp_lo:.4f}, {ece_temp_hi:.4f}] | "
        f"{mce_temp:.4f} | {brier_score(temp_scores, labels):.4f} | T={temp_T:.1f} |"
    )

    report_lines.append(format_bin_table(bins_platt, "Platt Scaling Bins"))
    report_lines.append(format_bin_table(bins_temp, "Temperature Scaling Bins"))

    # ── Cross-tabulation: SVG accepted vs gold ──
    report_lines.append("\n## Cross-Tabulation: SVG Accepted vs Gold Resolved\n")
    svg_accepted = {}
    with open(SVG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            svg_accepted[rec["instance_id"]] = rec["accepted"]

    tp = sum(1 for iid in common_ids if svg_accepted.get(iid) and gold_labels[iid] == 1)
    fp = sum(1 for iid in common_ids if svg_accepted.get(iid) and gold_labels[iid] == 0)
    fn = sum(1 for iid in common_ids if not svg_accepted.get(iid) and gold_labels[iid] == 1)
    tn = sum(1 for iid in common_ids if not svg_accepted.get(iid) and gold_labels[iid] == 0)

    report_lines.append("| | Gold Resolved | Gold Unresolved | Total |")
    report_lines.append("|---|---:|---:|---:|")
    report_lines.append(f"| SVG Accepted | {tp} | {fp} | {tp+fp} |")
    report_lines.append(f"| SVG Rejected | {fn} | {tn} | {fn+tn} |")
    report_lines.append(f"| Total | {tp+fn} | {fp+tn} | {n} |")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        report_lines.append(f"\n- SVG Precision: {precision:.3f}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        report_lines.append(f"- SVG Recall: {recall:.3f}")

    # ── Interpretation ──
    report_lines.append("\n## Interpretation\n")
    report_lines.append(
        "The SVG line_recall score distribution is extremely sparse: "
        f"{n - n_nonzero_svg}/{n} instances have score=0.0. "
        "This creates a degenerate calibration scenario where most of the ECE "
        "weight falls on the [0.0, 0.1) bin. The score is not a continuous "
        "confidence estimate in the traditional sense — it is a code overlap "
        "metric (line recall against gold patch)."
    )
    report_lines.append(
        "\n**Key caveat**: The SVG scores are from the SERA harness (Phase 0, "
        "verification-primitives), while gold labels are from the Claude Code + "
        "primitives production run (verification-primitives-swebench). These are "
        "different patches for the same instances. The SVG line_recall measures "
        "how close the SERA patch is to the gold patch, NOT the Claude Code patch."
    )

    # ── SWE-RM comparison context ──
    report_lines.append("\n## SWE-RM Comparison Context\n")
    report_lines.append(
        "SWE-RM (arXiv:2512.21919) found that two verifiers with similar ranking "
        "performance had 7x ECE difference (0.078 vs 0.541). The poorly calibrated "
        "one caused RL training collapse. Our SVG ECE result should be compared "
        "against these benchmarks:"
    )
    report_lines.append("\n| Verifier | ECE | RL Outcome |")
    report_lines.append("|----------|----:|-----------|")
    report_lines.append("| SWE-RM-LLM (well-calibrated) | 0.078 | Stable RL training |")
    report_lines.append("| SWE-RM-Verifier (poorly calibrated) | 0.541 | RL collapse |")
    report_lines.append(f"| **SVG line_recall (ours, raw)** | **{ece_uw:.3f}** | — |")
    report_lines.append(f"| **SVG line_recall (ours, Platt)** | **{ece_platt:.3f}** | — |")
    report_lines.append(f"| **SVG line_recall (ours, Temp)** | **{ece_temp:.3f}** | — |")

    # ── Write report ──
    report_text = "\n".join(report_lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ece_report.md"
    out_path.write_text(report_text)
    print(f"\nReport written to: {out_path}")
    print(f"\n{'='*60}")
    print(f"ECE (equal-width) = {ece_uw:.4f}  [{ece_lo:.4f}, {ece_hi:.4f}]")
    print(f"ECE (quantile)    = {ece_q:.4f}  [{ece_lo_q:.4f}, {ece_hi_q:.4f}]")
    print(f"ECE (Platt)       = {ece_platt:.4f}")
    print(f"ECE (Temp T={temp_T:.1f})  = {ece_temp:.4f}")
    print(f"Brier Score       = {brier_score(scores, labels):.4f}")
    print(f"Assessment: {assessment}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
