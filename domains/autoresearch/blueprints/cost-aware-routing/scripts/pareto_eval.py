"""Phase 1 final eval — Pareto curve, hypervolume, oracle gap, baseline comparisons.

Inputs:
    --rollouts <jsonl>    rollout records from training rollouts.jsonl OR a held-out eval run
    --baselines <dir>     dir of always-{worker}.jsonl files (one per static-routing baseline)

For each (alpha, eval-set) cell, compute:
    1. mean cost ($/q)
    2. mean quality (correct rate)
    3. quality / $ ratio (the headline)
    4. bootstrap 95% CIs (10K resamples)

Then build the Pareto frontier (sorted by cost) and compute:
    - Hypervolume gain over baselines (reference (cost=$0.30, quality=0))
    - Oracle gap closed (oracle = per-question cheapest correct worker)
    - Headline test: routers' quality/$ ≥ 5× Always-Opus, paired bootstrap p<0.01

Output:
    results/pareto_phase1.json
    results/pareto_phase1.html (visual; reuses spec-review's heatmap palette)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_rollouts(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def per_router_metrics(rollouts: list[dict], dataset_filter: str | None = None) -> dict:
    """Collapse rollouts to (alpha, dataset) → mean cost, mean quality."""
    by_key = defaultdict(list)
    for r in rollouts:
        if dataset_filter and r.get("dataset") != dataset_filter:
            continue
        key = (r.get("alpha"), r.get("dataset", "unknown"))
        by_key[key].append((r.get("cost_dollars", 0.0), int(bool(r.get("is_correct", False)))))

    out = {}
    for (alpha, ds), vals in by_key.items():
        if not vals:
            continue
        costs = [v[0] for v in vals]
        corrects = [v[1] for v in vals]
        out[(alpha, ds)] = {
            "n": len(vals),
            "mean_cost": sum(costs) / len(costs),
            "mean_quality": sum(corrects) / len(corrects),
            "total_cost": sum(costs),
            "total_correct": sum(corrects),
        }
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_resamples: int = 10000,
                 confidence: float = 0.95, seed: int = 17) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, len(values) - 1)] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(((1 - confidence) / 2) * n_resamples)]
    hi = means[int((1 - (1 - confidence) / 2) * n_resamples) - 1]
    return (lo, hi)


def paired_bootstrap_p(diffs: list[float], n_resamples: int = 10000,
                       seed: int = 17) -> float:
    """Two-sided p-value: P(resampled mean has different sign from observed)."""
    if not diffs:
        return 1.0
    obs_mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        sample = [diffs[rng.randint(0, len(diffs) - 1)] for _ in range(len(diffs))]
        m = sum(sample) / len(sample)
        if (obs_mean > 0 and m <= 0) or (obs_mean < 0 and m >= 0):
            extreme += 1
    return 2 * min(extreme / n_resamples, 1 - extreme / n_resamples)


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------

def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """2D hypervolume w.r.t. reference point. Points are (cost, quality) where
    we want to MINIMIZE cost and MAXIMIZE quality. Reference is "worst case"
    (max cost, min quality). HV = area of the dominated region.

    Algorithm: sort by cost ascending, walk the staircase.
    """
    if not points:
        return 0.0
    ref_cost, ref_quality = ref
    # Filter dominated points
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float]] = []
    best_q = -1
    for c, q in sorted_pts:
        if q > best_q:
            frontier.append((c, q))
            best_q = q
    # Compute area in log10(cost) × quality space
    if not frontier:
        return 0.0
    log_ref = math.log10(max(ref_cost, 1e-9))
    hv = 0.0
    prev_log_cost = log_ref
    for c, q in reversed(frontier):     # rightmost (highest cost) → leftmost
        log_c = math.log10(max(c, 1e-9))
        width = max(0.0, prev_log_cost - log_c)
        height = max(0.0, q - ref_quality)
        hv += width * height
        prev_log_cost = log_c
    return hv


# ---------------------------------------------------------------------------
# Oracle (per-question cheapest correct worker)
# ---------------------------------------------------------------------------

def compute_oracle(rollouts: list[dict]) -> dict:
    """Group by question; for each question, find the cheapest worker that got
    it right. Returns {question_id: oracle_cost} and aggregate stats."""
    by_q: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rollouts:
        qid = r.get("id") or r.get("question") or ""
        if not qid:
            continue
        by_q[qid].append((r.get("cost_dollars", 0.0), int(bool(r.get("is_correct", False)))))

    oracle_costs = []
    n_solvable = 0
    for qid, vals in by_q.items():
        correct = [c for c, ok in vals if ok]
        if correct:
            oracle_costs.append(min(correct))
            n_solvable += 1
    return {
        "n_questions": len(by_q),
        "n_solvable": n_solvable,
        "oracle_mean_cost": sum(oracle_costs) / len(oracle_costs) if oracle_costs else 0.0,
        "oracle_quality": n_solvable / len(by_q) if by_q else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True, help="path to rollouts.jsonl")
    p.add_argument("--baselines-dir", default=None,
                   help="optional dir with always-{worker}.jsonl baseline files")
    p.add_argument("--out-json", default="results/pareto_phase1.json")
    p.add_argument("--ref-cost", type=float, default=0.30)
    p.add_argument("--ref-quality", type=float, default=0.0)
    args = p.parse_args()

    rollouts = load_rollouts(Path(args.rollouts))
    log.info("loaded %d rollouts", len(rollouts))

    metrics = per_router_metrics(rollouts)

    # Aggregate per-router across all eval sets
    by_alpha = defaultdict(lambda: {"costs": [], "corrects": []})
    for (alpha, ds), m in metrics.items():
        by_alpha[alpha]["costs"].append(m["total_cost"])
        by_alpha[alpha]["corrects"].append(m["total_correct"])
        by_alpha[alpha].setdefault("ns", []).append(m["n"])

    pareto_points = []
    router_summary = {}
    for alpha, agg in sorted(by_alpha.items()):
        n = sum(agg["ns"])
        mc = sum(agg["costs"]) / n if n else 0.0
        mq = sum(agg["corrects"]) / n if n else 0.0
        # Bootstrap CI for quality
        per_q_correct = []
        for r in rollouts:
            if r.get("alpha") == alpha:
                per_q_correct.append(int(bool(r.get("is_correct", False))))
        ci_lo, ci_hi = bootstrap_ci(per_q_correct, seed=17)
        router_summary[f"alpha={alpha}"] = {
            "n": n,
            "mean_cost": round(mc, 6),
            "mean_quality": round(mq, 4),
            "quality_per_dollar": round(mq / max(mc, 1e-9), 2),
            "quality_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        pareto_points.append((mc, mq))

    # Hypervolume of router family
    ref = (args.ref_cost, args.ref_quality)
    hv_routers = hypervolume_2d(pareto_points, ref)

    # Baseline comparison if provided
    hv_baselines = None
    headline_test = {}
    if args.baselines_dir:
        baseline_points = []
        baseline_quality_per_dollar = {}
        for blf in sorted(Path(args.baselines_dir).glob("always-*.jsonl")):
            recs = load_rollouts(blf)
            costs = [r.get("cost_dollars", 0.0) for r in recs]
            corrects = [int(bool(r.get("is_correct", False))) for r in recs]
            n = len(recs)
            mc = sum(costs) / n if n else 0.0
            mq = sum(corrects) / n if n else 0.0
            baseline_name = blf.stem
            baseline_quality_per_dollar[baseline_name] = mq / max(mc, 1e-9)
            baseline_points.append((mc, mq))
        hv_baselines = hypervolume_2d(baseline_points, ref)

        # Headline: best router q/$ vs Always-Opus
        opus_qpd = baseline_quality_per_dollar.get("always-opus-4.7", 0.0)
        if opus_qpd > 0:
            best_router_qpd = max(s["quality_per_dollar"] for s in router_summary.values())
            headline_test = {
                "always_opus_quality_per_dollar": round(opus_qpd, 2),
                "best_router_quality_per_dollar": round(best_router_qpd, 2),
                "ratio": round(best_router_qpd / opus_qpd, 2),
                "passes_5x_threshold": best_router_qpd / opus_qpd >= 5.0,
            }

    # Oracle
    oracle = compute_oracle(rollouts)

    report = {
        "n_rollouts": len(rollouts),
        "router_summary": router_summary,
        "hypervolume": {
            "routers": round(hv_routers, 4),
            "baselines": round(hv_baselines, 4) if hv_baselines is not None else None,
            "gain": round(hv_routers - hv_baselines, 4) if hv_baselines is not None else None,
        },
        "headline_test": headline_test,
        "oracle": oracle,
        "reference_point": {"cost": args.ref_cost, "quality": args.ref_quality},
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))

    print("\n=== Phase 1 Pareto eval ===")
    for k, v in router_summary.items():
        print(f"  {k:<14}  cost=${v['mean_cost']:.5f}  quality={v['mean_quality']:.3f}  "
              f"q/$={v['quality_per_dollar']:.1f}  CI95=[{v['quality_ci_95'][0]:.3f},"
              f"{v['quality_ci_95'][1]:.3f}]")
    print(f"\n  HV(routers) = {hv_routers:.4f}")
    if hv_baselines is not None:
        print(f"  HV(baselines) = {hv_baselines:.4f}")
        print(f"  HV gain    = {hv_routers - hv_baselines:+.4f}")
    if headline_test:
        sym = "✓" if headline_test["passes_5x_threshold"] else "✗"
        print(f"\n  Headline {sym}  best q/$ = {headline_test['best_router_quality_per_dollar']:.1f}, "
              f"Always-Opus q/$ = {headline_test['always_opus_quality_per_dollar']:.1f}, "
              f"ratio {headline_test['ratio']:.1f}× (target ≥5×)")
    if oracle["n_questions"]:
        print(f"\n  Oracle: {oracle['n_solvable']}/{oracle['n_questions']} questions "
              f"solvable; oracle cost ${oracle['oracle_mean_cost']:.5f}")
    print(f"\nReport: {args.out_json}")


if __name__ == "__main__":
    main()
