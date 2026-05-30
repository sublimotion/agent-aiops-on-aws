"""Compute the oracle and best-static expected rewards for a sweep of alpha.

For each alpha:
  - For each source, find the ord that maximizes E[r] over the source's rollouts.
  - The "per-source oracle" picks that ord on every question of that source.
  - The "best-static" picks one ord across all sources.
  - Report the gap.

This tells us which alpha gives the largest oracle gap (most room for
the router to improve over static), and where the gap collapses (alpha
values where everything converges to a single static policy).
"""
from __future__ import annotations

import json
import sys

DEFAULT_MATH = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json"
DEFAULT_AIME = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json"
DEFAULT_WC = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json"

NAMES = ["gemma", "gpt-oss", "qwen3-32b", "qwen-coder", "mistral",
         "deepseek", "haiku", "sonnet", "opus"]

_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cn(c: float) -> float:
    return max(0.0, min(1.0, (c - _MIN_REF) / (_MAX_REF - _MIN_REF)))


def reward(is_correct: bool, cost_usd: float, alpha: float, floor: float = -1.0) -> float:
    if not is_correct:
        return 0.0
    return max(1.0 - alpha * cn(cost_usd), floor)


def per_source_e_r(rollouts: list[dict], alpha: float, correct_key: str) -> dict[int, float]:
    """ord -> mean reward across this source's rollouts at this alpha."""
    by_ord: dict[int, list[float]] = {}
    for r in rollouts:
        rs = reward(r[correct_key], r["cost_usd"], alpha)
        by_ord.setdefault(r["ord"], []).append(rs)
    return {ord_: sum(rs) / len(rs) for ord_, rs in by_ord.items()}


def main():
    sources = [
        ("math500", DEFAULT_MATH, "is_correct"),
        ("aime25", DEFAULT_AIME, "is_correct"),
        ("wildchat", DEFAULT_WC, "acceptable"),
    ]
    rollouts_by_src: dict[str, list[dict]] = {}
    for label, path, key in sources:
        rollouts_by_src[label] = json.load(open(path))["rollouts"]

    alphas = [0.3, 0.5, 1.0, 1.7, 3.0, 5.0]
    print(f"{'alpha':>5s}  per-source picks (E[r])"
          f"                                                       "
          f"oracle  best-static  gap")
    print("-" * 130)

    out: dict = {}
    for alpha in alphas:
        per_src_er = {}  # source -> {ord: E[r]}
        for label, _, key in sources:
            per_src_er[label] = per_source_e_r(rollouts_by_src[label], alpha, key)

        # Per-source best ord
        best_per_src = {}
        for label, _, _ in sources:
            ers = per_src_er[label]
            best_ord = max(ers, key=ers.get)
            best_per_src[label] = (best_ord, ers[best_ord])

        # Oracle E[r] = uniform mix of per-source bests, weighted by source size
        sizes = {s: len(rollouts_by_src[s]) // 9 for s, _, _ in sources}  # /9 because 9 ords per question
        total = sum(sizes.values())
        oracle_er = sum(best_per_src[s][1] * sizes[s] for s in sizes) / total

        # Best-static: pick a single ord that maximizes the weighted mix
        best_static_er = -10.0
        best_static_ord = None
        for ord_ in range(9):
            if any(ord_ not in per_src_er[s] for s, _, _ in sources):
                continue
            er_static = sum(per_src_er[s][ord_] * sizes[s] for s, _, _ in sources) / total
            if er_static > best_static_er:
                best_static_er = er_static
                best_static_ord = ord_

        picks_str = " ".join(
            f"{s[:4]}=ord_{best_per_src[s][0]}({NAMES[best_per_src[s][0]]}):{best_per_src[s][1]:+.3f}"
            for s, _, _ in sources
        )
        gap = oracle_er - best_static_er
        print(f"{alpha:>5.1f}  {picks_str}  "
              f"oracle={oracle_er:+.3f}  static=ord_{best_static_ord}({NAMES[best_static_ord]}):{best_static_er:+.3f}  "
              f"gap={gap:+.3f}")

        out[alpha] = {
            "per_source_best": {s: {"ord": best_per_src[s][0], "name": NAMES[best_per_src[s][0]],
                                    "e_r": round(best_per_src[s][1], 4)} for s, _, _ in sources},
            "per_source_e_r": {s: {ord_: round(er, 4) for ord_, er in per_src_er[s].items()}
                               for s, _, _ in sources},
            "oracle_e_r": round(oracle_er, 4),
            "best_static_ord": best_static_ord,
            "best_static_name": NAMES[best_static_ord],
            "best_static_e_r": round(best_static_er, 4),
            "oracle_gap": round(gap, 4),
        }

    out_path = "domains/autoresearch/blueprints/cost-aware-routing/results/runs/oracle_alpha_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # Print per-source full E[r] tables for the alphas with biggest gaps
    print("\nFull per-(source, ord) E[r] for alpha=1.0 and alpha=3.0:")
    for alpha in (1.0, 3.0):
        print(f"\n  alpha={alpha}")
        print(f"    {'src':<10s} " + " ".join(f"ord_{i}".rjust(7) for i in range(9)))
        for label, _, _ in sources:
            ers = out[alpha]["per_source_e_r"][label]
            cells = [f"{ers.get(i, 0):+7.3f}" for i in range(9)]
            # Bold the best per source via {} markers (text-only)
            best_ord = out[alpha]["per_source_best"][label]["ord"]
            cells[best_ord] = f"[{ers[best_ord]:+5.3f}]"
            print(f"    {label:<10s} " + " ".join(cells))


if __name__ == "__main__":
    main()
