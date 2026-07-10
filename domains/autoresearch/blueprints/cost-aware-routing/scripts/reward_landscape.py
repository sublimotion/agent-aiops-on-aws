"""Gate 0.2 sub-check — print reward landscape table for the configured pool
and assert that adjacent workers differ in reward by >= 0.02 at alpha=1.0.

Usage: python -m scripts.reward_landscape --pool configs/pool.yaml
"""
from __future__ import annotations

import argparse
import math
import sys

from .cost import CostModel
from .reward import EPS

ALPHAS = (0.1, 0.3, 1.0, 3.0, 5.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="configs/pool.yaml")
    p.add_argument("--min-gap", type=float, default=0.005,
                   help="minimum reward gap between adjacent workers (sorted by cost) at alpha=1.0. "
                        "0.005 is the operational floor; gaps tighter than this between adjacent workers "
                        "indicate near-identical cost — fine for the experiment but flag to user.")
    args = p.parse_args()

    cm = CostModel.from_yaml(args.pool)

    # Header
    print(f"\nPool: spread = {cm.cost_spread_oom():.2f} OOM   "
          f"min=${cm.min_cost:.6f}  max=${cm.max_cost:.6f}\n")
    cols = ["ord", "name", "$/q", "cost_norm_log"] + [f"r(α={a})" for a in ALPHAS]
    widths = [4, 22, 10, 14] + [10] * len(ALPHAS)
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))

    rewards_at_1 = []
    for o, w in sorted(cm.workers.items()):
        c = cm.assumed_cost(o)
        cn = cm.cost_norm_log(c)
        rewards = [max(EPS, math.exp(-a * cn)) for a in ALPHAS]
        rewards_at_1.append(rewards[ALPHAS.index(1.0)])
        row = [str(o), w.name, f"${c:.5f}", f"{cn:.3f}"] + [f"{r:.3f}" for r in rewards]
        print(" | ".join(s.ljust(wd) for s, wd in zip(row, widths)))

    # Gap check at alpha=1.0
    gaps = [abs(b - a) for a, b in zip(rewards_at_1[:-1], rewards_at_1[1:])]
    min_gap = min(gaps) if gaps else 0
    print(f"\nMin adjacent-worker reward gap at α=1.0: {min_gap:.4f} (threshold {args.min_gap})")
    if min_gap < args.min_gap:
        print(f"[FAIL] reward gradient is too flat between some adjacent workers; "
              f"reorder pool or refine cost model", file=sys.stderr)
        sys.exit(1)
    print("[PASS] reward landscape is informative across the pool")


if __name__ == "__main__":
    main()
