"""Cost model for cost-aware routing.

Per-call marginal cost is computed from the verified pricing in configs/pool.yaml.
Log-cost normalization spans [min_cost, max_cost] across the entire pool.

Usage:
    from cost import CostModel
    cm = CostModel.from_yaml("configs/pool.yaml")
    cost_dollars = cm.cost(ord=5, input_tok=300, output_tok=620)
    cost_norm = cm.cost_norm_log(cost_dollars)   # in [0, 1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class WorkerCost:
    ord: int
    name: str
    in_per_1m: float
    out_per_1m: float
    output_tok_assumed: int


class CostModel:
    def __init__(self, workers: Dict[int, WorkerCost], input_tok_assumed: int):
        self.workers = workers
        self.input_tok_assumed = input_tok_assumed
        # min/max for log normalization computed from the assumed-token budgets
        # (single source of truth used in reward landscape table).
        per_query = [self.cost(o, input_tok_assumed, w.output_tok_assumed)
                     for o, w in workers.items()]
        self.min_cost = min(per_query)
        self.max_cost = max(per_query)
        if self.min_cost <= 0:
            raise ValueError("min_cost must be > 0 for log normalization")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CostModel":
        cfg = yaml.safe_load(Path(path).read_text())
        workers = {
            w["ord"]: WorkerCost(
                ord=w["ord"],
                name=w["name"],
                in_per_1m=w["in_per_1m"],
                out_per_1m=w["out_per_1m"],
                output_tok_assumed=w["output_tok_assumed"],
            )
            for w in cfg["workers"]
        }
        return cls(workers, cfg["input_tok_assumed"])

    def cost(self, ord: int, input_tok: int, output_tok: int) -> float:
        """Marginal $ for a call. Use measured token counts at runtime."""
        w = self.workers[ord]
        return (input_tok * w.in_per_1m + output_tok * w.out_per_1m) / 1_000_000.0

    def assumed_cost(self, ord: int) -> float:
        """$ for a call using assumed token budgets — used for reward landscape table."""
        w = self.workers[ord]
        return self.cost(ord, self.input_tok_assumed, w.output_tok_assumed)

    def cost_norm_log(self, cost_dollars: float) -> float:
        """Log-normalize a $ cost to [0, 1] over the pool's [min, max]."""
        if cost_dollars <= 0:
            return 0.0
        num = math.log10(cost_dollars) - math.log10(self.min_cost)
        den = math.log10(self.max_cost) - math.log10(self.min_cost)
        return max(0.0, min(1.0, num / den))

    def cost_spread_oom(self) -> float:
        """Orders of magnitude between cheapest and most-expensive worker."""
        return math.log10(self.max_cost / self.min_cost)


if __name__ == "__main__":
    import sys
    cm = CostModel.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/pool.yaml")
    print(f"min={cm.min_cost:.6f}  max={cm.max_cost:.6f}  spread={cm.cost_spread_oom():.2f} OOM")
    for o, w in sorted(cm.workers.items()):
        c = cm.assumed_cost(o)
        cn = cm.cost_norm_log(c)
        print(f"  ord_{o:2d} {w.name:<22} ${c:.6f}  cost_norm_log={cn:.3f}")
