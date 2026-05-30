"""Closed-form cost-aware routing policy.

This is the inference-time component of the redesigned classifier-router
(see phase1-redesign-2026-05-28.md). It takes a (category, [complexity])
prediction from a classifier and returns the worker_id that maximizes
expected reward at the given alpha.

Architecture:

  question
     -> ModernBERT classifier (separate component, not here)
     -> {category: str, complexity: str | None}
     -> RouterPolicy.pick(category, complexity, alpha)
     -> worker_id (0..8)

The policy is a pure function over the QualityTable, which is built
offline from baseline rollouts. No RL, no training of the policy itself.
Adding/removing workers means re-fitting the quality table; no model
retraining.

This module is small (~150 lines) and unit-tested. It's ready to be
wired up to a ModernBERT classifier as soon as the classifier is trained.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Cost-normalization anchors — must match worker_pool.py and grpo_sim.py.
_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cost_normalized(cost_usd: float) -> float:
    """Map an actual cost to [0, 1] using the reference anchors."""
    return max(0.0, min(1.0, (cost_usd - _MIN_REF) / (_MAX_REF - _MIN_REF)))


def cost_aware_reward(is_correct: bool, cost_usd: float, alpha: float, floor: float = -1.0) -> float:
    """The reward function from the spec: max(1 - alpha*cn, -1) if correct else 0."""
    if not is_correct:
        return 0.0
    return max(1.0 - alpha * cost_normalized(cost_usd), floor)


@dataclass
class CellStats:
    """Aggregate stats for one (category, [complexity], worker) cell.

    Tracks per-rollout (is_correct, cost) so we can compute E[reward(alpha)]
    correctly — `mean(max(1-α·cn(c_i), -1))` is what we want, NOT
    `max(1-α·cn(mean(c)), -1)`. The floor and cn's linear-then-clamped
    shape make these differ.
    """
    n: int = 0
    n_correct: int = 0
    cost_sum: float = 0.0
    rollouts: list[tuple[bool, float]] = field(default_factory=list)

    def add(self, is_correct: bool, cost_usd: float) -> None:
        self.n += 1
        self.n_correct += int(is_correct)
        self.cost_sum += cost_usd
        self.rollouts.append((is_correct, cost_usd))

    @property
    def p_correct(self) -> float:
        return self.n_correct / max(self.n, 1)

    @property
    def avg_cost(self) -> float:
        return self.cost_sum / max(self.n, 1)

    def expected_reward(self, alpha: float, floor: float = -1.0) -> float:
        """Correct E[r] computed per-rollout then averaged."""
        if not self.rollouts:
            return 0.0
        total = 0.0
        for is_correct, cost_usd in self.rollouts:
            total += cost_aware_reward(is_correct, cost_usd, alpha, floor)
        return total / len(self.rollouts)


@dataclass
class QualityTable:
    """Per-(category, worker) success-rate and cost.

    The classifier emits a single 'category' string per query. This table
    maps category -> {worker_id: CellStats} for fast policy lookups.
    """
    cells: dict[str, dict[int, CellStats]] = field(default_factory=lambda: defaultdict(dict))
    # Categories the classifier knows about. Used for fallback when the
    # classifier emits a known-unknown.
    known_categories: set[str] = field(default_factory=set)

    def add_observation(self, category: str, worker_id: int, is_correct: bool, cost_usd: float) -> None:
        self.known_categories.add(category)
        cell = self.cells[category].setdefault(worker_id, CellStats())
        cell.add(is_correct, cost_usd)

    @classmethod
    def from_rollouts(cls, rollouts: Iterable[dict], category_fn) -> "QualityTable":
        """Build from a stream of rollout dicts.

        category_fn(rollout) -> category_str. Caller decides how to map
        (e.g. by source label, or by a more granular classifier output).
        """
        table = cls()
        for r in rollouts:
            category = category_fn(r)
            is_correct = bool(r.get("is_correct", r.get("acceptable", False)))
            cost = r["cost_usd"]
            table.add_observation(category, r["ord"], is_correct, cost)
        return table

    @classmethod
    def from_baselines(cls, paths_and_keys: list[tuple[str, str, str]]) -> "QualityTable":
        """Convenience: build from the existing always_x_*.json files.

        paths_and_keys: list of (category_name, json_path, correctness_field).
        Example: [
            ('math500', 'results/baselines/always_x_math500.json', 'is_correct'),
            ('aime25',  'results/baselines/always_x_aime25_n30.json', 'is_correct'),
            ('wildchat','results/baselines/always_x_wildchat_n50.json','acceptable'),
        ]
        """
        table = cls()
        for category, path, key in paths_and_keys:
            data = json.load(open(path))
            for r in data["rollouts"]:
                table.add_observation(category, r["ord"], bool(r[key]), r["cost_usd"])
        return table

    def expected_reward(self, category: str, worker_id: int, alpha: float) -> Optional[float]:
        cells = self.cells.get(category)
        if not cells or worker_id not in cells:
            return None
        cell = cells[worker_id]
        if cell.n == 0:
            return None
        return cell.expected_reward(alpha)

    def to_json(self) -> dict:
        return {
            "cells": {
                cat: {str(w): {"rollouts": [[bool(c), float(co)] for c, co in cell.rollouts]}
                      for w, cell in workers.items()}
                for cat, workers in self.cells.items()
            },
            "known_categories": sorted(self.known_categories),
        }

    @classmethod
    def from_json(cls, blob: dict) -> "QualityTable":
        table = cls()
        for cat, workers in blob["cells"].items():
            for w_str, s in workers.items():
                cell = CellStats()
                for is_correct, cost in s["rollouts"]:
                    cell.add(bool(is_correct), float(cost))
                table.cells[cat][int(w_str)] = cell
                table.known_categories.add(cat)
        return table


@dataclass
class RouterPolicy:
    """Closed-form cost-aware policy: argmax_w E[reward | category, w, alpha].

    No state, no training. Behavior is fully determined by the
    QualityTable + alpha + fallback_category.
    """
    table: QualityTable
    fallback_category: str = "math500"
    n_workers: int = 9

    def pick(self, category: str, alpha: float) -> int:
        """Return the worker_id with the highest expected reward.

        If category is unknown, fall back to the most common training category.
        If the table has no cell for some workers, those are skipped.

        Uses CellStats.expected_reward (per-rollout reward then mean), NOT
        p_correct × reward(avg_cost). The latter is a biased approximation
        that disagrees with the oracle calculation.
        """
        cat = category if category in self.table.known_categories else self.fallback_category
        cells = self.table.cells.get(cat, {})
        best_w = None
        best_er = float("-inf")
        for w in range(self.n_workers):
            if w not in cells:
                continue
            er = cells[w].expected_reward(alpha)
            if er > best_er:
                best_er = er
                best_w = w
        return best_w if best_w is not None else 0

    def pick_all(self, alpha: float) -> dict[str, int]:
        """Return the picked worker for every known category."""
        return {cat: self.pick(cat, alpha) for cat in sorted(self.table.known_categories)}


# --- Self-test ----------------------------------------------------------

def _self_test():
    """Validate against existing baselines: rebuild the per-source policy
    and confirm picks match what oracle_alpha_sweep.py reports.
    """
    table = QualityTable.from_baselines([
        ("math500", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("aime25", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("wildchat", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ])
    policy = RouterPolicy(table)

    # Cross-check against oracle_alpha_sweep.json
    oracle = json.load(open(
        "domains/autoresearch/blueprints/cost-aware-routing/results/runs/oracle_alpha_sweep.json"
    ))

    print("RouterPolicy self-test (per-source picks vs oracle_alpha_sweep):")
    print(f"  {'alpha':>5} {'src':>10} {'policy_pick':>12}  {'oracle_pick':>12}  match?")
    all_ok = True
    for alpha_str, info in oracle.items():
        alpha = float(alpha_str)
        for src, expected in info["per_source_best"].items():
            policy_pick = policy.pick(src, alpha)
            oracle_pick = expected["ord"]
            ok = policy_pick == oracle_pick
            all_ok &= ok
            print(f"  {alpha:>5.1f} {src:>10s} {policy_pick:>12d}  {oracle_pick:>12d}  {'OK' if ok else 'MISMATCH'}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    _self_test()
