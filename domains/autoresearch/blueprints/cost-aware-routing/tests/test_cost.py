"""Unit tests for the cost model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost import CostModel

POOL = str(Path(__file__).resolve().parents[1] / "configs" / "pool.yaml")


def test_loads_pool():
    cm = CostModel.from_yaml(POOL)
    assert cm.workers, "pool config loaded zero workers"
    assert cm.input_tok_assumed == 300


def test_cost_spread_meets_target():
    cm = CostModel.from_yaml(POOL)
    spread = cm.cost_spread_oom()
    # Spec target: ≥2 OOM (relaxed from 3 OOM after verifying real Bedrock pricing)
    # Current verified pool delivers ~3.2 OOM
    assert spread >= 2.0, f"cost spread only {spread:.2f} OOM — pool needs cheap or expensive specialist added"


def test_norm_endpoints():
    cm = CostModel.from_yaml(POOL)
    assert cm.cost_norm_log(cm.min_cost) == 0.0
    assert abs(cm.cost_norm_log(cm.max_cost) - 1.0) < 1e-9


def test_norm_monotonic():
    cm = CostModel.from_yaml(POOL)
    costs = sorted({cm.assumed_cost(o) for o in cm.workers})
    norms = [cm.cost_norm_log(c) for c in costs]
    assert all(b >= a - 1e-9 for a, b in zip(norms[:-1], norms[1:]))


def test_real_token_cost_matches_assumed():
    cm = CostModel.from_yaml(POOL)
    for o, w in cm.workers.items():
        delta = abs(cm.assumed_cost(o) - cm.cost(o, cm.input_tok_assumed, w.output_tok_assumed))
        assert delta < 1e-12
