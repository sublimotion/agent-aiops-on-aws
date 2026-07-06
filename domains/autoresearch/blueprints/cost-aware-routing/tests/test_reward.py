"""Unit tests for reward function. Gate 0.2 of Phase 0.

The rl-conductor reward function had 4 independent silent bugs (worth ~25-40pp
of plateau, unrecoverable without retraining). We are not repeating that.
Every property below MUST hold.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost import CostModel
from scripts.reward import (
    EPS, FORMAT_BONUS, parse_router_output, compute_reward,
)

POOL = str(Path(__file__).resolve().parents[1] / "configs" / "pool.yaml")


# ---------------------------------------------------------------------------
# parse_router_output
# ---------------------------------------------------------------------------

def test_parse_basic():
    assert parse_router_output("Answer: ord_5") == 5


def test_parse_with_thinking_prefix():
    raw = "<thinking>let me think about which worker</thinking>\nAnswer: ord_3"
    assert parse_router_output(raw) == 3


def test_parse_invalid():
    assert parse_router_output("I think we should use Sonnet.") is None


def test_parse_double_digit_ord():
    assert parse_router_output("Answer: ord_10") == 10


def test_parse_rejects_extra_args():
    # Phase 1 single-pick format ONLY. The Phase 2 multi-step format will
    # need its own parser; don't accidentally accept it here.
    assert parse_router_output("delegate ord_5 -- solve x") is None


# ---------------------------------------------------------------------------
# compute_reward — happy path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cm():
    return CostModel.from_yaml(POOL)


def test_correct_cheap_high_reward(cm):
    """Correct answer on cheapest worker should be near 1.0 (+ format bonus)."""
    r = compute_reward(
        router_output="Answer: ord_0",
        worker_response="The answer is \\boxed{42}.",
        gold_answer="42",
        dataset="math500",
        cost_model=cm,
        alpha=1.0,
        actual_input_tok=300, actual_output_tok=600,
    )
    assert r.is_correct
    assert r.format_valid
    # ord_0 is min_cost ⇒ cost_norm_log ≈ 0 ⇒ exp(0)=1 ⇒ reward ≈ 1 + 0.05
    assert abs(r.reward - (1.0 + FORMAT_BONUS)) < 0.01


def test_correct_expensive_lower_reward(cm):
    """Correct on Opus at α=1: reward < correct on Haiku."""
    r_opus = compute_reward(
        router_output="Answer: ord_10",
        worker_response="\\boxed{42}",
        gold_answer="42",
        dataset="math500",
        cost_model=cm, alpha=1.0,
        actual_input_tok=300, actual_output_tok=2000,
    )
    r_haiku = compute_reward(
        router_output="Answer: ord_8",
        worker_response="\\boxed{42}",
        gold_answer="42",
        dataset="math500",
        cost_model=cm, alpha=1.0,
        actual_input_tok=300, actual_output_tok=600,
    )
    assert r_opus.is_correct and r_haiku.is_correct
    assert r_opus.reward < r_haiku.reward


def test_eps_floor_high_alpha(cm):
    """At α=5 + correct on Opus, reward must be >= EPS + format bonus."""
    r = compute_reward(
        router_output="Answer: ord_10",
        worker_response="\\boxed{42}",
        gold_answer="42",
        dataset="math500",
        cost_model=cm, alpha=5.0,
        actual_input_tok=300, actual_output_tok=2000,
    )
    assert r.is_correct
    # raw exp(-5*1) = 0.0067 < EPS, so floor activates
    assert r.reward >= EPS + FORMAT_BONUS - 1e-6


def test_wrong_answer_zero_reward(cm):
    r = compute_reward(
        router_output="Answer: ord_5",
        worker_response="\\boxed{17}",
        gold_answer="42",
        dataset="math500",
        cost_model=cm, alpha=1.0,
        actual_input_tok=300, actual_output_tok=600,
    )
    assert not r.is_correct
    # Format bonus still applied even when answer is wrong
    assert r.reward == FORMAT_BONUS


def test_format_fail_zero_no_bonus(cm):
    r = compute_reward(
        router_output="I think Opus is the best choice here.",
        worker_response="\\boxed{42}",
        gold_answer="42",
        dataset="math500",
        cost_model=cm, alpha=1.0,
        actual_input_tok=300, actual_output_tok=600,
    )
    assert r.reward == 0.0
    assert not r.format_valid
    assert r.extraction_method == "format_fail"


# ---------------------------------------------------------------------------
# Reward landscape monotonicity (sanity check across alpha sweep)
# ---------------------------------------------------------------------------

def test_reward_monotonic_in_cost_for_fixed_alpha(cm):
    """At any α > 0, cheaper (correct) workers must dominate."""
    rewards = []
    for o in sorted(cm.workers):
        r = compute_reward(
            router_output=f"Answer: ord_{o}",
            worker_response="\\boxed{42}",
            gold_answer="42",
            dataset="math500",
            cost_model=cm, alpha=3.0,
            actual_input_tok=300, actual_output_tok=cm.workers[o].output_tok_assumed,
        )
        rewards.append((cm.assumed_cost(o), r.reward))
    rewards.sort()
    # Strictly non-increasing in cost (allow ties at the EPS floor)
    for (c1, r1), (c2, r2) in zip(rewards[:-1], rewards[1:]):
        assert r2 <= r1 + 1e-9, f"reward({c2})={r2} > reward({c1})={r1}"


def test_alpha_zero_collapses_to_quality_only(cm):
    """At α=0, exp(0)=1, so any correct rollout = 1 + format_bonus regardless of cost."""
    rs = []
    for o in (0, 5, 10):
        r = compute_reward(
            router_output=f"Answer: ord_{o}",
            worker_response="\\boxed{42}",
            gold_answer="42",
            dataset="math500",
            cost_model=cm, alpha=0.0,
            actual_input_tok=300, actual_output_tok=600,
        )
        rs.append(r.reward)
    assert all(abs(r - rs[0]) < 1e-9 for r in rs)
