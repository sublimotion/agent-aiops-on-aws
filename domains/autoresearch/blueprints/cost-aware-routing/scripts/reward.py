"""Cost-aware reward function with epsilon-floor and log-cost normalization.

reward = is_correct ? max(eps, exp(-alpha * cost_norm_log)) : 0

Plus format-validity bonus +0.05 (added to reward, not multiplied) when the
router output parses cleanly.

The reward function is called by the GRPO trainer per rollout. All graders,
extractors, and the cost model are read-only.

Pre-flight (Gate 0.2): every component below has unit tests in
tests/test_reward.py and tests/test_extractors.py. NEVER weaken a grader to
"fix" a low correct-rate — find the root cause.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .cost import CostModel
from .extractors import extract
from .graders import grade

# Phase 1 router output format. Phase 2 uses a different regex (see spec).
ROUTER_FORMAT_RE = re.compile(r"^Answer:\s*ord_(\d+)\s*$", re.MULTILINE)
FORMAT_BONUS = 0.05
EPS = 0.01


@dataclass
class RewardBreakdown:
    """Returned alongside scalar reward for logging / regrade."""
    reward: float
    is_correct: bool
    format_valid: bool
    extracted_answer: str
    extraction_method: str
    cost_dollars: float
    cost_norm_log: float
    worker_ord: Optional[int]
    fallback_to_judge: bool = False


def parse_router_output(text: str) -> Optional[int]:
    """Phase 1 parser. Returns ord int or None.

    Matches `^Answer:\\s*ord_(\\d+)\\s*$` MULTILINE — the router may emit a
    `<thinking>...</thinking>` prefix; we don't care, we just need the
    Answer: line. Note: this ignores any prose after the answer line, which
    is fine for Phase 1.
    """
    m = ROUTER_FORMAT_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def compute_reward(
    router_output: str,
    worker_response: str,
    gold_answer: str,
    dataset: str,
    cost_model: CostModel,
    alpha: float,
    actual_input_tok: int,
    actual_output_tok: int,
    judge_fn=None,                # callable(question, predicted, gold) -> bool, optional
    question_for_judge: str = "",
) -> RewardBreakdown:
    """Compute one rollout's reward.

    Args:
        router_output: raw text from the router LLM (the policy being trained).
        worker_response: raw text from the selected worker.
        gold_answer: ground truth (string for math/MCQ; test code for HumanEval).
        dataset: dataset name; routes to the right extractor + grader.
        cost_model: pool cost model loaded from configs/pool.yaml.
        alpha: cost-sensitivity coefficient.
        actual_input_tok / actual_output_tok: measured tokens for THIS call.
        judge_fn: optional LLM-judge fallback when the parser-grader returns False.
        question_for_judge: original question, passed to judge_fn.
    """
    worker_ord = parse_router_output(router_output)
    format_valid = worker_ord is not None

    if not format_valid:
        return RewardBreakdown(
            reward=0.0,
            is_correct=False,
            format_valid=False,
            extracted_answer="",
            extraction_method="format_fail",
            cost_dollars=0.0,
            cost_norm_log=0.0,
            worker_ord=None,
        )

    extracted, method = extract(worker_response, dataset)
    is_correct = grade(extracted, gold_answer, dataset) if extracted else False

    fallback_to_judge = False
    if not is_correct and judge_fn is not None and extracted:
        try:
            is_correct = bool(judge_fn(question_for_judge, extracted, gold_answer))
            fallback_to_judge = True
        except Exception:
            pass

    cost_d = cost_model.cost(worker_ord, actual_input_tok, actual_output_tok)
    cost_n = cost_model.cost_norm_log(cost_d)

    if is_correct:
        base = max(EPS, math.exp(-alpha * cost_n))
    else:
        base = 0.0

    reward = base + (FORMAT_BONUS if format_valid else 0.0)

    return RewardBreakdown(
        reward=reward,
        is_correct=is_correct,
        format_valid=format_valid,
        extracted_answer=extracted,
        extraction_method=method,
        cost_dollars=cost_d,
        cost_norm_log=cost_n,
        worker_ord=worker_ord,
        fallback_to_judge=fallback_to_judge,
    )
