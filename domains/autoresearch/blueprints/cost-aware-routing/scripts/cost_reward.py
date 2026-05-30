"""
Cost-aware reward function for cost-aware-routing GRPO Phase 1.

reward = is_correct ? max(1 - alpha * cost_normalized, -1) : 0

Correctness is judged by Haiku 4.5 (98% agreement with Sonnet 4.6 on
MATH500 per pre-flight). cost_normalized maps the actual $/call (using
measured input/output token counts) into [0, 1] using the 200/800
reference anchors from worker_pool.

This module is the GRPO reward backbone. The trainer calls
score_rollouts() once per minibatch with a list of rollouts; we batch
the Haiku judging concurrently to keep training step time bounded.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config

from worker_pool import (
    POOL,
    cost_normalized,
    invoke_worker,
    per_call_cost_usd,
)

REGION = "us-west-2"
JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
JUDGE_INPUT_PER_1M = 1.00   # $/1M input tokens
JUDGE_OUTPUT_PER_1M = 5.00  # $/1M output tokens

REWARD_FLOOR = -1.0  # see plan addendum §5

JUDGE_PROMPT = """You are grading a math/reasoning problem.

Question:
{question}

Gold answer:
{gold}

Student's response:
{response}

Decide whether the student's final answer is mathematically/logically
equivalent to the gold answer. Equivalent means: same value or claim
after simplification, regardless of formatting.

Reply with EXACTLY one token on the first line: CORRECT or INCORRECT
Then optionally one short sentence of justification."""


@dataclass
class Rollout:
    """One GRPO rollout."""
    question: str
    gold: str
    worker_ord: int          # which worker the router picked
    worker_response: str     # full response from the picked worker
    worker_input_tokens: int
    worker_output_tokens: int


@dataclass
class RewardResult:
    """Outcome of scoring one rollout."""
    is_correct: bool
    cost_usd: float
    cost_normalized: float
    reward: float
    judge_raw: str
    judge_input_tokens: int
    judge_output_tokens: int
    judge_cost_usd: float
    judge_elapsed_s: float


def _make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def _judge_one(client, question: str, gold: str, response: str) -> tuple[Optional[bool], dict]:
    """Returns (verdict, meta). verdict is True/False/None."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": JUDGE_PROMPT.format(
                question=question,
                gold=gold,
                response=response[-3000:],  # tail, in case of long responses
            )}],
        }],
    }
    t0 = time.time()
    try:
        resp = client.invoke_model(modelId=JUDGE_MODEL_ID, body=json.dumps(body))
        dur = time.time() - t0
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"].strip()
        first = text.splitlines()[0].strip().upper() if text else ""
        verdict: Optional[bool]
        if first.startswith("CORRECT"):
            verdict = True
        elif first.startswith("INCORRECT"):
            verdict = False
        else:
            verdict = None
        usage = payload.get("usage", {})
        meta = {
            "raw": text[:300],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_s": round(dur, 3),
            "error": None,
        }
        return verdict, meta
    except Exception as e:
        return None, {
            "raw": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def score_rollout(client, rollout: Rollout, alpha: float) -> RewardResult:
    """Score a single rollout."""
    verdict, jmeta = _judge_one(client, rollout.question, rollout.gold, rollout.worker_response)
    is_correct = bool(verdict)  # treat None (judge error) as INCORRECT — conservative

    cost_usd = per_call_cost_usd(
        rollout.worker_input_tokens, rollout.worker_output_tokens, rollout.worker_ord
    )
    cn = cost_normalized(cost_usd)

    if is_correct:
        reward = max(1.0 - alpha * cn, REWARD_FLOOR)
    else:
        reward = 0.0

    judge_cost = (
        jmeta["input_tokens"] * JUDGE_INPUT_PER_1M / 1e6
        + jmeta["output_tokens"] * JUDGE_OUTPUT_PER_1M / 1e6
    )
    return RewardResult(
        is_correct=is_correct,
        cost_usd=cost_usd,
        cost_normalized=cn,
        reward=reward,
        judge_raw=jmeta["raw"],
        judge_input_tokens=jmeta["input_tokens"],
        judge_output_tokens=jmeta["output_tokens"],
        judge_cost_usd=judge_cost,
        judge_elapsed_s=jmeta["elapsed_s"],
    )


def score_rollouts(rollouts: list[Rollout], alpha: float, workers: int = 16) -> list[RewardResult]:
    """Score a batch concurrently. Use during training between rollout-gen and policy update."""
    client = _make_client()
    results: list[Optional[RewardResult]] = [None] * len(rollouts)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(score_rollout, client, r, alpha): i for i, r in enumerate(rollouts)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results  # type: ignore


# --- Quick sanity test ----------------------------------------------------

def _self_test():
    """Run a small live sanity check on Bedrock. Expensive enough to gate."""
    client = _make_client()
    # Use Haiku itself as the worker so we know the response is reasonable.
    r = invoke_worker(
        client,
        ord_=6,  # Haiku
        prompt="What is 7 * 8? Reply with just the number.",
        max_tokens=32,
        temperature=0,
    )
    assert r["error"] is None, f"worker invoke failed: {r['error']}"
    rollout = Rollout(
        question="What is 7 * 8?",
        gold="56",
        worker_ord=6,
        worker_response=r["text"],
        worker_input_tokens=r["input_tokens"],
        worker_output_tokens=r["output_tokens"],
    )
    for alpha in [0.1, 1.0, 5.0]:
        res = score_rollout(client, rollout, alpha=alpha)
        print(
            f"alpha={alpha:>4.1f}  is_correct={res.is_correct}  "
            f"cost=${res.cost_usd:.5f}  cn={res.cost_normalized:.3f}  "
            f"reward={res.reward:+.3f}  judge_t={res.judge_elapsed_s}s"
        )


if __name__ == "__main__":
    _self_test()
