"""LLM-as-judge — Haiku 4.5 by default, Sonnet 4.6 for math escalation.

Two roles:
  1. Reward fallback: when parser-grader returns False, ask the judge
     "is `predicted` an equivalent answer to `gold`?". Used by
     reward.compute_reward via judge_fn.
  2. Gate 0.3 calibration: hand-grade 30 math items, run Haiku-judge on the
     same items, report agreement. If <90%, escalate to Sonnet judge for
     math-bucket reward fallback.

The judge is decoupled from training:
  - During Phase 1 training, reward computation only invokes the judge if
    the parser grader fails AND `judge_fn` is wired in (it's optional).
  - During Phase 0 calibration, run with --calibrate to get the agreement
    matrix.

Cost guardrails:
  - Haiku-judge: ~$0.001/call. ~256K judge calls during 5α training = ~$256.
  - Hard cap via env var COST_AWARE_JUDGE_MAX_CALLS (default 500K).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Defer boto import so unit tests can mock without installing boto3
import importlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


JUDGE_MODELS = {
    "haiku": ("anthropic.claude-haiku-4-5-20251001-v1:0", True),    # bedrock_id, needs_us_prefix
    "sonnet": ("anthropic.claude-sonnet-4-6", True),
}

JUDGE_PROMPT_FACTUAL = """You are evaluating whether a predicted answer matches a gold answer.

Gold: {gold}
Predicted: {predicted}

Are these equivalent? Differences in formatting, capitalization, or wording \
are OK if the underlying answer is the same. Different numeric values are \
NOT equivalent.

Respond with EXACTLY one word: "yes" or "no". No other text."""

JUDGE_PROMPT_MATH = """You are evaluating a math answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Is the predicted answer mathematically equivalent to the gold? Different \
forms of the same value (e.g., \\frac{{1}}{{2}} vs 0.5) are equivalent. \
Different numeric values are NOT equivalent.

Respond with EXACTLY one word: "yes" or "no". No other text."""


@dataclass
class JudgeStats:
    calls: int = 0
    cost_dollars: float = 0.0
    yes: int = 0
    no: int = 0
    errors: int = 0


class HaikuJudge:
    """Stateful judge wrapper. Tracks cost, enforces hard cap, retries on throttle."""

    def __init__(self, model: str = "haiku", region: str = "us-east-1"):
        if model not in JUDGE_MODELS:
            raise ValueError(f"Unknown judge: {model}")
        self.model = model
        self.bedrock_id, self.needs_us_prefix = JUDGE_MODELS[model]
        self.full_id = f"us.{self.bedrock_id}" if self.needs_us_prefix else self.bedrock_id
        self.region = region
        self._client = None
        self.stats = JudgeStats()
        self.max_calls = int(os.environ.get("COST_AWARE_JUDGE_MAX_CALLS", "500000"))
        self.in_cost_per_1m = 1.00 if model == "haiku" else 3.00     # docs-sourced
        self.out_cost_per_1m = 5.00 if model == "haiku" else 15.00

    def _get_client(self):
        if self._client is None:
            boto3 = importlib.import_module("boto3")
            from botocore.config import Config as BotoConfig
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=BotoConfig(read_timeout=30, retries={"max_attempts": 2, "mode": "standard"}),
            )
        return self._client

    def judge(
        self,
        predicted: str,
        gold: str,
        question: str = "",
        domain: str = "factual",
    ) -> Optional[bool]:
        """Returns True if judge says equivalent, False if not, None on error."""
        if self.stats.calls >= self.max_calls:
            log.warning("Judge call cap (%d) exceeded; refusing further calls.", self.max_calls)
            return None

        if domain == "math":
            prompt = JUDGE_PROMPT_MATH.format(question=question, gold=gold, predicted=predicted)
        else:
            prompt = JUDGE_PROMPT_FACTUAL.format(gold=gold, predicted=predicted)

        try:
            client = self._get_client()
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            })
            resp = client.invoke_model(modelId=self.full_id, body=body,
                                       contentType="application/json")
            data = json.loads(resp["body"].read())
            text_parts = [c.get("text", "") for c in data.get("content", [])
                          if c.get("type") == "text"]
            text = "".join(text_parts).strip().lower()
            usage = data.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            self.stats.cost_dollars += (
                in_tok * self.in_cost_per_1m + out_tok * self.out_cost_per_1m
            ) / 1_000_000.0
            self.stats.calls += 1

            if text.startswith("yes"):
                self.stats.yes += 1
                return True
            elif text.startswith("no"):
                self.stats.no += 1
                return False
            else:
                self.stats.errors += 1
                log.warning("Judge returned unparseable: %r", text[:50])
                return None
        except Exception as e:
            self.stats.errors += 1
            log.warning("Judge call failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Gate 0.3 calibration runner
# ---------------------------------------------------------------------------

def calibrate(human_graded: Path, judge_model: str = "haiku") -> dict:
    """human_graded is JSONL with rows: {question, predicted, gold, human_label: bool}.
    Returns agreement stats."""
    rows = [json.loads(l) for l in human_graded.read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"No rows in {human_graded}")

    judge = HaikuJudge(model=judge_model)
    agree = disagree = unparseable = 0
    fp = fn = 0
    samples = []
    for r in rows:
        domain = r.get("domain", "math")
        verdict = judge.judge(
            predicted=r["predicted"], gold=r["gold"],
            question=r.get("question", ""), domain=domain,
        )
        if verdict is None:
            unparseable += 1
            continue
        human = bool(r["human_label"])
        if verdict == human:
            agree += 1
        else:
            disagree += 1
            if verdict and not human:
                fp += 1   # judge says yes, human says no
            else:
                fn += 1
            samples.append({
                "question": r.get("question", "")[:200],
                "gold": r["gold"], "predicted": r["predicted"],
                "judge": verdict, "human": human,
            })

    n = agree + disagree
    return {
        "judge_model": judge_model,
        "n_graded": len(rows),
        "n_compared": n,
        "agreement": (agree / n) if n else 0.0,
        "false_positive_rate": (fp / n) if n else 0.0,
        "false_negative_rate": (fn / n) if n else 0.0,
        "unparseable": unparseable,
        "judge_cost_$": round(judge.stats.cost_dollars, 4),
        "disagreements": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", type=Path, help="JSONL of human-graded items for calibration")
    p.add_argument("--out", type=Path, default=Path("results/judge_calibration.json"))
    p.add_argument("--judge-model", default="haiku", choices=["haiku", "sonnet"])
    p.add_argument("--passing-threshold", type=float, default=0.90)
    args = p.parse_args()

    if not args.calibrate:
        print("Use --calibrate <human-graded.jsonl> to run Gate 0.3.")
        sys.exit(0)

    result = calibrate(args.calibrate, args.judge_model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n=== Gate 0.3 — Judge calibration ===")
    print(f"Judge model: {result['judge_model']}")
    print(f"Agreement:   {result['agreement']:.3f} (threshold {args.passing_threshold})")
    print(f"FP rate:     {result['false_positive_rate']:.3f}  (judge=yes, human=no)")
    print(f"FN rate:     {result['false_negative_rate']:.3f}  (judge=no, human=yes)")
    print(f"Unparseable: {result['unparseable']}")
    print(f"Cost:        ${result['judge_cost_$']:.4f}")
    print(f"Report:      {args.out}")

    if result["agreement"] < args.passing_threshold:
        print(f"\n[FAIL] agreement {result['agreement']:.3f} < {args.passing_threshold}")
        if args.judge_model == "haiku":
            print("→ Recommend retry with --judge-model sonnet (~5x cost, expected ~95-97% agreement).")
        sys.exit(1)
    print("[PASS]")


if __name__ == "__main__":
    main()
