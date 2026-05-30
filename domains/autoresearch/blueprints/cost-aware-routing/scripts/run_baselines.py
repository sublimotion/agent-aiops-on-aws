"""
Phase 1 Always-X baselines: per-worker accuracy and $/query on MATH500.

For Phase 1 success criterion #2 ("alpha=3.0 router achieves $/query <
Always-Sonnet AND accuracy >= Always-Haiku + 5pp") we need each worker's
solo performance. This script runs every worker on the same MATH500
subset (50 questions from rl-conductor v4 iter-074) and reports
(accuracy, $/query, judge_cost).

Output: results/baselines/always_x_math500_n{N}.json
        — per-worker stats + per-rollout records suitable for later
        Pareto-curve plotting alongside the trained router.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import time

import boto3
from botocore.config import Config

from cost_reward import (
    JUDGE_INPUT_PER_1M,
    JUDGE_OUTPUT_PER_1M,
    REWARD_FLOOR,
    Rollout,
    score_rollouts,
)
from worker_pool import POOL, invoke_worker, per_call_cost_usd

REGION = "us-west-2"
DEFAULT_INPUT = (
    "domains/autoresearch/blueprints/rl-conductor/results/greedy_eval/"
    "eval_greedy_v4_iter-0074_n50_paperfaithful.json"
)


def make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=120)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def load_questions(path: str, limit: int) -> list[dict]:
    raw = json.load(open(path))
    items = raw["results"][:limit]
    return [
        {
            "q_idx": it["q_idx"],
            "question_id": it.get("question_id"),
            "question": it["question"],
            "gold": it["gold"],
        }
        for it in items
    ]


def run_worker(client, ord_: int, questions: list[dict], max_workers: int = 8) -> list[Rollout]:
    """Invoke `ord_` on every question concurrently. Returns Rollout list."""
    out: list[None | Rollout] = [None] * len(questions)

    def _go(i: int, q: dict) -> tuple[int, Rollout]:
        prompt = (
            "Solve the following problem step by step. End your response with "
            "the final answer prefixed by 'Answer:'.\n\n"
            f"{q['question']}"
        )
        r = invoke_worker(
            client, ord_=ord_, prompt=prompt,
            max_tokens=1024, temperature=0.7,
        )
        return i, Rollout(
            question=q["question"],
            gold=q["gold"],
            worker_ord=ord_,
            worker_response=r["text"] if r["error"] is None else "",
            worker_input_tokens=r["input_tokens"],
            worker_output_tokens=r["output_tokens"],
        )

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_go, i, q) for i, q in enumerate(questions)]
        for fut in cf.as_completed(futures):
            i, ro = fut.result()
            out[i] = ro
    return out  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json",
    )
    ap.add_argument("--ords", type=str, default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--alpha-for-reward", type=float, default=1.0,
                    help="Alpha used to compute reference reward column. "
                         "Doesn't affect accuracy/cost; just a reference point.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ords = [int(x) for x in args.ords.split(",")]
    questions = load_questions(args.input, args.limit)
    print(f"Loaded {len(questions)} questions from {args.input}")

    client = make_client()

    per_worker = {}
    all_rollouts: list[dict] = []

    for ord_ in ords:
        w = POOL[ord_]
        print(f"\n=== ord_{ord_}: {w.name} ===")
        t0 = time.time()
        rollouts = run_worker(client, ord_, questions, max_workers=args.workers)
        wall = time.time() - t0
        print(f"  rollouts complete: {wall:.1f}s")

        # Score with cost-aware reward (alpha=1.0 default). is_correct is what
        # we really care about for accuracy; reward is just a sanity column.
        t0 = time.time()
        results = score_rollouts(rollouts, alpha=args.alpha_for_reward, workers=8)
        judge_wall = time.time() - t0
        print(f"  judging complete: {judge_wall:.1f}s")

        n = len(rollouts)
        n_correct = sum(1 for r in results if r.is_correct)
        accuracy = n_correct / n
        avg_cost = sum(r.cost_usd for r in results) / n
        avg_judge_cost = sum(r.judge_cost_usd for r in results) / n
        avg_reward = sum(r.reward for r in results) / n
        avg_in_tok = sum(ro.worker_input_tokens for ro in rollouts) / n
        avg_out_tok = sum(ro.worker_output_tokens for ro in rollouts) / n

        per_worker[ord_] = {
            "ord": ord_,
            "name": w.name,
            "model_id": w.model_id,
            "n": n,
            "n_correct": n_correct,
            "accuracy": round(accuracy, 4),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_judge_cost_usd": round(avg_judge_cost, 6),
            "avg_input_tokens": round(avg_in_tok, 1),
            "avg_output_tokens": round(avg_out_tok, 1),
            "avg_reward_alpha1": round(avg_reward, 4),
            "wall_s_rollouts": round(wall, 1),
            "wall_s_judge": round(judge_wall, 1),
        }
        print(f"  accuracy: {accuracy:.1%} ({n_correct}/{n})")
        print(f"  avg_cost: ${avg_cost:.5f}/q  avg_judge: ${avg_judge_cost:.5f}/q")
        print(f"  avg tokens: in={avg_in_tok:.0f} out={avg_out_tok:.0f}")
        print(f"  avg_reward(alpha=1): {avg_reward:+.3f}")

        for ro, res in zip(rollouts, results):
            all_rollouts.append({
                "ord": ord_,
                "name": w.name,
                "question": ro.question[:200],
                "gold": ro.gold,
                "response_tail": ro.worker_response[-400:],
                "input_tokens": ro.worker_input_tokens,
                "output_tokens": ro.worker_output_tokens,
                "cost_usd": round(res.cost_usd, 6),
                "is_correct": res.is_correct,
                "reward_alpha1": round(res.reward, 4),
                "judge_raw": res.judge_raw[:200],
            })

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "input": args.input,
        "n_questions": len(questions),
        "alpha_for_reward": args.alpha_for_reward,
        "per_worker": per_worker,
        "rollouts": all_rollouts,
    }, indent=2))

    print("\n=== PARETO TABLE (Always-X baselines) ===")
    print(f"  {'ord':>3}  {'name':18s}  {'accuracy':>9s}  {'$/query':>9s}  {'$/correct':>10s}")
    rows = sorted(per_worker.values(), key=lambda r: r["avg_cost_usd"])
    for row in rows:
        per_correct = row["avg_cost_usd"] / max(row["accuracy"], 0.01)
        print(f"  ord_{row['ord']}  {row['name']:18s}  {row['accuracy']:>8.1%}  "
              f"${row['avg_cost_usd']:>8.5f}  ${per_correct:>9.5f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
