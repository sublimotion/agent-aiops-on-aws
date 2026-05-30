"""
Always-X baselines on WildChat eval — open-domain Pareto axis.

Same shape as run_baselines.py (MATH500/AIME25) but uses the WildChat
quality judge (no gold answers, ACCEPTABLE/UNACCEPTABLE rubric).

For each ord, run all eval questions, judge with Haiku, report:
  acceptable_rate, $/query (worker), $/query (judge), avg tokens.

Output: results/baselines/always_x_wildchat_n50.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import time

import boto3
from botocore.config import Config

from worker_pool import POOL, invoke_worker, per_call_cost_usd
from wildchat_judge import HAIKU as JUDGE_MODEL_ID, JUDGE_PROMPT, call_judge, make_client

JUDGE_INPUT_PER_1M = 1.00
JUDGE_OUTPUT_PER_1M = 5.00


def run_worker(client, ord_: int, questions: list[dict], max_tokens: int = 2048) -> list[dict]:
    rows = [None] * len(questions)

    def _go(i, q):
        r = invoke_worker(client, ord_, q["question"], max_tokens=max_tokens, temperature=0.7)
        return i, q, r

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_go, i, q) for i, q in enumerate(questions)]
        for fut in cf.as_completed(futures):
            i, q, r = fut.result()
            rows[i] = (q, r)
    return rows  # type: ignore


def judge_batch(client, rows: list[tuple]) -> list[dict]:
    """Judge a batch of (question, response) tuples concurrently."""
    out = [None] * len(rows)

    def _go(i, q, r):
        if r["error"] is not None:
            return i, {"verdict": False, "raw": f"WORKER_ERROR: {r['error']}",
                       "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0}
        return i, call_judge(client, JUDGE_MODEL_ID, q["question"], r["text"])

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_go, i, q, r) for i, (q, r) in enumerate(rows)]
        for fut in cf.as_completed(futures):
            i, judgement = fut.result()
            out[i] = judgement
    return out  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="domains/autoresearch/blueprints/cost-aware-routing/data/lmsys_eval_100.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--ords", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json",
    )
    args = ap.parse_args()

    questions = []
    with open(args.input) as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= args.limit:
                break
    print(f"Loaded {len(questions)} WildChat questions")

    ords = [int(x) for x in args.ords.split(",")]
    client = make_client()
    per_worker = {}
    all_rollouts = []

    for ord_ in ords:
        w = POOL[ord_]
        print(f"\n=== ord_{ord_}: {w.name} ===")

        t0 = time.time()
        rollout_rows = run_worker(client, ord_, questions, max_tokens=args.max_tokens)
        wall_r = time.time() - t0
        print(f"  rollouts: {wall_r:.1f}s")

        t0 = time.time()
        judgements = judge_batch(client, rollout_rows)
        wall_j = time.time() - t0
        print(f"  judging: {wall_j:.1f}s")

        n = len(rollout_rows)
        n_acc = sum(1 for j in judgements if j["verdict"])
        accept_rate = n_acc / n
        avg_in = sum(r["input_tokens"] for _, r in rollout_rows) / n
        avg_out = sum(r["output_tokens"] for _, r in rollout_rows) / n
        avg_cost = sum(per_call_cost_usd(r["input_tokens"], r["output_tokens"], ord_)
                       for _, r in rollout_rows) / n
        avg_judge_cost = sum(
            j["input_tokens"] * JUDGE_INPUT_PER_1M / 1e6 +
            j["output_tokens"] * JUDGE_OUTPUT_PER_1M / 1e6
            for j in judgements
        ) / n
        per_worker[ord_] = {
            "ord": ord_, "name": w.name, "n": n, "n_acceptable": n_acc,
            "acceptable_rate": round(accept_rate, 4),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_judge_cost_usd": round(avg_judge_cost, 6),
            "avg_input_tokens": round(avg_in, 1),
            "avg_output_tokens": round(avg_out, 1),
            "wall_s_rollouts": round(wall_r, 1),
            "wall_s_judge": round(wall_j, 1),
        }
        print(f"  acceptable: {accept_rate:.1%} ({n_acc}/{n})")
        print(f"  avg_cost: ${avg_cost:.5f}/q  judge: ${avg_judge_cost:.5f}/q")
        print(f"  avg tokens: in={avg_in:.0f} out={avg_out:.0f}")

        for (q, r), j in zip(rollout_rows, judgements):
            all_rollouts.append({
                "ord": ord_, "name": w.name,
                "id": q.get("id"),
                "question": q["question"][:300],
                "response_tail": r["text"][-300:] if r["error"] is None else "",
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_usd": round(per_call_cost_usd(r["input_tokens"], r["output_tokens"], ord_), 6),
                "acceptable": j["verdict"],
                "judge_raw": j.get("raw", "")[:200],
            })

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "input": args.input,
        "n_questions": len(questions),
        "max_tokens": args.max_tokens,
        "judge": JUDGE_MODEL_ID,
        "per_worker": per_worker,
        "rollouts": all_rollouts,
    }, indent=2))

    print("\n=== PARETO TABLE (Always-X on WildChat) ===")
    rows = sorted(per_worker.values(), key=lambda r: r["avg_cost_usd"])
    print(f"  {'ord':>3}  {'name':18s}  {'accept':>7s}  {'$/query':>9s}  {'$/accepted':>11s}")
    for row in rows:
        per_acc = row["avg_cost_usd"] / max(row["acceptable_rate"], 0.01)
        print(f"  ord_{row['ord']}  {row['name']:18s}  {row['acceptable_rate']:>6.1%}  "
              f"${row['avg_cost_usd']:>8.5f}  ${per_acc:>10.5f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
