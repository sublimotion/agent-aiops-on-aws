"""Run all 9 workers on the NEW questions in augmented_baseline_500q.jsonl.

Skips questions already covered in the existing math500/aime25/wildchat
baselines (matched by source field). Uses category-appropriate judging:

  math, code      -> math judge prompt (gold-equivalence)
  factual, reasoning -> multiple-choice judge (extract letter, compare)
  open-domain     -> wildchat quality judge (truncation-tolerant)

Output appended to:
  results/baselines/always_x_augmented_<category>.json — one file per
    new category bucket (code/factual/reasoning/open-domain-extra).

Cost: 350 new questions x 9 workers = 3,150 worker calls
       + 3,150 judge calls. ~$25 total.
Time: ~25 minutes wall (16-way concurrency, similar to prior runs).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time

import boto3
from botocore.config import Config

sys.path.insert(0, "domains/autoresearch/blueprints/cost-aware-routing/scripts")
from worker_pool import POOL, invoke_worker, per_call_cost_usd
from cost_reward import JUDGE_PROMPT as MATH_JUDGE_PROMPT, JUDGE_INPUT_PER_1M, JUDGE_OUTPUT_PER_1M
from wildchat_judge import JUDGE_PROMPT as WC_JUDGE_PROMPT

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


# Multiple-choice judge: extract a letter from the response, compare to gold letter.
MC_JUDGE_PROMPT = """You are grading a multiple-choice question.

Question:
{question}

Gold answer letter: {gold}

Student's response:
{response}

Extract the LETTER (A/B/C/D/E/F/G/H/I/J) the student selected as their final answer.
If you cannot determine a clear answer letter, treat as INCORRECT.

Reply with EXACTLY one token on the first line: CORRECT or INCORRECT
Then optionally one short sentence of justification."""


def make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=120)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def call_judge(client, prompt_text: str, positive_token: str = "CORRECT") -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
    }
    t0 = time.time()
    try:
        resp = client.invoke_model(modelId=HAIKU, body=json.dumps(body))
        dur = time.time() - t0
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"].strip()
        first = text.splitlines()[0].strip().upper() if text else ""
        if first.startswith(positive_token):
            verdict = True
        elif first.startswith("UN" + positive_token) or first.startswith("IN" + positive_token):
            verdict = False
        else:
            verdict = None
        usage = payload.get("usage", {})
        return {
            "verdict": verdict,
            "raw": text[:200],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_s": round(dur, 2),
        }
    except Exception as e:
        return {"verdict": None, "raw": "", "input_tokens": 0, "output_tokens": 0,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def judge_response(client, q: dict, response_text: str) -> dict:
    """Pick the right judge for the category."""
    cat = q["category"]
    if cat in ("math", "code"):
        prompt = MATH_JUDGE_PROMPT.format(
            question=q["question"][:1500],
            gold=q["gold"][:500],
            response=response_text[-3000:],
        )
        return call_judge(client, prompt, "CORRECT")
    if cat in ("factual", "reasoning"):
        prompt = MC_JUDGE_PROMPT.format(
            question=q["question"][:1500],
            gold=q["gold"],
            response=response_text[-2000:],
        )
        return call_judge(client, prompt, "CORRECT")
    if cat == "open-domain":
        prompt = WC_JUDGE_PROMPT.format(
            question=q["question"][:2000],
            response=response_text[-3000:],
        )
        return call_judge(client, prompt, "ACCEPTABLE")
    raise ValueError(f"Unknown category: {cat}")


def _existing_baseline_ids() -> set[str]:
    """IDs of questions already covered by prior baselines.

    We bridge the existing math500/aime25/wildchat baselines (which used
    raw question text as id) to the augmented file's hashed scheme via
    short_id(category, text).
    """
    import hashlib

    def short_id(cat: str, text: str) -> str:
        return f"{cat}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"

    out: set[str] = set()
    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
    for path, cor_key in [
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ]:
        d = json.load(open(path))
        source = "math500" if "math500" in path else ("aime25" if "aime25" in path else "wildchat")
        cat = src_to_cat[source]
        for r in d["rollouts"]:
            out.add(short_id(cat, r["question"]))
    # Augmented 350-q baseline (already collected for the 480q dataset)
    try:
        aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
        for r in aug["rollouts"]:
            out.add(r["id"])
    except FileNotFoundError:
        pass
    return out


def load_new_questions(path: str) -> list[dict]:
    """Return only questions whose id is NOT in the existing baselines."""
    existing_ids = _existing_baseline_ids()
    print(f"  (skip set has {len(existing_ids)} already-baselined ids)")
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in existing_ids:
                out.append(r)
    return out


def run_worker_on_questions(client, ord_: int, questions: list[dict], max_tokens: int) -> list[dict]:
    """Concurrently invoke worker `ord_` on each question."""
    out: list[dict | None] = [None] * len(questions)

    def go(idx: int, q: dict):
        r = invoke_worker(client, ord_, q["question"], max_tokens=max_tokens, temperature=0.7)
        return idx, q, r

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(go, i, q) for i, q in enumerate(questions)]
        for fut in cf.as_completed(futures):
            i, q, r = fut.result()
            out[i] = (q, r)
    return out  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--ords", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json")
    args = ap.parse_args()

    questions = load_new_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} NEW questions (skipping math500/aime25/wildchat existing baselines)")
    by_cat = collections.Counter(q["category"] for q in questions)
    print(f"  by category: {dict(by_cat)}")

    ords = [int(x) for x in args.ords.split(",")]
    client = make_client()

    per_worker = {}
    all_rollouts: list[dict] = []

    for ord_ in ords:
        w = POOL[ord_]
        print(f"\n=== ord_{ord_}: {w.name} ===")

        # Long-form (open-domain, code) get more tokens; short-answer (factual, reasoning) get less.
        # Use 1024 as a reasonable default; the math reward is robust to truncation since
        # the gold answer extraction works on the response tail.
        t0 = time.time()
        rollout_rows = run_worker_on_questions(client, ord_, questions, max_tokens=1536)
        wall_r = time.time() - t0

        # Judge each
        t0 = time.time()
        judged: list[dict | None] = [None] * len(questions)

        def judge_one(i, q, r):
            if r["error"] is not None:
                return i, {"verdict": False, "raw": f"WORKER_ERROR: {r['error']}",
                           "input_tokens": 0, "output_tokens": 0}
            return i, judge_response(client, q, r["text"])

        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(judge_one, i, q, r) for i, (q, r) in enumerate(rollout_rows)]
            for fut in cf.as_completed(futs):
                i, j = fut.result()
                judged[i] = j
        wall_j = time.time() - t0

        n = len(rollout_rows)
        n_correct = sum(1 for j in judged if j["verdict"])
        avg_in = sum(r["input_tokens"] for _, r in rollout_rows) / n
        avg_out = sum(r["output_tokens"] for _, r in rollout_rows) / n
        avg_cost = sum(per_call_cost_usd(r["input_tokens"], r["output_tokens"], ord_)
                       for _, r in rollout_rows) / n
        avg_judge_cost = sum(j["input_tokens"] * JUDGE_INPUT_PER_1M / 1e6
                             + j["output_tokens"] * JUDGE_OUTPUT_PER_1M / 1e6
                             for j in judged) / n

        per_worker[ord_] = {
            "ord": ord_, "name": w.name, "n": n, "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_judge_cost_usd": round(avg_judge_cost, 6),
            "avg_input_tokens": round(avg_in, 1),
            "avg_output_tokens": round(avg_out, 1),
            "wall_s_rollouts": round(wall_r, 1),
            "wall_s_judge": round(wall_j, 1),
        }

        # Per-category accuracy on this ord
        per_cat = collections.defaultdict(lambda: {"n": 0, "n_correct": 0})
        for (q, r), j in zip(rollout_rows, judged):
            per_cat[q["category"]]["n"] += 1
            per_cat[q["category"]]["n_correct"] += int(bool(j["verdict"]))
        per_worker[ord_]["per_category_accuracy"] = {
            c: round(s["n_correct"] / max(s["n"], 1), 3) for c, s in per_cat.items()
        }

        print(f"  n={n}, accuracy={n_correct/n:.1%}, avg_cost=${avg_cost:.5f}/q")
        print(f"  per-category accuracy: {per_worker[ord_]['per_category_accuracy']}")

        for (q, r), j in zip(rollout_rows, judged):
            all_rollouts.append({
                "ord": ord_, "name": w.name,
                "id": q["id"],
                "category": q["category"],
                "source": q["source"],
                "question": q["question"][:300],
                "gold": q["gold"][:200],
                "response_tail": r["text"][-300:] if r["error"] is None else "",
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_usd": round(per_call_cost_usd(r["input_tokens"], r["output_tokens"], ord_), 6),
                "is_correct": bool(j["verdict"]),
                "judge_raw": j.get("raw", "")[:200],
            })

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "input": args.input,
        "n_questions": len(questions),
        "per_worker": per_worker,
        "rollouts": all_rollouts,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
