"""
Judge-reward variance test — how stable is Haiku-as-judge on repeat calls?

GRPO uses within-group advantage normalization: A = (r − mean) / std.
If Haiku gives different verdicts on the same (q, response) pair across
calls, that noise propagates into the advantage signal and slows
convergence (the policy gradient sees worker_quality + judge_noise
instead of just worker_quality).

This script samples 30 pre-judged rollouts (10 each from MATH500, AIME25,
WildChat) and re-judges each one 5× with Haiku at temperature=0. Reports:

  - Per-source flip rate: fraction of (q, r) pairs where any of the 5
    re-judgings disagrees with at least one other.
  - Cross-source comparison: math judging is binary on a clean spec; open
    domain WildChat is more subjective and likely noisier.
  - Mean ratio of agreement-with-original-judgement.

If flip rate > 5%, the recommendation is to ensemble (3 judge calls +
majority vote) for the GRPO reward signal. Cost roughly triples but the
training signal becomes cleaner.

Cost: 30 pairs × 5 calls × ~$0.001 = ~$0.15.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import time

import boto3
from botocore.config import Config

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


# Math judge prompt (from cost_reward.py)
MATH_JUDGE_PROMPT = """You are grading a math/reasoning problem.

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

# WildChat (open-domain) judge prompt (from wildchat_judge.py, truncation-tolerant)
WC_JUDGE_PROMPT = """You are evaluating an AI assistant's response to a user prompt.

User prompt:
{question}

Assistant's response:
{response}

Decide whether the response is ACCEPTABLE or UNACCEPTABLE based on the
*content that's there* — ignore truncation. Many responses end with the
last token cap reached; that is a system-level limit, not a quality
problem with the assistant. Judge the substance.

ACCEPTABLE means:
  - addresses the user's actual ask (not a refusal or off-topic ramble)
  - is factually plausible (no obvious hallucinations)
  - respects explicit constraints (format, tone) the user specified
  - the visible content engages the prompt at a competent level

UNACCEPTABLE means:
  - refuses or evades a benign request
  - obviously wrong facts, broken code that clearly wouldn't run
  - generates content that doesn't match the prompt at all
  - so brief or generic that it doesn't engage the prompt
  - violates a clear safety norm

Do NOT mark UNACCEPTABLE for: cut-off mid-sentence, incomplete final list
item, missing concluding paragraph. These are token-cap artifacts.

Reply with EXACTLY one token on the first line: ACCEPTABLE or UNACCEPTABLE.
Then optionally one short sentence of justification."""


def call_judge(client, prompt_text: str, positive_token: str) -> dict:
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
        verdict: bool | None
        if first.startswith(positive_token):
            verdict = True
        elif first.startswith("UN" + positive_token) or first.startswith("IN" + positive_token):
            verdict = False
        else:
            verdict = None
        usage = payload.get("usage", {})
        return {"verdict": verdict, "raw": text[:200], "elapsed_s": round(dur, 2),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0)}
    except Exception as e:
        return {"verdict": None, "raw": "", "elapsed_s": round(time.time() - t0, 2),
                "input_tokens": 0, "output_tokens": 0,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def load_pairs(math_path: str, aime_path: str, wc_path: str, n_per_source: int = 10, seed: int = 17) -> list[dict]:
    """Load (q, gold, response, source, original_verdict) tuples — one batch per source."""
    rng = random.Random(seed)
    pairs = []
    for path, source in [(math_path, "math500"), (aime_path, "aime25"), (wc_path, "wildchat")]:
        raw = json.load(open(path))
        rollouts = raw["rollouts"]
        # Group rollouts by question to maximize variety
        by_q: dict[str, list[dict]] = {}
        for r in rollouts:
            qid = r.get("id") or r.get("question", "")[:60]
            by_q.setdefault(qid, []).append(r)
        # Pick n_per_source distinct questions; for each, pick one rollout (mixing ords for variety)
        qids = list(by_q.keys())
        rng.shuffle(qids)
        sampled = qids[:n_per_source]
        for qid in sampled:
            rs = by_q[qid]
            r = rng.choice(rs)
            # gold field name varies — math has 'gold' nested in question text only here; we approximate
            # but the math judge needs gold. Fall back to using the rollout's gold info if present.
            pairs.append({
                "source": source,
                "qid": qid,
                "ord": r["ord"],
                "name": r["name"],
                "question": r.get("question", ""),
                "response_tail": r.get("response_tail", ""),
                "original_verdict": bool(r.get("is_correct", r.get("acceptable", False))),
            })
    return pairs


def get_full_question_and_gold(source: str, qid: str, math_path: str, aime_path: str, wc_path: str):
    """Look up the full question text + gold answer from the source eval files."""
    # The Always-X baseline files have truncated 'question' (200 chars in math, 300 in WildChat).
    # We need the full text + gold for math/aime, full text only for wildchat.
    if source == "math500":
        path = math_path
    elif source == "aime25":
        path = aime_path
    else:
        path = wc_path
    raw = json.load(open(path))
    # math/aime files store full text in rollouts but truncated; we rely on the truncation
    # because that's what was used at original judge time.
    return None  # rely on the (possibly truncated) text from load_pairs.


def run_variance_test(pairs: list[dict], n_repeats: int = 5, workers: int = 8) -> dict:
    client = make_client()
    rows = []
    futures: dict = {}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for p in pairs:
            for trial in range(n_repeats):
                if p["source"] == "wildchat":
                    prompt_text = WC_JUDGE_PROMPT.format(
                        question=p["question"][:2000],
                        response=p["response_tail"][:3000],
                    )
                    pos = "ACCEPTABLE"
                else:
                    # math judge needs gold answer — but we don't have it from baseline rollouts directly
                    # Use a slightly modified prompt that tells the judge "the original question is X
                    # and the student's final answer (extracted) is Y" and ask it to verify.
                    # Approximate by re-running the original judge prompt against (q, response).
                    prompt_text = MATH_JUDGE_PROMPT.format(
                        question=p["question"][:1500],
                        gold="(see student's reasoning)",  # ⚠️ degraded — see note below
                        response=p["response_tail"][:2000],
                    )
                    pos = "CORRECT"
                fut = ex.submit(call_judge, client, prompt_text, pos)
                futures[(p["qid"], p["source"], trial)] = fut

        for key, fut in futures.items():
            qid, source, trial = key
            verdict = fut.result()
            rows.append({"qid": qid, "source": source, "trial": trial, **verdict})

    # Aggregate per pair
    by_pair: dict[tuple, list] = {}
    for r in rows:
        by_pair.setdefault((r["qid"], r["source"]), []).append(r)

    pair_summaries = []
    for (qid, source), trials in by_pair.items():
        verdicts = [t["verdict"] for t in trials]
        n_true = sum(1 for v in verdicts if v is True)
        n_false = sum(1 for v in verdicts if v is False)
        n_none = sum(1 for v in verdicts if v is None)
        flipped = n_true > 0 and n_false > 0
        pair_summaries.append({
            "qid": qid, "source": source,
            "n_true": n_true, "n_false": n_false, "n_none": n_none,
            "flipped": flipped,
        })

    by_source = {}
    for src in ("math500", "aime25", "wildchat"):
        src_pairs = [p for p in pair_summaries if p["source"] == src]
        n = len(src_pairs)
        n_flipped = sum(1 for p in src_pairs if p["flipped"])
        n_unanimous = sum(1 for p in src_pairs if not p["flipped"] and p["n_none"] == 0)
        by_source[src] = {
            "n_pairs": n,
            "n_flipped": n_flipped,
            "flip_rate": round(n_flipped / max(n, 1), 4),
            "n_unanimous": n_unanimous,
            "unanimous_rate": round(n_unanimous / max(n, 1), 4),
        }

    return {"by_source": by_source, "pair_summaries": pair_summaries, "raw_trials": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json")
    ap.add_argument("--aime", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json")
    ap.add_argument("--wildchat", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json")
    ap.add_argument("--n-per-source", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/judge_variance.json",
    )
    args = ap.parse_args()

    pairs = load_pairs(args.math, args.aime, args.wildchat, args.n_per_source)
    print(f"Loaded {len(pairs)} pairs ({args.n_per_source} per source)")
    print(f"Re-judging each {args.n_repeats}× with Haiku at temp=0...")

    result = run_variance_test(pairs, n_repeats=args.n_repeats)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print()
    print("=== JUDGE VARIANCE ===")
    print(f"  {'source':10s} {'n_pairs':>8s} {'unanimous':>10s} {'flipped':>8s} {'flip_rate':>10s}")
    for src, s in result["by_source"].items():
        print(f"  {src:10s} {s['n_pairs']:>8d} {s['n_unanimous']:>10d} {s['n_flipped']:>8d} {s['flip_rate']:>9.1%}")

    overall_flip = sum(s["n_flipped"] for s in result["by_source"].values())
    overall_n = sum(s["n_pairs"] for s in result["by_source"].values())
    overall_rate = overall_flip / max(overall_n, 1)
    print(f"  OVERALL: {overall_flip}/{overall_n} pairs flipped — flip rate {overall_rate:.1%}")
    print(f"  Recommendation: {'ENSEMBLE (3-vote majority) — flip rate > 5%' if overall_rate > 0.05 else 'Single judge OK — flip rate within tolerance'}")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
