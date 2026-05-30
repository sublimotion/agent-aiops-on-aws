"""
Open-domain quality judge for WildChat-style prompts.

WildChat prompts have no gold answers, but the cost-aware reward function
needs a binary `is_correct` signal that works alongside MATH/AIME judging.
We use Haiku (and Sonnet for validation) with a quality rubric:

  ACCEPTABLE — would satisfy a typical user
  UNACCEPTABLE — refuses, hallucinates, ignores constraints, or is
                 noticeably lower quality than what a competent assistant
                 would produce

Single-bit, with the floor calibrated so a strong response from a
mid-tier worker (Mistral, DeepSeek) clears it. The bar is *not* "best
possible answer" — that would collapse all reward to Opus and defeat
the cost-aware design.

Validation: Haiku vs Sonnet agreement on n=20 sampled WildChat (q, response)
pairs. Target ≥90% agreement (lower bar than math's 98% — open-domain is
inherently more subjective).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import time

import boto3
from botocore.config import Config

from worker_pool import POOL, invoke_worker

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-4-6"

JUDGE_PROMPT = """You are evaluating an AI assistant's response to a user prompt.

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
  - violates a clear safety norm (the assistant should refuse but didn't)

Do NOT mark UNACCEPTABLE for: cut-off mid-sentence, incomplete final list
item, missing concluding paragraph. These are token-cap artifacts, not
quality problems.

Reply with EXACTLY one token on the first line: ACCEPTABLE or UNACCEPTABLE.
Then optionally one short sentence of justification."""


def make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def call_judge(client, model_id: str, question: str, response: str) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": JUDGE_PROMPT.format(
            question=question[:2000],
            response=response[:3000],
        )}]}],
    }
    t0 = time.time()
    try:
        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        dur = time.time() - t0
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"].strip()
        first = text.splitlines()[0].strip().upper() if text else ""
        verdict = (
            True if first.startswith("ACCEPTABLE")
            else False if first.startswith("UNACCEPTABLE")
            else None
        )
        usage = payload.get("usage", {})
        return {
            "verdict": verdict,
            "raw": text[:300],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_s": round(dur, 2),
        }
    except Exception as e:
        return {
            "verdict": None,
            "raw": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def calibration(eval_jsonl: str, n: int = 20, worker_ord: int = 4) -> dict:
    """Generate responses from one mid-tier worker, judge with both Haiku & Sonnet."""
    client = make_client()
    rows = []
    questions: list[dict] = []
    with open(eval_jsonl) as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= n:
                break

    print(f"Generating responses from ord_{worker_ord} ({POOL[worker_ord].name})...")
    response_rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        # max_tokens=2048 because WildChat prompts often ask for long-form
        # output (60-item lists, full essays, complete stories). 1024 caused
        # ~half of responses to truncate and get UNACCEPTABLE judgments.
        futures = {
            ex.submit(invoke_worker, client, worker_ord, q["question"], 2048, 0.7): q
            for q in questions
        }
        for fut in cf.as_completed(futures):
            q = futures[fut]
            r = fut.result()
            response_rows.append({"q": q, "response": r})

    print(f"Judging with Haiku and Sonnet ({n} responses × 2 judges)...")
    out = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        all_futures = {}
        for row in response_rows:
            q = row["q"]
            text = row["response"]["text"]
            f_h = ex.submit(call_judge, client, HAIKU, q["question"], text)
            f_s = ex.submit(call_judge, client, SONNET, q["question"], text)
            all_futures[(q["id"], "h")] = f_h
            all_futures[(q["id"], "s")] = f_s

        results: dict[tuple, dict] = {}
        for key, fut in all_futures.items():
            results[key] = fut.result()

    rows = []
    for row in response_rows:
        q = row["q"]
        h = results[(q["id"], "h")]
        s = results[(q["id"], "s")]
        rows.append({
            "id": q["id"],
            "question": q["question"][:200],
            "response_tail": row["response"]["text"][-300:],
            "haiku": h,
            "sonnet": s,
        })

    haiku_v = [r["haiku"]["verdict"] for r in rows]
    sonnet_v = [r["sonnet"]["verdict"] for r in rows]
    n_both = sum(1 for h, s in zip(haiku_v, sonnet_v) if h is not None and s is not None)
    n_agree = sum(1 for h, s in zip(haiku_v, sonnet_v) if h is not None and s is not None and h == s)
    haiku_acc = sum(1 for v in haiku_v if v) / max(len(rows), 1)
    sonnet_acc = sum(1 for v in sonnet_v if v) / max(len(rows), 1)

    summary = {
        "n": len(rows),
        "judged_by_both": n_both,
        "agreement": n_agree / max(n_both, 1),
        "haiku_acceptable_rate": haiku_acc,
        "sonnet_acceptable_rate": sonnet_acc,
        "responder_worker_ord": worker_ord,
        "responder_worker_name": POOL[worker_ord].name,
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-jsonl",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/lmsys_eval_100.jsonl",
    )
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--responder-ord", type=int, default=4,
                    help="Which worker generates the responses for calibration "
                         "(default 4 = Mistral Large 3, mid-tier).")
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/wildchat_judge_calibration.json",
    )
    args = ap.parse_args()

    result = calibration(args.eval_jsonl, n=args.n, worker_ord=args.responder_ord)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    print()
    print("=== WILDCHAT JUDGE CALIBRATION ===")
    print(f"  n={s['n']} responses from ord_{s['responder_worker_ord']} ({s['responder_worker_name']})")
    print(f"  Haiku acceptable rate:  {s['haiku_acceptable_rate']:.1%}")
    print(f"  Sonnet acceptable rate: {s['sonnet_acceptable_rate']:.1%}")
    print(f"  Haiku vs Sonnet agreement: {s['agreement']:.1%} ({int(s['agreement']*s['judged_by_both'])}/{s['judged_by_both']})")
    # Open-domain target is 85% (math target was 98%; subjective tasks
    # have legitimate borderline cases). See plan addendum §11.
    print(f"  Pass threshold: ≥85% (open-domain) -> {'PASS' if s['agreement'] >= 0.85 else 'FAIL — re-tune rubric'}")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
