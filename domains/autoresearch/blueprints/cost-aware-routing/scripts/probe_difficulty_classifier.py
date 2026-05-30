"""Probe Haiku 4.5 as a difficulty classifier.

For path A (training-time difficulty hint) to work, we need a classifier
that labels questions as easy/hard at >90% accuracy. The simulator
showed that 25% noise crashes per-difficulty performance to shared-policy
levels.

This probe sends each unique question from the 130-question baseline pool
to Haiku and measures agreement with the source label:
  math500 + wildchat -> "easy" (cheap workers do well)
  aime25 -> "hard" (only Opus does well)

If Haiku achieves >90%, path A is viable. If <80%, we need to rethink.

Cost: 130 questions × ~$0.0015 each = ~$0.20.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import time

import boto3
from botocore.config import Config

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CLASSIFIER_PROMPT = """You are a difficulty classifier for an LLM router.

Question:
{question}

Classify this question as one of:
  EASY  — most competent LLMs (including small/cheap ones) can answer correctly.
  HARD  — requires strong reasoning; small/cheap models often fail.

Reply with EXACTLY one token on the first line: EASY or HARD
Optionally add one short sentence of justification."""


def call_classifier(client, question: str) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text",
            "text": CLASSIFIER_PROMPT.format(question=question[:2000])}]}],
    }
    t0 = time.time()
    try:
        resp = client.invoke_model(modelId=HAIKU, body=json.dumps(body))
        dur = time.time() - t0
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"].strip()
        first = text.splitlines()[0].strip().upper() if text else ""
        if first.startswith("EASY"):
            verdict = "easy"
        elif first.startswith("HARD"):
            verdict = "hard"
        else:
            verdict = None
        usage = payload.get("usage", {})
        return {
            "verdict": verdict, "raw": text[:200],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_s": round(dur, 2),
        }
    except Exception as e:
        return {"verdict": None, "raw": "", "input_tokens": 0, "output_tokens": 0,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def load_unique_questions() -> list[dict]:
    """One per unique question across the three baselines."""
    seen: dict[str, dict] = {}
    sources = [
        ("math500", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json"),
        ("aime25", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json"),
        ("wildchat", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json"),
    ]
    for label, path in sources:
        for r in json.load(open(path))["rollouts"]:
            qid = r.get("id") or r.get("question", "")[:60]
            key = (label, qid)
            if key in seen:
                continue
            seen[key] = {"source": label, "qid": qid, "question": r["question"],
                         "true_difficulty": "hard" if label == "aime25" else "easy"}
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions for cheaper testing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/difficulty_classifier_probe.json",
    )
    args = ap.parse_args()

    questions = load_unique_questions()
    if args.limit:
        questions = questions[:args.limit]
    print(f"Probing {len(questions)} questions...")

    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60)
    client = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)

    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call_classifier, client, q["question"]): q for q in questions}
        for done, fut in enumerate(cf.as_completed(futures), 1):
            q = futures[fut]
            res = fut.result()
            rows.append({**q, **res})
            if done % 20 == 0:
                print(f"  {done}/{len(questions)} done")

    # Aggregate
    by_source: dict[str, dict] = {}
    for r in rows:
        s = by_source.setdefault(r["source"], {"n": 0, "correct": 0, "errors": 0,
                                               "easy_predicted": 0, "hard_predicted": 0})
        s["n"] += 1
        if r["verdict"] is None:
            s["errors"] += 1
            continue
        s[f"{r['verdict']}_predicted"] += 1
        if r["verdict"] == r["true_difficulty"]:
            s["correct"] += 1

    summary = {}
    for src, s in by_source.items():
        summary[src] = {
            **s,
            "accuracy": round(s["correct"] / max(s["n"] - s["errors"], 1), 4),
        }

    overall_n = sum(s["n"] for s in by_source.values())
    overall_errors = sum(s["errors"] for s in by_source.values())
    overall_correct = sum(s["correct"] for s in by_source.values())
    overall_acc = overall_correct / max(overall_n - overall_errors, 1)
    summary["__overall__"] = {
        "n": overall_n, "correct": overall_correct, "errors": overall_errors,
        "accuracy": round(overall_acc, 4),
    }

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print()
    print("=== HAIKU DIFFICULTY CLASSIFIER PROBE ===")
    for src, s in summary.items():
        if src == "__overall__":
            continue
        print(f"  {src:10s} n={s['n']:>3d} acc={s['accuracy']:.1%} "
              f"(easy_pred={s.get('easy_predicted', 0):>3d}, hard_pred={s.get('hard_predicted', 0):>3d}, "
              f"errors={s['errors']:>2d})")
    o = summary["__overall__"]
    print(f"\n  OVERALL: {o['correct']}/{o['n']} = {o['accuracy']:.1%}")
    target = 0.90
    print(f"  Target: ≥{target:.0%} -> {'PASS' if o['accuracy'] >= target else 'FAIL'}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
