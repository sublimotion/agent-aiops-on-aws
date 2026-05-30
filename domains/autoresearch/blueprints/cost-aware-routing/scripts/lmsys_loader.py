"""
Real-world user-prompt loader for cost-aware-routing Phase 1 training.

LMSYS Chatbot Arena Conversations (lmsys/chatbot_arena_conversations) is the
ideal source — real user prompts paired with model-vs-model preferences —
but it is gated on Hugging Face. WildChat-1M (allenai/WildChat-1M) is the
closest public substitute: 1M real user conversations to ChatGPT/GPT-4
collected via a public proxy. We use it to inject "production-distribution"
queries into Phase 1 training and OOD eval.

Filters:
  - language == "English"           (avoid translation tasks dominating)
  - toxic == False                  (skip moderation-flagged content)
  - redacted == False               (skip PII-redacted prompts; they read weird)
  - 50 <= len(first_user_msg) <= 2000 (avoid one-liners and walls of text)
  - turn == 1                       (single-turn; routing is single-pick)

Sample 400 deterministically with seed=17, split 300 train / 100 eval.

Output JSONL records:
  {"id": "wildchat_<hash>", "question": "...", "source": "wildchat", "split": "train" | "eval"}

No gold answers — these prompts are scored by LLM-as-judge in Phase 1
(Haiku for reward, Sonnet for held-out validation). The judge prompt for
open-ended questions evaluates "is this a high-quality, helpful response"
on a 0/1 scale, similar to the math judge but with a quality rubric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets", file=sys.stderr)
    raise


import re

# Heuristic markers that the user is asking for a long-form response that
# would exceed our 1024-2048 max_tokens budget and cause truncation. We
# drop these; they would inflate "UNACCEPTABLE" rates from truncation
# rather than worker quality. ~26% of otherwise-good prompts match.
LONG_OUTPUT_SIGNALS = [
    re.compile(r"\b(?:[2-9]\d\d?|1\d\d\d?)\b"),  # 200, 1500, etc. (numbers ≥ 200 often = list size or word count)
    re.compile(r"list of \d+", re.I),
    re.compile(r"comprehensive", re.I),
    re.compile(r"detailed analysis", re.I),
    re.compile(r"\d+ ?page", re.I),
    re.compile(r"complete the.*?\b(?:novel|book|essay|story|guide|report)\b", re.I),
    re.compile(r"\b(?:full(?:-length)?|entire|complete)\s+(?:novel|book|essay|story|guide|report|analysis|report)\b", re.I),
]


def passes_filter(ex: dict, min_len: int = 50, max_len: int = 2000) -> tuple[bool, str | None]:
    """Returns (keep, reason_for_skip)."""
    if ex.get("language") != "English":
        return False, "non-english"
    if ex.get("toxic"):
        return False, "toxic"
    if ex.get("redacted"):
        return False, "redacted"
    if ex.get("turn") != 1:
        return False, "multi-turn"
    convo = ex.get("conversation") or []
    if not convo:
        return False, "no-conversation"
    first = convo[0]
    if first.get("role") != "user":
        return False, "non-user-first"
    text = first.get("content") or ""
    if len(text) < min_len:
        return False, "too-short"
    if len(text) > max_len:
        return False, "too-long"
    for pat in LONG_OUTPUT_SIGNALS:
        if pat.search(text):
            return False, "long-output-expected"
    return True, None


def short_id(text: str) -> str:
    return "wildchat_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def load_questions(
    n: int = 400,
    seed: int = 17,
    min_len: int = 50,
    max_len: int = 2000,
    scan_limit: int = 50_000,
) -> list[dict]:
    """Stream WildChat-1M, filter, and reservoir-sample n English single-turn prompts.

    scan_limit caps how many records we scan; 50K should comfortably yield
    400 keepers given the filter rates we observe (~40% pass).
    """
    rng = random.Random(seed)
    print(f"Loading allenai/WildChat-1M (streaming)...", flush=True)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    kept: list[dict] = []
    skipped: dict[str, int] = {}
    seen_ids: set[str] = set()

    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > scan_limit:
            break
        ok, reason = passes_filter(ex, min_len=min_len, max_len=max_len)
        if not ok:
            skipped[reason or "unknown"] = skipped.get(reason or "unknown", 0) + 1
            continue
        text = ex["conversation"][0]["content"]
        qid = short_id(text)
        if qid in seen_ids:
            skipped["duplicate"] = skipped.get("duplicate", 0) + 1
            continue
        seen_ids.add(qid)
        kept.append({"id": qid, "question": text, "source": "wildchat"})
        if scanned % 5000 == 0:
            print(f"  scanned {scanned} kept {len(kept)}", flush=True)

    print(f"Done scanning ({scanned} examples).")
    print(f"Kept {len(kept)}, skipped: {skipped}")

    if len(kept) < n:
        raise RuntimeError(
            f"Only kept {len(kept)} after filter; need {n}. "
            f"Increase --scan-limit or relax filters."
        )

    rng.shuffle(kept)
    sample = kept[:n]
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scan-limit", type=int, default=50_000)
    ap.add_argument("--train-frac", type=float, default=0.75)  # 300/400
    ap.add_argument(
        "--out-train",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/lmsys_train_300.jsonl",
    )
    ap.add_argument(
        "--out-eval",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/lmsys_eval_100.jsonl",
    )
    args = ap.parse_args()

    sample = load_questions(n=args.n, seed=args.seed, scan_limit=args.scan_limit)
    n_train = int(args.n * args.train_frac)
    train, evalset = sample[:n_train], sample[n_train:]
    for r in train:
        r["split"] = "train"
    for r in evalset:
        r["split"] = "eval"

    out_train = pathlib.Path(args.out_train)
    out_eval = pathlib.Path(args.out_eval)
    for path, rows in [(out_train, train), (out_eval, evalset)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} rows to {path}")

    # Brief stats
    avg_len = sum(len(r["question"]) for r in sample) / len(sample)
    print(f"\nAvg question length: {avg_len:.0f} chars")
    print(f"Sample first 3 train questions:")
    for r in train[:3]:
        print(f"  [{r['id']}] {r['question'][:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
