"""Assemble ~5,000-question category-labeled corpus for Phase 2 scale-up.

Targets:
  math      — MATH500 (500) + AIME25 (30) = 530
  code      — HumanEval (164) + MBPP (974 sampled to 470) = 634
  factual   — MMLU-Pro {business, law, health, economics, history} sampled to 1000
  reasoning — MMLU-Pro {physics, chemistry, engineering, computer science} sampled to 1000
  open-dom  — WildChat-1M filtered to 1000

Total: ~5,164.

Output: data/augmented_baseline_5000q.jsonl with rows {id, question, gold,
        category, source}. Existing 480q rows are reused (already in jsonl)
        and supplemented with new ones. The 480q file is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re


def short_id(category: str, text: str) -> str:
    return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def load_math500_full() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for ex in ds:
        out.append({
            "id": short_id("math", ex["problem"]),
            "question": ex["problem"],
            "gold": str(ex.get("answer", ex.get("solution", ""))),
            "category": "math",
            "source": "math500",
        })
    return out


def load_aime25() -> list[dict]:
    """Reuse the existing 30 from rl-conductor cherry-pick."""
    raw = json.load(open(
        "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json"
    ))
    seen = set()
    out = []
    for r in raw["rollouts"]:
        qid = short_id("math", r["question"])
        if qid in seen:
            continue
        seen.add(qid)
        out.append({
            "id": qid,
            "question": r["question"],
            "gold": str(r.get("gold", "")),
            "category": "math",
            "source": "aime25",
        })
    return out


def load_humaneval(n: int = 164) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        prompt = ex["prompt"].rstrip()
        out.append({
            "id": short_id("code", prompt),
            "question": "Complete the following Python function:\n\n" + prompt,
            "gold": ex["canonical_solution"].rstrip(),
            "category": "code",
            "source": "humaneval",
        })
    return out


def load_mbpp(n: int = 470, seed: int = 17) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for idx in indices:
        ex = ds[idx]
        prompt = ex.get("prompt") or ex.get("text", "")
        if not prompt:
            continue
        out.append({
            "id": short_id("code", prompt),
            "question": f"Write a Python solution:\n\n{prompt}",
            "gold": ex.get("code", ""),
            "category": "code",
            "source": "mbpp",
        })
    return out


def load_mmlu_pro(n: int, subjects: list[str], category: str, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    pool = [ex for ex in ds if ex["category"] in subjects]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    out = []
    for ex in sample:
        opts = ex["options"]
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        question_text = (
            f"{ex['question']}\n\n{opt_text}\n\n"
            f"Reply with the letter of the correct answer."
        )
        out.append({
            "id": short_id(category, question_text),
            "question": question_text,
            "gold": ex["answer"],
            "category": category,
            "source": f"mmlu-pro/{ex['category']}",
        })
    return out


LONG_OUT_PATS = [
    re.compile(r"\b(?:[2-9]\d\d?|1\d\d\d?)\b"),
    re.compile(r"list of \d+", re.I),
    re.compile(r"comprehensive", re.I),
    re.compile(r"\d+ ?page", re.I),
]


def load_wildchat(n: int = 1000, seed: int = 17) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(seed)
    for ex in ds:
        if len(out) >= n:
            break
        if ex.get("language") != "English" or ex.get("toxic") or ex.get("redacted"):
            continue
        if ex.get("turn") != 1:
            continue
        c = ex.get("conversation") or []
        if not c or c[0].get("role") != "user":
            continue
        text = c[0].get("content") or ""
        if not (50 <= len(text) <= 2000):
            continue
        if any(pat.search(text) for pat in LONG_OUT_PATS):
            continue
        h = short_id("open-domain", text)
        if h in seen:
            continue
        seen.add(h)
        out.append({
            "id": h,
            "question": text,
            "gold": "",
            "category": "open-domain",
            "source": "wildchat",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_5000q.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rows: list[dict] = []
    seen_ids: set[str] = set()

    def add(items, label):
        added = 0
        for r in items:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            rows.append(r)
            added += 1
        print(f"  +{added} {label}")

    print("Loading math (MATH500 + AIME25)...")
    add(load_math500_full(), "math500 (full 500)")
    add(load_aime25(), "aime25 (30)")

    print("Loading code (HumanEval + MBPP)...")
    add(load_humaneval(n=164), "humaneval (164)")
    try:
        add(load_mbpp(n=470, seed=args.seed), "mbpp (470)")
    except Exception as e:
        print(f"  !! mbpp failed: {type(e).__name__}: {e}")

    print("Loading factual (MMLU-Pro mixed)...")
    add(load_mmlu_pro(
        n=1000, subjects=["business", "law", "health", "economics", "history"],
        category="factual", seed=args.seed,
    ), "mmlu-pro factual")

    print("Loading reasoning (MMLU-Pro hard)...")
    add(load_mmlu_pro(
        n=1000, subjects=["physics", "chemistry", "engineering", "computer science"],
        category="reasoning", seed=args.seed + 1,
    ), "mmlu-pro reasoning")

    print("Loading open-domain (WildChat)...")
    add(load_wildchat(n=1000, seed=args.seed), "wildchat (1000)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    by_cat = collections.Counter(r["category"] for r in rows)
    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"By category: {dict(by_cat)}")


if __name__ == "__main__":
    main()
