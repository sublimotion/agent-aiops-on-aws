"""Assemble 480-question category-labeled baseline set for the redesign.

Pulls from:
  math (existing): MATH500 (50) + AIME25 (30) = 80
  code (new):      HumanEval (sample 100)
  factual (new):   MMLU-Pro {business, law, health, economics, history} (sample 100)
  reasoning (new): MMLU-Pro {physics, chemistry, engineering, computer science} (sample 100)
  open-domain (new): WildChat-1M filtered (sample 50 more, beyond existing 50)

Output: data/augmented_500q.jsonl with rows
  {id, question, gold, category, source}

`gold` is empty for open-domain (judged via wildchat_judge); for code it
contains the canonical solution (judged via test execution OR a simpler
"is this code that would compile" judge if test exec is too heavy);
for math/factual/reasoning it's the reference answer (judged via math/
multiple-choice judge).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random


def _short_id(category: str, text: str) -> str:
    return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def load_existing_math() -> list[dict]:
    """80 questions: 50 MATH500 + 30 AIME25 already in baseline files."""
    out = []
    seen = set()
    for path in [
        "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json",
        "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json",
    ]:
        d = json.load(open(path))
        for r in d["rollouts"]:
            qid = r.get("id") or r["question"][:60]
            if qid in seen:
                continue
            seen.add(qid)
            out.append({
                "id": _short_id("math", r["question"]),
                "question": r["question"],
                "gold": str(r.get("gold", "")),
                "category": "math",
                "source": "math500" if "math500" in path else "aime25",
            })
    return out


def load_existing_wildchat() -> list[dict]:
    """50 questions from existing wildchat baseline."""
    seen = set()
    rows = []
    d = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json"))
    for r in d["rollouts"]:
        qid = r.get("id") or r["question"][:60]
        if qid in seen:
            continue
        seen.add(qid)
        rows.append({
            "id": _short_id("open-domain", r["question"]),
            "question": r["question"],
            "gold": "",
            "category": "open-domain",
            "source": "wildchat",
        })
    return rows


def load_humaneval(n: int = 100, seed: int = 17) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for idx in indices:
        ex = ds[idx]
        prompt = ex["prompt"].rstrip()
        # Canonical answer for our judge: the canonical_solution as a code completion.
        gold = ex["canonical_solution"].rstrip()
        out.append({
            "id": _short_id("code", prompt),
            "question": "Complete the following Python function:\n\n" + prompt,
            "gold": gold,
            "category": "code",
            "source": "humaneval",
        })
    return out


def load_mmlu_pro(n: int, subjects: list[str], category: str, seed: int = 17) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    pool = [ex for ex in ds if ex["category"] in subjects]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    out = []
    for ex in sample:
        opts = ex["options"]
        # Format as a multiple-choice question; gold is the letter answer.
        opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        question_text = (
            f"{ex['question']}\n\n{opt_text}\n\n"
            f"Reply with the letter of the correct answer."
        )
        gold = ex["answer"]  # already a letter (A/B/C/...)
        out.append({
            "id": _short_id(category, question_text),
            "question": question_text,
            "gold": gold,
            "category": category,
            "source": f"mmlu-pro/{ex['category']}",
        })
    return out


def load_wildchat_extra(n: int = 50, seed: int = 17) -> list[dict]:
    """Pull n MORE wildchat questions beyond the existing 50. Excludes the
    50 we've already baselined."""
    from datasets import load_dataset
    import re
    LONG_OUT_PATS = [
        re.compile(r"\b(?:[2-9]\d\d?|1\d\d\d?)\b"),
        re.compile(r"list of \d+", re.I),
        re.compile(r"comprehensive", re.I),
        re.compile(r"\d+ ?page", re.I),
    ]
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    existing_ids: set[str] = set()
    # Load existing wildchat IDs to avoid re-using.
    d = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json"))
    for r in d["rollouts"]:
        existing_ids.add(r.get("id") or "")
    rng = random.Random(seed + 1)
    out: list[dict] = []
    seen: set[str] = set()
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
        h = "wildchat_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        if h in existing_ids or h in seen:
            continue
        seen.add(h)
        out.append({
            "id": h,
            "question": text,
            "gold": "",
            "category": "open-domain",
            "source": "wildchat-extra",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    print("Loading existing math (MATH500 + AIME25)...")
    rows = load_existing_math()
    print(f"  +{len(rows)} math")

    print("Loading existing wildchat...")
    wc = load_existing_wildchat()
    rows.extend(wc)
    print(f"  +{len(wc)} open-domain (existing)")

    print("Loading HumanEval (100)...")
    he = load_humaneval(n=100, seed=args.seed)
    rows.extend(he)
    print(f"  +{len(he)} code")

    print("Loading MMLU-Pro factual (100)...")
    fact = load_mmlu_pro(
        n=100,
        subjects=["business", "law", "health", "economics", "history"],
        category="factual",
        seed=args.seed,
    )
    rows.extend(fact)
    print(f"  +{len(fact)} factual")

    print("Loading MMLU-Pro reasoning (100)...")
    reasoning = load_mmlu_pro(
        n=100,
        subjects=["physics", "chemistry", "engineering", "computer science"],
        category="reasoning",
        seed=args.seed + 1,
    )
    rows.extend(reasoning)
    print(f"  +{len(reasoning)} reasoning")

    print("Loading WildChat extra (50)...")
    wc_extra = load_wildchat_extra(n=50, seed=args.seed)
    rows.extend(wc_extra)
    print(f"  +{len(wc_extra)} open-domain (new)")

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
