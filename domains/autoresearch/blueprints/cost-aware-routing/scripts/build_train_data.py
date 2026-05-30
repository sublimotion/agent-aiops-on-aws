"""Assemble cost-aware-routing Phase 1 training data.

Produces a single JSONL where each row has:
  {id: str, question: str, gold: str | "", source: str}

Sources (per spec):
  MATH500 train      - 300 questions from rl-conductor's train.jsonl (math only)
  MMLU train         - 300 questions, scored by exact match on choice letter
  HumanEval train    - 300 questions, gold = canonical solution
  LiveCodeBench train - 300 questions
  AIME25 train split - 20 hard math questions (n=20 train / n=10 eval)
  WildChat train     - 300 real user prompts (no gold; LLM-judge scored)

For the smoke run, only MATH500 + AIME25 + WildChat are required (the
three datasets we have measured baselines for). MMLU, HumanEval, and
LCB are pulled in Phase 1 production but are NOT necessary for a smoke
run that validates the GRPO loop converges.

Run:
  python3 build_train_data.py --smoke   # 620 questions: MATH+AIME+WildChat
  python3 build_train_data.py           # 1,520 questions: full Phase 1 mix
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys


def load_math500_train(rl_conductor_train_path: str, n: int = 300, seed: int = 17) -> list[dict]:
    """rl-conductor train.jsonl has MATH500 + MMLU + LCB entries.
    For now, take the first n records (they were pre-shuffled by rl-conductor)."""
    rng = random.Random(seed)
    rows = []
    with open(rl_conductor_train_path) as f:
        for line in f:
            r = json.loads(line)
            # rl-conductor format: {question, answer, type}
            q_type = r.get("type", "math")
            if q_type != "math":
                continue
            rows.append({
                "id": f"math500_{len(rows):04d}",
                "question": r["question"],
                "gold": r["answer"],
                "source": "math500",
            })
    rng.shuffle(rows)
    return rows[:n]


def load_aime25_train(aime_baseline_path: str, n: int = 20, seed: int = 17) -> list[dict]:
    """Take the first n unique questions from rl-conductor's AIME25 cherry-pick."""
    raw = json.load(open(aime_baseline_path))
    seen: set = set()
    rows = []
    for r in raw["rollouts"]:
        qid = r.get("id") or r.get("question", "")[:60]
        if qid in seen:
            continue
        seen.add(qid)
        rows.append({
            "id": f"aime25_{len(rows):04d}",
            "question": r["question"],
            "gold": str(r.get("gold", "")),  # AIME baselines may not preserve gold
            "source": "aime25",
        })
        if len(rows) >= n:
            break
    return rows


def load_wildchat_train(wildchat_jsonl_path: str, n: int = 300) -> list[dict]:
    """Already-prepared WildChat train split (lmsys_train_300.jsonl)."""
    rows = []
    with open(wildchat_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "id": r["id"],
                "question": r["question"],
                "gold": "",  # no gold; LLM-judge scored
                "source": "wildchat",
            })
            if len(rows) >= n:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke-run mix: MATH500+AIME25+WildChat only (620 q).")
    ap.add_argument(
        "--rl-conductor-train",
        default="domains/autoresearch/blueprints/cost-aware-routing/vendor/rl-conductor-phase1/data/train.jsonl",
    )
    ap.add_argument(
        "--aime-baseline",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json",
    )
    ap.add_argument(
        "--wildchat",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/lmsys_train_300.jsonl",
    )
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/train.jsonl",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    rows += load_math500_train(args.rl_conductor_train, n=300)
    rows += load_aime25_train(args.aime_baseline, n=20)
    rows += load_wildchat_train(args.wildchat, n=300)

    if not args.smoke:
        # Phase 1 production mix would also pull MMLU, HumanEval, LCB here.
        # For now, skip — the smoke mix is sufficient to validate GRPO convergence
        # and the three datasets where we have baselines for the success criteria.
        print("(Note: --smoke and full mix are currently identical; MMLU/HumanEval/LCB "
              "pending; spec lists them at 1,520 total.)", file=sys.stderr)

    random.Random(17).shuffle(rows)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"By source: {by_source}")


if __name__ == "__main__":
    main()
