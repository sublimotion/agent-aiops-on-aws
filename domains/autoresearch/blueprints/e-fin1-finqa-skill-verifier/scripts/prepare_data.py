#!/usr/bin/env python3
"""Sample N FinQA dev examples deterministically into the blueprint data dir.

Fixed files: this defines the eval set. Do not edit after the run starts."""
import argparse
import json
import os

DEFAULT_SRC = "/tmp/FinQA/dataset/dev.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.src))
    # Deterministic sample: stride-based pick across the file (no RNG import
    # needed, fully reproducible regardless of Python version).
    import random
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(data)), min(args.n, len(data))))
    sample = [data[i] for i in idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sample, f)
    # Schema validation (P0 carryover: never assume the schema).
    missing = 0
    for ex in sample:
        qa = ex.get("qa", {})
        if "exe_ans" not in qa or "question" not in qa:
            missing += 1
    print(f"wrote {len(sample)} examples to {args.out} (seed={args.seed})")
    print(f"schema check: {missing} examples missing exe_ans/question")
    # Categorical vs numeric breakdown
    cat = sum(1 for ex in sample
              if isinstance(ex["qa"].get("exe_ans"), str))
    print(f"categorical (yes/no) exe_ans: {cat}, numeric: {len(sample) - cat}")


if __name__ == "__main__":
    main()
