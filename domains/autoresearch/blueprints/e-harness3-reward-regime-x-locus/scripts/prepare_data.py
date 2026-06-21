#!/usr/bin/env python3
"""
Build a verified DBBench eval set for E_harness2.

Keep only SELECT-family tasks whose GOLD sql, run in our SQLite env, reproduces
the gold `label` (the eval-harness-trust gate). Stratify-sample n per the seed.
This is the deterministic oracle the spec calls for, minus the MySQL-only
mutation path the official harness leaves unimplemented for SQLite.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import dbbench_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(C.ROOT, "data", "db_out_new.jsonl"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "data", "dbbench_eval.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src)]
    sel = [r for r in rows if C.is_select_family(r)]
    print(f"total={len(rows)} select_family={len(sel)}")

    # Eval-harness-trust gate: gold SQL must reproduce gold label in SQLite.
    verified = [r for r in sel if C.gold_label_matches(r)]
    print(f"verified self-contained oracle: {len(verified)}/{len(sel)} "
          f"({len(verified)/len(sel):.1%})")

    by_type = defaultdict(list)
    for r in verified:
        by_type[r["type"][0]].append(r)
    print("verified by type:", {k: len(v) for k, v in sorted(by_type.items())})

    # Stratified sample: proportional to availability, deterministic.
    rng = random.Random(args.seed)
    for v in by_type.values():
        rng.shuffle(v)
    # round-robin across types until we reach n (keeps diversity)
    order = sorted(by_type, key=lambda k: -len(by_type[k]))
    picked, idx = [], defaultdict(int)
    while len(picked) < args.n and any(idx[t] < len(by_type[t]) for t in order):
        for t in order:
            if idx[t] < len(by_type[t]):
                picked.append(by_type[t][idx[t]])
                idx[t] += 1
                if len(picked) >= args.n:
                    break

    # attach a stable task_id
    for i, r in enumerate(picked):
        r["task_id"] = f"{r['type'][0]}_{r.get('index', i)}_{i}"

    print(f"\nsampled n={len(picked)} type dist:",
          dict(Counter(r["type"][0] for r in picked)))
    json.dump(picked, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
