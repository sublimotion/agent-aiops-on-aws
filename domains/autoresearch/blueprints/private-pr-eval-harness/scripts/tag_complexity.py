#!/usr/bin/env python3
"""Phase 1 — Gate 3: tag each task low/medium/high complexity + store features.

Auditable heuristic (spec: publish the rule). Features already captured by the
miner: net_lines, n_files, n_test_files, and #source_files (cross-module proxy).
A pure-heuristic tier is the sanctioned fallback to an LLM classifier — we use it
here so the distribution is fully inspectable and reproducible.

Tiers (tuned to the Databricks ~25/60/15 low/med/high prior; adjust after
inspecting the histogram):
  low    : net_lines <= 25  AND n_files <= 2
  high   : net_lines >= 150 OR  n_files >= 6
  medium : everything else

Usage:
  python3 tag_complexity.py --in results/candidates.jsonl --out results/candidates.jsonl
  (in-place augmentation; also prints the histogram vs the declared prior)
"""
import argparse, json, pathlib
from collections import Counter

PRIOR = {"low": 0.25, "medium": 0.60, "high": 0.15}


def tier(r):
    nl, nf = r["net_lines"], r["n_files"]
    if nl <= 25 and nf <= 2:
        return "low"
    if nl >= 150 or nf >= 6:
        return "high"
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/candidates.jsonl")
    ap.add_argument("--out", default="results/candidates.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    for r in rows:
        r["complexity_tier"] = tier(r)
        r["complexity_features"] = {
            "net_lines": r["net_lines"], "n_files": r["n_files"],
            "n_test_files": r["n_test_files"],
            "n_source_files": len(r.get("source_files", []))}

    pathlib.Path(args.out).write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    n = len(rows)
    c = Counter(r["complexity_tier"] for r in rows)
    print(f"[tag] tagged {n} tasks; rule published in this file's docstring")
    print(f"{'tier':<8}{'n':>5}{'share':>8}{'prior':>8}{'delta':>8}")
    for t in ("low", "medium", "high"):
        share = c[t] / n if n else 0
        print(f"{t:<8}{c[t]:>5}{share:>7.0%}{PRIOR[t]:>7.0%}"
              f"{share - PRIOR[t]:>+8.0%}")
    # spec Phase-4 sanity: flag if any tier is >10pp off the prior
    skew = [t for t in PRIOR if abs(c[t] / n - PRIOR[t]) > 0.10]
    if skew:
        print(f"[tag] WARNING tiers off-prior by >10pp: {skew} "
              f"— sample toward the prior or note the headline won't generalize")


if __name__ == "__main__":
    main()
