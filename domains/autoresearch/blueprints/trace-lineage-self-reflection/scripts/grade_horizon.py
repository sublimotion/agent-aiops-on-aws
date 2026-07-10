#!/usr/bin/env python3
"""Mechanical oracle for the horizon-ledger task. Zero LLM judgment.

Two independent metrics:
  execution_accuracy = accounts at correct final balance / K   (the horizon/p^H signal)
  summary_consistent = does summary.md hold the true total       (the coupling/drift signal)

The interesting drift case: execution_accuracy high but summary_consistent False
= agent tracked the accounts but forgot to keep the derived summary in sync
(exactly what lineage.py value/reference-drift should have flagged).

Usage: grade_horizon.py --task DIR [--json out.json]
"""
import argparse
import json
import os
import re


def read(path):
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def first_int(text):
    if text is None:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def grade(task_dir):
    with open(os.path.join(task_dir, "task.json")) as f:
        m = json.load(f)
    final = m["final_balances"]
    k = m["k"]

    per_account = []
    correct = 0
    for key, truth in final.items():
        got = first_int(read(os.path.join(task_dir, "accounts", f"{key}.txt")))
        ok = (got == truth)
        correct += ok
        per_account.append({"account": key, "expected": truth, "got": got, "ok": ok})

    # summary consistency: does summary.md hold the true total of ACTUAL file balances?
    summary_text = read(os.path.join(task_dir, "summary.md"))
    summary_val = first_int(summary_text)
    # actual total from the files the agent left behind (not the ground truth) —
    # summary is "consistent" if it matches the accounts as they currently stand.
    actual_total = 0
    have_all = True
    for key in final:
        v = first_int(read(os.path.join(task_dir, "accounts", f"{key}.txt")))
        if v is None:
            have_all = False
        else:
            actual_total += v
    summary_matches_files = (summary_val == actual_total) if have_all else False
    summary_matches_truth = (summary_val == m["final_total"])

    return {
        "task_type": "horizon_ledger", "seed": m["seed"], "k": k, "turns": m["turns"],
        "execution_accuracy": round(correct / k, 4) if k else 0.0,
        "accounts_correct": correct,
        "summary_value": summary_val,
        "summary_true_total": m["final_total"],
        "summary_actual_files_total": actual_total if have_all else None,
        "summary_consistent_with_files": summary_matches_files,  # coupling/drift metric
        "summary_matches_truth": summary_matches_truth,
        # headline drift: accounts (mostly) right but summary out of sync
        "drift_summary_stale": (correct >= 1 and not summary_matches_files),
        "per_account": per_account,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    v = grade(args.task)
    out = json.dumps(v, indent=2)
    if args.json:
        with open(args.json, "w") as f:
            f.write(out)
    print(out)


if __name__ == "__main__":
    main()
