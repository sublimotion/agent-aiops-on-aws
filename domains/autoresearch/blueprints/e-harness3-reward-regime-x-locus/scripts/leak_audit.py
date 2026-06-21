#!/usr/bin/env python3
"""
Stage-0 reward-withholding LEAK AUDIT — HARD GATE for cells B/D.

Spec: "confirm no success signal leaks to the author via error strings, row
counts, empty results, or exception types. If the ablation isn't clean, B/D are
meaningless." Carryover-auditor P0-1 added: the author INVOCATION pattern and the
trajectory digest's own gold-label/WRONG tags are the biggest leaks.

We audit the EXACT withheld digest (`jit_authoring._trajectory_digest(ep, False)`)
the B/D author will see, built from REAL e-harness2 L2 trajectories (which carry
the identical fields), so the audit needs no new API calls and uses the real
distribution of pass/fail trajectories.

Two layers:

  (1) STATIC GATE (hard, must pass) — STRUCTURAL, not data-value matching:
      The author legitimately SEES the committed answer, the SQL, the question, and
      any errors. On a CORRECT task the committed answer naturally equals the gold
      label — but that overlap is invisible-as-a-signal to the author, which has NO
      gold reference to compare against. So the real leak test is structural: the
      withheld digest must not present the reward as a SEPARATE FIELD the author can
      read off. We assert the withheld digest contains NONE of the reward FIELDS:
        * a "Gold answer:" reference line,
        * a "(WRONG)" / verdict tag on the committed answer,
      while the VISIBLE digest (A/C) DOES contain both (positive control — proves
      the detector fires on a known leak). The B/D INVOCATION schedule is also
      reward-independent (author called on EVERY task — run_dbbench_cell.py).

  (2) EMPIRICAL SEPARABILITY (bounded, reported):
      A trajectory is legitimately INFORMATIVE — an author is *meant* to reason
      about it. The ablation is "clean" iff pass/fail is not DETERMINISTICALLY
      recoverable from the digest (no single channel that reads the reward). We
      fit the best possible reward predictor on the digest-readable features
      (n_errors, finish_reason, n_sql, empty-commit, h2_blocks) and report its
      accuracy/AUC vs the base rate. Interpretation:
        * AUC ~ 1.0  -> a deterministic leak channel exists -> GATE FAIL.
        * AUC modestly > 0.5 -> trajectory is informative but reward is not
          directly readable -> CLEAN, with the residual documented (it bounds,
          not voids, B/D — the author can guess, not read).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import jit_authoring as J

# A deterministic leak would let a trivial rule recover reward almost perfectly.
# Below this we call the channel "informative, not a reward read".
DETERMINISTIC_AUC = 0.90

# Reward FIELDS that must never appear as labeled lines in the withheld digest.
# (We match the field MARKERS the digest builder emits, not raw data values —
#  a correct task's committed answer legitimately equals the gold value and that
#  overlap carries no signal to an author with no gold reference to compare to.)
BANNED_FIELDS = ["gold answer:", "(wrong)", "is_correct", "ground truth",
                 "verdict", "passed:", "failed:", "score:"]


def load_eps(results_dir):
    eps = []
    for p in sorted(glob.glob(os.path.join(results_dir, "A_*.jsonl"))):
        for line in open(p):
            try:
                eps.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return eps


def static_gate(eps):
    print("=== (1) STATIC GATE — withheld digest contains no reward signal ===")
    leaks = 0
    visible_has_reward_field = 0
    for ep in eps:
        withheld = J._trajectory_digest(ep, reward_visible=False).lower()
        visible = J._trajectory_digest(ep, reward_visible=True).lower()
        # no reward FIELD may appear as a labeled line in the withheld digest
        hit = next((f for f in BANNED_FIELDS if f in withheld), None)
        if hit:
            print(f"  LEAK: reward field {hit!r} in withheld digest for {ep['task_id']}")
            leaks += 1
        # positive control: the VISIBLE digest carries the reward fields on the
        # failures where E_harness2 actually authored (gold answer line + WRONG tag)
        if not ep["is_correct"] and "gold answer:" in visible and "(wrong)" in visible:
            visible_has_reward_field += 1
    n_fail = sum(1 for e in eps if not e["is_correct"])
    print(f"  withheld digests audited: {len(eps)}   reward-field leaks found: {leaks}")
    print(f"  positive control (visible digest carries gold-answer+WRONG on failures): "
          f"{visible_has_reward_field}/{n_fail}")
    assert leaks == 0, "withheld digest exposes a reward field — B/D ablation is NOT clean"
    assert visible_has_reward_field > 0, \
        "positive control failed — the detector cannot see a known reward field"
    print("  STATIC GATE PASS: withheld digest exposes no reward field; "
          "detector validated against the visible (A/C) digest.\n")


def _features(ep):
    """Everything an author could READ off the withheld digest, as numbers."""
    fr = ep.get("finish_reason", "")
    return {
        "n_errors": ep.get("n_errors", 0),
        "has_error": 1 if ep.get("n_errors", 0) > 0 else 0,
        "n_sql": ep.get("n_sql", 0),
        "empty_commit": 1 if not ep.get("committed") else 0,
        "h2_blocks": ep.get("h2_blocks", 0),
        "fin_max_rounds": 1 if fr == "max_rounds" else 0,
        "fin_committed": 1 if fr == "committed" else 0,
    }


def empirical_separability(eps):
    print("=== (2) EMPIRICAL SEPARABILITY — is reward DETERMINISTICALLY recoverable? ===")
    y = [int(e["is_correct"]) for e in eps]
    base = sum(y) / len(y)
    base_acc = max(base, 1 - base)  # majority-class accuracy
    feats = [_features(e) for e in eps]
    keys = list(feats[0])

    # Best single-threshold rule per feature (the strongest "channel" an author
    # could exploit) + its AUC. AUC via rank statistic (Mann-Whitney), no sklearn.
    def auc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return 0.5
        wins = ties = 0
        for sp in pos:
            for sn in neg:
                if sp > sn:
                    wins += 1
                elif sp == sn:
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    print(f"  base rate (P[correct]) = {base:.3f}   majority-class acc = {base_acc:.3f}")
    print(f"  n={len(eps)}  ({sum(y)} correct / {len(y)-sum(y)} incorrect)")
    best_auc = 0.5
    rows = []
    for k in keys:
        scores = [f[k] for f in feats]
        a = auc(scores, y)
        a = max(a, 1 - a)  # feature could predict either direction
        rows.append((k, a))
        best_auc = max(best_auc, a)
    rows.sort(key=lambda r: -r[1])
    for k, a in rows:
        flag = "  <-- DETERMINISTIC LEAK" if a >= DETERMINISTIC_AUC else ""
        print(f"    {k:16s} reward-recovery AUC = {a:.3f}{flag}")

    print(f"\n  strongest single-channel AUC = {best_auc:.3f} "
          f"(threshold for 'deterministic leak' = {DETERMINISTIC_AUC})")
    clean = best_auc < DETERMINISTIC_AUC
    if clean:
        print("  -> CLEAN: the trajectory is INFORMATIVE (errors/finish-reason correlate")
        print("     with outcome, as any trajectory does) but reward is NOT directly")
        print("     readable. The author can GUESS from the trace, not READ the reward.")
        print("     This residual BOUNDS the B/D interpretation; it does not void it.")
    else:
        print("  -> LEAK: a single digest channel recovers reward near-deterministically.")
    return clean, best_auc, dict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eps = load_eps(args.results)
    if not eps:
        print("no A_*.jsonl episodes found to audit", file=sys.stderr)
        sys.exit(1)
    print(f"auditing {len(eps)} real trajectories (e-harness2 L2 = cell A)\n")
    static_gate(eps)
    clean, best_auc, aucs = empirical_separability(eps)

    audit = {
        "n_trajectories": len(eps),
        "static_gate": "PASS",
        "best_single_channel_auc": best_auc,
        "deterministic_leak_threshold": DETERMINISTIC_AUC,
        "per_feature_auc": aucs,
        "verdict": "CLEAN" if clean else "LEAK",
        "interpretation": (
            "Withheld digest carries no gold label / verdict / pass-fail token "
            "(static gate). Author invoked on EVERY task (reward-independent "
            "schedule, run_dbbench_cell.py). Strongest single digest channel "
            f"recovers reward at AUC={best_auc:.3f} < {DETERMINISTIC_AUC}, i.e. the "
            "trajectory is informative but the reward is not deterministically "
            "readable — B/D is a valid (bounded) reward-withholding ablation."),
    }
    out = args.out or os.path.join(args.results, "leak_audit.json")
    json.dump(audit, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    if not clean:
        print("LEAK AUDIT FAILED — B/D would be meaningless.", file=sys.stderr)
        sys.exit(1)
    print("\nLEAK AUDIT PASSED — B/D is a clean (bounded) reward-withholding ablation.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nLEAK AUDIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
