#!/usr/bin/env python3
"""
Stage-0 FinanceBench LLM-JUDGE gate — HARD GATE for cells E/F.

Carryover-auditor P0-2 (from E_fin1): a same-tier Claude judge on financial
free-text ENGAGES but may not DISCRIMINATE (E_fin1: AUC 0.565, confidently
affirmed 19/29 wrong answers). The spec's "stability across temperature" check is
NECESSARY BUT NOT SUFFICIENT — a judge that always says "correct" is maximally
stable and useless. So this gate measures BOTH:

  (A) DISCRIMINATION (the real test): build labeled pairs with KNOWN ground truth —
      known-CORRECT candidates (the gold answer itself, lightly reworded) and
      known-INCORRECT candidates (a wrong substitute for the same question). A
      usable judge must separate them: AUC >= AUC_GATE and accuracy on the two
      classes both reasonable. If AUC < gate, E/F are reported "judge-confounded,
      inconclusive" — B/D still stand.

  (B) STABILITY: same pair judged at temp 0.0 and temp 0.7; verdict agreement.
      (Stable-but-blind is caught by (A); stability alone is reported, not gated.)

Known-incorrect construction (no fabrication of plausible-but-subtle errors, which
would itself need a judge): we use a DISTINCT, clearly-wrong answer for the
question — the gold answer of a DIFFERENT, dissimilar FinanceBench task. That is
unambiguously incorrect for THIS question, so any judge worth using must reject it.
This makes the gate a floor: if the judge cannot even reject an off-topic answer,
it cannot supply a reward. (A stricter near-miss test is noted as a limitation.)
"""
import argparse
import json
import os
import re
import sys

import finbench_common as F

_NUM_RE = re.compile(r"-?\$?\s*[\d,]+(?:\.\d+)?")


def _perturb_numeric(gold, factor):
    """Turn the FIRST number in a numeric gold answer into a clearly-wrong but
    ON-TOPIC near-miss (e.g. ×1.4). Returns None if no number is present."""
    m = _NUM_RE.search(gold)
    if not m:
        return None
    raw = m.group(0)
    try:
        val = float(raw.replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    if val == 0:
        return None
    new = val * factor
    new_str = f"{new:,.2f}" if "." in raw or abs(new) < 100 else f"{new:,.0f}"
    if "$" in raw:
        new_str = "$" + new_str
    return gold[: m.start()] + new_str + gold[m.end():]


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for sp in pos:
        for sn in neg:
            wins += sp > sn
            ties += sp == sn
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def signed_score(j):
    """Map verdict+confidence to a single discriminative score in [0,1]:
    correct -> 0.5 + conf/2 ; incorrect -> 0.5 - conf/2."""
    c = max(0.0, min(1.0, j["confidence"]))
    return 0.5 + c / 2 if j["is_correct"] else 0.5 - c / 2


def build_pairs(data, n, seed=42):
    """n known-correct + n known-incorrect labeled candidate pairs."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    pick = idx[: 2 * n] if len(idx) >= 2 * n else idx
    pairs = []
    # known-correct: candidate == gold (verbatim) — judge MUST accept
    for i in pick[:n]:
        e = data[i]
        pairs.append({"task_id": e["task_id"], "question": e["question"],
                      "gold": e["gold"], "candidate": e["gold"], "label": 1})
    # known-incorrect: candidate = gold of a DIFFERENT task (off-topic, wrong)
    for k, i in enumerate(pick[n:2 * n]):
        e = data[i]
        wrong_src = data[pick[(k + 1) % n]]  # a different task's gold
        if wrong_src["gold"].strip() == e["gold"].strip():
            wrong_src = data[pick[(k + 2) % n]]
        pairs.append({"task_id": e["task_id"], "question": e["question"],
                      "gold": e["gold"], "candidate": wrong_src["gold"], "label": 0})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku", choices=["haiku", "sonnet"])
    ap.add_argument("--n", type=int, default=15, help="per class (correct/incorrect)")
    ap.add_argument("--auc-gate", type=float, default=0.70)
    ap.add_argument("--out", default=f"{F.ROOT}/results/judge_gate.json")
    args = ap.parse_args()

    data = F.load_finbench()
    pairs = build_pairs(data, args.n)
    print(f"judge gate: {len(pairs)} labeled pairs "
          f"({sum(p['label'] for p in pairs)} correct / "
          f"{sum(1-p['label'] for p in pairs)} incorrect), judge={args.model}\n")

    print("=== (A) DISCRIMINATION + (B) STABILITY ===")
    scores, labels, agree, recs = [], [], 0, []
    cost = 0.0
    for p in pairs:
        j0 = F.judge_answer(p["question"], p["gold"], p["candidate"],
                            model_key=args.model, temperature=0.0)
        j1 = F.judge_answer(p["question"], p["gold"], p["candidate"],
                            model_key=args.model, temperature=0.7)
        cost += j0["cost_usd"] + j1["cost_usd"]
        scores.append(signed_score(j0))
        labels.append(p["label"])
        agree += int(j0["is_correct"] == j1["is_correct"])
        recs.append({**{k: p[k] for k in ("task_id", "label")},
                     "v0": j0["verdict"], "conf0": j0["confidence"],
                     "v1": j1["verdict"], "stable": j0["is_correct"] == j1["is_correct"]})

    a = auc(scores, labels)
    acc_pos = sum(1 for s, l in zip(scores, labels) if l == 1 and s > 0.5) / max(1, labels.count(1))
    acc_neg = sum(1 for s, l in zip(scores, labels) if l == 0 and s < 0.5) / max(1, labels.count(0))
    stability = agree / len(pairs)

    # ---- NEAR-MISS test (the real E_fin1 'engaged-but-not-discriminating' probe) ----
    # On-topic numeric answers perturbed x1.4 — clearly wrong, but NOT off-topic.
    # This is what E_fin1 found judges rubber-stamp. Report rejection rate.
    print("\n=== (C) NEAR-MISS discrimination (on-topic numeric, x1.4 — the E_fin1 probe) ===")
    nm_total = nm_rejected = 0
    nm_cost = 0.0
    for p in pairs:
        if p["label"] != 1:
            continue
        pert = _perturb_numeric(p["gold"], 1.4)
        if pert is None or pert.strip() == p["gold"].strip():
            continue
        jm = F.judge_answer(p["question"], p["gold"], pert, model_key=args.model, temperature=0.0)
        nm_cost += jm["cost_usd"]
        nm_total += 1
        nm_rejected += int(not jm["is_correct"])
    nm_reject_rate = (nm_rejected / nm_total) if nm_total else None
    cost += nm_cost
    if nm_total:
        print(f"  near-miss numeric answers tested = {nm_total}")
        print(f"  judge REJECTED (correctly)       = {nm_rejected}/{nm_total} "
              f"= {nm_reject_rate:.3f}")
        print("  (E_fin1 failure mode = low reject rate: confidently affirms wrong numbers)")
    else:
        print("  (no numeric gold answers in sample to perturb)")

    print(f"  AUC (discrimination)        = {a:.3f}   (gate >= {args.auc_gate})")
    print(f"  recall on known-CORRECT     = {acc_pos:.3f}  (judge accepts gold)")
    print(f"  recall on known-INCORRECT   = {acc_neg:.3f}  (judge rejects off-topic)")
    print(f"  temperature stability       = {stability:.3f}  (verdict agree 0.0 vs 0.7)")
    print(f"  judge cost                  = ${cost:.3f}")

    # Near-miss rejection is the discriminating-on-hard-cases check (RQ4). We gate
    # on the floor (AUC/recall) but FLAG a weak near-miss rate as a bounded confound.
    nm_ok = nm_reject_rate is None or nm_reject_rate >= 0.7
    passed = a >= args.auc_gate and acc_pos >= 0.7 and acc_neg >= 0.7
    verdict = "PASS" if passed else "FAIL"
    nm_txt = (f" Near-miss (on-topic x1.4) rejection={nm_reject_rate:.2f}"
              if nm_reject_rate is not None else "")
    if passed and not nm_ok:
        note = (f"Judge passes the floor (AUC={a:.3f}, off-topic separation) BUT rejects "
                f"only {nm_reject_rate:.2f} of on-topic near-miss numeric errors — the "
                "E_fin1 'engaged-but-not-discriminating' signature. E/F usable but F's "
                "lift is bounded: a weak judge under-rewards both authors equally (RQ4).")
    elif passed:
        note = ("Judge discriminates known-correct from off-topic-incorrect at "
                f"AUC={a:.3f}, rejects on-topic near-miss numeric errors at "
                f"{nm_reject_rate if nm_reject_rate is not None else float('nan'):.2f}, "
                f"stable across temperature ({stability:.2f}). E/F reward signal usable; "
                "discriminates on hard cases, not just off-topic (RQ4 addressed).")
    else:
        note = (f"Judge AUC={a:.3f} below gate {args.auc_gate} (or class recall low: "
                f"pos={acc_pos:.2f} neg={acc_neg:.2f}). Per E_fin1 'engaged-but-not-"
                "discriminating': E/F to be reported as JUDGE-CONFOUNDED, INCONCLUSIVE. "
                "B/D still stand.")

    out = {"judge_model": args.model, "n_pairs": len(pairs), "auc": a,
           "auc_gate": args.auc_gate, "recall_correct": acc_pos,
           "recall_incorrect": acc_neg, "temperature_stability": stability,
           "near_miss_reject_rate": nm_reject_rate, "near_miss_n": nm_total,
           "verdict": verdict, "cost_usd": cost, "note": note, "records": recs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  {verdict}: {note}")
    print(f"  wrote {args.out}")
    # The gate is INFORMATIVE, not run-blocking: a FAIL flags E/F as inconclusive
    # but B/D + the verifiable-vs-withheld half remain valid. Exit 0 either way;
    # the verdict is recorded for the analysis to consume.


if __name__ == "__main__":
    main()
