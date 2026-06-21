#!/usr/bin/env python3
"""Stage 7: score both verifiers against exact-match ground truth.

Computes precision / recall / accuracy on the CONFIDENT subset for:
  - Confirmatory: confident = verdict likely_correct (rating >=4)
  - Adversarial: confident = 4/4 unanimous likely_correct

Plus AUC (using a continuous score where available), cost/eval, and the
verifier-model calibration check (does adversarial avoid defaulting to
'uncertain' off-coding?).

The "precision on confident subset" is the headline metric, directly
comparable to the coding 0.40 (confirmatory) -> 0.92 (adversarial) result.
"""
import argparse
import json


def precision_recall(recs, is_confident):
    """precision = P(match | confident); recall = P(confident | match)."""
    confident = [r for r in recs if is_confident(r)]
    matches = [r for r in recs if r["match"]]
    tp = sum(1 for r in confident if r["match"])
    prec = tp / len(confident) if confident else float("nan")
    rec = tp / len(matches) if matches else float("nan")
    return {
        "n_confident": len(confident),
        "n_match": len(matches),
        "tp": tp,
        "fp": len(confident) - tp,
        "precision": prec,
        "recall": rec,
    }


def auc_score(recs, score_fn):
    """Mann-Whitney AUC: P(score(pos) > score(neg))."""
    pos = [score_fn(r) for r in recs if r["match"]]
    neg = [score_fn(r) for r in recs if not r["match"]]
    pos = [s for s in pos if s is not None]
    neg = [s for s in neg if s is not None]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) and x == x else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run_verifiers.py output")
    ap.add_argument("--out", default=None, help="write aggregate JSON here")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.results) if l.strip()]
    n = len(recs)
    base_rate = sum(1 for r in recs if r["match"]) / n

    # --- Confirmatory: confident = likely_correct (rating >= 4) ---
    def conf_confident(r):
        return r.get("conf_verdict") == "likely_correct"
    conf = precision_recall(recs, conf_confident)
    conf_auc = auc_score(recs, lambda r: r.get("conf_rating"))

    # --- Adversarial: confident = 4/4 unanimous likely_correct ---
    def adv_confident(r):
        return bool(r.get("adv_confident"))
    adv = precision_recall(recs, adv_confident)

    # Adversarial continuous score for AUC: count of likely_correct votes (0-4)
    def adv_votes(r):
        vs = r.get("adv_verdicts", [])
        return sum(1 for v in vs if v == "likely_correct")
    adv_auc = auc_score(recs, adv_votes)

    # --- Calibration check: verdict distributions ---
    def vdist(getter):
        d = {}
        for r in recs:
            v = getter(r)
            d[v] = d.get(v, 0) + 1
        return d
    conf_dist = vdist(lambda r: r.get("conf_verdict"))
    # Adversarial: distribution of per-call verdicts (4n total) + vote histogram
    adv_call_dist = {}
    vote_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for r in recs:
        for v in r.get("adv_verdicts", []):
            adv_call_dist[v] = adv_call_dist.get(v, 0) + 1
        vote_hist[adv_votes(r)] = vote_hist.get(adv_votes(r), 0) + 1

    # --- Cost ---
    gen_cost = sum(r.get("cost_usd", 0) for r in recs)
    conf_cost = sum(r.get("conf_cost", 0) for r in recs)
    adv_cost = sum(r.get("adv_cost", 0) for r in recs)

    # --- Report ---
    print("=" * 64)
    print("E_fin1 — FinQA Skill-Verifier Replication: RESULTS")
    print("=" * 64)
    print(f"n = {n}   base rate (agent accuracy) = {fmt(base_rate)}")
    print()
    print("PRIMARY: precision on confident subset (adversarial vs confirmatory)")
    print(f"  Confirmatory : precision={fmt(conf['precision'])} "
          f"recall={fmt(conf['recall'])} "
          f"(confident={conf['n_confident']}, tp={conf['tp']}, fp={conf['fp']})")
    print(f"  Adversarial  : precision={fmt(adv['precision'])} "
          f"recall={fmt(adv['recall'])} "
          f"(confident={adv['n_confident']}, tp={adv['tp']}, fp={adv['fp']})")
    if conf['precision'] == conf['precision'] and conf['precision'] > 0:
        lift = adv['precision'] / conf['precision']
        print(f"  LIFT (adv/conf precision) = {fmt(lift, 2)}x   "
              f"(coding domain was 2.3x: 0.40 -> 0.92)")
    print(f"  delta precision = {fmt(adv['precision'] - conf['precision'])}")
    print()
    print("SECONDARY: AUC")
    print(f"  Confirmatory AUC (rating 1-5) = {fmt(conf_auc)}")
    print(f"  Adversarial  AUC (votes 0-4)  = {fmt(adv_auc)}")
    print()
    print("CALIBRATION CHECK (does adversarial avoid defaulting to 'uncertain'?)")
    print(f"  Confirmatory verdict dist: {conf_dist}")
    print(f"  Adversarial per-call verdict dist (of {4*n} calls): {adv_call_dist}")
    print(f"  Adversarial vote histogram (# likely_correct of 4): {vote_hist}")
    print()
    print("COST")
    print(f"  generation: ${gen_cost:.4f}  (${gen_cost/n:.5f}/example)")
    print(f"  confirmatory verifier: ${conf_cost:.4f}  (${conf_cost/n:.5f}/eval)")
    print(f"  adversarial verifier:  ${adv_cost:.4f}  (${adv_cost/n:.5f}/eval)")
    print(f"  adversarial vs $0.03 ceiling: "
          f"{'UNDER' if adv_cost/n <= 0.03 else 'OVER'} "
          f"(${adv_cost/n:.5f}/eval)")
    print("=" * 64)

    agg = {
        "n": n,
        "base_rate": base_rate,
        "confirmatory": {**conf, "auc": conf_auc, "verdict_dist": conf_dist},
        "adversarial": {**adv, "auc": adv_auc,
                        "call_verdict_dist": adv_call_dist,
                        "vote_histogram": vote_hist},
        "precision_lift": (adv['precision'] / conf['precision']
                           if conf['precision'] else None),
        "precision_delta": adv['precision'] - conf['precision'],
        "cost": {
            "gen_total": gen_cost, "gen_per": gen_cost / n,
            "conf_total": conf_cost, "conf_per": conf_cost / n,
            "adv_total": adv_cost, "adv_per": adv_cost / n,
            "adv_under_ceiling": adv_cost / n <= 0.03,
        },
    }
    if args.out:
        json.dump(agg, open(args.out, "w"), indent=2)
        print(f"\nwrote aggregate to {args.out}")


if __name__ == "__main__":
    main()
