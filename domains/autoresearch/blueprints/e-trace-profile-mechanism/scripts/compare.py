#!/usr/bin/env python3
"""
E_trace-profile STEP 4 — three-point comparison: does behavioral-RF AUC track
trace-richness? Emits richness_vs_auc.json + a plain-text table.
RUN WITH python3.13.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

prof = json.loads((RESULTS / "profile.json").read_text())
rf = json.loads((RESULTS / "rf_results.json").read_text())

# richness scalar = output_tokens CV (primary), mean (secondary)
points = []
# coding: token stats not on disk; multi-edit agentic by construction -> richest
points.append({
    "point": "coding (Phase-3)", "n": 300, "out_tok_mean": None, "out_tok_cv": None,
    "best_beh_d": None, "rf_auc": 0.756, "rf_ci95": None, "ci_width": None,
    "richness_note": "multi-edit agentic; top-3 feats (cost/tokens-per-edit/loop) 95% imp",
})
fq = prof["finqa"]["richness"]["output_tokens"]
fq_d = max((abs(v["cohens_d"]) for v in prof["finqa"]["separation"].values()
            if v["cohens_d"] is not None))
points.append({
    "point": "FinQA", "n": 100, "out_tok_mean": fq["mean"], "out_tok_cv": fq["cv"],
    "best_beh_d": round(fq_d, 3), "rf_auc": 0.569, "rf_ci95": [0.430, 0.709],
    "ci_width": round(0.709 - 0.430, 3), "richness_note": "single-call, short, low-variance",
})
for cell, c in prof["financebench"].items():
    ot = c["richness"]["output_tokens"]
    d = max((abs(v["cohens_d"]) for v in c["separation"].values()
             if v["cohens_d"] is not None))
    r = rf["per_cell"][cell]
    points.append({
        "point": f"FB:{cell} (per-cell)", "n": r["n"],
        "out_tok_mean": ot["mean"], "out_tok_cv": ot["cv"], "best_beh_d": round(d, 3),
        "rf_auc": r["auc"], "rf_ci95": r["ci95"], "ci_width": r["ci_width"],
        "richness_note": "single-call graded Q&A; jit_* inert",
    })
for k, label in [("haiku_n300", "FB:haiku (n=300 within-model)"),
                 ("sonnet_n300", "FB:sonnet (n=300 within-model)")]:
    r = rf["pooled_within_model"][k]
    points.append({
        "point": label, "n": r["n"], "out_tok_mean": None, "out_tok_cv": None,
        "best_beh_d": None, "rf_auc": r["auc"], "rf_ci95": r["ci95"],
        "ci_width": r["ci_width"], "richness_note": "same pricing within model; tighter CI",
    })

out = {
    "richness_scalar": "output_tokens CV (primary), mean (secondary); coding richest by construction (multi-edit)",
    "ordering_richness": "coding > FinanceBench (270-336 tok, CV .36-.58) > FinQA (148 tok, CV .29)",
    "ordering_separation": "coding > FinanceBench (|d| .66-1.04) > FinQA (|d| <= .57)",
    "points": points,
}
(RESULTS / "richness_vs_auc.json").write_text(json.dumps(out, indent=2))
print("Wrote", RESULTS / "richness_vs_auc.json\n")

hdr = f"{'point':32s} {'n':>4s} {'out_tok':>8s} {'CV':>5s} {'|d|':>5s} {'RF_AUC':>7s} {'CI95':>18s} {'width':>6s}"
print(hdr); print("-" * len(hdr))
for p in points:
    om = f"{p['out_tok_mean']:.0f}" if p["out_tok_mean"] else "--"
    cv = f"{p['out_tok_cv']:.2f}" if p["out_tok_cv"] else "--"
    d = f"{p['best_beh_d']:.2f}" if p["best_beh_d"] else "--"
    ci = str(p["rf_ci95"]) if p["rf_ci95"] else "--"
    w = f"{p['ci_width']:.2f}" if p["ci_width"] else "--"
    print(f"{p['point']:32s} {p['n']:>4d} {om:>8s} {cv:>5s} {d:>5s} {p['rf_auc']:>7.3f} {ci:>18s} {w:>6s}")
