#!/usr/bin/env python3
"""
E_trace-profile STEP 1 — trace-richness profile for FinQA + FinanceBench.

Profile-FIRST anti-confirmation gate: compute trace-richness distributions and
per-behavioral-feature pass-vs-fail separation BEFORE fitting any RF. The
prediction (prediction.md) is written from THIS profile alone and committed
before profile.py's sibling rf_financebench.py is ever run.

RUN WITH python3.13 (macOS python3.14 sklearn is broken — carried lesson).

Richness scalars reported per domain/cell:
  - output_tokens mean, sd, CV (coefficient of variation = sd/mean)
  - input_tokens mean, sd, CV
  - a process-substrate flag: do loop/edit/revision-style features carry variance?

Separation per behavioral feature: standardized mean difference (Cohen's d,
pooled-SD) of pass vs fail. |d| is the descriptive effect size; sign tells
direction. At n~100-150 these are descriptive, not significance claims.
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BLUEPRINT = HERE.parent
RESULTS = BLUEPRINT / "results"
RESULTS.mkdir(exist_ok=True)

ROOT = BLUEPRINT.parent  # domains/autoresearch/blueprints
FINQA_CSV = ROOT / "e-fin2-finqa-behavioral-features" / "results" / "features.csv"
FB_DIR = ROOT / "e-harness3-reward-regime-x-locus" / "results"
FB_CELLS = ["E_haiku", "E_sonnet", "F_haiku_v-haiku", "F_sonnet_v-haiku"]

# FinQA behavioral feature set (verbatim from e-fin2 BEH_FEATURES).
FINQA_BEH = [
    "beh_output_tokens", "beh_input_tokens", "beh_cost_usd",
    "beh_reasoning_words", "beh_tokens_per_word", "beh_revision_count",
    "beh_num_mentions", "beh_abstain", "beh_latency_ms",
]


def cohens_d(pass_vals, fail_vals):
    """Standardized mean difference (pass - fail), pooled SD. Sign = direction."""
    p = np.asarray(pass_vals, float)
    f = np.asarray(fail_vals, float)
    p = p[~np.isnan(p)]
    f = f[~np.isnan(f)]
    if len(p) < 2 or len(f) < 2:
        return None
    np_, nf = len(p), len(f)
    sp = np.sqrt(((np_ - 1) * p.var(ddof=1) + (nf - 1) * f.var(ddof=1)) / (np_ + nf - 2))
    if sp == 0:
        return 0.0
    return float((p.mean() - f.mean()) / sp)


def richness(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return {}
    m, sd = float(v.mean()), float(v.std(ddof=1))
    return {
        "mean": round(m, 2),
        "sd": round(sd, 2),
        "cv": round(sd / m, 4) if m else None,
        "min": round(float(v.min()), 2),
        "max": round(float(v.max()), 2),
    }


def profile_finqa():
    df = pd.read_csv(FINQA_CSV)
    label = df["gold_pass"].astype(int)
    out = {
        "n": int(len(df)),
        "n_pass": int(label.sum()),
        "n_fail": int((1 - label).sum()),
        "base_rate": round(float(label.mean()), 4),
        "label": "gold_pass (E_fin1 exact-match)",
        "richness": {
            "output_tokens": richness(df["beh_output_tokens"]),
            "input_tokens": richness(df["beh_input_tokens"]),
            "cost_usd": richness(df["beh_cost_usd"]),
        },
        "separation": {},
    }
    pass_mask = label == 1
    for f in FINQA_BEH:
        if f not in df.columns:
            continue
        s = pd.to_numeric(df[f], errors="coerce")
        d = cohens_d(s[pass_mask], s[~pass_mask])
        # near-constant flag: fraction at the modal value
        nonzero = float((s != 0).mean())
        out["separation"][f] = {
            "pass_mean": round(float(s[pass_mask].mean()), 4),
            "fail_mean": round(float(s[~pass_mask].mean()), 4),
            "cohens_d": round(d, 4) if d is not None else None,
            "frac_nonzero": round(nonzero, 4),
        }
    return out


def profile_financebench():
    cells = {}
    for cell in FB_CELLS:
        rows = [json.loads(l) for l in (FB_DIR / f"{cell}.jsonl").open()]
        df = pd.DataFrame(rows)
        label = df["is_correct"].astype(int)
        # Behavioral feature set derivable from single-call graded Q&A traces.
        # NOTE: judge_conf EXCLUDED — is_correct derives from the judge, that's
        # label leakage, not a behavioral/process signal.
        df["tokens_ratio"] = df["output_tokens"] / df["input_tokens"].clip(lower=1)
        beh_cols = ["input_tokens", "output_tokens", "cost_usd", "tokens_ratio",
                    "jit_state_chars", "jit_notes_total"]
        pass_mask = label == 1
        sep = {}
        for f in beh_cols:
            if f not in df.columns:
                continue
            s = pd.to_numeric(df[f], errors="coerce")
            d = cohens_d(s[pass_mask], s[~pass_mask])
            sep[f] = {
                "pass_mean": round(float(s[pass_mask].mean()), 4),
                "fail_mean": round(float(s[~pass_mask].mean()), 4),
                "cohens_d": round(d, 4) if d is not None else None,
                "frac_nonzero": round(float((s != 0).mean()), 4),
            }
        cells[cell] = {
            "n": int(len(df)),
            "n_pass": int(label.sum()),
            "n_fail": int((1 - label).sum()),
            "base_rate": round(float(label.mean()), 4),
            "model": rows[0]["model"],
            "cell_meta": {"cell": rows[0]["cell"], "locus": rows[0]["locus"],
                          "regime": rows[0]["regime"]},
            "richness": {
                "output_tokens": richness(df["output_tokens"]),
                "input_tokens": richness(df["input_tokens"]),
                "cost_usd": richness(df["cost_usd"]),
            },
            "separation": sep,
        }
    return cells


def main():
    finqa = profile_finqa()
    fb = profile_financebench()

    # Coding reference (Phase-3, repo-verified phase3_report.md).
    coding_ref = {
        "auc": 0.756,
        "n": 300,
        "top3_importance_sum": 0.948,
        "top_features": {
            "beh_total_cost_usd": 0.4201,
            "beh_tokens_per_edit": 0.3272,
            "beh_loop_count": 0.2007,
        },
        "note": ("Coding traces are multi-edit agentic SWE-bench runs: the top-3 "
                 "features (cost, tokens-per-EDIT, loop-count) require an edit/loop "
                 "substrate that single-call Q&A lacks entirely."),
    }

    out = {
        "_meta": {
            "spec": "E_trace-profile (mechanism test, profile-first)",
            "env": "python3.13 sklearn-stack",
            "claim_under_test": ("behavioral RF (coding 0.756) is a trace-richness / "
                                 "thrash detector; discriminates iff traces are long "
                                 "and variable enough to carry a process signal"),
            "richness_scalar_def": ("output_tokens CV (coefficient of variation) as the "
                                    "primary trace-richness scalar; output_tokens mean as "
                                    "secondary. Higher = richer/more-variable trace."),
            "caveat": ("FinanceBench (e-harness3) traces are SINGLE-CALL graded Q&A, NOT "
                       "full agentic 10-K retrieval. No edit/loop/revision substrate. They "
                       "may UNDERSTATE FinanceBench's true trace richness — if nearly as "
                       "sparse as FinQA, the mechanism test is INCONCLUSIVE, not negative."),
        },
        "coding_reference": coding_ref,
        "finqa": finqa,
        "financebench": fb,
    }
    (RESULTS / "profile.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", RESULTS / "profile.json")

    # ── console summary ──
    print("\n=== TRACE-RICHNESS PROFILE ===")
    print(f"\nCODING (Phase-3 ref): AUC 0.756, n=300. Multi-edit agentic; top-3 "
          f"beh feats = 95% importance (cost/tokens-per-edit/loop-count).")
    fq = finqa["richness"]["output_tokens"]
    print(f"\nFinQA: n={finqa['n']} base_rate={finqa['base_rate']}")
    print(f"  output_tokens mean={fq['mean']} sd={fq['sd']} CV={fq['cv']}")
    print(f"  best |cohens_d| behavioral feature:")
    ds = [(k, v["cohens_d"]) for k, v in finqa["separation"].items() if v["cohens_d"] is not None]
    for k, d in sorted(ds, key=lambda kv: -abs(kv[1]))[:4]:
        print(f"     {k:22s} d={d:+.3f}")
    # structurally-absent flag
    print(f"  process-substrate features near-constant (frac_nonzero):")
    for k in ["beh_revision_count", "beh_abstain"]:
        print(f"     {k:22s} frac_nonzero={finqa['separation'][k]['frac_nonzero']}")

    print(f"\nFinanceBench (single-call graded Q&A):")
    for cell, c in fb.items():
        ot = c["richness"]["output_tokens"]
        ds = [(k, v["cohens_d"]) for k, v in c["separation"].items() if v["cohens_d"] is not None]
        best = sorted(ds, key=lambda kv: -abs(kv[1]))[:3]
        beststr = ", ".join(f"{k}={d:+.2f}" for k, d in best)
        print(f"  {cell:18s} n={c['n']} base={c['base_rate']} | "
              f"out_tok mean={ot['mean']} CV={ot['cv']} | top d: {beststr}")

    # richness ordering
    print("\n=== RICHNESS ORDERING (output_tokens) ===")
    rows = [("coding", None, None)]  # coding token stats not on disk; multi-edit by construction
    rows.append(("FinQA", fq["mean"], fq["cv"]))
    for cell, c in fb.items():
        ot = c["richness"]["output_tokens"]
        rows.append((f"FB:{cell}", ot["mean"], ot["cv"]))
    for name, m, cv in rows:
        print(f"  {name:22s} mean_out_tok={m} CV={cv}")


if __name__ == "__main__":
    main()
