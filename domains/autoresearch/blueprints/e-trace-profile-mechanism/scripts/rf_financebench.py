#!/usr/bin/env python3
"""
E_trace-profile STEP 3 — fit the behavioral RF on FinanceBench (e-harness3
single-call graded-Q&A traces), verbatim Phase-3 recipe.

GATE: prediction.md + profile.json were committed to git BEFORE this script ran.

RUN WITH python3.13 (macOS python3.14 sklearn broken — carried lesson).

Recipe (verbatim from e-fin2/analyze.py, which copies learned-verifier):
  RandomForestClassifier(n_estimators=200, max_depth=7,
                         class_weight="balanced", random_state=42)
  nan -> -999; 5-fold stratified CV; pooled out-of-fold probabilities;
  bootstrap 95% CI (2000 resamples).

Fit PER CELL (single model) — the clean analog of FinQA's single-model design,
and avoids a cost_usd model-detector confound when pooling Haiku+Sonnet (~10x
price gap). A pooled-all fit is ALSO reported but flagged as confounded.

Behavioral feature set (pre-committed in prediction.md):
  [input_tokens, output_tokens, cost_usd, tokens_ratio, jit_state_chars,
   jit_notes_total]
judge_conf / judge_verdict EXCLUDED (label leakage: is_correct derives from judge).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BLUEPRINT = HERE.parent
RESULTS = BLUEPRINT / "results"
ROOT = BLUEPRINT.parent
FB_DIR = ROOT / "e-harness3-reward-regime-x-locus" / "results"
FB_CELLS = ["E_haiku", "E_sonnet", "F_haiku_v-haiku", "F_sonnet_v-haiku"]

RNG = 42
np.random.seed(RNG)

BEH_FEATURES = ["input_tokens", "output_tokens", "cost_usd", "tokens_ratio",
                "jit_state_chars", "jit_notes_total"]

PHASE3_CODING_AUC = 0.756
FINQA_AUC = 0.569
FINQA_CI = [0.430, 0.709]


def load_cell(cell):
    rows = [json.loads(l) for l in (FB_DIR / f"{cell}.jsonl").open()]
    df = pd.DataFrame(rows)
    df["tokens_ratio"] = df["output_tokens"] / df["input_tokens"].clip(lower=1)
    df["label"] = df["is_correct"].astype(int)
    df["model"] = df["model"]
    return df


def rf_oof(df, features, label="label", n_splits=5):
    sub = df.dropna(subset=[label]).copy()
    X = np.nan_to_num(sub[features].values.astype(float), nan=-999)
    y = sub[label].astype(int).values
    minority = min((y == 0).sum(), (y == 1).sum())
    splits = min(n_splits, minority) if minority >= 2 else 0
    if splits < 2:
        return None
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RNG)
    yt, yp, imps = [], [], []
    for tr, va in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=200, max_depth=7,
                                   class_weight="balanced", random_state=RNG)
        m.fit(X[tr], y[tr])
        yp.extend(m.predict_proba(X[va])[:, 1])
        yt.extend(y[va])
        imps.append(m.feature_importances_)
    yt, yp = np.array(yt), np.array(yp)
    imp = dict(zip(features, np.mean(imps, axis=0)))
    return yt, yp, imp, splits


def auc_with_ci(y_true, y_prob, n_boot=2000):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return None
    rng = np.random.default_rng(RNG)
    boots, n = [], len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"auc": round(float(auc), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_width": round(float(hi - lo), 4),
            "n_boot_valid": len(boots)}


def fit(df, features, name):
    res = rf_oof(df, features)
    if not res:
        return {"name": name, "auc": None, "note": "too few minority to CV"}
    yt, yp, imp, splits = res
    r = auc_with_ci(yt, yp) or {"auc": None, "note": "single-class OOF"}
    r["name"] = name
    r["n"] = int(len(yt))
    r["n_pass"] = int(yt.sum())
    r["n_fail"] = int((1 - yt).sum())
    r["cv_splits"] = splits
    r["importances"] = {k: round(float(v), 4)
                        for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    return r


def main():
    out = {
        "_meta": {
            "spec": "E_trace-profile STEP 3 (RF fit on FinanceBench)",
            "gate": "prediction.md committed at 5e63a3f BEFORE this fit",
            "rf_recipe": "RF(n_estimators=200, max_depth=7, class_weight=balanced, "
                         "seed=42); nan->-999; 5-fold stratified pooled OOF; "
                         "bootstrap 95% CI (2000)",
            "features": BEH_FEATURES,
            "leakage_excluded": ["judge_conf", "judge_verdict", "judge_reason"],
            "fit_unit": "per cell (single model), to mirror FinQA single-model "
                        "design and avoid cost_usd model-detector confound on pooling",
            "env": "python3.13 sklearn-stack",
            "reference_points": {"coding": PHASE3_CODING_AUC,
                                 "finqa": FINQA_AUC, "finqa_ci95": FINQA_CI},
        },
        "per_cell": {},
    }

    cells_df = {}
    for cell in FB_CELLS:
        df = load_cell(cell)
        cells_df[cell] = df
        out["per_cell"][cell] = fit(df, BEH_FEATURES, f"financebench_{cell}")

    # Pooled-all (flagged confounded: cost_usd separates Haiku vs Sonnet by ~10x).
    pooled = pd.concat(cells_df.values(), ignore_index=True)
    out["pooled_all_CONFOUNDED"] = fit(pooled, BEH_FEATURES, "financebench_pooled_all")
    out["pooled_all_CONFOUNDED"]["warning"] = (
        "cost_usd separates Haiku (~$0.003) vs Sonnet (~$0.007) by model pricing; "
        "a pooled fit can detect MODEL, not process quality. Per-cell is the "
        "honest unit. Reported for completeness only.")

    # Pooled-within-model (Haiku cells together, Sonnet cells together): same
    # pricing within group, so cost_usd is not a model tag. Larger n per fit.
    haiku = pd.concat([cells_df["E_haiku"], cells_df["F_haiku_v-haiku"]], ignore_index=True)
    sonnet = pd.concat([cells_df["E_sonnet"], cells_df["F_sonnet_v-haiku"]], ignore_index=True)
    out["pooled_within_model"] = {
        "haiku_n300": fit(haiku, BEH_FEATURES, "financebench_haiku_pooled"),
        "sonnet_n300": fit(sonnet, BEH_FEATURES, "financebench_sonnet_pooled"),
        "note": "same pricing within model => cost_usd not a model tag; n=300 each, "
                "tighter CI than per-cell n=150.",
    }

    (RESULTS / "rf_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", RESULTS / "rf_results.json")

    print("\n=== FinanceBench behavioral-RF (verbatim Phase-3 recipe) ===")
    print(f"Reference: coding 0.756 (n=300) | FinQA 0.569 CI{FINQA_CI} (n=100)")
    print(f"PRE-REGISTERED prediction: per-cell AUC 0.62-0.72 (~0.67)\n")
    print(f"{'cell':20s} {'n':>4s} {'pass/fail':>10s} {'AUC':>7s} {'CI95':>18s} {'width':>6s}")
    for cell, r in out["per_cell"].items():
        if r.get("auc") is None:
            print(f"{cell:20s} -- {r.get('note')}")
            continue
        pf = f"{r['n_pass']}/{r['n_fail']}"
        print(f"{cell:20s} {r['n']:>4d} {pf:>10s} "
              f"{r['auc']:>7.3f} {str(r['ci95']):>18s} {r['ci_width']:>6.3f}")
    print("\nPooled-within-model (n=300, cost_usd not a model tag):")
    for k in ["haiku_n300", "sonnet_n300"]:
        r = out["pooled_within_model"][k]
        print(f"  {k:14s} AUC={r['auc']:.3f} CI{r['ci95']} width={r['ci_width']:.3f} "
              f"n={r['n']} ({r['n_pass']}/{r['n_fail']})")
    pc = out["pooled_all_CONFOUNDED"]
    print(f"\nPooled-all (CONFOUNDED, model-detector): AUC={pc['auc']:.3f} "
          f"CI{pc['ci95']} — top feat {list(pc['importances'].items())[0]}")

    print("\nPer-cell mean importances (top feature):")
    for cell, r in out["per_cell"].items():
        if r.get("importances"):
            top = list(r["importances"].items())[:2]
            print(f"  {cell:20s} {top}")


if __name__ == "__main__":
    main()
