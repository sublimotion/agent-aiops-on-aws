#!/usr/bin/env python3
"""
E_harness1 — Harness x Behavioral-Verifier Interaction.

Question: does improving the harness DESTROY the failure signatures the Phase-3
behavioral RF reads (substitutes), or do failures RELOCATE to new signatures
(complements)?

Data-only, local. No GPU, no API generation, no new trajectories.

Harness-quality axis is operationalized via the verification-scaffold composition
inside the Phase-3 corpus (combined_features.csv, the Claude-Code VP production eval,
n=300). The composition one-hots encode pre/post harness improvement:
  - WEAK harness    = beh_comp_ignore   (no verification; the 0%-tool-adoption arm)
  - IMPROVED harness= beh_comp_full_pipeline (gen->run->review; the 83% two-stage arm)

The agent-harness SERA/LangGraph/Aider eval has trajectory metadata but NO RF
features (cost / tokens_per_edit / loop_count) and NO gold labels, so it supports
distribution-shift reporting only, not RF re-fit. Handled in a separate section.

Reuses the exact Phase-3 RF recipe from learned-verifier/scripts/train_combined.py:
  RandomForestClassifier(n_estimators=200, max_depth=7, class_weight="balanced",
                         random_state=42); nan->-999; 5-fold stratified, pooled OOF.

Phase-3 baseline (repo-verified, phase3_report.md):
  selected_4 RF AUC=0.756; importances cost 0.420 / tokens_per_edit 0.327 /
  loop_count 0.201 / svg_accepted 0.052.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BLUEPRINT = HERE.parent
RESULTS = BLUEPRINT / "results"
RESULTS.mkdir(exist_ok=True)

BLUEPRINTS = BLUEPRINT.parent
COMBINED = BLUEPRINTS / "learned-verifier" / "results" / "combined_features.csv"
AGENT_HARNESS = BLUEPRINTS / "agent-harness" / "results"

RNG = 42
np.random.seed(RNG)

# The Phase-3 selected_4 feature set (svg_accepted is the only non-behavioral one).
SELECTED_4 = ["beh_total_cost_usd", "beh_tokens_per_edit", "beh_loop_count", "svg_accepted"]
BEH_3 = ["beh_total_cost_usd", "beh_tokens_per_edit", "beh_loop_count"]

# The "4 behavioral feature distributions" named in the spec (cost, edit%, loop, tool-adoption)
# plus tokens_per_edit (the actual RF feature) -> report all to bridge spec & RF.
DIST_FEATURES = [
    "beh_total_cost_usd",
    "beh_tokens_per_edit",
    "beh_loop_count",
    "beh_action_pct_edit",      # edit%
    "beh_adversarial_review_used",  # tool-adoption (review tool)
]


# ─── RF recipe (verbatim from train_combined.py) ──────────────────────────────

def rf_oof(df, features, label="gold_pass", n_splits=5):
    """5-fold stratified CV, pooled out-of-fold probs. Returns (y_true, y_prob, importances)."""
    sub = df.dropna(subset=[label]).copy()
    X = sub[features].values
    y = sub[label].astype(int).values
    X = np.nan_to_num(X, nan=-999)

    # Need at least n_splits of the minority class
    minority = min((y == 0).sum(), (y == 1).sum())
    splits = min(n_splits, minority) if minority >= 2 else 0
    if splits < 2:
        return None  # too few of one class to CV

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RNG)
    yt, yp = [], []
    last = None
    for tr, va in skf.split(X, y):
        m = RandomForestClassifier(
            n_estimators=200, max_depth=7, class_weight="balanced", random_state=RNG
        )
        m.fit(X[tr], y[tr])
        yp.extend(m.predict_proba(X[va])[:, 1])
        yt.extend(y[va])
        last = m
    yt, yp = np.array(yt), np.array(yp)
    imp = dict(zip(features, last.feature_importances_))
    return yt, yp, imp, splits


def auc_with_ci(y_true, y_prob, n_boot=2000):
    """Pooled AUC + bootstrap 95% CI."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return None
    rng = np.random.default_rng(RNG)
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"auc": round(float(auc), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot_valid": len(boots)}


def precision_at_recall(y_true, y_prob, min_recall=0.30):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    valid = r >= min_recall
    return float(np.max(p[valid])) if valid.any() else 0.0


# ─── Distribution shift ───────────────────────────────────────────────────────

def describe(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 6),
        "std": round(float(s.std(ddof=1)) if len(s) > 1 else 0.0, 6),
        "var": round(float(s.var(ddof=1)) if len(s) > 1 else 0.0, 6),
        "median": round(float(s.median()), 6),
        "cv": round(float(s.std(ddof=1) / s.mean()), 4) if len(s) > 1 and s.mean() != 0 else None,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out = {"_meta": {
        "spec": "E_harness1",
        "corpus": str(COMBINED.relative_to(BLUEPRINTS.parent.parent)) if False else "learned-verifier/results/combined_features.csv",
        "phase3_baseline": {"selected_4_rf_auc": 0.756,
                            "importances": {"beh_total_cost_usd": 0.4201,
                                            "beh_tokens_per_edit": 0.3272,
                                            "beh_loop_count": 0.2007,
                                            "svg_accepted": 0.0519}},
        "harness_quality_axis": "verification-scaffold composition (beh_comp_*) inside the VP production eval",
    }}

    df = pd.read_csv(COMBINED)
    out["_meta"]["n_total"] = int(len(df))
    out["_meta"]["pool_pass_rate"] = round(float(df.gold_pass.mean()), 4)

    # ── Define conditions ──
    # Primary axis: WEAK (no verification) vs IMPROVED (full pipeline)
    cond = {
        "weak_ignore":    df[df["beh_comp_ignore"] == 1].copy(),
        "improved_full_pipeline": df[df["beh_comp_full_pipeline"] == 1].copy(),
    }
    # Coarse alternative axis matching the pivot-analysis +46.3pp lever (tool used vs not)
    cond_tool = {
        "tool_not_used": df[df["beh_tool_used"] == 0].copy(),
        "tool_used":     df[df["beh_tool_used"] == 1].copy(),
    }

    out["condition_sizes"] = {
        k: {"n": int(len(v)), "pass": int(v.gold_pass.sum()),
            "fail": int((1 - v.gold_pass).sum()),
            "pass_rate": round(float(v.gold_pass.mean()), 4)}
        for k, v in {**cond, **cond_tool}.items()
    }

    # ── (2) Distribution shift per condition ──
    dist = {}
    for axis_name, conds in [("verification_scaffold", cond), ("tool_adoption", cond_tool)]:
        dist[axis_name] = {}
        for feat in DIST_FEATURES:
            dist[axis_name][feat] = {cname: describe(c[feat]) for cname, c in conds.items()}
        # compression ratios (improved / weak) on var
        names = list(conds.keys())
        weak, improved = names[0], names[1]
        dist[axis_name]["_var_ratio_improved_over_weak"] = {}
        for feat in DIST_FEATURES:
            vw = dist[axis_name][feat][weak].get("var")
            vi = dist[axis_name][feat][improved].get("var")
            ratio = round(vi / vw, 4) if vw not in (None, 0) and vi is not None else None
            dist[axis_name]["_var_ratio_improved_over_weak"][feat] = ratio
    out["distribution_shift"] = dist

    # ── (3) Per-condition RF AUC (selected_4 and beh_3) ──
    rf = {}
    # Pooled baseline reproduction (sanity check vs 0.756)
    for fset_name, fset in [("selected_4", SELECTED_4), ("beh_3", BEH_3)]:
        res = rf_oof(df, fset)
        if res:
            yt, yp, imp, splits = res
            r = auc_with_ci(yt, yp)
            r["p_at_r30"] = round(precision_at_recall(yt, yp), 4)
            r["importances"] = {k: round(float(v), 4) for k, v in imp.items()}
            r["n"] = int(len(yt))
            r["cv_splits"] = splits
            rf[f"POOLED__{fset_name}"] = r

    for cname, c in cond.items():
        for fset_name, fset in [("selected_4", SELECTED_4), ("beh_3", BEH_3)]:
            res = rf_oof(c, fset)
            key = f"{cname}__{fset_name}"
            if not res:
                rf[key] = {"auc": None, "note": "too few of one class to CV"}
                continue
            yt, yp, imp, splits = res
            r = auc_with_ci(yt, yp)
            if r is None:
                rf[key] = {"auc": None, "note": "single-class OOF"}
                continue
            r["p_at_r30"] = round(precision_at_recall(yt, yp), 4)
            r["importances"] = {k: round(float(v), 4) for k, v in imp.items()}
            r["n"] = int(len(yt))
            r["cv_splits"] = splits
            rf[key] = r
    out["per_condition_rf"] = rf

    # ── (4) Failure-relocation probe ──
    # "Wrong output despite clean trajectory" within the IMPROVED harness.
    # Clean trajectory := below-median loop_count AND below-median cost (failure
    # signatures suppressed). Among improved-harness cases, do residual signals
    # (v009 rubric, debate, svg, read_edit) still flag the clean failures?
    imp_df = cond["improved_full_pipeline"].dropna(subset=["beh_loop_count", "beh_total_cost_usd"]).copy()
    loop_med = imp_df["beh_loop_count"].median()
    cost_med = imp_df["beh_total_cost_usd"].median()
    clean = imp_df[(imp_df["beh_loop_count"] <= loop_med) & (imp_df["beh_total_cost_usd"] <= cost_med)]
    dirty = imp_df[~imp_df.index.isin(clean.index)]

    reloc = {
        "definition": "clean := loop_count<=median AND cost<=median within improved harness",
        "improved_n": int(len(imp_df)),
        "loop_median": round(float(loop_med), 4),
        "cost_median": round(float(cost_med), 6),
        "clean": {"n": int(len(clean)), "pass": int(clean.gold_pass.sum()),
                  "fail": int((1 - clean.gold_pass).sum()),
                  "pass_rate": round(float(clean.gold_pass.mean()), 4)},
        "dirty": {"n": int(len(dirty)), "pass": int(dirty.gold_pass.sum()),
                  "fail": int((1 - dirty.gold_pass).sum()),
                  "pass_rate": round(float(dirty.gold_pass.mean()), 4)},
    }
    # Among clean-trajectory cases, can residual (non-RF) signals separate fail from pass?
    residual_feats = ["v009_mean_score", "v009_lc_count", "debate_score",
                      "svg_line_recall", "svg_accepted", "enew1_read_edit_ratio",
                      "enew2_total_errors", "beh_action_pct_edit"]
    residual_feats = [f for f in residual_feats if f in df.columns]
    reloc["clean_failure_vs_pass_separation"] = {}
    if clean.gold_pass.nunique() == 2 and len(clean) >= 8:
        for f in residual_feats:
            cp = pd.to_numeric(clean.loc[clean.gold_pass == 1, f], errors="coerce").dropna()
            cf = pd.to_numeric(clean.loc[clean.gold_pass == 0, f], errors="coerce").dropna()
            if len(cp) >= 2 and len(cf) >= 2:
                reloc["clean_failure_vs_pass_separation"][f] = {
                    "pass_mean": round(float(cp.mean()), 4),
                    "fail_mean": round(float(cf.mean()), 4),
                    "delta": round(float(cp.mean() - cf.mean()), 4),
                }
        # AUC of residual-signal RF restricted to clean-trajectory cases
        res = rf_oof(clean, residual_feats)
        if res:
            yt, yp, imp, splits = res
            r = auc_with_ci(yt, yp)
            if r:
                r["importances"] = {k: round(float(v), 4) for k, v in imp.items()}
                r["n"] = int(len(yt))
                reloc["clean_residual_rf"] = r
        # And does the BEHAVIORAL RF still flag clean failures? (the substitute test)
        res_b = rf_oof(clean, BEH_3)
        if res_b:
            yt, yp, imp, splits = res_b
            rb = auc_with_ci(yt, yp)
            if rb:
                rb["n"] = int(len(yt))
                reloc["clean_behavioral_rf"] = rb
    out["failure_relocation"] = reloc

    # ── (5) Per-condition forward selection ──
    candidate = [c for c in df.columns if c not in ("instance_id", "gold_pass")
                 and df[c].notna().any() and pd.api.types.is_numeric_dtype(df[c])]
    fsel = {}
    for cname, c in list(cond.items()) + [("POOLED", df)]:
        fsel[cname] = forward_select(c, candidate, max_features=6)
    out["forward_selection"] = fsel

    # ── Verdict ──
    out["verdict"] = derive_verdict(out)

    (RESULTS / "harness_interaction_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", RESULTS / "harness_interaction_results.json")
    return out


def forward_select(c, candidate, max_features=6, metric="auc"):
    """Greedy forward selection within a condition, optimizing pooled OOF AUC."""
    sub = c.dropna(subset=["gold_pass"])
    if sub.gold_pass.nunique() < 2 or min((sub.gold_pass == 0).sum(), (sub.gold_pass == 1).sum()) < 3:
        return {"note": "too few of one class", "n": int(len(sub))}
    selected, history, best = [], [], 0.0
    for step in range(max_features):
        best_feat, best_auc = None, best
        for f in candidate:
            if f in selected:
                continue
            res = rf_oof(sub, selected + [f])
            if not res:
                continue
            yt, yp, _, _ = res
            try:
                a = roc_auc_score(yt, yp)
            except ValueError:
                continue
            if a > best_auc:
                best_auc, best_feat = a, f
        if best_feat is None:
            break
        selected.append(best_feat)
        history.append({"step": step + 1, "added": best_feat, "auc": round(best_auc, 4)})
        best = best_auc
        if len(history) >= 3 and history[-1]["auc"] <= history[-3]["auc"]:
            break
    return {"selected": selected, "history": history, "n": int(len(sub))}


def derive_verdict(out):
    rf = out["per_condition_rf"]
    weak = rf.get("weak_ignore__selected_4", {}).get("auc")
    weak_b = rf.get("weak_ignore__beh_3", {}).get("auc")
    impr = rf.get("improved_full_pipeline__selected_4", {}).get("auc")
    impr_b = rf.get("improved_full_pipeline__beh_3", {}).get("auc")
    reloc = out["failure_relocation"]
    clean_pass = reloc["clean"]["pass_rate"]
    clean_resid = reloc.get("clean_residual_rf", {}).get("auc")
    clean_beh = reloc.get("clean_behavioral_rf", {}).get("auc")
    return {
        "weak_auc_selected4": weak, "weak_auc_beh3": weak_b,
        "improved_auc_selected4": impr, "improved_auc_beh3": impr_b,
        "clean_trajectory_pass_rate": clean_pass,
        "clean_residual_signal_auc": clean_resid,
        "clean_behavioral_signal_auc": clean_beh,
        "interpretation": "See report markdown — computed verdict written there.",
    }


if __name__ == "__main__":
    main()
