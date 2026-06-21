#!/usr/bin/env python3
"""
E_fin2 — fit the Phase-3 RandomForest recipe on FinQA behavioral/process
features and run the head-to-head vs the E_fin1 skill-verifier.

RUN WITH python3.13 (macOS python3.14 has a broken sklearn install —
"No module named sklearn.utils._estimator_html_repr"; lesson from E_harness1).

RF recipe (verbatim from learned-verifier/scripts/train_combined.py, as
reproduced in E_harness1/scripts/analyze_harness_interaction.py):
  RandomForestClassifier(n_estimators=200, max_depth=7,
                         class_weight="balanced", random_state=42)
  nan -> -999; 5-fold stratified CV; pooled out-of-fold probabilities.

Phase-3 coding baseline (repo-verified, learned-verifier/results/phase3_report.md):
  selected_4 RF AUC = 0.756
  importances: beh_total_cost_usd 0.420, beh_tokens_per_edit 0.327,
               beh_loop_count 0.201, svg_accepted 0.052
  ordering: behavioral-only RF 0.730 > v009-only 0.682 = debate-only 0.682
  difficulty-conditioning regressed AUC 0.756 -> 0.743 (enew_report.md) -> NOT re-run here.
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
FEATURES = RESULTS / "features.csv"

RNG = 42
np.random.seed(RNG)

PHASE3_BASELINE = {
    "selected_4_rf_auc": 0.756,
    "behavioral_only_rf_auc": 0.730,
    "v009_only_auc": 0.682,
    "debate_only_auc": 0.682,
    "importances": {"beh_total_cost_usd": 0.420, "beh_tokens_per_edit": 0.327,
                    "beh_loop_count": 0.201, "svg_accepted": 0.052},
    "difficulty_conditioning_regressed_to": 0.743,
    "source": "learned-verifier/results/phase3_report.md + enew_report.md (repo-verified)",
}

# ── Feature families (see extract_features.py) ──
# Agent-trajectory behavioral = the DIRECT Phase-3 analog (process signal from
# the agent's own run).
BEH_FEATURES = [
    "beh_output_tokens", "beh_input_tokens", "beh_cost_usd",
    "beh_reasoning_words", "beh_tokens_per_word", "beh_revision_count",
    "beh_num_mentions", "beh_abstain", "beh_latency_ms",
]
# A 4-feature behavioral subset chosen to mirror the Phase-3 selected_4 shape
# (cost, tokens-per-unit-work, a loop/thrash count, a binary state flag).
BEH_4 = ["beh_cost_usd", "beh_tokens_per_word", "beh_revision_count", "beh_abstain"]

# Gold-program structural = TASK-side difficulty proxy (not agent behavior).
PROG_FEATURES = [
    "prog_op_count", "prog_op_diversity", "prog_has_chain", "prog_uses_const",
    "prog_uses_table_op", "prog_sign_flip", "prog_magnitude_blowup",
    "prog_max_abs_interm",
]

# Skill-verifier (E_fin1) signals for the head-to-head.
SKILL_FEATURES = [
    "skill_conf_rating", "skill_adv_lc_count", "skill_adv_li_count",
    "skill_adv_unc_count", "skill_adv_confident",
]


def rf_oof(df, features, label="gold_pass", n_splits=5):
    sub = df.dropna(subset=[label]).copy()
    X = sub[features].values.astype(float)
    y = sub[label].astype(int).values
    X = np.nan_to_num(X, nan=-999)
    minority = min((y == 0).sum(), (y == 1).sum())
    splits = min(n_splits, minority) if minority >= 2 else 0
    if splits < 2:
        return None
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RNG)
    yt, yp = [], []
    last = None
    for tr, va in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=200, max_depth=7,
                                   class_weight="balanced", random_state=RNG)
        m.fit(X[tr], y[tr])
        yp.extend(m.predict_proba(X[va])[:, 1])
        yt.extend(y[va])
        last = m
    yt, yp = np.array(yt), np.array(yp)
    imp = dict(zip(features, last.feature_importances_))
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
            "n_boot_valid": len(boots)}


def precision_at_recall(y_true, y_prob, min_recall=0.30):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    valid = r >= min_recall
    return float(np.max(p[valid])) if valid.any() else 0.0


def fit_report(df, features, name):
    res = rf_oof(df, features)
    if not res:
        return {"name": name, "auc": None, "note": "too few of one class to CV"}
    yt, yp, imp, splits = res
    r = auc_with_ci(yt, yp)
    if r is None:
        return {"name": name, "auc": None, "note": "single-class OOF"}
    r["name"] = name
    r["features"] = features
    r["p_at_r30"] = round(precision_at_recall(yt, yp), 4)
    r["importances"] = {k: round(float(v), 4)
                        for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    r["n"] = int(len(yt))
    r["cv_splits"] = splits
    return r


def forward_select(df, candidate, max_features=6):
    sub = df.dropna(subset=["gold_pass"])
    if sub.gold_pass.nunique() < 2 or min((sub.gold_pass == 0).sum(),
                                          (sub.gold_pass == 1).sum()) < 3:
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
        history.append({"step": step + 1, "added": best_feat,
                        "auc": round(best_auc, 4)})
        best = best_auc
        if len(history) >= 3 and history[-1]["auc"] <= history[-3]["auc"]:
            break
    return {"selected": selected, "history": history, "n": int(len(sub))}


def univariate_table(df, features, label="gold_pass"):
    """Per-feature pass/fail mean + single-feature AUC (direction & strength)."""
    out = {}
    yp = df[label].values
    passmask = df[label] == 1
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        valid = s.notna()
        if valid.sum() < 4:
            continue
        sv = s[valid]
        yv = df.loc[valid, label].values
        try:
            a = roc_auc_score(yv, sv.values)
        except ValueError:
            a = None
        out[f] = {
            "pass_mean": round(float(s[passmask].mean()), 4),
            "fail_mean": round(float(s[~passmask].mean()), 4),
            "auc_single": round(float(a), 4) if a is not None else None,
        }
    return out


def main():
    df = pd.read_csv(FEATURES)
    out = {
        "_meta": {
            "spec": "E_fin2",
            "corpus": "FinQA czyssrs/FinQA dev (n=100, E_fin1 sample, seed=42)",
            "rf_recipe": "RF(n_estimators=200, max_depth=7, class_weight=balanced, "
                         "seed=42); nan->-999; 5-fold stratified pooled OOF",
            "phase3_baseline": PHASE3_BASELINE,
            "n": int(len(df)),
            "base_rate": round(float(df.gold_pass.mean()), 4),
            "n_pass": int(df.gold_pass.sum()),
            "n_fail": int((1 - df.gold_pass).sum()),
            "env": "python3.13 sklearn 1.8.0 (python3.14 sklearn broken)",
            "feature_family_note": (
                "beh_* = agent-trajectory process signal (direct Phase-3 analog); "
                "prog_* = gold-program structure (task-side difficulty proxy, NOT "
                "agent behavior)."),
        }
    }

    # ── (1) RF AUC: behavioral, program-structural, skill, and combinations ──
    rf = {}
    rf["behavioral_all"] = fit_report(df, BEH_FEATURES, "behavioral_all (9 agent-trajectory feats)")
    rf["behavioral_4"] = fit_report(df, BEH_4, "behavioral_4 (cost/tok-per-word/revision/abstain)")
    rf["program_structural"] = fit_report(df, PROG_FEATURES, "program_structural (8 gold-program feats)")
    rf["program_plus_behavioral"] = fit_report(df, PROG_FEATURES + BEH_FEATURES, "program+behavioral")
    out["rf_auc"] = rf

    # ── (2) Univariate feature table (direction + single-feature AUC) ──
    out["univariate"] = univariate_table(df, PROG_FEATURES + BEH_FEATURES)

    # ── (3) Forward selection over all process features ──
    candidate = PROG_FEATURES + BEH_FEATURES
    out["forward_selection"] = forward_select(df, candidate, max_features=6)

    # ── (4) Head-to-head: behavioral vs skill-verifier (E_fin1) vs combined ──
    has_skill = all(c in df.columns for c in SKILL_FEATURES)
    h2h = {}
    h2h["behavioral_only"] = fit_report(df, BEH_FEATURES, "behavioral_only")
    h2h["program_only"] = fit_report(df, PROG_FEATURES, "program_only")
    if has_skill:
        h2h["skill_verifier_only"] = fit_report(df, SKILL_FEATURES, "skill_verifier_only (E_fin1 conf+adv)")
        h2h["behavioral_plus_skill"] = fit_report(df, BEH_FEATURES + SKILL_FEATURES, "behavioral+skill")
        h2h["all_combined"] = fit_report(df, PROG_FEATURES + BEH_FEATURES + SKILL_FEATURES, "all_combined")
        # Also: the raw E_fin1 adversarial confident flag as a standalone classifier (its AUC).
        try:
            adv_auc = roc_auc_score(df.gold_pass.values, df.skill_adv_lc_count.values)
            conf_auc = roc_auc_score(df.gold_pass.values, df.skill_conf_rating.values)
            h2h["raw_signal_auc"] = {
                "adv_lc_count_auc": round(float(adv_auc), 4),
                "conf_rating_auc": round(float(conf_auc), 4),
                "note": "single-signal AUC of the E_fin1 verifier outputs (no RF)",
            }
        except ValueError:
            pass
    out["head_to_head"] = h2h

    # ── (5) Difficulty/strategy paradox — DOCUMENT ONLY (do not re-run conditioning) ──
    # Coding prior (enew_report.md): difficulty-conditioning regressed AUC
    # 0.756 -> 0.743; the cross-sectional reversal was about agent strategy, not
    # difficulty, and difficulty variance was low. FinQA difficulty variance is
    # even lower (program op-count: 56% are 1-op). We stratify and REPORT whether
    # any feature reverses sign by difficulty bucket; we do NOT fit a
    # difficulty-conditioned RF as a fix.
    df["difficulty_bucket"] = np.where(df.prog_op_count <= 1, "easy_1op", "hard_2plus")
    diff = {"buckets": {}, "method": "stratify by gold-program op-count (<=1 vs >=2); "
                                     "DOCUMENT sign reversals only, no conditioned RF (regressed in coding)"}
    for b, g in df.groupby("difficulty_bucket"):
        diff["buckets"][b] = {
            "n": int(len(g)),
            "pass_rate": round(float(g.gold_pass.mean()), 4),
            "n_pass": int(g.gold_pass.sum()),
            "n_fail": int((1 - g.gold_pass).sum()),
        }
    # Sign-reversal check on the behavioral-4 features (pass-fail delta per bucket)
    reversal = {}
    for f in BEH_4 + ["prog_op_count", "beh_output_tokens"]:
        per = {}
        for b, g in df.groupby("difficulty_bucket"):
            if g.gold_pass.nunique() < 2:
                per[b] = None
                continue
            d = float(g.loc[g.gold_pass == 1, f].mean() - g.loc[g.gold_pass == 0, f].mean())
            per[b] = round(d, 4)
        # pooled delta
        dp = float(df.loc[df.gold_pass == 1, f].mean() - df.loc[df.gold_pass == 0, f].mean())
        vals = [v for v in per.values() if v is not None]
        paradox = (len(vals) == 2 and (vals[0] > 0) != (vals[1] > 0)
                   and abs(vals[0]) > 1e-9 and abs(vals[1]) > 1e-9)
        reversal[f] = {"pooled_pass_minus_fail": round(dp, 4),
                       "per_bucket_pass_minus_fail": per,
                       "sign_reversal": bool(paradox)}
    diff["sign_reversal_check"] = reversal
    diff["paradox_detected"] = any(v["sign_reversal"] for v in reversal.values())
    out["difficulty_paradox"] = diff

    # ── (6) Verdict ──
    beh_auc = rf["behavioral_all"]["auc"]
    prog_auc = rf["program_structural"]["auc"]
    skill_auc = h2h.get("skill_verifier_only", {}).get("auc")
    best_process = max([a for a in (beh_auc, prog_auc) if a is not None], default=None)
    out["verdict"] = {
        "behavioral_all_auc": beh_auc,
        "behavioral_all_ci95": rf["behavioral_all"].get("ci95"),
        "program_structural_auc": prog_auc,
        "skill_verifier_auc": skill_auc,
        "coding_baseline_auc": PHASE3_BASELINE["selected_4_rf_auc"],
        "gap_vs_coding_baseline": round(0.756 - best_process, 4) if best_process is not None else None,
        "expected_band_055_065": "behavioral_all" if (beh_auc is not None and 0.55 <= beh_auc <= 0.65) else "outside",
        "scoping": None,  # filled by report
    }

    (RESULTS / "rf_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", RESULTS / "rf_results.json")
    # Console summary
    print(f"\nn={out['_meta']['n']} base_rate={out['_meta']['base_rate']} "
          f"({out['_meta']['n_pass']} pass / {out['_meta']['n_fail']} fail)")
    print(f"\nCoding baseline (Phase-3 selected_4 RF): AUC 0.756")
    print(f"  behavioral_all  AUC = {beh_auc}  CI {rf['behavioral_all'].get('ci95')}")
    print(f"  behavioral_4    AUC = {rf['behavioral_4']['auc']}")
    print(f"  program_struct  AUC = {prog_auc}  CI {rf['program_structural'].get('ci95')}")
    print(f"  prog+behavioral AUC = {rf['program_plus_behavioral']['auc']}")
    if skill_auc is not None:
        print(f"\nHead-to-head:")
        print(f"  behavioral_only AUC = {h2h['behavioral_only']['auc']}")
        print(f"  skill_verifier  AUC = {skill_auc}")
        print(f"  behavioral+skill AUC = {h2h['behavioral_plus_skill']['auc']}")
        print(f"  all_combined    AUC = {h2h['all_combined']['auc']}")
        print(f"  raw adv_lc_count AUC = {h2h.get('raw_signal_auc',{}).get('adv_lc_count_auc')}")
    print(f"\nForward-selected: {out['forward_selection'].get('selected')}")
    print(f"Difficulty paradox detected: {out['difficulty_paradox']['paradox_detected']}")


if __name__ == "__main__":
    main()
