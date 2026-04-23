#!/usr/bin/env python3
"""
Phase 2b: Feature selection to maximize P@R≥30%.

The full model (36 features) has lower P@R≥30% than behavioral_only (16 features),
suggesting noisy features hurt precision. This script finds the optimal feature subset.

Reads: results/features.csv
Writes: results/selection_results.json (appended to training_results.json)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE = Path(__file__).resolve().parent.parent


def precision_at_recall(y_true, y_prob, min_recall):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid = recalls >= min_recall
    if not valid.any():
        return 0.0, 1.0
    best_idx = np.argmax(precisions[valid])
    indices = np.where(valid)[0]
    idx = indices[best_idx]
    return float(precisions[idx]), float(thresholds[idx]) if idx < len(thresholds) else 0.5


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def evaluate_features(X, y, feature_names, model_name="RandomForest", n_splits=5):
    """Quick 5-fold CV evaluation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_prob = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_f = np.nan_to_num(X_train, nan=-999)
        X_val_f = np.nan_to_num(X_val, nan=-999)

        if model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=200, max_depth=7,
                class_weight="balanced", random_state=42,
            )
        elif model_name == "XGBoost" and HAS_XGBOOST:
            scale_pos = sum(y_train == 0) / max(sum(y_train == 1), 1)
            model = XGBClassifier(
                max_depth=5, n_estimators=150, learning_rate=0.1,
                scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False,
            )
        else:
            return None

        model.fit(X_train_f, y_train)
        y_prob = model.predict_proba(X_val_f)[:, 1]
        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = (all_y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        auc = 0.5

    pr30, _ = precision_at_recall(all_y_true, all_y_prob, 0.30)
    pr50, _ = precision_at_recall(all_y_true, all_y_prob, 0.50)
    ece = compute_ece(all_y_true, all_y_prob)
    brier = brier_score_loss(all_y_true, all_y_prob)

    return {
        "features": list(feature_names),
        "n_features": len(feature_names),
        "auc": auc,
        "f1": f1_score(all_y_true, all_y_pred),
        "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "pr30": pr30,
        "pr50": pr50,
        "ece": ece,
        "brier": brier,
    }


def main():
    df = pd.read_csv(BASE / "results" / "features.csv")
    y = df["gold_pass"].values
    print(f"Loaded {len(df)} instances")

    # Top features from RF importance (ordered)
    top_features = [
        "elapsed_s", "total_cost_usd", "action_entropy", "context_growth_rate",
        "review_score_max", "action_pct_bash", "total_actions", "review_score_mean",
        "action_pct_search", "patch_len", "diff_size_chars", "loop_count",
        "action_pct_edit", "v009_confidence", "first_edit_pct",
    ]

    # Filter to available features
    top_features = [f for f in top_features if f in df.columns]

    results = {}

    # 1. Forward feature selection: add features one at a time
    print("\n═══ Forward Feature Selection (RF) ═══")
    best_subset = []
    best_pr30 = 0
    forward_trace = []

    for i in range(min(len(top_features), 15)):
        best_next = None
        best_next_result = None

        for feat in top_features:
            if feat in best_subset:
                continue
            trial = best_subset + [feat]
            X = df[trial].values
            result = evaluate_features(X, y, trial, "RandomForest")
            if result and result["pr30"] > (best_next_result["pr30"] if best_next_result else 0):
                best_next = feat
                best_next_result = result

        if best_next is None:
            break

        best_subset.append(best_next)
        forward_trace.append({
            "step": i + 1,
            "added": best_next,
            "features": list(best_subset),
            **best_next_result,
        })

        marker = " ***" if best_next_result["pr30"] > best_pr30 else ""
        if best_next_result["pr30"] > best_pr30:
            best_pr30 = best_next_result["pr30"]

        print(f"  +{best_next:25s} → AUC={best_next_result['auc']:.3f} "
              f"P@R30={best_next_result['pr30']:.3f} "
              f"ECE={best_next_result['ece']:.3f}{marker}")

    results["forward_selection_rf"] = forward_trace

    # Find the peak P@R30 step
    peak_step = max(forward_trace, key=lambda x: x["pr30"])
    print(f"\n  Peak P@R30: {peak_step['pr30']:.3f} at {peak_step['step']} features: {peak_step['features']}")

    # 2. Try the same with XGBoost
    if HAS_XGBOOST:
        print("\n═══ Forward Feature Selection (XGBoost) ═══")
        best_subset_xgb = []
        best_pr30_xgb = 0
        forward_trace_xgb = []

        for i in range(min(len(top_features), 15)):
            best_next = None
            best_next_result = None

            for feat in top_features:
                if feat in best_subset_xgb:
                    continue
                trial = best_subset_xgb + [feat]
                X = df[trial].values
                result = evaluate_features(X, y, trial, "XGBoost")
                if result and result["pr30"] > (best_next_result["pr30"] if best_next_result else 0):
                    best_next = feat
                    best_next_result = result

            if best_next is None:
                break

            best_subset_xgb.append(best_next)
            forward_trace_xgb.append({
                "step": i + 1,
                "added": best_next,
                "features": list(best_subset_xgb),
                **best_next_result,
            })

            marker = " ***" if best_next_result["pr30"] > best_pr30_xgb else ""
            if best_next_result["pr30"] > best_pr30_xgb:
                best_pr30_xgb = best_next_result["pr30"]

            print(f"  +{best_next:25s} → AUC={best_next_result['auc']:.3f} "
                  f"P@R30={best_next_result['pr30']:.3f} "
                  f"ECE={best_next_result['ece']:.3f}{marker}")

        results["forward_selection_xgb"] = forward_trace_xgb

        peak_xgb = max(forward_trace_xgb, key=lambda x: x["pr30"])
        print(f"\n  Peak P@R30: {peak_xgb['pr30']:.3f} at {peak_xgb['step']} features: {peak_xgb['features']}")

    # 3. Behavioral + v009 hybrid (add v009 to behavioral set)
    print("\n═══ Behavioral + v009 Hybrid ═══")
    behavioral = [
        "diff_size_chars", "files_modified", "num_turns", "total_cost_usd",
        "elapsed_s", "total_actions", "action_pct_edit", "action_pct_search",
        "action_pct_bash", "first_edit_pct", "first_edit_action",
        "loop_count", "action_entropy", "parkinson_ratio",
        "context_growth_rate", "patch_len",
    ]
    behavioral = [f for f in behavioral if f in df.columns]

    for extra_set_name, extra_features in [
        ("behavioral_only", []),
        ("behavioral+v009", ["v009_verdict", "v009_confidence"]),
        ("behavioral+review", ["review_score_mean", "review_score_max"]),
        ("behavioral+v009+review", ["v009_verdict", "v009_confidence", "review_score_mean", "review_score_max"]),
    ]:
        feats = behavioral + [f for f in extra_features if f in df.columns]
        X = df[feats].values
        r = evaluate_features(X, y, feats, "RandomForest")
        results[f"hybrid_rf_{extra_set_name}"] = r
        print(f"  {extra_set_name:30s}: AUC={r['auc']:.3f} P@R30={r['pr30']:.3f} ECE={r['ece']:.3f}")

        if HAS_XGBOOST:
            r2 = evaluate_features(X, y, feats, "XGBoost")
            results[f"hybrid_xgb_{extra_set_name}"] = r2
            print(f"  {extra_set_name+' (XGB)':30s}: AUC={r2['auc']:.3f} P@R30={r2['pr30']:.3f} ECE={r2['ece']:.3f}")

    # Save results
    out_path = BASE / "results" / "selection_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    # Summary
    print("\n═══ Summary ═══")
    all_configs = []
    for k, v in results.items():
        if isinstance(v, list):
            for item in v:
                all_configs.append((f"{k}_step{item['step']}", item))
        elif isinstance(v, dict) and "pr30" in v:
            all_configs.append((k, v))

    # Sort by P@R30
    all_configs.sort(key=lambda x: x[1]["pr30"], reverse=True)
    print(f"\n  Top 5 configs by P@R≥30%:")
    for name, cfg in all_configs[:5]:
        print(f"    {name:45s}: P@R30={cfg['pr30']:.3f} AUC={cfg['auc']:.3f} ECE={cfg['ece']:.3f}")

    # Check if any beat 0.85
    above_85 = [(n, c) for n, c in all_configs if c["pr30"] >= 0.85]
    if above_85:
        print(f"\n  {len(above_85)} configs achieve P@R≥30% ≥ 0.85!")
        for name, cfg in above_85:
            print(f"    {name}: P@R30={cfg['pr30']:.3f}")
    else:
        print(f"\n  No config achieves P@R≥30% ≥ 0.85 (best: {all_configs[0][1]['pr30']:.3f})")


if __name__ == "__main__":
    main()
