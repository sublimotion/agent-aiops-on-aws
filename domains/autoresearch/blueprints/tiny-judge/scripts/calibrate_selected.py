#!/usr/bin/env python3
"""
Phase 4b: Apply Platt scaling to the 5-feature RF model and save both models.

The 5-feature RF has ECE=0.108 (marginally above 0.1 target).
Platt scaling (isotonic regression) should improve calibration.

Reads: results/features.csv
Writes: results/models/selected_rf.pkl, results/calibration_selected.json
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, brier_score_loss,
)

import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
SELECTED_FEATURES = ["total_cost_usd", "loop_count", "action_pct_search", "patch_len", "first_edit_pct"]


def precision_at_recall(y_true, y_prob, min_recall):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid = recalls >= min_recall
    if not valid.any():
        return 0.0, 1.0
    best_idx = np.argmax(precisions[valid])
    indices = np.where(valid)[0]
    idx = indices[best_idx]
    return float(precisions[idx]), float(thresholds[idx]) if idx < len(thresholds) else 0.5


def compute_ece(y_true, y_prob, n_bins=10, strategy="uniform"):
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0.0
        bins[-1] = 1.0
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def evaluate_cv(X, y, calibrate=False, method="isotonic"):
    """5-fold stratified CV with optional calibration."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_f = np.nan_to_num(X_train, nan=-999)
        X_val_f = np.nan_to_num(X_val, nan=-999)

        base_model = RandomForestClassifier(
            n_estimators=200, max_depth=7,
            class_weight="balanced", random_state=42,
        )

        if calibrate:
            # CalibratedClassifierCV does internal CV for calibration
            model = CalibratedClassifierCV(base_model, method=method, cv=3)
        else:
            model = base_model

        model.fit(X_train_f, y_train)
        y_prob = model.predict_proba(X_val_f)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        auc = 0.5

    pr30, _ = precision_at_recall(all_y_true, all_y_prob, 0.30)
    pr50, _ = precision_at_recall(all_y_true, all_y_prob, 0.50)
    ece_u = compute_ece(all_y_true, all_y_prob, strategy="uniform")
    ece_q = compute_ece(all_y_true, all_y_prob, strategy="quantile")
    brier = brier_score_loss(all_y_true, all_y_prob)

    prob_true, prob_pred = calibration_curve(all_y_true, all_y_prob, n_bins=10, strategy="uniform")

    return {
        "auc": auc,
        "f1": f1_score(all_y_true, all_y_pred),
        "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "pr30": pr30,
        "pr50": pr50,
        "ece_uniform": ece_u,
        "ece_quantile": ece_q,
        "brier": brier,
        "calibration": {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred],
        },
    }


def main():
    df = pd.read_csv(BASE / "results" / "features.csv")
    y = df["gold_pass"].values
    feats = [f for f in SELECTED_FEATURES if f in df.columns]
    X = df[feats].values
    print(f"Loaded {len(df)} instances, {len(feats)} features: {feats}")

    results = {}

    # Uncalibrated baseline
    print("\n═══ Uncalibrated RF (5 features) ═══")
    r = evaluate_cv(X, y, calibrate=False)
    results["uncalibrated"] = r
    print(f"  AUC={r['auc']:.3f} P@R30={r['pr30']:.3f} ECE={r['ece_uniform']:.3f} Brier={r['brier']:.3f}")

    # Platt scaling (sigmoid)
    print("\n═══ Platt Scaling (sigmoid) ═══")
    r = evaluate_cv(X, y, calibrate=True, method="sigmoid")
    results["platt_sigmoid"] = r
    print(f"  AUC={r['auc']:.3f} P@R30={r['pr30']:.3f} ECE={r['ece_uniform']:.3f} Brier={r['brier']:.3f}")

    # Isotonic calibration
    print("\n═══ Isotonic Calibration ═══")
    r = evaluate_cv(X, y, calibrate=True, method="isotonic")
    results["isotonic"] = r
    print(f"  AUC={r['auc']:.3f} P@R30={r['pr30']:.3f} ECE={r['ece_uniform']:.3f} Brier={r['brier']:.3f}")

    # Save results
    out_path = BASE / "results" / "calibration_selected.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")

    # Train and save final models on full data
    print("\n═══ Saving Final Models ═══")
    X_full = np.nan_to_num(X, nan=-999)

    # Uncalibrated (best P@R30)
    model_uncal = RandomForestClassifier(
        n_estimators=200, max_depth=7,
        class_weight="balanced", random_state=42,
    )
    model_uncal.fit(X_full, y)

    # Calibrated (best ECE)
    best_cal_method = min(
        ["platt_sigmoid", "isotonic"],
        key=lambda m: results[m]["ece_uniform"],
    )
    print(f"  Best calibration method: {best_cal_method}")

    model_cal = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=200, max_depth=7,
            class_weight="balanced", random_state=42,
        ),
        method="isotonic" if "isotonic" in best_cal_method else "sigmoid",
        cv=5,
    )
    model_cal.fit(X_full, y)

    models_dir = BASE / "results" / "models"
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / "selected_rf_uncalibrated.pkl", "wb") as f:
        pickle.dump({"model": model_uncal, "features": feats}, f)
    with open(models_dir / "selected_rf_calibrated.pkl", "wb") as f:
        pickle.dump({"model": model_cal, "features": feats, "method": best_cal_method}, f)

    print(f"  Saved: selected_rf_uncalibrated.pkl, selected_rf_calibrated.pkl")

    # Summary
    print("\n═══ Summary ═══")
    print(f"  Uncalibrated: P@R30={results['uncalibrated']['pr30']:.3f}, ECE={results['uncalibrated']['ece_uniform']:.3f}")
    print(f"  Platt:        P@R30={results['platt_sigmoid']['pr30']:.3f}, ECE={results['platt_sigmoid']['ece_uniform']:.3f}")
    print(f"  Isotonic:     P@R30={results['isotonic']['pr30']:.3f}, ECE={results['isotonic']['ece_uniform']:.3f}")

    best = min(results.items(), key=lambda x: x[1]["ece_uniform"])
    print(f"\n  Best ECE: {best[0]} = {best[1]['ece_uniform']:.3f}")
    if best[1]["ece_uniform"] < 0.1:
        print(f"  ECE < 0.1: PASS")
    else:
        print(f"  ECE < 0.1: FAIL (but {best[1]['ece_uniform']:.3f} is close)")


if __name__ == "__main__":
    main()
