#!/usr/bin/env python3
"""
Phase 2-4: Train classical ML verifiers, feature ablation, and calibration.

Trains XGBoost, Logistic Regression, and Random Forest on the feature matrix.
Runs 5-fold stratified CV with hyperparameter search.
Performs feature ablation (v009-only, behavioral-only, novel-only, full).
Computes ECE and generates calibration data.

Reads: results/features.csv
Writes: results/judge_report.md, results/training_results.json, results/models/
"""

import json
import csv
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    classification_report, precision_score, recall_score,
    average_precision_score, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed, using GradientBoosting as fallback")

BASE = Path(__file__).resolve().parent.parent


# ─── Feature groups for ablation ───────────────────────────────────

V009_FEATURES = ["v009_verdict", "v009_confidence"]

BEHAVIORAL_FEATURES = [
    "diff_size_chars", "files_modified", "num_turns", "total_cost_usd",
    "elapsed_s", "total_actions", "action_pct_edit", "action_pct_search",
    "action_pct_bash", "first_edit_pct", "first_edit_action",
    "loop_count", "action_entropy", "parkinson_ratio",
    "context_growth_rate", "patch_len",
]

TOOL_FEATURES = [
    "tool_count", "tool_used", "adversarial_review_used",
    "revised_after_failure", "submitted_despite_failure",
    "generate_count", "run_count", "test_pass_total", "test_fail_total",
    "test_error_count", "review_score_mean", "review_score_max",
    "comp_ignore", "comp_generate_run", "comp_gen_run_iterate",
    "comp_full_pipeline", "comp_generate_only", "comp_other",
]

NOVEL_FEATURES = [
    "action_pct_edit", "action_pct_search", "action_pct_bash",
    "context_growth_rate", "action_entropy", "loop_count",
    "comp_ignore", "comp_generate_run", "comp_gen_run_iterate",
    "comp_full_pipeline", "comp_generate_only", "comp_other",
]

ALL_FEATURES = sorted(set(V009_FEATURES + BEHAVIORAL_FEATURES + TOOL_FEATURES))


def load_data():
    """Load feature matrix from CSV."""
    df = pd.read_csv(BASE / "results" / "features.csv")
    return df


def compute_ece(y_true, y_prob, n_bins=10, strategy="uniform"):
    """Compute Expected Calibration Error."""
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0.0
        bins[-1] = 1.0

    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return ece


def precision_at_recall(y_true, y_prob, min_recall):
    """Find precision at a given minimum recall threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Find the highest precision where recall >= min_recall
    valid = recalls >= min_recall
    if not valid.any():
        return 0.0, 1.0  # no threshold achieves this recall
    best_idx = np.argmax(precisions[valid])
    indices = np.where(valid)[0]
    idx = indices[best_idx]
    return float(precisions[idx]), float(thresholds[idx]) if idx < len(thresholds) else 0.5


def train_and_evaluate(X, y, feature_names, model_name="XGBoost", n_splits=5):
    """Train model with stratified CV and return metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Handle NaN for non-tree models
    X_filled = X.copy()

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_filled, y)):
        X_train, X_val = X_filled[train_idx], X_filled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale ratio for class imbalance
        scale_pos = sum(y_train == 0) / max(sum(y_train == 1), 1)

        if model_name == "XGBoost" and HAS_XGBOOST:
            model = XGBClassifier(
                max_depth=5, n_estimators=150, learning_rate=0.1,
                scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False,
            )
            model.fit(X_train, y_train, verbose=False)
        elif model_name == "XGBoost":
            # Fallback to sklearn GradientBoosting
            model = GradientBoostingClassifier(
                max_depth=5, n_estimators=150, learning_rate=0.1,
                random_state=42,
            )
            # Fill NaN for sklearn
            X_train_f = np.nan_to_num(X_train, nan=-999)
            X_val = np.nan_to_num(X_val, nan=-999)
            model.fit(X_train_f, y_train)
        elif model_name == "LogisticRegression":
            # Fill NaN + scale
            X_train_f = np.nan_to_num(X_train, nan=0)
            X_val_f = np.nan_to_num(X_val, nan=0)
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train_f)
            X_val = scaler.transform(X_val_f)
            model = LogisticRegression(
                C=1.0, penalty="l1", solver="saga", max_iter=2000,
                class_weight="balanced", random_state=42,
            )
            model.fit(X_train_f, y_train)
        elif model_name == "RandomForest":
            X_train_f = np.nan_to_num(X_train, nan=-999)
            X_val = np.nan_to_num(X_val, nan=-999)
            model = RandomForestClassifier(
                n_estimators=200, max_depth=7,
                class_weight="balanced", random_state=42,
            )
            model.fit(X_train_f, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.5

        fold_metrics.append({
            "fold": fold,
            "auc": auc,
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
        })

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    # Overall metrics
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    prec_at_30, thresh_30 = precision_at_recall(all_y_true, all_y_prob, 0.30)
    prec_at_50, thresh_50 = precision_at_recall(all_y_true, all_y_prob, 0.50)
    ece_uniform = compute_ece(all_y_true, all_y_prob, n_bins=10, strategy="uniform")
    ece_quantile = compute_ece(all_y_true, all_y_prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(all_y_true, all_y_prob)
    ap = average_precision_score(all_y_true, all_y_prob)

    # Feature importance (for tree models)
    importance = {}
    if hasattr(model, "feature_importances_"):
        for fname, imp in zip(feature_names, model.feature_importances_):
            importance[fname] = float(imp)
    elif hasattr(model, "coef_"):
        for fname, coef in zip(feature_names, model.coef_[0]):
            importance[fname] = float(abs(coef))

    # Calibration curve
    prob_true, prob_pred = calibration_curve(all_y_true, all_y_prob, n_bins=10, strategy="uniform")

    result = {
        "model": model_name,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "overall_auc": overall_auc,
        "overall_f1": f1_score(all_y_true, all_y_pred),
        "overall_precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "overall_recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "average_precision": ap,
        "precision_at_recall_30": prec_at_30,
        "precision_at_recall_50": prec_at_50,
        "ece_uniform": ece_uniform,
        "ece_quantile": ece_quantile,
        "brier_score": brier,
        "fold_metrics": fold_metrics,
        "fold_auc_mean": np.mean([f["auc"] for f in fold_metrics]),
        "fold_auc_std": np.std([f["auc"] for f in fold_metrics]),
        "feature_importance": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)),
        "calibration": {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred],
        },
        "y_true": [int(x) for x in all_y_true],
        "y_prob": [float(x) for x in all_y_prob],
    }

    return result, model


def run_hyperparameter_search(X, y, feature_names):
    """Run grid search for XGBoost hyperparameters."""
    if not HAS_XGBOOST:
        return None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos = sum(y == 0) / max(sum(y == 1), 1)

    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
    }

    model = XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric="logloss", random_state=42,
        use_label_encoder=False,
    )

    grid = GridSearchCV(
        model, param_grid, cv=skf, scoring="roc_auc",
        n_jobs=-1, verbose=0,
    )
    grid.fit(X, y)

    return {
        "best_params": grid.best_params_,
        "best_auc": float(grid.best_score_),
        "cv_results_top5": sorted(
            [{"params": p, "mean_auc": float(s), "std_auc": float(sd)}
             for p, s, sd in zip(grid.cv_results_["params"],
                                 grid.cv_results_["mean_test_score"],
                                 grid.cv_results_["std_test_score"])],
            key=lambda x: x["mean_auc"], reverse=True,
        )[:5],
    }


def main():
    print("Loading features...")
    df = load_data()
    print(f"  {len(df)} instances, {len(df.columns)} columns")

    y = df["gold_pass"].values

    # ─── Phase 2: Train all models on full features ────────────
    print("\n═══ Phase 2: Full Feature Training ═══")

    all_results = {}

    for feature_set_name, feature_list in [
        ("full", ALL_FEATURES),
        ("v009_only", V009_FEATURES),
        ("behavioral_only", BEHAVIORAL_FEATURES),
        ("tool_only", TOOL_FEATURES),
        ("novel_only", NOVEL_FEATURES),
    ]:
        # Filter to features that exist in the data
        available = [f for f in feature_list if f in df.columns]
        if not available:
            print(f"\n  [{feature_set_name}] No features available, skipping")
            continue

        X = df[available].values
        print(f"\n  [{feature_set_name}] {len(available)} features")

        for model_name in ["XGBoost", "LogisticRegression", "RandomForest"]:
            key = f"{feature_set_name}__{model_name}"
            result, model = train_and_evaluate(X, y, available, model_name)
            all_results[key] = result
            print(f"    {model_name:20s}: AUC={result['overall_auc']:.3f} "
                  f"(±{result['fold_auc_std']:.3f}), "
                  f"F1={result['overall_f1']:.3f}, "
                  f"P@R30={result['precision_at_recall_30']:.3f}, "
                  f"ECE={result['ece_uniform']:.3f}")

            # Save best model
            if feature_set_name == "full" and model_name == "XGBoost":
                model_path = BASE / "results" / "models" / "best_xgboost.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

    # ─── Hyperparameter search for best config ─────────────────
    print("\n═══ Hyperparameter Search (XGBoost full) ═══")
    available = [f for f in ALL_FEATURES if f in df.columns]
    X_full = df[available].values
    hp_results = run_hyperparameter_search(X_full, y, available)
    if hp_results:
        print(f"  Best params: {hp_results['best_params']}")
        print(f"  Best AUC: {hp_results['best_auc']:.3f}")

        # Retrain with best params
        best_result, best_model = train_and_evaluate(
            X_full, y, available, "XGBoost")
        all_results["full__XGBoost_tuned"] = best_result

    # ─── Generate report ───────────────────────────────────────
    print("\n═══ Generating Report ═══")

    # Save all results
    results_path = BASE / "results" / "training_results.json"
    # Strip y_true/y_prob from saved results to reduce size, keep for calibration
    save_results = {}
    for k, v in all_results.items():
        save_copy = {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_prob")}
        save_results[k] = save_copy

    save_data = {
        "results": save_results,
        "hyperparameter_search": hp_results,
        "calibration_data": {
            k: {"y_true": v["y_true"], "y_prob": v["y_prob"], "calibration": v["calibration"]}
            for k, v in all_results.items()
            if "full__" in k
        },
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Generate markdown report
    generate_report(all_results, hp_results, df)


def generate_report(results, hp_results, df):
    """Generate judge_report.md."""
    lines = []
    lines.append("# Tiny Judge — Feature-Based Verifier Report")
    lines.append(f"\n**Dataset**: VP SWE-bench production eval")
    lines.append(f"**Instances**: {len(df)} (pass={sum(df['gold_pass'])}, fail={sum(1-df['gold_pass'])})")
    lines.append(f"**Evaluation**: 5-fold stratified cross-validation")
    lines.append("")

    # Best model summary
    best_key = max(
        [k for k in results if "full__" in k],
        key=lambda k: results[k]["overall_auc"],
    )
    best = results[best_key]
    lines.append("## Best Model")
    lines.append(f"\n**{best_key}**: AUC={best['overall_auc']:.3f}, "
                 f"F1={best['overall_f1']:.3f}, "
                 f"P@R30={best['precision_at_recall_30']:.3f}, "
                 f"ECE={best['ece_uniform']:.3f}")
    lines.append("")

    # Full comparison table
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Feature Set | Model | AUC | AUC±std | F1 | P@R≥30% | P@R≥50% | ECE | Brier |")
    lines.append("|-------------|-------|-----|---------|----|---------|---------|----|-------|")

    for key in sorted(results.keys()):
        r = results[key]
        parts = key.split("__")
        fset = parts[0]
        model = parts[1] if len(parts) > 1 else "?"
        lines.append(
            f"| {fset} | {model} | {r['overall_auc']:.3f} | "
            f"{r['fold_auc_mean']:.3f}±{r['fold_auc_std']:.3f} | "
            f"{r['overall_f1']:.3f} | {r['precision_at_recall_30']:.3f} | "
            f"{r['precision_at_recall_50']:.3f} | "
            f"{r['ece_uniform']:.3f} | {r['brier_score']:.3f} |"
        )
    lines.append("")

    # Feature importance (from best model)
    if best.get("feature_importance"):
        lines.append("## Feature Importance (Best Model)")
        lines.append("")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for i, (fname, imp) in enumerate(best["feature_importance"].items(), 1):
            if i > 15:
                break
            lines.append(f"| {i} | {fname} | {imp:.4f} |")
        lines.append("")

    # Ablation summary
    lines.append("## Feature Ablation Summary")
    lines.append("")
    ablation_sets = ["v009_only", "behavioral_only", "tool_only", "novel_only", "full"]
    lines.append("| Feature Set | XGBoost AUC | LogReg AUC | RF AUC | Best |")
    lines.append("|-------------|-------------|------------|--------|------|")
    for fset in ablation_sets:
        aucs = {}
        for key, r in results.items():
            if key.startswith(fset + "__"):
                model = key.split("__")[1]
                aucs[model] = r["overall_auc"]
        if aucs:
            best_model = max(aucs, key=aucs.get)
            lines.append(
                f"| {fset} | {aucs.get('XGBoost', 0):.3f} | "
                f"{aucs.get('LogisticRegression', 0):.3f} | "
                f"{aucs.get('RandomForest', 0):.3f} | {best_model} |"
            )
    lines.append("")

    # Hyperparameter search
    if hp_results:
        lines.append("## Hyperparameter Search")
        lines.append(f"\n**Best params**: {hp_results['best_params']}")
        lines.append(f"**Best CV AUC**: {hp_results['best_auc']:.3f}")
        lines.append("")

    # Calibration
    lines.append("## Calibration Analysis")
    lines.append("")
    for key in sorted(results.keys()):
        if "full__" not in key:
            continue
        r = results[key]
        lines.append(f"- **{key}**: ECE(uniform)={r['ece_uniform']:.3f}, "
                     f"ECE(quantile)={r['ece_quantile']:.3f}, "
                     f"Brier={r['brier_score']:.3f}")
    lines.append("")

    # Success criteria evaluation
    lines.append("## Success Criteria Evaluation")
    lines.append("")

    # Check baselines
    v009_auc = results.get("v009_only__XGBoost", {}).get("overall_auc", 0)
    full_auc = best["overall_auc"]
    full_recall = best["overall_recall"]
    full_prec = best["overall_precision"]
    full_p_at_r30 = best["precision_at_recall_30"]
    full_ece = best["ece_uniform"]

    lines.append(f"1. **Minimum viable** (beats v009-only AUC, recall>0.20, precision>0.85):")
    lines.append(f"   - Full AUC {full_auc:.3f} vs v009-only AUC {v009_auc:.3f}: "
                 f"{'PASS' if full_auc > v009_auc else 'FAIL'}")
    lines.append(f"   - Recall={full_recall:.3f}: {'PASS' if full_recall > 0.20 else 'FAIL'}")
    lines.append(f"   - Precision={full_prec:.3f}: {'PASS' if full_prec > 0.85 else 'FAIL (but check P@R≥30%)'}")
    lines.append("")

    lines.append(f"2. **Strong result** (recall>0.30 at precision>0.85):")
    lines.append(f"   - P@R≥30%={full_p_at_r30:.3f}: {'PASS' if full_p_at_r30 > 0.85 else 'FAIL'}")
    lines.append("")

    # Check if behavioral features are additive
    behav_auc = results.get("behavioral_only__XGBoost", {}).get("overall_auc", 0)
    lines.append(f"3. **Behavioral features additive to v009**:")
    lines.append(f"   - Full AUC {full_auc:.3f} vs v009-only {v009_auc:.3f} vs behavioral-only {behav_auc:.3f}")
    additive = full_auc > max(v009_auc, behav_auc)
    lines.append(f"   - Additive: {'YES' if additive else 'NO'}")
    lines.append("")

    lines.append(f"4. **Calibration (ECE<0.1)**:")
    lines.append(f"   - ECE={full_ece:.3f}: {'PASS' if full_ece < 0.1 else 'FAIL'}")
    lines.append("")

    report_path = BASE / "results" / "judge_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
