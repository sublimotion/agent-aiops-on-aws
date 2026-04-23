#!/usr/bin/env python3
"""
Phase 3: Train combined learned verifier on all 5 signal sources.

Extends tiny-judge's approach (5-fold stratified CV, RF/XGBoost/LogReg)
with expanded feature set: behavioral + v009 + debate + SVG.

Feature groups for ablation:
- behavioral_only: beh_* features (same as tiny-judge)
- v009_only: v009_* features
- debate_only: debate_* features
- svg_only: svg_* features
- behavioral_v009: behavioral + v009 (tiny-judge extension)
- all_signals: everything
- selected: forward-selected best subset

Reads: results/combined_features.csv
Writes: results/phase3_report.md, results/phase3_results.json
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier

BASE = Path(__file__).resolve().parent.parent


# ─── Feature groups ────────────────────────────────────────────

def get_feature_groups(columns):
    """Define feature groups from available columns."""
    beh = sorted([c for c in columns if c.startswith("beh_")])
    v009 = sorted([c for c in columns if c.startswith("v009_")])
    debate = sorted([c for c in columns if c.startswith("debate_")])
    svg = sorted([c for c in columns if c.startswith("svg_")])

    return {
        "behavioral_only": beh,
        "v009_only": v009,
        "debate_only": debate,
        "svg_only": svg,
        "beh_v009": beh + v009,
        "beh_debate": beh + debate,
        "beh_v009_debate": beh + v009 + debate,
        "all_signals": beh + v009 + debate + svg,
    }


# ─── Metrics ───────────────────────────────────────────────────

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
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def precision_at_recall(y_true, y_prob, min_recall):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid = recalls >= min_recall
    if not valid.any():
        return 0.0, 1.0
    best_idx = np.argmax(precisions[valid])
    indices = np.where(valid)[0]
    idx = indices[best_idx]
    return float(precisions[idx]), float(thresholds[idx]) if idx < len(thresholds) else 0.5


# ─── Training ──────────────────────────────────────────────────

def train_and_evaluate(X, y, feature_names, model_name="RandomForest", n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    fold_metrics = []
    last_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
            X_train_f = np.nan_to_num(X_train, nan=-999)
            X_val = np.nan_to_num(X_val, nan=-999)
            model = GradientBoostingClassifier(
                max_depth=5, n_estimators=150, learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_train_f, y_train)
        elif model_name == "LogisticRegression":
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

        last_model = model
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
            "auc": round(auc, 4),
            "f1": round(f1_score(y_val, y_pred), 4),
        })

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    prec_at_30, _ = precision_at_recall(all_y_true, all_y_prob, 0.30)
    prec_at_50, _ = precision_at_recall(all_y_true, all_y_prob, 0.50)
    ece_uniform = compute_ece(all_y_true, all_y_prob, strategy="uniform")
    brier = brier_score_loss(all_y_true, all_y_prob)

    importance = {}
    if hasattr(last_model, "feature_importances_"):
        for fname, imp in zip(feature_names, last_model.feature_importances_):
            importance[fname] = round(float(imp), 5)
    elif hasattr(last_model, "coef_"):
        for fname, coef in zip(feature_names, last_model.coef_[0]):
            importance[fname] = round(float(abs(coef)), 5)

    return {
        "model": model_name,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "auc": round(overall_auc, 4),
        "f1": round(f1_score(all_y_true, all_y_pred), 4),
        "precision": round(precision_score(all_y_true, all_y_pred, zero_division=0), 4),
        "recall": round(recall_score(all_y_true, all_y_pred, zero_division=0), 4),
        "p_at_r30": round(prec_at_30, 4),
        "p_at_r50": round(prec_at_50, 4),
        "ece": round(ece_uniform, 4),
        "brier": round(brier, 4),
        "fold_auc_mean": round(np.mean([f["auc"] for f in fold_metrics]), 4),
        "fold_auc_std": round(np.std([f["auc"] for f in fold_metrics]), 4),
        "fold_metrics": fold_metrics,
        "feature_importance": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)),
        "y_true": [int(x) for x in all_y_true],
        "y_prob": [float(round(x, 6)) for x in all_y_prob],
    }, last_model


def forward_selection(df, y, candidate_features, model_cls="RandomForest",
                      max_features=12, metric="p_at_r30"):
    """Greedy forward feature selection optimizing the given metric."""
    selected = []
    history = []
    best_metric = 0.0

    for step in range(max_features):
        best_feat = None
        best_result = None

        for feat in candidate_features:
            if feat in selected:
                continue
            trial = selected + [feat]
            X_trial = df[trial].values
            result, _ = train_and_evaluate(X_trial, y, trial, model_cls)
            val = result[metric]
            if val > best_metric:
                best_metric = val
                best_feat = feat
                best_result = result

        if best_feat is None:
            break

        selected.append(best_feat)
        history.append({
            "step": step + 1,
            "added": best_feat,
            metric: best_metric,
            "auc": best_result["auc"],
            "ece": best_result["ece"],
            "f1": best_result["f1"],
        })
        print(f"    Step {step+1}: +{best_feat} → {metric}={best_metric:.4f}, AUC={best_result['auc']:.4f}")

        # Stop if metric hasn't improved for 2 steps
        if len(history) >= 3 and history[-1][metric] <= history[-3][metric]:
            print(f"    Early stop: no improvement in 2 steps")
            break

    return selected, history


def run_calibration(df, y, features, model_name="RandomForest"):
    """Run calibration analysis (Platt + isotonic) on the given feature set."""
    X = df[features].values
    X_filled = np.nan_to_num(X, nan=-999)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calibration_results = {}

    for cal_method in [None, "sigmoid", "isotonic"]:
        all_y_true = []
        all_y_prob = []

        for train_idx, val_idx in skf.split(X_filled, y):
            X_train, X_val = X_filled[train_idx], X_filled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if model_name == "RandomForest":
                base_model = RandomForestClassifier(
                    n_estimators=200, max_depth=7,
                    class_weight="balanced", random_state=42,
                )
            elif model_name == "XGBoost" and HAS_XGBOOST:
                scale_pos = sum(y_train == 0) / max(sum(y_train == 1), 1)
                base_model = XGBClassifier(
                    max_depth=5, n_estimators=150, learning_rate=0.1,
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss", random_state=42,
                    use_label_encoder=False,
                )

            if cal_method:
                model = CalibratedClassifierCV(base_model, method=cal_method, cv=3)
            else:
                model = base_model

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            all_y_true.extend(y_val)
            all_y_prob.extend(y_prob)

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)

        prec_at_30, _ = precision_at_recall(all_y_true, all_y_prob, 0.30)
        ece = compute_ece(all_y_true, all_y_prob)
        brier = brier_score_loss(all_y_true, all_y_prob)

        method_name = cal_method or "uncalibrated"
        calibration_results[method_name] = {
            "p_at_r30": round(prec_at_30, 4),
            "auc": round(roc_auc_score(all_y_true, all_y_prob), 4),
            "ece": round(ece, 4),
            "brier": round(brier, 4),
        }
        print(f"    {method_name:15s}: P@R30={prec_at_30:.4f}, ECE={ece:.4f}, Brier={brier:.4f}")

    return calibration_results


def generate_report(all_results, selection_history, calibration, df):
    """Generate phase3_report.md."""
    lines = []
    lines.append("# Phase 3: Combined Learned Verifier Report")
    lines.append(f"\n**Dataset**: VP SWE-bench production eval")
    lines.append(f"**Instances**: {len(df)} (pass={int(df['gold_pass'].sum())}, fail={int((1-df['gold_pass']).sum())})")
    lines.append(f"**Evaluation**: 5-fold stratified cross-validation")
    lines.append(f"**Signal sources**: Behavioral (tiny-judge) + v009 rubric + Debate verdicts + SVG consensus")
    lines.append("")

    # Find best model
    best_key = max(all_results.keys(), key=lambda k: all_results[k].get("auc", 0))
    best = all_results[best_key]

    # Tiny-judge baseline for comparison
    tj_behavioral = all_results.get("behavioral_only__RandomForest", {})
    tj_v009 = all_results.get("v009_only__RandomForest", {})

    lines.append("## Best Model")
    lines.append(f"\n**{best_key}**: AUC={best['auc']:.3f}, F1={best['f1']:.3f}, "
                 f"P@R30={best['p_at_r30']:.3f}, ECE={best['ece']:.3f}")
    lines.append("")

    # Comparison to tiny-judge baseline
    lines.append("## Improvement Over Baselines")
    lines.append("")
    lines.append("| Baseline | AUC | P@R≥30% | ECE | F1 |")
    lines.append("|----------|-----|---------|-----|----|")
    if tj_v009:
        lines.append(f"| v009-only (RF) | {tj_v009.get('auc', 0):.3f} | {tj_v009.get('p_at_r30', 0):.3f} | {tj_v009.get('ece', 0):.3f} | {tj_v009.get('f1', 0):.3f} |")
    if tj_behavioral:
        lines.append(f"| Behavioral-only (RF) | {tj_behavioral.get('auc', 0):.3f} | {tj_behavioral.get('p_at_r30', 0):.3f} | {tj_behavioral.get('ece', 0):.3f} | {tj_behavioral.get('f1', 0):.3f} |")
    lines.append(f"| **{best_key}** | **{best['auc']:.3f}** | **{best['p_at_r30']:.3f}** | **{best['ece']:.3f}** | **{best['f1']:.3f}** |")
    lines.append("")

    # Full comparison table
    lines.append("## Full Model Comparison")
    lines.append("")
    lines.append("| Feature Set | Model | AUC | AUC±std | F1 | P@R≥30% | P@R≥50% | ECE | Brier |")
    lines.append("|-------------|-------|-----|---------|----|---------|---------|----|-------|")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        parts = key.split("__")
        fset = parts[0]
        model = parts[1] if len(parts) > 1 else "?"
        lines.append(
            f"| {fset} | {model} | {r['auc']:.3f} | "
            f"{r['fold_auc_mean']:.3f}±{r['fold_auc_std']:.3f} | "
            f"{r['f1']:.3f} | {r['p_at_r30']:.3f} | {r['p_at_r50']:.3f} | "
            f"{r['ece']:.3f} | {r['brier']:.3f} |"
        )
    lines.append("")

    # Feature importance
    if best.get("feature_importance"):
        lines.append("## Feature Importance (Best Model)")
        lines.append("")
        lines.append("| Rank | Feature | Importance | Signal Source |")
        lines.append("|------|---------|------------|--------------|")
        for i, (fname, imp) in enumerate(best["feature_importance"].items(), 1):
            if i > 20:
                break
            if fname.startswith("beh_"):
                source = "Behavioral"
            elif fname.startswith("v009_"):
                source = "v009 Rubric"
            elif fname.startswith("debate_"):
                source = "Debate"
            elif fname.startswith("svg_"):
                source = "SVG"
            else:
                source = "Other"
            lines.append(f"| {i} | {fname} | {imp:.4f} | {source} |")
        lines.append("")

    # Forward selection
    if selection_history:
        lines.append("## Forward Feature Selection")
        lines.append("")
        lines.append("| Step | Added Feature | P@R≥30% | AUC | ECE |")
        lines.append("|------|---------------|---------|-----|-----|")
        for h in selection_history:
            lines.append(f"| {h['step']} | {h['added']} | {h['p_at_r30']:.3f} | {h['auc']:.3f} | {h['ece']:.3f} |")
        lines.append("")

    # Calibration
    if calibration:
        lines.append("## Calibration Analysis")
        lines.append("")
        lines.append("| Method | P@R≥30% | AUC | ECE | Brier |")
        lines.append("|--------|---------|-----|-----|-------|")
        for method, cal in calibration.items():
            lines.append(f"| {method} | {cal['p_at_r30']:.3f} | {cal['auc']:.3f} | {cal['ece']:.3f} | {cal['brier']:.3f} |")
        lines.append("")

    # Signal contribution analysis
    lines.append("## Signal Contribution Analysis")
    lines.append("")
    lines.append("Does adding each signal improve over behavioral-only?")
    lines.append("")
    beh_auc = tj_behavioral.get("auc", 0) if tj_behavioral else 0
    for fset_key in ["beh_v009", "beh_debate", "beh_v009_debate", "all_signals"]:
        for model_name in ["RandomForest"]:
            key = f"{fset_key}__{model_name}"
            if key in all_results:
                r = all_results[key]
                delta = r["auc"] - beh_auc
                lines.append(f"- **{fset_key}**: AUC={r['auc']:.3f} (Δ={delta:+.3f} vs behavioral-only), "
                             f"P@R30={r['p_at_r30']:.3f}, ECE={r['ece']:.3f}")
    lines.append("")

    # Success criteria
    lines.append("## Success Criteria")
    lines.append("")
    lines.append(f"1. **Combined beats behavioral-only**: AUC {best['auc']:.3f} vs {beh_auc:.3f} → "
                 f"{'PASS' if best['auc'] > beh_auc else 'FAIL'} (Δ={best['auc']-beh_auc:+.3f})")
    lines.append(f"2. **P@R≥30% > 0.85**: {best['p_at_r30']:.3f} → {'PASS' if best['p_at_r30'] > 0.85 else 'FAIL'}")
    lines.append(f"3. **ECE < 0.1 (RL-ready)**: {best['ece']:.3f} → {'PASS' if best['ece'] < 0.1 else 'FAIL'}")
    if calibration:
        best_cal_ece = min(c["ece"] for c in calibration.values())
        lines.append(f"4. **Post-calibration ECE < 0.1**: {best_cal_ece:.3f} → {'PASS' if best_cal_ece < 0.1 else 'FAIL'}")
    lines.append("")

    report_path = BASE / "results" / "phase3_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {report_path}")


def main():
    print("Loading combined features...")
    df = pd.read_csv(BASE / "results" / "combined_features.csv")
    print(f"  {len(df)} instances × {len(df.columns)} columns")

    y = df["gold_pass"].values
    feature_cols = [c for c in df.columns if c not in ("instance_id", "gold_pass")]

    # Define feature groups
    groups = get_feature_groups(feature_cols)
    print(f"\nFeature groups:")
    for name, feats in groups.items():
        print(f"  {name}: {len(feats)} features")

    # ─── Train all configurations ──────────────────────────────
    print("\n═══ Training All Configurations ═══\n")
    all_results = {}

    for group_name, features in groups.items():
        if not features:
            print(f"  [{group_name}] No features, skipping")
            continue

        available = [f for f in features if f in df.columns and df[f].notna().any()]
        if not available:
            print(f"  [{group_name}] No non-null features, skipping")
            continue

        X = df[available].values
        print(f"  [{group_name}] {len(available)} features")

        for model_name in ["RandomForest", "LogisticRegression", "XGBoost"]:
            key = f"{group_name}__{model_name}"
            result, model = train_and_evaluate(X, y, available, model_name)
            # Don't save y_true/y_prob in all_results for report (save separately)
            result_clean = {k: v for k, v in result.items() if k not in ("y_true", "y_prob")}
            all_results[key] = result_clean
            print(f"    {model_name:20s}: AUC={result['auc']:.3f} "
                  f"(±{result['fold_auc_std']:.3f}), "
                  f"F1={result['f1']:.3f}, "
                  f"P@R30={result['p_at_r30']:.3f}, "
                  f"ECE={result['ece']:.3f}")

    # ─── Forward selection ─────────────────────────────────────
    print("\n═══ Forward Feature Selection ═══\n")
    candidate_features = [f for f in feature_cols if f in df.columns and df[f].notna().any()]
    selected_features, selection_history = forward_selection(
        df, y, candidate_features, "RandomForest", max_features=12
    )

    if selected_features:
        print(f"\n  Selected {len(selected_features)} features: {selected_features}")
        X_selected = df[selected_features].values
        for model_name in ["RandomForest", "XGBoost", "LogisticRegression"]:
            key = f"selected_{len(selected_features)}__{model_name}"
            result, model = train_and_evaluate(X_selected, y, selected_features, model_name)
            result_clean = {k: v for k, v in result.items() if k not in ("y_true", "y_prob")}
            all_results[key] = result_clean
            print(f"    {model_name:20s}: AUC={result['auc']:.3f}, "
                  f"P@R30={result['p_at_r30']:.3f}, ECE={result['ece']:.3f}")

        # Save best selected model
        best_selected_key = max(
            [k for k in all_results if k.startswith("selected_")],
            key=lambda k: all_results[k]["auc"],
        )
        best_selected_result, best_model = train_and_evaluate(
            X_selected, y, selected_features, "RandomForest"
        )
        model_path = BASE / "results" / "models" / "phase3_best.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": best_model, "features": selected_features}, f)
        print(f"\n  Best model saved: {model_path}")

    # ─── Calibration analysis ──────────────────────────────────
    print("\n═══ Calibration Analysis ═══\n")

    # Calibrate the best all_signals model
    all_signal_feats = groups.get("all_signals", [])
    all_signal_avail = [f for f in all_signal_feats if f in df.columns and df[f].notna().any()]

    calibration = {}
    if all_signal_avail:
        print("  All-signals RF:")
        calibration["all_signals_RF"] = run_calibration(df, y, all_signal_avail, "RandomForest")

    if selected_features:
        print("  Selected RF:")
        calibration["selected_RF"] = run_calibration(df, y, selected_features, "RandomForest")

    # ─── Generate report ───────────────────────────────────────
    print("\n═══ Generating Report ═══")
    generate_report(all_results, selection_history, calibration.get("selected_RF", calibration.get("all_signals_RF", {})), df)

    # Save full results
    results_path = BASE / "results" / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": all_results,
            "selection_history": selection_history,
            "selected_features": selected_features if selected_features else [],
            "calibration": calibration,
        }, f, indent=2, default=str)
    print(f"  Full results: {results_path}")


if __name__ == "__main__":
    main()
