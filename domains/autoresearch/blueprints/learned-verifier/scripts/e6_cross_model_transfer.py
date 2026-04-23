#!/usr/bin/env python3
"""
E6: Cross-Model Transfer Validation.

Tests whether the 4-feature RF trained on Claude traces (AUC=0.756)
generalizes to Qwen3.5 traces. Uses a 3-feature ablation (dropping
svg_accepted, which is Claude-only) as the primary test.

Approach:
  A. Train RF on Claude 3 features → evaluate on Qwen3.5 (zero-shot transfer)
  B. Leave-one-model-out CV on combined Claude + Qwen3.5 data
  C. Per-model-family ensemble vs single RF

Data sources:
  - Claude: learned-verifier/results/combined_features.csv (n=300)
  - Qwen3.5 SERA: agent-swarm/results/swarm_phase1_qwen35-397b_sera.jsonl (n=50, 7 pass)
  - Qwen3.5 OpenCode: agent-swarm/results/swarm_phase1_qwen35-397b_opencode.jsonl
    + gold labels from verifier-reward/results/t4_cross_verifier_haiku_qwen35.jsonl (n=43, 4 pass)

Output: results/e6_cross_model_report.md, results/e6_cross_model_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
SWARM = BASE.parent / "agent-swarm"
VERIFIER = BASE.parent / "verifier-reward"

# The 3 features that can be computed from both Claude and Qwen3.5 traces
# (svg_accepted excluded — Claude-only)
TRANSFER_FEATURES = ["total_cost_usd", "tokens_per_edit", "loop_count"]

# Full 4-feature set for Claude-only baseline
FULL_FEATURES = ["total_cost_usd", "tokens_per_edit", "svg_accepted", "loop_count"]


def load_claude_data():
    """Load Claude combined features (n=300)."""
    path = BASE / "results" / "combined_features.csv"
    df = pd.read_csv(path)

    # Map to normalized feature names
    out = pd.DataFrame({
        "instance_id": df["instance_id"],
        "gold_pass": df["gold_pass"],
        "total_cost_usd": df["beh_total_cost_usd"],
        "tokens_per_edit": df["beh_tokens_per_edit"],
        "svg_accepted": df["svg_accepted"],
        "loop_count": df["beh_loop_count"],
        "model_family": "claude",
    })
    print(f"  Claude: {len(out)} instances, {int(out['gold_pass'].sum())} pass")
    return out


def load_qwen35_sera():
    """Load Qwen3.5 x SERA traces (n=50, 7 pass)."""
    path = SWARM / "results" / "swarm_phase1_qwen35-397b_sera.jsonl"
    rows = [json.loads(l) for l in open(path)]

    out_rows = []
    for r in rows:
        # Approximate feature mapping:
        # total_cost_usd: self-hosted, use tokens as proxy
        #   Claude Haiku rate ($0.80/$4.00 per MTok) as normalization baseline
        tokens = r.get("tokens_consumed", 0)
        input_tok = r.get("input_tokens", 0)
        output_tok = r.get("output_tokens", 0)

        # If input/output breakdown available, use it; else split 90/10
        if input_tok > 0 or output_tok > 0:
            cost_proxy = (input_tok * 0.80 + output_tok * 4.00) / 1_000_000
        else:
            cost_proxy = (tokens * 0.9 * 0.80 + tokens * 0.1 * 4.00) / 1_000_000

        # tokens_per_edit: total tokens / 1 if fix generated (no per-tool breakdown)
        fix = r.get("fix_generated", False)
        tokens_per_edit = tokens if fix else np.nan

        # loop_count: turns_used is the closest proxy
        loop_count = r.get("turns_used", 0)

        out_rows.append({
            "instance_id": r["instance_id"],
            "gold_pass": 1 if r.get("tests_pass") else 0,
            "total_cost_usd": cost_proxy,
            "tokens_per_edit": tokens_per_edit,
            "svg_accepted": np.nan,  # not available
            "loop_count": loop_count,
            "model_family": "qwen35",
            "harness": "sera",
        })

    df = pd.DataFrame(out_rows)
    print(f"  Qwen3.5 SERA: {len(df)} instances, {int(df['gold_pass'].sum())} pass")
    return df


def load_qwen35_opencode():
    """Load Qwen3.5 x OpenCode traces, joined with T4 gold labels."""
    trace_path = SWARM / "results" / "swarm_phase1_qwen35-397b_opencode.jsonl"
    gold_path = VERIFIER / "results" / "t4_cross_verifier_haiku_qwen35.jsonl"

    # Load traces
    traces = {}
    for l in open(trace_path):
        r = json.loads(l)
        traces[r["instance_id"]] = r

    # Load gold labels from T4
    gold = {}
    for l in open(gold_path):
        r = json.loads(l)
        gold[r["instance_id"]] = 1 if r.get("gold_pass") else 0

    out_rows = []
    for iid, label in gold.items():
        r = traces.get(iid)
        if r is None:
            continue

        tokens = r.get("tokens_consumed", 0)
        input_tok = r.get("input_tokens", 0)
        output_tok = r.get("output_tokens", 0)

        if input_tok > 0 or output_tok > 0:
            cost_proxy = (input_tok * 0.80 + output_tok * 4.00) / 1_000_000
        else:
            cost_proxy = (tokens * 0.9 * 0.80 + tokens * 0.1 * 4.00) / 1_000_000

        fix = r.get("fix_generated", False)
        tokens_per_edit = tokens if fix else np.nan
        loop_count = r.get("turns_used", 0)

        out_rows.append({
            "instance_id": iid,
            "gold_pass": label,
            "total_cost_usd": cost_proxy,
            "tokens_per_edit": tokens_per_edit,
            "svg_accepted": np.nan,
            "loop_count": loop_count,
            "model_family": "qwen35",
            "harness": "opencode",
        })

    df = pd.DataFrame(out_rows)
    print(f"  Qwen3.5 OpenCode: {len(df)} instances, {int(df['gold_pass'].sum())} pass")
    return df


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


def precision_at_recall(y_true, y_prob, min_recall):
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= min_recall
    if not valid.any():
        return 0.0
    return float(np.max(precisions[valid]))


def train_rf(X_train, y_train):
    X_filled = np.nan_to_num(X_train, nan=-999)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=7,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_filled, y_train)
    return rf


def evaluate_rf(rf, X_test, y_test):
    X_filled = np.nan_to_num(X_test, nan=-999)
    y_prob = rf.predict_proba(X_filled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    return {
        "auc": round(auc, 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "p_at_r30": round(precision_at_recall(y_test, y_prob, 0.30), 4),
        "ece": round(compute_ece(y_test, y_prob), 4),
        "n": len(y_test),
        "n_pass": int(y_test.sum()),
        "n_fail": int((1 - y_test).sum()),
    }


def cv_evaluate(df, features, n_splits=5):
    """5-fold stratified CV on a single dataset."""
    X = df[features].values
    y = df["gold_pass"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_prob = [], []

    for train_idx, val_idx in skf.split(X, y):
        rf = train_rf(X[train_idx], y[train_idx])
        X_val = np.nan_to_num(X[val_idx], nan=-999)
        y_prob = rf.predict_proba(X_val)[:, 1]
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        auc = 0.5

    return {
        "auc": round(auc, 4),
        "p_at_r30": round(precision_at_recall(all_y_true, all_y_prob, 0.30), 4),
        "ece": round(compute_ece(all_y_true, all_y_prob), 4),
        "n": len(all_y_true),
    }


def main():
    print("Loading data...\n")
    claude_df = load_claude_data()
    qwen_sera = load_qwen35_sera()
    qwen_oc = load_qwen35_opencode()
    qwen_df = pd.concat([qwen_sera, qwen_oc], ignore_index=True)
    print(f"\n  Qwen3.5 combined: {len(qwen_df)} instances, {int(qwen_df['gold_pass'].sum())} pass")

    results = {}

    # ─── Baseline: Claude 4-feature RF (5-fold CV) ────────────
    print("\n═══ Baseline: Claude 4-Feature RF ═══")
    baseline_4f = cv_evaluate(claude_df, FULL_FEATURES)
    results["claude_4feat_cv"] = baseline_4f
    print(f"  AUC={baseline_4f['auc']}, P@R30={baseline_4f['p_at_r30']}")

    # ─── Baseline: Claude 3-feature RF (no svg_accepted) ──────
    print("\n═══ Claude 3-Feature Ablation ═══")
    baseline_3f = cv_evaluate(claude_df, TRANSFER_FEATURES)
    results["claude_3feat_cv"] = baseline_3f
    print(f"  AUC={baseline_3f['auc']}, P@R30={baseline_3f['p_at_r30']}")
    print(f"  AUC drop from removing svg_accepted: {baseline_4f['auc'] - baseline_3f['auc']:+.4f}")

    # ─── Approach A: Zero-shot transfer ────────────────────────
    print("\n═══ Approach A: Zero-Shot Transfer (Claude RF → Qwen3.5) ═══")

    # Train on all Claude data
    X_claude = claude_df[TRANSFER_FEATURES].values
    y_claude = claude_df["gold_pass"].values
    rf_claude = train_rf(X_claude, y_claude)

    # Feature importance from Claude RF
    importances = dict(zip(TRANSFER_FEATURES, rf_claude.feature_importances_))
    print(f"  Claude RF feature importance: {json.dumps({k: round(v, 4) for k, v in importances.items()})}")

    # Evaluate on Qwen3.5
    X_qwen = qwen_df[TRANSFER_FEATURES].values
    y_qwen = qwen_df["gold_pass"].values
    transfer_result = evaluate_rf(rf_claude, X_qwen, y_qwen)
    results["transfer_claude_to_qwen"] = transfer_result
    print(f"  Qwen3.5 AUC={transfer_result['auc']} (n={transfer_result['n']}, "
          f"{transfer_result['n_pass']} pass, {transfer_result['n_fail']} fail)")

    # Breakdown by harness
    for harness in ["sera", "opencode"]:
        mask = qwen_df["harness"] == harness
        if mask.sum() < 5:
            continue
        sub = qwen_df[mask]
        sub_result = evaluate_rf(rf_claude, sub[TRANSFER_FEATURES].values, sub["gold_pass"].values)
        results[f"transfer_claude_to_qwen_{harness}"] = sub_result
        print(f"    {harness}: AUC={sub_result['auc']} (n={sub_result['n']}, {sub_result['n_pass']} pass)")

    # ─── Feature distribution comparison ───────────────────────
    print("\n═══ Feature Distribution Comparison ═══")
    for feat in TRANSFER_FEATURES:
        c_vals = claude_df[feat].dropna()
        q_vals = qwen_df[feat].dropna()
        print(f"  {feat}:")
        print(f"    Claude:  mean={c_vals.mean():.2f}, std={c_vals.std():.2f}, "
              f"median={c_vals.median():.2f}")
        print(f"    Qwen3.5: mean={q_vals.mean():.2f}, std={q_vals.std():.2f}, "
              f"median={q_vals.median():.2f}")
        # Ratio of means as distributional shift indicator
        if c_vals.mean() > 0:
            print(f"    Shift ratio (qwen/claude): {q_vals.mean() / c_vals.mean():.2f}x")

    # ─── Approach B: Leave-one-model-out CV ────────────────────
    print("\n═══ Approach B: Leave-One-Model-Out CV ═══")

    combined = pd.concat([claude_df, qwen_df], ignore_index=True)

    # Train on Claude, test on Qwen (already done above — same as transfer)
    # Train on Qwen, test on Claude
    X_qwen_train = qwen_df[TRANSFER_FEATURES].values
    y_qwen_train = qwen_df["gold_pass"].values
    rf_qwen = train_rf(X_qwen_train, y_qwen_train)
    reverse_result = evaluate_rf(rf_qwen, X_claude, y_claude)
    results["transfer_qwen_to_claude"] = reverse_result
    print(f"  Qwen RF → Claude: AUC={reverse_result['auc']}")
    print(f"  Claude RF → Qwen: AUC={transfer_result['auc']}")

    # Bidirectional mean
    bidi_auc = (transfer_result["auc"] + reverse_result["auc"]) / 2
    results["bidirectional_mean_auc"] = round(bidi_auc, 4)
    print(f"  Bidirectional mean AUC: {bidi_auc:.4f}")

    # ─── Approach C: Per-model ensemble vs single RF ───────────
    print("\n═══ Approach C: Per-Model Ensemble vs Single RF ═══")

    # Single RF on combined data (5-fold CV)
    single_rf = cv_evaluate(combined, TRANSFER_FEATURES)
    results["combined_single_rf_cv"] = single_rf
    print(f"  Single RF (combined, 5-fold CV): AUC={single_rf['auc']}")

    # Per-model ensemble: route to model-specific RF at inference
    # Simulate via model-stratified evaluation
    ensemble_y_true, ensemble_y_prob = [], []

    # Claude portion: 5-fold CV with Claude-only RF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_c = claude_df[TRANSFER_FEATURES].values
    y_c = claude_df["gold_pass"].values
    for train_idx, val_idx in skf.split(X_c, y_c):
        rf = train_rf(X_c[train_idx], y_c[train_idx])
        X_val = np.nan_to_num(X_c[val_idx], nan=-999)
        ensemble_y_true.extend(y_c[val_idx])
        ensemble_y_prob.extend(rf.predict_proba(X_val)[:, 1])

    # Qwen portion: 5-fold CV with Qwen-only RF (or LOO if too few)
    X_q = qwen_df[TRANSFER_FEATURES].values
    y_q = qwen_df["gold_pass"].values
    n_pass_q = int(y_q.sum())
    if n_pass_q >= 5:
        n_splits_q = min(5, n_pass_q)
        skf_q = StratifiedKFold(n_splits=n_splits_q, shuffle=True, random_state=42)
        for train_idx, val_idx in skf_q.split(X_q, y_q):
            rf = train_rf(X_q[train_idx], y_q[train_idx])
            X_val = np.nan_to_num(X_q[val_idx], nan=-999)
            ensemble_y_true.extend(y_q[val_idx])
            ensemble_y_prob.extend(rf.predict_proba(X_val)[:, 1])
    else:
        # Too few passes for stratified CV — use leave-one-out
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        for train_idx, val_idx in loo.split(X_q):
            rf = train_rf(X_q[train_idx], y_q[train_idx])
            X_val = np.nan_to_num(X_q[val_idx], nan=-999)
            ensemble_y_true.extend(y_q[val_idx])
            ensemble_y_prob.extend(rf.predict_proba(X_val)[:, 1])

    ensemble_y_true = np.array(ensemble_y_true)
    ensemble_y_prob = np.array(ensemble_y_prob)

    try:
        ensemble_auc = roc_auc_score(ensemble_y_true, ensemble_y_prob)
    except ValueError:
        ensemble_auc = 0.5

    ensemble_result = {
        "auc": round(ensemble_auc, 4),
        "p_at_r30": round(precision_at_recall(ensemble_y_true, ensemble_y_prob, 0.30), 4),
        "ece": round(compute_ece(ensemble_y_true, ensemble_y_prob), 4),
        "n": len(ensemble_y_true),
    }
    results["per_model_ensemble_cv"] = ensemble_result
    print(f"  Per-model ensemble: AUC={ensemble_result['auc']}")
    print(f"  Single RF:          AUC={single_rf['auc']}")
    print(f"  Ensemble delta:     {ensemble_result['auc'] - single_rf['auc']:+.4f}")

    # ─── Summary ───────────────────────────────────────────────
    print("\n═══ Summary ═══\n")

    summary = {
        "claude_4feat_baseline": baseline_4f["auc"],
        "claude_3feat_ablation": baseline_3f["auc"],
        "svg_drop": round(baseline_4f["auc"] - baseline_3f["auc"], 4),
        "transfer_auc": transfer_result["auc"],
        "reverse_transfer_auc": reverse_result["auc"],
        "bidirectional_mean": round(bidi_auc, 4),
        "single_rf_combined": single_rf["auc"],
        "per_model_ensemble": ensemble_result["auc"],
    }
    results["summary"] = summary

    # Success criteria
    print("  Success Criteria:")
    print(f"    Cross-model AUC > 0.65: {transfer_result['auc']:.3f} → "
          f"{'PASS' if transfer_result['auc'] > 0.65 else 'FAIL'}")
    print(f"    Per-family ensemble AUC > 0.72: {ensemble_result['auc']:.3f} → "
          f"{'PASS' if ensemble_result['auc'] > 0.72 else 'FAIL'}")
    print(f"    Model-agnostic 3-feat AUC > 0.70: {baseline_3f['auc']:.3f} → "
          f"{'PASS' if baseline_3f['auc'] > 0.70 else 'FAIL'}")

    # ─── Generate report ───────────────────────────────────────
    generate_report(results, claude_df, qwen_df)

    # Save results
    out_path = BASE / "results" / "e6_cross_model_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {out_path}")


def generate_report(results, claude_df, qwen_df):
    """Generate e6_cross_model_report.md."""
    s = results["summary"]
    t = results["transfer_claude_to_qwen"]

    lines = [
        "# E6: Cross-Model Transfer Validation Results",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"**Claude data**: {len(claude_df)} instances ({int(claude_df['gold_pass'].sum())} pass)",
        f"**Qwen3.5 data**: {len(qwen_df)} instances ({int(qwen_df['gold_pass'].sum())} pass)",
        f"**Features**: 3-feature ablation (no svg_accepted)",
        "",
        "## Key Results",
        "",
        "| Metric | Value | Target | Status |",
        "|--------|-------|--------|--------|",
        f"| Cross-model AUC (Claude RF → Qwen3.5) | {t['auc']:.3f} | > 0.65 | {'PASS' if t['auc'] > 0.65 else 'FAIL'} |",
        f"| Per-family ensemble AUC | {s['per_model_ensemble']:.3f} | > 0.72 | {'PASS' if s['per_model_ensemble'] > 0.72 else 'FAIL'} |",
        f"| Model-agnostic 3-feat AUC (Claude CV) | {s['claude_3feat_ablation']:.3f} | > 0.70 | {'PASS' if s['claude_3feat_ablation'] > 0.70 else 'FAIL'} |",
        "",
        "## Ablation: svg_accepted Drop",
        "",
        f"- 4-feature RF (with svg): AUC = {s['claude_4feat_baseline']:.3f}",
        f"- 3-feature RF (no svg):   AUC = {s['claude_3feat_ablation']:.3f}",
        f"- Drop: {s['svg_drop']:+.3f}",
        "",
        "## Transfer Results",
        "",
        "| Direction | AUC | n | Passes |",
        "|-----------|-----|---|--------|",
        f"| Claude RF → Qwen3.5 | {s['transfer_auc']:.3f} | {t['n']} | {t['n_pass']} |",
        f"| Qwen3.5 RF → Claude | {s['reverse_transfer_auc']:.3f} | {len(claude_df)} | {int(claude_df['gold_pass'].sum())} |",
        f"| Bidirectional mean | {s['bidirectional_mean']:.3f} | — | — |",
        "",
    ]

    # Harness breakdown
    for harness in ["sera", "opencode"]:
        key = f"transfer_claude_to_qwen_{harness}"
        if key in results:
            r = results[key]
            lines.append(f"- {harness}: AUC={r['auc']:.3f} (n={r['n']}, {r['n_pass']} pass)")
    lines.append("")

    lines.extend([
        "## Ensemble vs Single RF",
        "",
        f"- Single RF (combined data, 5-fold CV): AUC = {s['single_rf_combined']:.3f}",
        f"- Per-model ensemble: AUC = {s['per_model_ensemble']:.3f}",
        f"- Delta: {s['per_model_ensemble'] - s['single_rf_combined']:+.3f}",
        "",
        "## Feature Approximation Notes",
        "",
        "Qwen3.5 features are approximations from swarm-level aggregates:",
        "- `total_cost_usd`: tokens × Haiku rate proxy (self-hosted, no real pricing)",
        "- `tokens_per_edit`: total tokens if fix_generated (no per-tool breakdown; Claude uses tokens/n_edits)",
        "- `loop_count`: turns_used (Claude uses repeated-action loop detection)",
        "- `svg_accepted`: excluded (Claude-only SVG pipeline)",
        "",
        "These approximations may attenuate the transfer signal. A positive result",
        "under these conditions is a strong signal; a negative result may reflect",
        "feature misalignment rather than genuine non-transferability.",
        "",
    ])

    report_path = BASE / "results" / "e6_cross_model_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
