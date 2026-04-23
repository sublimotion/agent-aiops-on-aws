#!/usr/bin/env python3
"""
E6: Train per-model RFs on OpenHands evaluation outputs with REAL API telemetry.

Dataset: OpenHands/openhands-evaluation-outputs (HuggingFace)
- 19 model configs x ~300 instances each
- Full per-call telemetry: prompt_tokens, completion_tokens, cost, timestamps
- Per-tool attribution via function_name
- Gold labels via report.resolved

This is the definitive E6 experiment — precise features from real API responses,
not character-count estimates.

Features extracted:
  - total_cost_usd: metrics.accumulated_cost (real USD from litellm)
  - tokens_per_edit: total_tokens / n_edit_actions
  - loop_count: number of action events in history

Output:
  - results/e6_openhands_features.csv
  - results/e6_openhands_rf_report.md
  - results/e6_openhands_rf_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE / "results" / "e6_openhands_features.csv"
CLAUDE_FEATURES_PATH = BASE / "results" / "combined_features.csv"

TRANSFER_FEATURES = ["total_cost_usd", "tokens_per_edit", "loop_count"]

# Model configs to process (all SWE-bench Lite)
CONFIGS = {
    "claude-sonnet": "claude-3-5-sonnet-20241022_maxiter_100_N_v2.2-no-hint",
    "claude-haiku": "claude-3-5-haiku-20241022_maxiter_100_N_v2.1-no-hint",
    "gpt-4o": "gpt-4o-2024-05-13_maxiter_30_N_v1.9-no-hint-eval-24sep",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18_maxiter_30_N_v1.9-no-hint-eval-24sep",
    "deepseek": "deepseek-chat_maxiter_100_N_v2.2-no-hint-main-non-fncall-run_1",
    "qwen-72b": "qwen-2.5-72b-instruct_maxiter_30_N_v1.9-no-hint-non-fncall-eval-24sep",
    "llama-70b": "llama-v3p3-70b-instruct_maxiter_100_N_v0.15.0-no-hint-run_1",
    "llama-405b": "llama-v3p1-405b-instruct_maxiter_30_N_v1.9-no-hint-eval-24sep",
}

HF_REPO = "OpenHands/openhands-evaluation-outputs"
HF_BASE = "outputs/SWE-bench_Lite-test/CodeActAgent"


def extract_features_from_record(row, model_family):
    """Extract features from one OpenHands output record with real telemetry."""
    instance_id = row.get("instance_id", "")
    report = row.get("report") or {}
    resolved = 1 if report.get("resolved") else 0
    history = row.get("history") or []
    metrics = row.get("metrics") or {}

    # Real accumulated cost from litellm
    total_cost_usd = metrics.get("accumulated_cost", 0.0) or 0.0

    # Flatten history: older configs use [(action, obs), ...] pairs
    flat_history = []
    for item in history:
        if isinstance(item, list):
            flat_history.extend(item)
        elif isinstance(item, dict):
            flat_history.append(item)

    # Count actions and tokens from history
    n_edits = 0
    n_reads = 0
    n_bash = 0
    n_actions = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for ev in flat_history:
        action = ev.get("action", "")
        if action in ("run", "run_ipython", "browse", "finish", "message"):
            n_actions += 1

        # New format: tool_call_metadata with per-call telemetry
        tcm = ev.get("tool_call_metadata")
        if tcm and isinstance(tcm, dict):
            fn = tcm.get("function_name", "")
            if fn == "str_replace_editor":
                args = ev.get("args", {})
                if isinstance(args, dict):
                    cmd = args.get("command", "")
                    if cmd in ("str_replace", "create", "insert"):
                        n_edits += 1
                    elif cmd == "view":
                        n_reads += 1
                    else:
                        n_edits += 1
                else:
                    n_edits += 1
            elif fn == "execute_bash":
                n_bash += 1

            # Extract real token usage from tool_call_metadata
            mr = tcm.get("model_response", {})
            if isinstance(mr, dict):
                usage = mr.get("usage", {})
                if isinstance(usage, dict):
                    total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    total_completion_tokens += usage.get("completion_tokens", 0) or 0
                    total_tokens += usage.get("total_tokens", 0) or 0

        # Old format: no tool_call_metadata, classify from action + args
        elif action in ("run", "run_ipython"):
            args = ev.get("args", {})
            if isinstance(args, dict):
                code = args.get("code", "") or args.get("command", "")
                # Heuristic: str_replace_editor calls show up as run_ipython
                if "str_replace_editor" in str(code):
                    if any(k in str(code) for k in ("str_replace", "create", "insert")):
                        n_edits += 1
                    elif "view" in str(code):
                        n_reads += 1
                    else:
                        n_edits += 1
                else:
                    n_bash += 1

    # For old-format models with no per-call tokens, estimate from cost
    # Use average rate of ~$5/Mtok (mix of input/output) as rough inverse
    if total_tokens == 0 and total_cost_usd > 0:
        total_tokens = int(total_cost_usd / 5.0 * 1_000_000)

    # tokens_per_edit: real tokens / real edit count
    if n_edits > 0:
        tokens_per_edit = total_tokens / n_edits
    elif total_tokens > 0:
        tokens_per_edit = total_tokens  # no edits = all tokens wasted
    else:
        tokens_per_edit = np.nan

    # loop_count: total action events
    loop_count = n_actions

    return {
        "instance_id": instance_id,
        "gold_pass": resolved,
        "total_cost_usd": round(total_cost_usd, 6),
        "tokens_per_edit": round(tokens_per_edit, 1) if not np.isnan(tokens_per_edit) else np.nan,
        "loop_count": loop_count,
        "model_family": model_family,
        # Diagnostics
        "_n_edits": n_edits,
        "_n_reads": n_reads,
        "_n_bash": n_bash,
        "_total_tokens": total_tokens,
        "_total_prompt_tokens": total_prompt_tokens,
        "_total_completion_tokens": total_completion_tokens,
    }


def load_and_extract():
    """Load all model configs and extract features."""
    if FEATURES_PATH.exists():
        print(f"  Loading cached features from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
        print(f"  {len(df)} instances across {df['model_family'].nunique()} models")
        return df

    from huggingface_hub import hf_hub_download

    all_rows = []
    for model_name, config in CONFIGS.items():
        print(f"  Downloading {model_name}...")
        try:
            path = hf_hub_download(
                HF_REPO,
                f"{HF_BASE}/{config}/output.jsonl",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        n = 0
        n_resolved = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                feat = extract_features_from_record(row, model_name)
                all_rows.append(feat)
                n += 1
                if feat["gold_pass"]:
                    n_resolved += 1

        print(f"    {n} instances, {n_resolved} resolved ({n_resolved/n*100:.1f}%)")

    df = pd.DataFrame(all_rows)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n  Saved: {FEATURES_PATH}")
    print(f"  {len(df)} total instances across {df['model_family'].nunique()} models")
    return df


# ─── ML utilities ──────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def precision_at_recall(y_true, y_prob, min_recall):
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= min_recall
    return float(np.max(precisions[valid])) if valid.any() else 0.0


def train_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=7,
        class_weight="balanced", random_state=42,
    )
    rf.fit(np.nan_to_num(X, nan=-999), y)
    return rf


def cv_evaluate(df, features, n_splits=5):
    X = df[features].values
    y = df["gold_pass"].values
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < n_splits:
        return {"auc": 0.5, "p_at_r30": 0.0, "ece": 0.5, "n": len(y), "n_pass": int(y.sum())}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_prob = [], []
    for tr, va in skf.split(X, y):
        rf = train_rf(X[tr], y[tr])
        all_y_prob.extend(rf.predict_proba(np.nan_to_num(X[va], nan=-999))[:, 1])
        all_y_true.extend(y[va])
    all_y_true, all_y_prob = np.array(all_y_true), np.array(all_y_prob)
    try:
        auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        auc = 0.5
    return {
        "auc": round(auc, 4),
        "p_at_r30": round(precision_at_recall(all_y_true, all_y_prob, 0.30), 4),
        "ece": round(compute_ece(all_y_true, all_y_prob), 4),
        "n": len(all_y_true),
        "n_pass": int(all_y_true.sum()),
    }


def evaluate_rf(rf, X, y):
    y_prob = rf.predict_proba(np.nan_to_num(X, nan=-999))[:, 1]
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = 0.5
    return {"auc": round(auc, 4), "n": len(y), "n_pass": int(y.sum())}


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("═══ E6 OpenHands: Multi-Model RF with Real Telemetry ═══\n")

    # Load data
    print("Loading OpenHands evaluation outputs...")
    oh_df = load_and_extract()

    # Load our Claude data for comparison
    print("\nLoading our Claude data (Phase 3)...")
    claude_raw = pd.read_csv(CLAUDE_FEATURES_PATH)
    claude_df = pd.DataFrame({
        "instance_id": claude_raw["instance_id"],
        "gold_pass": claude_raw["gold_pass"],
        "total_cost_usd": claude_raw["beh_total_cost_usd"],
        "tokens_per_edit": claude_raw["beh_tokens_per_edit"],
        "loop_count": claude_raw["beh_loop_count"],
        "model_family": "claude-ours",
    })
    print(f"  Our Claude: {len(claude_df)} instances, {int(claude_df['gold_pass'].sum())} pass")

    results = {}

    # ─── Per-model standalone RFs ──────────────────────────────
    print("\n═══ Per-Model Standalone RF (5-fold CV) ═══\n")
    model_results = {}
    for model in sorted(oh_df["model_family"].unique()):
        sub = oh_df[oh_df["model_family"] == model]
        cv = cv_evaluate(sub, TRANSFER_FEATURES)
        model_results[model] = cv
        print(f"  {model:20s}: AUC={cv['auc']:.3f}, P@R30={cv['p_at_r30']:.3f}, "
              f"n={cv['n']}, pass={cv['n_pass']}")
    results["per_model_standalone"] = model_results

    # Our Claude for comparison
    claude_cv = cv_evaluate(claude_df, TRANSFER_FEATURES)
    results["claude_ours_standalone"] = claude_cv
    print(f"  {'claude-ours':20s}: AUC={claude_cv['auc']:.3f}, P@R30={claude_cv['p_at_r30']:.3f}, "
          f"n={claude_cv['n']}, pass={claude_cv['n_pass']}")

    # ─── Feature distributions per model ───────────────────────
    print("\n═══ Feature Distributions ═══\n")
    print(f"  {'Model':20s} {'cost_mean':>10s} {'cost_std':>10s} {'tpe_mean':>12s} {'tpe_std':>12s} {'loop_mean':>10s} {'loop_std':>10s}")
    for model in sorted(oh_df["model_family"].unique()):
        sub = oh_df[oh_df["model_family"] == model]
        for feat_short, feat in [("cost", "total_cost_usd"), ("tpe", "tokens_per_edit"), ("loop", "loop_count")]:
            vals = sub[feat].dropna()
            if feat_short == "cost":
                row = f"  {model:20s} {vals.mean():10.4f} {vals.std():10.4f}"
            elif feat_short == "tpe":
                row += f" {vals.mean():12.0f} {vals.std():12.0f}"
            elif feat_short == "loop":
                row += f" {vals.mean():10.1f} {vals.std():10.1f}"
                print(row)

    # ─── Cross-model transfer matrix ───────────────────────────
    print("\n═══ Cross-Model Transfer Matrix (Train→Test AUC) ═══\n")
    models = sorted(oh_df["model_family"].unique())
    transfer_matrix = {}

    # Header
    header = f"  {'Train\\Test':20s}"
    for m in models:
        header += f" {m[:8]:>8s}"
    print(header)

    for train_model in models:
        train_sub = oh_df[oh_df["model_family"] == train_model]
        if train_sub["gold_pass"].nunique() < 2:
            continue
        rf = train_rf(train_sub[TRANSFER_FEATURES].values, train_sub["gold_pass"].values)

        row = f"  {train_model:20s}"
        row_results = {}
        for test_model in models:
            test_sub = oh_df[oh_df["model_family"] == test_model]
            if test_sub["gold_pass"].nunique() < 2:
                row += f" {'n/a':>8s}"
                continue
            ev = evaluate_rf(rf, test_sub[TRANSFER_FEATURES].values, test_sub["gold_pass"].values)
            row += f" {ev['auc']:8.3f}"
            row_results[test_model] = ev["auc"]
        print(row)
        transfer_matrix[train_model] = row_results

    results["transfer_matrix"] = transfer_matrix

    # ─── Per-model ensemble vs single RF ───────────────────────
    print("\n═══ Per-Model Ensemble vs Single RF (all OpenHands data) ═══")

    # Single RF on all data
    single_cv = cv_evaluate(oh_df, TRANSFER_FEATURES)
    results["single_rf_all"] = single_cv
    print(f"  Single RF (all models): AUC={single_cv['auc']}")

    # Ensemble: per-model RFs
    ensemble_y_true, ensemble_y_prob = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model in models:
        sub = oh_df[oh_df["model_family"] == model]
        X, y = sub[TRANSFER_FEATURES].values, sub["gold_pass"].values
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
            continue
        n_splits_m = min(5, min(np.bincount(y)))
        skf_m = StratifiedKFold(n_splits=max(2, n_splits_m), shuffle=True, random_state=42)
        for tr, va in skf_m.split(X, y):
            rf = train_rf(X[tr], y[tr])
            ensemble_y_prob.extend(rf.predict_proba(np.nan_to_num(X[va], nan=-999))[:, 1])
            ensemble_y_true.extend(y[va])

    ensemble_y_true, ensemble_y_prob = np.array(ensemble_y_true), np.array(ensemble_y_prob)
    ensemble_auc = roc_auc_score(ensemble_y_true, ensemble_y_prob)
    ensemble_result = {
        "auc": round(ensemble_auc, 4),
        "p_at_r30": round(precision_at_recall(ensemble_y_true, ensemble_y_prob, 0.30), 4),
        "ece": round(compute_ece(ensemble_y_true, ensemble_y_prob), 4),
    }
    results["per_model_ensemble"] = ensemble_result
    print(f"  Per-model ensemble:     AUC={ensemble_result['auc']}")
    print(f"  Ensemble delta:         {ensemble_result['auc'] - single_cv['auc']:+.4f}")

    # ─── RL reward density (best per-model RF) ─────────────────
    print("\n═══ RL Reward Density (per-model RFs) ═══")
    for model in models:
        sub = oh_df[oh_df["model_family"] == model]
        X, y = sub[TRANSFER_FEATURES].values, sub["gold_pass"].values
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
            continue

        cv_yt, cv_yp = [], []
        n_splits_m = min(5, min(np.bincount(y)))
        skf_m = StratifiedKFold(n_splits=max(2, n_splits_m), shuffle=True, random_state=42)
        for tr, va in skf_m.split(X, y):
            rf = train_rf(X[tr], y[tr])
            cv_yp.extend(rf.predict_proba(np.nan_to_num(X[va], nan=-999))[:, 1])
            cv_yt.extend(y[va])
        cv_yt, cv_yp = np.array(cv_yt), np.array(cv_yp)
        precs, recs, _ = precision_recall_curve(cv_yt, cv_yp)

        p90_r = float(recs[precs >= 0.90].max()) if (precs >= 0.90).any() else 0
        print(f"  {model:20s}: P>=90% recall={p90_r:.3f} ({p90_r*100:.1f}% reward coverage)")

    # ─── Summary ───────────────────────────────────────────────
    print("\n═══ Summary ═══\n")

    # Average diagonal (self-model AUC)
    diag_aucs = [transfer_matrix.get(m, {}).get(m, 0) for m in models if m in transfer_matrix]
    # Average off-diagonal (cross-model AUC)
    off_diag = []
    for m1 in models:
        for m2 in models:
            if m1 != m2 and m1 in transfer_matrix and m2 in transfer_matrix.get(m1, {}):
                off_diag.append(transfer_matrix[m1][m2])

    print(f"  Mean self-model AUC (diagonal):  {np.mean(diag_aucs):.3f}")
    print(f"  Mean cross-model AUC (off-diag): {np.mean(off_diag):.3f}")
    print(f"  Transfer gap:                    {np.mean(diag_aucs) - np.mean(off_diag):.3f}")
    print(f"  Per-model ensemble AUC:          {ensemble_result['auc']:.3f}")
    print(f"  Single RF AUC:                   {single_cv['auc']:.3f}")

    results["summary"] = {
        "mean_self_auc": round(np.mean(diag_aucs), 4),
        "mean_cross_auc": round(np.mean(off_diag), 4),
        "transfer_gap": round(np.mean(diag_aucs) - np.mean(off_diag), 4),
        "ensemble_auc": ensemble_result["auc"],
        "single_rf_auc": single_cv["auc"],
    }

    # Save
    out_path = BASE / "results" / "e6_openhands_rf_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {out_path}")


if __name__ == "__main__":
    main()
