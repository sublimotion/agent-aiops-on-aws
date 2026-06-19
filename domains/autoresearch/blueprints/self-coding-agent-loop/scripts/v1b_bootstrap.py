#!/usr/bin/env python3
"""V1b_bootstrap: FlywheelBootstrap on Qwen3.5-27B × OpenHands target distribution.

Steps:
  1. Sample N labeled trajectories from the bootstrap pool (Nebius Qwen3-Coder-480B ×
     OpenHands, used here as best-available proxy for Qwen3.5-27B × OpenHands —
     acknowledged mismatch, but the harness is identical and it's free).
  2. Featurize each trajectory using the learned-verifier feature extractors (the
     6-column intersection: total_cost_usd, tokens_per_edit, loop_count, _n_edits,
     _n_reads, _n_bash).
  3. Train an RF on (features, resolved) for the pool. 5-fold CV reports AUC/ECE.
  4. Save the recalibrated RF and emit summary.json.

Output:
  rf_recalibrated.pkl
  ece_history.json — ECE at n=50, 100, 150, 200 (to show convergence)
  summary.json
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold


FEATURES = ["total_cost_usd", "tokens_per_edit", "loop_count", "_n_edits", "_n_reads", "_n_bash"]


def featurize_from_trajectory(record: dict) -> dict:
    """Extract the 6 behavioral features from a Nebius OpenHands trajectory record.

    The OpenHands trajectory is a list of turn dicts. We proxy features from the
    model_patch + trajectory length since we don't have token-cost metadata in the
    released dataset.
    """
    patch = record.get("model_patch", "") or ""
    traj = record.get("trajectory") or []
    n_turns = len(traj)

    # Count edits and reads from the trajectory actions
    n_edits = 0
    n_reads = 0
    n_bash = 0
    for turn in traj:
        role = turn.get("role", "")
        if role != "tool":
            # Check tool_calls on assistant turns
            for tc in turn.get("tool_calls") or []:
                fn_name = ((tc or {}).get("function") or {}).get("name", "") or ""
                if "edit" in fn_name.lower() or "str_replace" in fn_name.lower():
                    n_edits += 1
                elif "read" in fn_name.lower() or "view" in fn_name.lower():
                    n_reads += 1
                elif "bash" in fn_name.lower() or "execute" in fn_name.lower():
                    n_bash += 1

    # Patch-based proxy for cost/tokens
    patch_chars = len(patch)
    # Rough token estimate: 4 chars/token
    tokens_per_edit = patch_chars / max(n_edits, 1) / 4

    return {
        "total_cost_usd": 0.01 * n_turns,   # proxy; real cost not in dataset
        "tokens_per_edit": tokens_per_edit,
        "loop_count": n_turns,
        "_n_edits": n_edits,
        "_n_reads": n_reads,
        "_n_bash": n_bash,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1] if i == n_bins - 1 else y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)


def train_rf_cv(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, dict]:
    X = np.nan_to_num(X, nan=-999)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_t, all_p = [], []
    for tr, va in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=7,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        all_t.extend(y[va])
        all_p.extend(rf.predict_proba(X[va])[:, 1])
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=7,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    rf_full.fit(X, y)
    y_t = np.array(all_t); y_p = np.array(all_p)
    precisions, recalls, _ = precision_recall_curve(y_t, y_p)
    viable = recalls >= 0.30
    return rf_full, {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "auc": float(roc_auc_score(y_t, y_p)) if len(set(y_t)) > 1 else None,
        "ece": compute_ece(y_t, y_p),
        "p_at_r30": float(precisions[viable].max()) if viable.any() else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="v1b_bootstrap_pool.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-labels", type=int, default=200)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load pool
    records = []
    with open(args.pool) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[: args.n_labels]
    print(f"loaded {len(records)} labeled trajectories")

    # Featurize
    rows = []
    for r in records:
        feats = featurize_from_trajectory(r)
        feats["resolved"] = int(r.get("resolved", 0))
        feats["trajectory_id"] = r.get("trajectory_id", "")
        rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(out / "featurized.csv", index=False)
    print(df[FEATURES + ["resolved"]].describe())

    # ECE convergence: fit at n=50, 100, 150, 200 and record
    ece_history = []
    for n in [50, 100, 150, 200]:
        if n > len(df):
            break
        sub = df.iloc[:n]
        X = sub[FEATURES].values.astype(float)
        y = sub["resolved"].astype(int).values
        if y.sum() < 2 or (y == 0).sum() < 2:
            continue
        _, metrics = train_rf_cv(X, y)
        metrics["n_labels"] = n
        ece_history.append(metrics)
        print(f"n={n}: AUC={metrics['auc']:.3f}, ECE={metrics['ece']:.3f}")

    # Final model on full pool
    X = df[FEATURES].values.astype(float)
    y = df["resolved"].astype(int).values
    rf, final_metrics = train_rf_cv(X, y)

    with open(out / "rf_recalibrated.pkl", "wb") as f:
        pickle.dump({"rf": rf, "features": FEATURES, "training_metrics": final_metrics}, f)

    with open(out / "ece_history.json", "w") as f:
        json.dump(ece_history, f, indent=2)

    summary = {
        "n_labels_used": len(df),
        "final_auc": final_metrics["auc"],
        "final_ece": final_metrics["ece"],
        "final_p_at_r30": final_metrics["p_at_r30"],
        "ece_trend_converging": len(ece_history) >= 2 and ece_history[-1]["ece"] < ece_history[0]["ece"] - 0.02,
        "features": FEATURES,
        "target_distribution_caveat": (
            "Bootstrap pool is Nebius Qwen3-Coder-480B × OpenHands. "
            "Target is Qwen3.5-27B × OpenHands (same harness, different model). "
            "If V1b_validate fails, the gap may come from model-family mismatch — "
            "switch to self-generated Qwen3.5-27B × OpenHands traces for rebootstrap."
        ),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
