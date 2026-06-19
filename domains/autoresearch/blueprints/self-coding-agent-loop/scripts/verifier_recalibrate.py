#!/usr/bin/env python3
"""verifier_recalibrate.py — per-round Loop 1 recalibration.

After round N's gold eval lands, retrain the RF on cumulative (gen, gold) labels from
rounds 1..N. Measure ECE/precision on drift_audit_300 at Gen_N. Append to drift_trajectory.json.

Input:
  --round N
  --control-gold  round_N/control_gold_results.jsonl     (labels for round_N_control)
  --drift-gold    round_N/drift_audit_gold_results.jsonl (labels for drift_audit_300 at Gen_N)
  --control-predictions   round_N/control_predictions.jsonl (agent patches)
  --drift-predictions     round_N/drift_audit_predictions.jsonl (agent patches)
  --prior-rf    round_{N-1}/rf.pkl OR v1b_bootstrap/rf_recalibrated.pkl for round 1
  --output-dir  round_N/

Output:
  round_N/rf.pkl                   — Gen_N verifier (RF retrained on cumulative labels)
  round_N/verifier_metrics.json    — ECE, precision, agreement on drift_audit_300 at Gen_N
  round_N/drift_trajectory.json    — appended-to running file: [{round, metrics...}, ...]

v009 rubric is intentionally NOT retrained here — reward-hacking guardrail.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from v1b_bootstrap import FEATURES, featurize_from_trajectory, compute_ece


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def join_predictions_with_gold(pred_path: Path, gold_path: Path) -> pd.DataFrame:
    preds = load_jsonl(pred_path)
    gold = {r["instance_id"]: r for r in load_jsonl(gold_path)}
    rows = []
    for p in preds:
        iid = p["instance_id"]
        if iid not in gold:
            continue
        row = {
            "instance_id": iid,
            "resolved": int(gold[iid].get("resolved", 0)),
            "error": int(gold[iid].get("error", 0)),
        }
        # Featurize from the prediction record — assumes round_runner writes trajectory-shaped output
        row.update(featurize_from_trajectory(p))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    X = np.nan_to_num(X, nan=-999)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=7, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def eval_verifier(rf: RandomForestClassifier, df: pd.DataFrame) -> dict:
    X = np.nan_to_num(df[FEATURES].values.astype(float), nan=-999)
    y = df["resolved"].astype(int).values
    prob = rf.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    if len(set(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, prob))

    precisions, recalls, _ = precision_recall_curve(y, prob)

    def p_at_r(r_target):
        mask = recalls >= r_target
        return float(precisions[mask].max()) if mask.any() else 0.0

    agreement = float((pred == y).mean())
    gold_pass_rate = float(y.mean())
    verifier_pass_rate = float(pred.mean())

    return {
        "n": int(len(y)),
        "gold_pass_rate": gold_pass_rate,
        "verifier_pass_rate": verifier_pass_rate,
        "agreement": agreement,
        "auc": auc,
        "ece": compute_ece(y, prob),
        "p_at_r30": p_at_r(0.30),
        "p_at_r50": p_at_r(0.50),
        "precision_at_0.5": float((pred & y).sum() / pred.sum()) if pred.sum() else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--control-predictions", required=True)
    p.add_argument("--control-gold", required=True)
    p.add_argument("--drift-predictions", required=True)
    p.add_argument("--drift-gold", required=True)
    p.add_argument("--prior-rf", required=True, help="Round_{N-1}/rf.pkl or v1b_bootstrap/rf_recalibrated.pkl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--drift-trajectory", default=None,
                   help="Running drift trajectory json (default: output-dir/../drift_trajectory.json)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    drift_traj_path = Path(args.drift_trajectory) if args.drift_trajectory else out.parent / "drift_trajectory.json"

    print(f"[verifier_recalibrate] round {args.round}")

    # Step 1: featurize + join predictions with gold for both sets
    ctrl_df = join_predictions_with_gold(Path(args.control_predictions), Path(args.control_gold))
    drift_df = join_predictions_with_gold(Path(args.drift_predictions), Path(args.drift_gold))
    print(f"  control_N: n={len(ctrl_df)} gold_pass={ctrl_df['resolved'].mean():.3f}")
    print(f"  drift_audit: n={len(drift_df)} gold_pass={drift_df['resolved'].mean():.3f}")

    # Step 2: evaluate the PRIOR verifier on drift_audit (measures drift since last round)
    with open(args.prior_rf, "rb") as f:
        prior_blob = pickle.load(f)
    prior_rf = prior_blob["rf"] if isinstance(prior_blob, dict) and "rf" in prior_blob else prior_blob
    print(f"[verifier_recalibrate] evaluating PRIOR verifier on drift_audit (drift signal)")
    prior_drift_metrics = eval_verifier(prior_rf, drift_df)
    print(f"  prior on drift: auc={prior_drift_metrics['auc']}, ece={prior_drift_metrics['ece']:.3f}, agreement={prior_drift_metrics['agreement']:.3f}")

    # Step 3: retrain RF on cumulative labels (round_1..N gold evals)
    # Accumulate by reading this round's control + all prior rounds' controls
    cumulative_train_dfs = [ctrl_df]
    for prev_n in range(1, args.round):
        prev_ctrl_path = out.parent / f"round_{prev_n}" / "control_gold_results.jsonl"
        prev_pred_path = out.parent / f"round_{prev_n}" / "control_predictions.jsonl"
        if prev_ctrl_path.exists() and prev_pred_path.exists():
            prev_df = join_predictions_with_gold(prev_pred_path, prev_ctrl_path)
            cumulative_train_dfs.append(prev_df)
    train_df = pd.concat(cumulative_train_dfs, ignore_index=True)
    print(f"[verifier_recalibrate] retraining RF on {len(train_df)} cumulative labels from rounds 1..{args.round}")

    X_train = np.nan_to_num(train_df[FEATURES].values.astype(float), nan=-999)
    y_train = train_df["resolved"].astype(int).values
    if len(set(y_train)) < 2:
        print("[verifier_recalibrate] WARN: single-class training data; skipping retrain, using prior")
        new_rf = prior_rf
    else:
        new_rf = fit_rf(X_train, y_train)

    # Step 4: evaluate the NEW verifier on drift_audit (this round's data point)
    new_drift_metrics = eval_verifier(new_rf, drift_df)
    print(f"[verifier_recalibrate] NEW verifier on drift_audit (Gen_{args.round} data point):")
    print(f"  auc={new_drift_metrics['auc']}, ece={new_drift_metrics['ece']:.3f}, agreement={new_drift_metrics['agreement']:.3f}")
    print(f"  p@r30={new_drift_metrics['p_at_r30']:.3f}, gold_rate={new_drift_metrics['gold_pass_rate']:.3f}")

    # Step 5: save Gen_N verifier
    with open(out / "rf.pkl", "wb") as f:
        pickle.dump({"rf": new_rf, "features": FEATURES, "round": args.round,
                     "n_cumulative_train": len(train_df)}, f)

    # Step 6: write verifier_metrics.json for this round
    metrics = {
        "round": args.round,
        "prior_verifier_on_drift_audit": prior_drift_metrics,
        "new_verifier_on_drift_audit": new_drift_metrics,
        "n_cumulative_train": int(len(train_df)),
        "control_gold_pass_rate": float(ctrl_df["resolved"].mean()),
        "drift_audit_gold_pass_rate": float(drift_df["resolved"].mean()),
    }
    with open(out / "verifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 7: append to drift trajectory
    if drift_traj_path.exists():
        with open(drift_traj_path) as f:
            trajectory = json.load(f)
    else:
        trajectory = []
    # Remove any existing entry for this round (idempotent re-run)
    trajectory = [t for t in trajectory if t.get("round") != args.round]
    trajectory.append({
        "round": args.round,
        "gen_id": f"Gen_{args.round}",
        "drift_audit_metrics": new_drift_metrics,
        "model_gold_pass_on_drift": new_drift_metrics["gold_pass_rate"],
        "verifier_ece_on_drift": new_drift_metrics["ece"],
        "verifier_agreement_on_drift": new_drift_metrics["agreement"],
        "verifier_p_at_r30_on_drift": new_drift_metrics["p_at_r30"],
    })
    trajectory.sort(key=lambda t: t["round"])
    with open(drift_traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[verifier_recalibrate] drift trajectory: {drift_traj_path} ({len(trajectory)} points)")


if __name__ == "__main__":
    main()
