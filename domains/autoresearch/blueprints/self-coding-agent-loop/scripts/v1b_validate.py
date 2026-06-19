#!/usr/bin/env python3
"""V1b_validate: check that the recalibrated RF reaches V1b-unlock level on the
calibration set. V1b-unlock = precision >= 0.70 at any usable recall point.

Input:
  --rf rf_recalibrated.pkl    (from v1b_bootstrap)
  --calibration <jsonl>       (the same bootstrap pool or a held-out subset)
Output:
  summary.json with decision: pass / fail / extend_bootstrap
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score


# Reuse the featurizer from v1b_bootstrap
import sys
sys.path.insert(0, str(Path(__file__).parent))
from v1b_bootstrap import featurize_from_trajectory, FEATURES, compute_ece


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rf", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--v1b-unlock-precision", type=float, default=0.70)
    p.add_argument("--sft-ready-precision", type=float, default=0.85)
    p.add_argument("--rl-ready-precision", type=float, default=0.90)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.rf, "rb") as f:
        blob = pickle.load(f)
    rf = blob["rf"]
    features = blob["features"]

    records = []
    with open(args.calibration) as f:
        for line in f:
            records.append(json.loads(line))
    rows = []
    for r in records:
        feats = featurize_from_trajectory(r)
        feats["resolved"] = int(r.get("resolved", 0))
        rows.append(feats)
    df = pd.DataFrame(rows)

    X = np.nan_to_num(df[features].values.astype(float), nan=-999)
    y = df["resolved"].astype(int).values
    prob = rf.predict_proba(X)[:, 1]

    auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else None
    ece = compute_ece(y, prob)
    precisions, recalls, thresholds = precision_recall_curve(y, prob)

    def p_at_recall(target_r: float) -> float:
        mask = recalls >= target_r
        return float(precisions[mask].max()) if mask.any() else 0.0

    p_at_r30 = p_at_recall(0.30)

    # Decision
    if p_at_r30 >= args.rl_ready_precision and ece < 0.1:
        level = "rl_ready"
        decision = "pass_rl_ready_unlocks_arm_d_phase2"
    elif p_at_r30 >= args.sft_ready_precision and ece < 0.3:
        level = "sft_ready"
        decision = "pass_sft_ready_unlocks_arms_c_e"
    elif p_at_r30 >= args.v1b_unlock_precision:
        level = "v1b_unlock"
        decision = "pass_v1b_unlock_arms_c_d_e_may_start"
    else:
        level = "below_threshold"
        # Decide whether to extend bootstrap or stop
        # Look at ECE history if present
        ece_history_path = Path(args.rf).parent / "ece_history.json"
        if ece_history_path.exists():
            with open(ece_history_path) as f:
                history = json.load(f)
            if len(history) >= 2 and history[-1]["ece"] < history[0]["ece"] - 0.02:
                decision = "fail_ece_trending_down_extend_bootstrap_200_more_labels"
            else:
                decision = "fail_ece_flat_rubric_bottleneck_run_v1b_rubric_rebuild"
        else:
            decision = "fail_extend_bootstrap_to_400_labels"

    summary = {
        "n_calibration": int(len(df)),
        "auc": auc,
        "ece": ece,
        "p_at_r30": p_at_r30,
        "readiness_level": level,
        "decision": decision,
        "thresholds": {
            "v1b_unlock_precision": args.v1b_unlock_precision,
            "sft_ready_precision": args.sft_ready_precision,
            "rl_ready_precision": args.rl_ready_precision,
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
