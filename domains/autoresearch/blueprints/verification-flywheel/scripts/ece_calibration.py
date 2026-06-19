#!/usr/bin/env python3
"""
ECE Calibration Experiment — Verification Flywheel Phase 4.

Measures Expected Calibration Error for each verification signal to determine
RL/post-training feasibility. Uses existing Phase 1 cascade results + saved
RF models + fresh CoderForge traces.

Thresholds (from SWE-RM paper, see RLVR_AND_VERIFICATION.md):
  ECE < 0.1  → RL-ready (safe for GRPO reward signal)
  ECE 0.1-0.3 → best-of-N only (ranking safe, reward unsafe)
  ECE > 0.3  → ranking only (cannot use for RL)

Signals measured:
  1. Multiprompt Haiku (3-prompt consensus) — from Phase 1 cascade results
  2. RF (4-feature) — re-run on 1,200 traces with saved models
  3. Platt-calibrated RF — post-hoc calibration
  4. Platt-calibrated Multiprompt — post-hoc calibration

Usage:
  python3 ece_calibration.py                 # run full analysis
  python3 ece_calibration.py --n-traces 400  # use more traces for RF
  python3 ece_calibration.py --bootstrap 500 # more bootstrap samples
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from coderforge_adapter import from_coderforge_row

from learned_verifier.classifiers.rf_verifier import RFVerifier
from learned_verifier.metrics import compute_ece, evaluate
from learned_verifier.telemetry import extract_rf_features

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = RESULTS_DIR / "models"


def load_phase1_cascade():
    """Load Phase 1 cascade results with confidence + gold labels."""
    path = RESULTS_DIR / "phase1_cold_start.json"
    with open(path) as f:
        data = json.load(f)
    return data["cascade_results"]


def load_coderforge_traces(split, n, offset=0):
    """Load CoderForge traces and extract features + gold labels."""
    from datasets import load_dataset

    ds = load_dataset(
        "togethercomputer/CoderForge-Preview",
        "trajectories",
        split=split,
        streaming=True,
    )
    traces, labels = [], []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if len(traces) >= n:
            break
        trace, reward, _ = from_coderforge_row(row)
        traces.append(trace)
        labels.append(int(reward))
    return traces, labels


def get_rf_features(traces):
    """Extract 4-feature vectors from traces."""
    X = []
    for t in traces:
        feats = extract_rf_features(t)
        X.append([
            feats.get("beh_total_cost_usd", 0.15),
            feats.get("beh_tokens_per_edit", 5000.0),
            feats.get("beh_loop_count", 0.0),
            feats.get("svg_accepted", 0.0),
        ])
    return np.array(X)


def bootstrap_ece(y_true, y_prob, n_bootstrap=200, n_bins=10):
    """Bootstrap 95% CI for ECE."""
    eces = []
    rng = np.random.default_rng(42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ece = compute_ece(y_true[idx], y_prob[idx], n_bins=n_bins)
        eces.append(ece)
    return np.percentile(eces, 2.5), np.percentile(eces, 97.5)


def platt_calibrate(y_true_train, y_prob_train, y_prob_test):
    """Apply Platt scaling (logistic regression on logits)."""
    # Convert probs to logits, clipping to avoid inf
    eps = 1e-6
    logits_train = np.log(np.clip(y_prob_train, eps, 1 - eps) /
                          np.clip(1 - y_prob_train, eps, 1 - eps))
    logits_test = np.log(np.clip(y_prob_test, eps, 1 - eps) /
                         np.clip(1 - y_prob_test, eps, 1 - eps))

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits_train.reshape(-1, 1), y_true_train)

    calibrated = lr.predict_proba(logits_test.reshape(-1, 1))[:, 1]
    return calibrated, lr


def reliability_diagram(y_true, y_prob, n_bins=10, label=""):
    """Generate ASCII reliability diagram data + text."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        count = mask.sum()
        if count > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
        else:
            acc = 0
            conf = (bins[i] + bins[i + 1]) / 2
        bin_data.append({
            "bin_low": round(bins[i], 2),
            "bin_high": round(bins[i + 1], 2),
            "count": int(count),
            "accuracy": round(float(acc), 4),
            "confidence": round(float(conf), 4),
            "gap": round(abs(float(acc) - float(conf)), 4),
        })

    # ASCII diagram
    lines = [f"\n  Reliability Diagram: {label}", "  " + "-" * 52]
    lines.append(f"  {'Bin':>10} | {'Count':>5} | {'Acc':>6} | {'Conf':>6} | {'Gap':>5} | Visual")
    lines.append("  " + "-" * 52)
    for b in bin_data:
        bar_acc = int(b["accuracy"] * 20)
        bar_conf = int(b["confidence"] * 20)
        visual = ""
        for j in range(21):
            if j == bar_acc and j == bar_conf:
                visual += "X"  # overlap
            elif j == bar_acc:
                visual += "*"  # accuracy
            elif j == bar_conf:
                visual += "|"  # confidence
            else:
                visual += "."
        lines.append(
            f"  {b['bin_low']:.1f}-{b['bin_high']:.1f} | {b['count']:>5} | "
            f"{b['accuracy']:.3f} | {b['confidence']:.3f} | {b['gap']:.3f} | {visual}"
        )
    lines.append("  " + "-" * 52)
    lines.append("  Legend: * = accuracy, | = confidence, X = overlap")

    return bin_data, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ECE Calibration Experiment")
    parser.add_argument("--n-traces", type=int, default=1200,
                        help="Number of traces for RF evaluation")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Number of bootstrap samples for CI")
    args = parser.parse_args()

    print("=" * 60)
    print("ECE CALIBRATION EXPERIMENT")
    print("=" * 60)
    print(f"Goal: Determine RL-readiness of verification signals")
    print(f"Threshold: ECE < 0.1 for RL, ECE < 0.3 for best-of-N")
    print()

    results = {}

    # =========================================================
    # Signal 1: Multiprompt Haiku (from Phase 1 cascade results)
    # =========================================================
    print("--- Signal 1: Multiprompt Haiku ---")
    cascade_results = load_phase1_cascade()

    # Separate by tier
    mp_results = [r for r in cascade_results if r["tier"] == "multiprompt"]
    debate_results = [r for r in cascade_results if r["tier"] == "debate"]

    # Multiprompt confidence → probability
    # Confidence mapping: accept verdicts use confidence as-is,
    # reject verdicts use (1 - confidence) as accept probability
    mp_probs = []
    mp_labels = []
    for r in mp_results:
        if r["verdict"] == "accept":
            mp_probs.append(r["confidence"])
        else:
            # reject with confidence c → accept prob is (1 - c)
            # But confidence=0.0 for reject means "certain reject" → accept prob ~0
            mp_probs.append(1.0 - r["confidence"] if r["confidence"] > 0 else 0.0)
        mp_labels.append(r["gold_label"])

    mp_probs = np.array(mp_probs)
    mp_labels = np.array(mp_labels)

    mp_metrics = evaluate(mp_labels, mp_probs)
    mp_ece_lo, mp_ece_hi = bootstrap_ece(mp_labels, mp_probs, n_bootstrap=args.bootstrap)
    mp_bins, mp_diagram = reliability_diagram(mp_labels, mp_probs, label="Multiprompt Haiku")

    print(f"  n = {len(mp_labels)}")
    print(f"  Base rate: {mp_labels.mean():.3f}")
    print(f"  AUC: {mp_metrics['auc']}")
    print(f"  ECE: {mp_metrics['ece']} [{mp_ece_lo:.4f}, {mp_ece_hi:.4f}]")
    print(f"  Brier: {mp_metrics['brier']}")
    print(f"  RL-ready: {mp_metrics['rl_ready']}")
    print(f"  Best-of-N ready: {mp_metrics['bon_ready']}")
    print(mp_diagram)

    results["multiprompt"] = {
        **mp_metrics,
        "n": len(mp_labels),
        "base_rate": round(float(mp_labels.mean()), 4),
        "ece_95ci": [round(mp_ece_lo, 4), round(mp_ece_hi, 4)],
        "reliability_bins": mp_bins,
    }

    # Also compute for ALL cascade results (multiprompt + debate combined)
    print("\n--- Signal 1b: Full Cascade (multiprompt + debate) ---")
    all_probs = []
    all_labels = []
    for r in cascade_results:
        if r["verdict"] == "accept":
            all_probs.append(r["confidence"])
        else:
            all_probs.append(1.0 - r["confidence"] if r["confidence"] > 0 else 0.0)
        all_labels.append(r["gold_label"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    cascade_metrics = evaluate(all_labels, all_probs)
    cascade_ece_lo, cascade_ece_hi = bootstrap_ece(all_labels, all_probs, n_bootstrap=args.bootstrap)
    cascade_bins, cascade_diagram = reliability_diagram(all_labels, all_probs, label="Full Cascade")

    print(f"  n = {len(all_labels)}")
    print(f"  ECE: {cascade_metrics['ece']} [{cascade_ece_lo:.4f}, {cascade_ece_hi:.4f}]")
    print(f"  RL-ready: {cascade_metrics['rl_ready']}")
    print(cascade_diagram)

    results["cascade_full"] = {
        **cascade_metrics,
        "n": len(all_labels),
        "base_rate": round(float(all_labels.mean()), 4),
        "ece_95ci": [round(cascade_ece_lo, 4), round(cascade_ece_hi, 4)],
        "reliability_bins": cascade_bins,
    }

    # =========================================================
    # Signal 2: RF (4-feature) — re-run predictions on traces
    # =========================================================
    print("\n--- Signal 2: RF Classifier (4-feature) ---")

    # Load best RF from Phase 2
    rf_model_path = None
    for i in range(5, 0, -1):
        path = MODELS_DIR / f"rf_cycle{i}.pkl"
        if path.exists():
            rf_model_path = path
            break
    if rf_model_path is None:
        rf_model_path = MODELS_DIR / "rf_phase1.pkl"

    print(f"  Loading RF from {rf_model_path}")
    rf = RFVerifier(model_path=str(rf_model_path))

    print(f"  Loading {args.n_traces} CoderForge traces (SWE_Rebench)...")
    traces, gold_labels = load_coderforge_traces("SWE_Rebench", args.n_traces, offset=0)
    gold_labels = np.array(gold_labels)
    print(f"  Loaded {len(traces)} traces, base rate: {gold_labels.mean():.3f}")

    # RF predictions
    rf_probs = np.array([float(rf.predict_proba(t)) for t in traces])
    # Ensure binary labels (int), filter out any non-binary (-1 etc.)
    gold_labels = gold_labels.astype(int)
    valid_mask = (gold_labels == 0) | (gold_labels == 1)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Filtering {n_invalid} non-binary labels (values: {np.unique(gold_labels[~valid_mask])})")
        traces = [t for t, v in zip(traces, valid_mask) if v]
        rf_probs = rf_probs[valid_mask]
        gold_labels = gold_labels[valid_mask]
    # Clip probabilities to valid range
    rf_probs = np.clip(rf_probs, 0.0, 1.0)
    rf_metrics = evaluate(gold_labels, rf_probs)
    rf_ece_lo, rf_ece_hi = bootstrap_ece(gold_labels, rf_probs, n_bootstrap=args.bootstrap)
    rf_bins, rf_diagram = reliability_diagram(gold_labels, rf_probs, label="RF (4-feature)")

    print(f"  AUC: {rf_metrics['auc']}")
    print(f"  ECE: {rf_metrics['ece']} [{rf_ece_lo:.4f}, {rf_ece_hi:.4f}]")
    print(f"  Brier: {rf_metrics['brier']}")
    print(f"  RL-ready: {rf_metrics['rl_ready']}")
    print(rf_diagram)

    results["rf"] = {
        **rf_metrics,
        "n": len(traces),
        "base_rate": round(float(gold_labels.mean()), 4),
        "ece_95ci": [round(rf_ece_lo, 4), round(rf_ece_hi, 4)],
        "model_path": str(rf_model_path),
        "reliability_bins": rf_bins,
    }

    # =========================================================
    # Signal 3: Platt-calibrated RF
    # =========================================================
    print("\n--- Signal 3: Platt-calibrated RF ---")

    # 50/50 train/test split for calibration
    n_half = len(traces) // 2
    idx = np.random.default_rng(42).permutation(len(traces))
    train_idx, test_idx = idx[:n_half], idx[n_half:]

    rf_probs_train = rf_probs[train_idx]
    rf_probs_test = rf_probs[test_idx]
    gold_train = gold_labels[train_idx]
    gold_test = gold_labels[test_idx]

    cal_rf_probs, platt_model = platt_calibrate(gold_train, rf_probs_train, rf_probs_test)

    cal_rf_metrics = evaluate(gold_test, cal_rf_probs)
    cal_rf_ece_lo, cal_rf_ece_hi = bootstrap_ece(gold_test, cal_rf_probs, n_bootstrap=args.bootstrap)
    cal_rf_bins, cal_rf_diagram = reliability_diagram(gold_test, cal_rf_probs, label="Platt-calibrated RF")

    print(f"  n (test) = {len(gold_test)}")
    print(f"  Platt params: a={platt_model.coef_[0][0]:.4f}, b={platt_model.intercept_[0]:.4f}")
    print(f"  AUC: {cal_rf_metrics['auc']}")
    print(f"  ECE: {cal_rf_metrics['ece']} [{cal_rf_ece_lo:.4f}, {cal_rf_ece_hi:.4f}]")
    print(f"  Brier: {cal_rf_metrics['brier']}")
    print(f"  RL-ready: {cal_rf_metrics['rl_ready']}")
    print(f"  Improvement: ECE {rf_metrics['ece']} → {cal_rf_metrics['ece']}")
    print(cal_rf_diagram)

    results["rf_platt"] = {
        **cal_rf_metrics,
        "n": len(gold_test),
        "base_rate": round(float(gold_test.mean()), 4),
        "ece_95ci": [round(cal_rf_ece_lo, 4), round(cal_rf_ece_hi, 4)],
        "platt_a": round(platt_model.coef_[0][0], 4),
        "platt_b": round(platt_model.intercept_[0], 4),
        "reliability_bins": cal_rf_bins,
    }

    # =========================================================
    # Signal 4: Platt-calibrated Multiprompt
    # =========================================================
    print("\n--- Signal 4: Platt-calibrated Multiprompt ---")

    # Use leave-one-out style: 5-fold cross-val calibration
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cal_mp_probs_all = np.zeros_like(mp_probs)

    for fold, (train_i, test_i) in enumerate(kf.split(mp_probs)):
        cal_probs, _ = platt_calibrate(mp_labels[train_i], mp_probs[train_i], mp_probs[test_i])
        cal_mp_probs_all[test_i] = cal_probs

    cal_mp_metrics = evaluate(mp_labels, cal_mp_probs_all)
    cal_mp_ece_lo, cal_mp_ece_hi = bootstrap_ece(mp_labels, cal_mp_probs_all, n_bootstrap=args.bootstrap)
    cal_mp_bins, cal_mp_diagram = reliability_diagram(mp_labels, cal_mp_probs_all, label="Platt-calibrated Multiprompt")

    print(f"  n = {len(mp_labels)} (5-fold cross-validated)")
    print(f"  AUC: {cal_mp_metrics['auc']}")
    print(f"  ECE: {cal_mp_metrics['ece']} [{cal_mp_ece_lo:.4f}, {cal_mp_ece_hi:.4f}]")
    print(f"  Brier: {cal_mp_metrics['brier']}")
    print(f"  RL-ready: {cal_mp_metrics['rl_ready']}")
    print(f"  Improvement: ECE {mp_metrics['ece']} → {cal_mp_metrics['ece']}")
    print(cal_mp_diagram)

    results["multiprompt_platt"] = {
        **cal_mp_metrics,
        "n": len(mp_labels),
        "base_rate": round(float(mp_labels.mean()), 4),
        "ece_95ci": [round(cal_mp_ece_lo, 4), round(cal_mp_ece_hi, 4)],
        "reliability_bins": cal_mp_bins,
    }

    # =========================================================
    # Summary & RL-Readiness Verdict
    # =========================================================
    print("\n" + "=" * 60)
    print("ECE CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Signal':<25} {'ECE':>8} {'95% CI':>18} {'AUC':>8} {'Brier':>8} {'RL?':>5} {'BoN?':>5}")
    print("-" * 78)

    for name, label in [
        ("multiprompt", "Multiprompt Haiku"),
        ("cascade_full", "Full Cascade"),
        ("rf", "RF (4-feature)"),
        ("rf_platt", "RF + Platt"),
        ("multiprompt_platt", "Multiprompt + Platt"),
    ]:
        r = results[name]
        ci = f"[{r['ece_95ci'][0]:.3f}, {r['ece_95ci'][1]:.3f}]"
        rl = "YES" if r["rl_ready"] else "no"
        bon = "YES" if r["bon_ready"] else "no"
        print(f"  {label:<23} {r['ece']:>8.4f} {ci:>18} {r['auc']:>8.4f} {r['brier']:>8.4f} {rl:>5} {bon:>5}")

    # RL-readiness verdict
    print("\n" + "=" * 60)
    print("RL-READINESS VERDICT")
    print("=" * 60)

    any_rl_ready = any(results[k]["rl_ready"] for k in results)
    any_bon_ready = any(results[k]["bon_ready"] for k in results)

    if any_rl_ready:
        rl_signals = [k for k in results if results[k]["rl_ready"]]
        print(f"\n  RL-READY signals: {', '.join(rl_signals)}")
        print("  → Can proceed with GRPO using these signals as reward")
        print("  → Recommended: rejection sampling SFT first, then GRPO")
    elif any_bon_ready:
        bon_signals = [k for k in results if results[k]["bon_ready"]]
        print(f"\n  BEST-OF-N ready signals: {', '.join(bon_signals)}")
        print("  → Safe for best-of-N selection and rejection sampling SFT")
        print("  → NOT safe for RL (ECE too high, risk of reward hacking)")
        print("  → Recommendation: use best-of-N / rejection sampling SFT")
        print("    (Shopify pattern), collect calibration data for RL later")
    else:
        print("\n  NO signals are calibrated enough for RL or best-of-N")
        print("  → All signals are ranking-only (ECE > 0.3)")
        print("  → Recommendation: collect more diverse evaluation data")

    # Specific recommendations
    print("\n  Recommendations:")
    for name, label in [
        ("multiprompt", "Multiprompt"),
        ("rf", "RF"),
        ("multiprompt_platt", "Multiprompt+Platt"),
        ("rf_platt", "RF+Platt"),
    ]:
        r = results[name]
        if r["rl_ready"]:
            print(f"    {label}: ECE={r['ece']:.3f} — RL-ready, "
                  f"safe for GRPO reward signal")
        elif r["bon_ready"]:
            print(f"    {label}: ECE={r['ece']:.3f} — best-of-N only, "
                  f"Platt calibration {'helped' if 'platt' in name else 'may help'}")
        else:
            print(f"    {label}: ECE={r['ece']:.3f} — ranking only, "
                  f"needs more data or better features")

    # Save
    out_path = RESULTS_DIR / "phase4_ece_calibration.json"
    # Remove reliability_bins for cleaner output (large)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != "reliability_bins"}
        save_results[k]["reliability_bins_count"] = len(v.get("reliability_bins", []))

    with open(out_path, "w") as f:
        json.dump({"phase": 4, "experiment": "ece_calibration", "signals": save_results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
