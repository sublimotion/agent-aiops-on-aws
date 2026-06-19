#!/usr/bin/env python3
"""
Verification Flywheel Demo — Phases 1-3.

Demonstrates the self-bootstrapping evaluation flywheel using
CoderForge-Preview trajectories (155K Docker-verified, 1,655 repos).

No GPU required. Uses:
  - Tier 0: RF classifier (CPU, free, <1ms)
  - Tier 1: Multiprompt content verifier (Bedrock API, $0.029/patch)
  - Gold labels: Docker-verified reward field from CoderForge

Phases:
  1. Cold Start Bootstrap — train initial RF from 200 content-verifier-labeled traces
  2. Flywheel Iteration — 5 cycles, measure cost reduction as RF handles more
  3. OOD Generalization — test RF on SWE_Smith split (trained on SWE_Rebench)

Usage:
  python3 flywheel_demo.py --phase 1       # cold start only
  python3 flywheel_demo.py --phase 2       # flywheel iteration (requires phase 1)
  python3 flywheel_demo.py --phase 3       # OOD test (requires phase 2)
  python3 flywheel_demo.py --all           # run all phases
  python3 flywheel_demo.py --phase 1 --rf-only  # skip LLM calls, use gold labels as silver

Skill reference: verification-cascade (learned-verifier blueprint)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
from datasets import load_dataset

# Local adapter
sys.path.insert(0, str(Path(__file__).parent))
from coderforge_adapter import from_coderforge_messages, from_coderforge_row

from learned_verifier.cascade import Cascade
from learned_verifier.classifiers.rf_verifier import RFVerifier
from learned_verifier.schemas import TraceInput, Verdict
from learned_verifier.telemetry import extract_rf_features

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 200  # traces per flywheel cycle
N_FLYWHEEL_CYCLES = 5


def load_coderforge_split(split: str, n: int, offset: int = 0):
    """Load n rows from a CoderForge split, starting at offset."""
    ds = load_dataset(
        "togethercomputer/CoderForge-Preview",
        "trajectories",
        split=split,
        streaming=True,
    )
    rows = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if len(rows) >= n:
            break
        rows.append(row)
    return rows


def extract_features_and_labels(rows):
    """Convert CoderForge rows to (traces, gold_labels, problems, diffs)."""
    traces = []
    labels = []
    problems = []
    diffs = []

    for row in rows:
        trace, reward, diff = from_coderforge_row(row)
        traces.append(trace)
        labels.append(int(reward))
        problems.append(trace.problem_statement or "")
        diffs.append(diff)

    return traces, labels, problems, diffs


def build_rf_feature_matrix(traces):
    """Extract RF feature vectors from traces."""
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


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute accuracy, precision, recall, F1, and optionally AUC."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "n": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "base_rate": round(sum(y_true) / len(y_true), 4) if y_true else 0,
    }

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["auc"] = None

    return metrics


def run_cascade_batch(cascade, traces, problems, diffs, gold_labels):
    """Run cascade on a batch, return verdicts and cost."""
    results = []
    total_cost = 0.0
    tier_counts = {"rf": 0, "multiprompt": 0, "debate": 0, "uncertain": 0}

    for i, (trace, problem, diff) in enumerate(zip(traces, problems, diffs)):
        t0 = time.time()
        try:
            result = cascade.verify(
                trace=trace,
                problem=problem if problem else None,
                diff=diff if diff else None,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}", flush=True)
            results.append({
                "instance_id": trace.instance_id,
                "verdict": "uncertain",
                "confidence": 0.5,
                "gold_label": gold_labels[i],
                "tier": "error",
                "cost_usd": 0,
                "latency_s": 0,
                "error": str(e),
            })
            tier_counts["error"] = tier_counts.get("error", 0) + 1
            continue
        elapsed = time.time() - t0

        verdict_int = 1 if result.verdict == Verdict.ACCEPT else 0
        tier = result.tier_name or "uncertain"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_cost += result.cost_usd

        results.append({
            "instance_id": trace.instance_id,
            "verdict": result.verdict.value,
            "confidence": round(result.confidence, 4),
            "gold_label": gold_labels[i],
            "tier": tier,
            "cost_usd": round(result.cost_usd, 6),
            "latency_s": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(traces)}] cost=${total_cost:.2f}, "
                  f"tiers: {dict(tier_counts)}", flush=True)

    return results, total_cost, tier_counts


def phase1_cold_start(args):
    """Phase 1: Bootstrap RF from 200 traces with content verifier labels."""
    print("=" * 60)
    print("PHASE 1: Cold Start Bootstrap")
    print("=" * 60)

    print(f"\nLoading {BATCH_SIZE} CoderForge trajectories (SWE_Rebench split)...")
    rows = load_coderforge_split("SWE_Rebench", BATCH_SIZE, offset=0)
    traces, gold_labels, problems, diffs = extract_features_and_labels(rows)

    base_rate = sum(gold_labels) / len(gold_labels)
    print(f"Loaded {len(traces)} traces, base rate: {base_rate:.2%}")

    if args.rf_only:
        # Use gold labels as silver labels (skip LLM calls)
        print("\n--rf-only mode: using gold Docker labels as silver labels (no API calls)")
        silver_labels = gold_labels
        cascade_results = None
        total_cost = 0.0
    else:
        # Run full cascade WITHOUT RF (cold start — no RF exists yet)
        print("\nRunning cascade WITHOUT RF (cold start, Tier 1 for all)...")
        cascade = Cascade.default(
            provider=args.provider,
            model=args.model,
            include_rf=False,
        )
        cascade_results, total_cost, tier_counts = run_cascade_batch(
            cascade, traces, problems, diffs, gold_labels,
        )
        silver_labels = [
            1 if r["verdict"] == "accept" else 0
            for r in cascade_results
        ]
        print(f"\nCascade complete: ${total_cost:.2f} total, "
              f"${total_cost/len(traces):.4f}/patch")

    # Train RF on silver labels
    print("\nTraining RF on silver labels...")
    X = build_rf_feature_matrix(traces)
    y = np.array(silver_labels)

    rf = RFVerifier()
    rf.train(X, y)

    model_path = MODELS_DIR / "rf_phase1.pkl"
    rf.save(str(model_path))
    print(f"RF saved to {model_path}")

    # Evaluate RF against gold labels
    rf_probs = [rf.predict_proba(t) for t in traces]
    rf_preds = [1 if p >= 0.5 else 0 for p in rf_probs]
    rf_metrics = compute_metrics(gold_labels, rf_preds, rf_probs)

    print(f"\nRF performance (vs gold Docker labels):")
    print(f"  AUC:       {rf_metrics.get('auc', 'N/A')}")
    print(f"  Accuracy:  {rf_metrics['accuracy']}")
    print(f"  Precision: {rf_metrics['precision']}")
    print(f"  Recall:    {rf_metrics['recall']}")
    print(f"  F1:        {rf_metrics['f1']}")

    # Silver vs gold agreement
    if not args.rf_only:
        silver_vs_gold = compute_metrics(gold_labels, silver_labels)
        print(f"\nSilver labels (cascade) vs gold Docker labels:")
        print(f"  Accuracy:  {silver_vs_gold['accuracy']}")
        print(f"  Precision: {silver_vs_gold['precision']}")
        print(f"  Recall:    {silver_vs_gold['recall']}")

    # Save results
    phase1_results = {
        "phase": 1,
        "n_traces": len(traces),
        "base_rate": base_rate,
        "total_cost": round(total_cost, 4),
        "cost_per_patch": round(total_cost / len(traces), 6) if total_cost > 0 else 0,
        "rf_metrics": rf_metrics,
        "rf_only": args.rf_only,
    }
    if cascade_results:
        phase1_results["cascade_results"] = cascade_results

    results_path = RESULTS_DIR / "phase1_cold_start.json"
    with open(results_path, "w") as f:
        json.dump(phase1_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return rf


def phase2_flywheel(args, rf=None):
    """Phase 2: Flywheel iteration — 5 cycles, measure cost reduction."""
    print("\n" + "=" * 60)
    print("PHASE 2: Flywheel Iteration")
    print("=" * 60)

    if rf is None:
        model_path = MODELS_DIR / "rf_phase1.pkl"
        if not model_path.exists():
            print(f"ERROR: {model_path} not found. Run --phase 1 first.")
            return None
        rf = RFVerifier(model_path=str(model_path))
        print(f"Loaded RF from {model_path}")

    cycle_results = []
    accumulated_traces = []
    accumulated_labels = []

    for cycle in range(N_FLYWHEEL_CYCLES):
        offset = BATCH_SIZE + (cycle * BATCH_SIZE)  # skip phase 1 data
        print(f"\n--- Cycle {cycle + 1}/{N_FLYWHEEL_CYCLES} "
              f"(offset={offset}, n={BATCH_SIZE}) ---")

        rows = load_coderforge_split("SWE_Rebench", BATCH_SIZE, offset=offset)
        if len(rows) < BATCH_SIZE:
            print(f"  Only {len(rows)} rows available, stopping.")
            break

        traces, gold_labels, problems, diffs = extract_features_and_labels(rows)
        base_rate = sum(gold_labels) / len(gold_labels)

        if args.rf_only:
            # RF-only evaluation (no cascade calls)
            rf_probs = [rf.predict_proba(t) for t in traces]
            rf_preds = [1 if p >= 0.5 else 0 for p in rf_probs]
            rf_metrics = compute_metrics(gold_labels, rf_preds, rf_probs)

            # Use gold labels as silver for retraining
            silver_labels = gold_labels
            total_cost = 0.0
            rf_resolved_pct = 1.0
            tier_counts = {"rf": len(traces)}
        else:
            # Run cascade WITH RF
            cascade = Cascade.default(
                provider=args.provider,
                model=args.model,
                include_rf=True,
            )
            # Inject our trained RF
            cascade.steps[0].verifier = rf

            cascade_results, total_cost, tier_counts = run_cascade_batch(
                cascade, traces, problems, diffs, gold_labels,
            )
            silver_labels = [
                1 if r["verdict"] == "accept" else 0
                for r in cascade_results
            ]

            # RF resolution rate (RF reports as "rf_4feature")
            rf_resolved = tier_counts.get("rf", 0) + tier_counts.get("rf_4feature", 0)
            rf_resolved_pct = rf_resolved / len(traces) if traces else 0

            # RF-only metrics (for comparison)
            rf_probs = [rf.predict_proba(t) for t in traces]
            rf_preds = [1 if p >= 0.5 else 0 for p in rf_probs]
            rf_metrics = compute_metrics(gold_labels, rf_preds, rf_probs)

        # Accumulate for retraining
        accumulated_traces.extend(traces)
        accumulated_labels.extend(silver_labels)

        # Retrain RF on accumulated silver labels
        X_all = build_rf_feature_matrix(accumulated_traces)
        y_all = np.array(accumulated_labels)
        rf.train(X_all, y_all)
        rf.save(str(MODELS_DIR / f"rf_cycle{cycle + 1}.pkl"))

        cycle_result = {
            "cycle": cycle + 1,
            "n_traces": len(traces),
            "accumulated_traces": len(accumulated_traces),
            "base_rate": round(base_rate, 4),
            "total_cost": round(total_cost, 4),
            "cost_per_patch": round(total_cost / len(traces), 6) if total_cost > 0 else 0,
            "rf_resolved_pct": round(rf_resolved_pct, 4),
            "tier_counts": dict(tier_counts),
            "rf_metrics_vs_gold": rf_metrics,
        }
        cycle_results.append(cycle_result)

        print(f"  Base rate: {base_rate:.2%}")
        print(f"  Cost: ${total_cost:.2f} (${total_cost/len(traces):.4f}/patch)")
        print(f"  RF resolved: {rf_resolved_pct:.1%}")
        print(f"  RF AUC (vs gold): {rf_metrics.get('auc', 'N/A')}")
        print(f"  RF accuracy: {rf_metrics['accuracy']}")
        print(f"  Accumulated training data: {len(accumulated_traces)}")

    # Summary
    print("\n" + "=" * 60)
    print("FLYWHEEL SUMMARY")
    print("=" * 60)
    print(f"{'Cycle':<8} {'Cost/patch':>12} {'RF resolved':>13} {'RF AUC':>10} {'Accuracy':>10}")
    print("-" * 53)
    for c in cycle_results:
        auc_val = c['rf_metrics_vs_gold'].get('auc')
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
        print(f"{c['cycle']:<8} ${c['cost_per_patch']:>10.4f} "
              f"{c['rf_resolved_pct']:>12.1%} "
              f"{auc_str:>10} "
              f"{c['rf_metrics_vs_gold']['accuracy']:>10}")

    # Save
    results_path = RESULTS_DIR / "phase2_flywheel.json"
    with open(results_path, "w") as f:
        json.dump({"phase": 2, "cycles": cycle_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return rf


def phase3_ood(args, rf=None):
    """Phase 3: OOD Generalization — test RF trained on SWE_Rebench against SWE_Smith."""
    print("\n" + "=" * 60)
    print("PHASE 3: OOD Generalization (SWE_Rebench → SWE_Smith)")
    print("=" * 60)

    # Load best RF from phase 2
    if rf is None:
        best_model = None
        for i in range(N_FLYWHEEL_CYCLES, 0, -1):
            path = MODELS_DIR / f"rf_cycle{i}.pkl"
            if path.exists():
                best_model = path
                break
        if best_model is None:
            best_model = MODELS_DIR / "rf_phase1.pkl"
        if not best_model.exists():
            print(f"ERROR: No RF model found. Run --phase 1 or --phase 2 first.")
            return
        rf = RFVerifier(model_path=str(best_model))
        print(f"Loaded RF from {best_model}")

    # Evaluate on SWE_Smith (OOD)
    print(f"\nLoading {BATCH_SIZE} CoderForge trajectories (SWE_Smith split, OOD)...")
    rows = load_coderforge_split("SWE_Smith", BATCH_SIZE, offset=0)
    traces, gold_labels, problems, diffs = extract_features_and_labels(rows)
    base_rate = sum(gold_labels) / len(gold_labels)
    print(f"Loaded {len(traces)} traces, base rate: {base_rate:.2%}")

    # RF predictions on OOD data
    rf_probs = [rf.predict_proba(t) for t in traces]
    rf_preds = [1 if p >= 0.5 else 0 for p in rf_probs]
    ood_metrics = compute_metrics(gold_labels, rf_preds, rf_probs)

    print(f"\nRF performance on OOD split (SWE_Smith):")
    print(f"  AUC:       {ood_metrics.get('auc', 'N/A')}")
    print(f"  Accuracy:  {ood_metrics['accuracy']}")
    print(f"  Precision: {ood_metrics['precision']}")
    print(f"  Recall:    {ood_metrics['recall']}")
    print(f"  F1:        {ood_metrics['f1']}")

    # Partial cold start: retrain on 200 SWE_Smith traces
    print(f"\nPartial cold start: retraining RF on {BATCH_SIZE} SWE_Smith traces...")
    X_ood = build_rf_feature_matrix(traces)
    y_ood = np.array(gold_labels)  # use gold as silver for demo
    rf_retrained = RFVerifier()
    rf_retrained.train(X_ood, y_ood)
    rf_retrained.save(str(MODELS_DIR / "rf_ood_retrained.pkl"))

    # Evaluate retrained RF on next batch from SWE_Smith
    print(f"Loading next {BATCH_SIZE} SWE_Smith traces for evaluation...")
    rows2 = load_coderforge_split("SWE_Smith", BATCH_SIZE, offset=BATCH_SIZE)
    traces2, gold_labels2, _, _ = extract_features_and_labels(rows2)

    rf_probs2 = [rf_retrained.predict_proba(t) for t in traces2]
    rf_preds2 = [1 if p >= 0.5 else 0 for p in rf_probs2]
    retrained_metrics = compute_metrics(gold_labels2, rf_preds2, rf_probs2)

    print(f"\nRetrained RF performance on SWE_Smith (after cold start):")
    print(f"  AUC:       {retrained_metrics.get('auc', 'N/A')}")
    print(f"  Accuracy:  {retrained_metrics['accuracy']}")
    print(f"  Precision: {retrained_metrics['precision']}")
    print(f"  Recall:    {retrained_metrics['recall']}")
    print(f"  F1:        {retrained_metrics['f1']}")

    # Recovery ratio
    if ood_metrics.get("auc") and retrained_metrics.get("auc"):
        recovery = retrained_metrics["auc"] / max(ood_metrics["auc"], 0.001)
        print(f"\n  AUC recovery ratio: {recovery:.2f}x "
              f"({ood_metrics['auc']} → {retrained_metrics['auc']})")

    # Save
    results_path = RESULTS_DIR / "phase3_ood.json"
    with open(results_path, "w") as f:
        json.dump({
            "phase": 3,
            "ood_metrics": ood_metrics,
            "retrained_metrics": retrained_metrics,
            "base_rate_ood": round(base_rate, 4),
            "base_rate_eval": round(sum(gold_labels2) / len(gold_labels2), 4),
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Verification Flywheel Demo")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run specific phase")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases sequentially")
    parser.add_argument("--rf-only", action="store_true",
                        help="Skip LLM calls, use gold labels as silver")
    parser.add_argument("--provider", default="bedrock",
                        choices=["bedrock", "anthropic"],
                        help="LLM provider for content verifier")
    parser.add_argument("--model", default="haiku",
                        choices=["haiku", "sonnet"],
                        help="Model for content verifier")
    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.print_help()
        sys.exit(1)

    rf = None

    if args.all or args.phase == 1:
        rf = phase1_cold_start(args)

    if args.all or args.phase == 2:
        rf = phase2_flywheel(args, rf)

    if args.all or args.phase == 3:
        phase3_ood(args, rf)


if __name__ == "__main__":
    main()
