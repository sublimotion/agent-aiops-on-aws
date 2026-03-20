#!/usr/bin/env python3
"""
Phase 0: Behavioral Signal Check

Joins Phase 1 turn-level data (configs A-F) with eval_sera labels (23 instances)
to test whether SERA's behavioral telemetry predicts test outcomes.

Features extracted per instance (aggregated from turns):
- total_turns: number of turns used
- first_edit_turn: turn where first edit action occurred
- parkinson_ratio: first_edit_turn / total_turns (late = bad)
- repeat_rate: fraction of turns with repeated_action=True
- context_growth_rate: (final_context_tokens - initial_context_tokens) / initial
- action_pct_search: fraction of turns with search actions
- action_pct_edit: fraction of turns with edit actions
- action_pct_read: fraction of turns with read/view actions
- edit_to_search_ratio: edit turns / search turns
- avg_duration_s: mean turn duration

Models:
- Logistic regression with LOOCV (leave-one-out cross-validation)
- Per-feature univariate AUC
- Feature correlation with tests_pass
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data" / "phase0"


def load_eval_labels():
    """Load eval_sera.jsonl -> {instance_id: tests_pass}"""
    labels = {}
    with open(DATA_DIR / "eval_sera.jsonl") as f:
        for line in f:
            d = json.loads(line)
            labels[d["instance_id"]] = int(d["tests_pass"])
    return labels


def load_turns(config: str):
    """Load phase1_{config}_turns.jsonl -> list of turn dicts"""
    turns = []
    with open(DATA_DIR / f"phase1_{config}_turns.jsonl") as f:
        for line in f:
            turns.append(json.loads(line))
    return turns


def extract_features(turns_for_instance: list) -> dict:
    """Extract behavioral features from a list of turns for one instance."""
    if not turns_for_instance:
        return None

    sorted_turns = sorted(turns_for_instance, key=lambda t: t["turn"])
    n = len(sorted_turns)

    # Action type categorization
    search_actions = {"search_files", "search_dir", "find_file", "grep", "search"}
    edit_actions = {"str_replace_editor", "edit", "write", "insert", "replace", "str_replace"}
    read_actions = {"read", "view", "open_file", "cat"}
    run_actions = {"run_command", "bash", "execute", "terminal"}

    action_types = [t.get("action_type", "") for t in sorted_turns]
    n_search = sum(1 for a in action_types if a in search_actions or "search" in str(a).lower())
    n_edit = sum(1 for a in action_types if a in edit_actions or "edit" in str(a).lower() or "replace" in str(a).lower())
    n_read = sum(1 for a in action_types if a in read_actions or "read" in str(a).lower() or "view" in str(a).lower())
    n_run = sum(1 for a in action_types if a in run_actions or "bash" in str(a).lower() or "command" in str(a).lower())

    # First edit turn
    first_edit = n  # default to max if no edit
    for i, a in enumerate(action_types):
        if a in edit_actions or "edit" in str(a).lower() or "replace" in str(a).lower():
            first_edit = i + 1  # 1-indexed
            break

    # Repeated actions
    repeat_count = sum(1 for t in sorted_turns if t.get("repeated_action"))

    # Context growth
    context_tokens = [t.get("context_tokens", 0) or 0 for t in sorted_turns]
    initial_ctx = context_tokens[0] if context_tokens[0] > 0 else 1
    final_ctx = context_tokens[-1] if context_tokens else initial_ctx
    context_growth = (final_ctx - initial_ctx) / initial_ctx if initial_ctx > 0 else 0

    # Duration
    durations = [t.get("duration_s", 0) or 0 for t in sorted_turns]
    avg_duration = sum(durations) / n if n > 0 else 0

    return {
        "total_turns": n,
        "first_edit_turn": first_edit,
        "parkinson_ratio": first_edit / n if n > 0 else 1.0,
        "repeat_rate": repeat_count / n if n > 0 else 0,
        "context_growth_rate": context_growth,
        "action_pct_search": n_search / n if n > 0 else 0,
        "action_pct_edit": n_edit / n if n > 0 else 0,
        "action_pct_read": n_read / n if n > 0 else 0,
        "action_pct_run": n_run / n if n > 0 else 0,
        "edit_to_search_ratio": n_edit / max(n_search, 1),
        "avg_duration_s": avg_duration,
    }


def run_signal_check():
    labels = load_eval_labels()
    print(f"Eval labels: {len(labels)} instances ({sum(labels.values())} pass, {len(labels)-sum(labels.values())} fail)")
    print()

    # Aggregate features across all configs for each instance
    all_features = defaultdict(list)

    for config in ["A", "B", "C", "D", "E", "F"]:
        turns = load_turns(config)

        # Group by instance
        by_instance = defaultdict(list)
        for t in turns:
            by_instance[t["instance_id"]].append(t)

        for iid in labels:
            if iid in by_instance:
                feats = extract_features(by_instance[iid])
                if feats:
                    feats["config"] = config
                    all_features[iid].append(feats)

    print(f"Instances with features: {len(all_features)}")
    print(f"Total feature rows (instance x config): {sum(len(v) for v in all_features.values())}")
    print()

    # Use config D (30 turns, highest signal) as the primary analysis
    # But also show aggregated across configs
    for analysis_name, get_features in [
        ("Config D only (30 turns)", lambda iid: next((f for f in all_features[iid] if f["config"] == "D"), None)),
        ("Mean across configs A-F", lambda iid: _mean_features(all_features[iid])),
    ]:
        print(f"=" * 60)
        print(f"Analysis: {analysis_name}")
        print(f"=" * 60)

        X_rows = []
        y = []
        instance_ids = []
        feature_names = None

        for iid in sorted(labels.keys()):
            feats = get_features(iid)
            if feats is None:
                continue
            if feature_names is None:
                feature_names = [k for k in feats.keys() if k != "config"]
            X_rows.append([feats[k] for k in feature_names])
            y.append(labels[iid])
            instance_ids.append(iid)

        if not X_rows:
            print("  No data available for this analysis")
            continue

        n_samples = len(X_rows)
        n_pos = sum(y)
        n_neg = n_samples - n_pos
        print(f"  Samples: {n_samples} ({n_pos} pass, {n_neg} fail)")
        print()

        # Per-feature analysis
        print("  Per-feature point-biserial correlation with tests_pass:")
        print(f"  {'Feature':<25} {'Corr':>7} {'p-val':>8} {'AUC':>6}  {'Pass mean':>10} {'Fail mean':>10}")
        print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*6}  {'-'*10} {'-'*10}")

        try:
            from scipy import stats
            has_scipy = True
        except ImportError:
            has_scipy = False
            print("  (scipy not available — showing means only)")

        for fi, fname in enumerate(feature_names):
            col = [row[fi] for row in X_rows]
            pass_vals = [col[i] for i in range(n_samples) if y[i] == 1]
            fail_vals = [col[i] for i in range(n_samples) if y[i] == 0]
            pass_mean = sum(pass_vals) / len(pass_vals) if pass_vals else 0
            fail_mean = sum(fail_vals) / len(fail_vals) if fail_vals else 0

            if has_scipy:
                corr, pval = stats.pointbiserialr(y, col)
                # Compute AUC manually
                auc = _manual_auc(y, col)
                marker = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {fname:<25} {corr:>7.3f} {pval:>8.4f} {auc:>6.3f}  {pass_mean:>10.3f} {fail_mean:>10.3f}  {marker}")
            else:
                print(f"  {fname:<25} {'n/a':>7} {'n/a':>8} {'n/a':>6}  {pass_mean:>10.3f} {fail_mean:>10.3f}")

        # Logistic regression with LOOCV
        if has_scipy:
            print()
            _run_loocv_logistic(X_rows, y, feature_names, instance_ids)

        print()


def _mean_features(feature_list):
    """Average features across configs."""
    if not feature_list:
        return None
    keys = [k for k in feature_list[0].keys() if k != "config"]
    result = {}
    for k in keys:
        vals = [f[k] for f in feature_list if k in f]
        result[k] = sum(vals) / len(vals) if vals else 0
    return result


def _manual_auc(y_true, scores):
    """Compute AUC from binary labels and continuous scores."""
    pairs = list(zip(scores, y_true))
    pairs.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    prev_fp = 0
    prev_tp = 0
    auc = 0.0
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    prev_score = None
    for score, label in pairs:
        if score != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def _run_loocv_logistic(X_rows, y, feature_names, instance_ids):
    """Run leave-one-out CV with logistic regression."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        print("  (sklearn not available — skipping logistic regression)")
        return

    n = len(X_rows)
    predictions = []
    probabilities = []

    for i in range(n):
        X_train = X_rows[:i] + X_rows[i + 1:]
        y_train = y[:i] + y[i + 1:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform([X_rows[i]])

        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)[0]
        prob = model.predict_proba(X_test_scaled)[0][1]
        predictions.append(pred)
        probabilities.append(prob)

    # Metrics
    accuracy = sum(1 for p, t in zip(predictions, y) if p == t) / n
    try:
        auc = roc_auc_score(y, probabilities)
    except ValueError:
        auc = float("nan")

    # Calibration (ECE with 5 bins)
    ece = _compute_ece(y, probabilities, n_bins=5)

    print(f"  Logistic Regression (LOOCV, n={n}):")
    print(f"    Accuracy:  {accuracy:.3f} ({sum(1 for p, t in zip(predictions, y) if p == t)}/{n})")
    print(f"    AUC:       {auc:.3f}")
    print(f"    ECE:       {ece:.3f}")
    print(f"    Baseline:  {max(sum(y), n - sum(y)) / n:.3f} (always predict majority)")

    # Show misclassifications
    print(f"\n  Misclassifications:")
    for i in range(n):
        if predictions[i] != y[i]:
            print(f"    {instance_ids[i]}: predicted={predictions[i]}, actual={y[i]}, prob={probabilities[i]:.3f}")

    # Feature importance (from full-data fit)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rows)
    model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model.fit(X_scaled, y)

    print(f"\n  Feature coefficients (full model):")
    coefs = list(zip(feature_names, model.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        direction = "+" if coef > 0 else "-"
        print(f"    {name:<25} {coef:>7.3f}  ({direction} pass rate)")


def _compute_ece(y_true, y_prob, n_bins=5):
    """Expected Calibration Error."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = [i for i in range(n) if lo <= y_prob[i] < hi or (b == n_bins - 1 and y_prob[i] == hi)]
        if not mask:
            continue
        bin_acc = sum(y_true[i] for i in mask) / len(mask)
        bin_conf = sum(y_prob[i] for i in mask) / len(mask)
        ece += len(mask) / n * abs(bin_acc - bin_conf)
    return ece


def analyze_svg_results():
    """Analyze SVG production-run1 results as a standalone verifier."""
    svg_path = DATA_DIR / "svg_results_production_run1.jsonl"
    if not svg_path.exists():
        print("SVG results not found — skipping")
        return

    print("=" * 60)
    print("SVG Consensus Verifier Analysis (production-run1, n=300)")
    print("=" * 60)

    results = []
    with open(svg_path) as f:
        for line in f:
            results.append(json.loads(line))

    n = len(results)
    fix_gen = sum(1 for r in results if r.get("fix_generated"))
    tests_pass = sum(1 for r in results if r.get("tests_pass"))
    accepted = sum(1 for r in results if r.get("accepted"))

    print(f"  Total: {n}")
    print(f"  fix_generated: {fix_gen} ({fix_gen/n*100:.1f}%)")
    print(f"  tests_pass: {tests_pass} ({tests_pass/n*100:.1f}%)")
    print(f"  SVG accepted (recall >= 0.8): {accepted} ({accepted/n*100:.1f}%)")

    # Confusion matrix: SVG accepted vs tests_pass
    tp = sum(1 for r in results if r.get("accepted") and r.get("tests_pass"))
    fp = sum(1 for r in results if r.get("accepted") and not r.get("tests_pass"))
    fn = sum(1 for r in results if not r.get("accepted") and r.get("tests_pass"))
    tn = sum(1 for r in results if not r.get("accepted") and not r.get("tests_pass"))

    print(f"\n  SVG as binary classifier (threshold=0.8 recall):")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1:        {f1:.3f}")
    print(f"    Accuracy:  {(tp+tn)/n:.3f}")

    # line_recall as continuous predictor
    has_recall = [(r.get("line_recall", 0) or 0, int(r.get("tests_pass", False))) for r in results if r.get("fix_generated")]
    if has_recall:
        recalls, labels = zip(*has_recall)
        auc = _manual_auc(list(labels), list(recalls))
        print(f"\n  line_recall as continuous predictor (among fix_generated={len(has_recall)}):")
        print(f"    AUC: {auc:.3f}")

        # Mean recall by outcome
        pass_recalls = [r for r, l in has_recall if l == 1]
        fail_recalls = [r for r, l in has_recall if l == 0]
        print(f"    Mean recall (pass): {sum(pass_recalls)/len(pass_recalls):.3f} (n={len(pass_recalls)})")
        print(f"    Mean recall (fail): {sum(fail_recalls)/len(fail_recalls):.3f} (n={len(fail_recalls)})")

    print()


if __name__ == "__main__":
    print("Phase 0: Behavioral Signal Check")
    print("=" * 60)
    print()
    run_signal_check()
    analyze_svg_results()
