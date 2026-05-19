#!/usr/bin/env python3
"""
Compare benchmark artifacts — regression detection and cross-config comparison.

Usage:
    # Compare two artifacts
    ./compare.py artifact_a.json artifact_b.json

    # Series comparison (incremental optimization layers)
    ./compare.py --series baseline.json +eagle3.json +eagle3+mla.json +fullstack.json

    # Regression detection
    ./compare.py --regression --baseline baseline.json --candidate new.json --threshold 5
"""

import argparse
import json
import sys
from pathlib import Path


CORE_METRICS = [
    ("ttft_ms.p50", "TTFT p50", "ms", "lower"),
    ("ttft_ms.p99", "TTFT p99", "ms", "lower"),
    ("tpot_ms.p50", "TPOT p50", "ms", "lower"),
    ("tpot_ms.p99", "TPOT p99", "ms", "lower"),
    ("output_toks_per_s", "Output tok/s", "tok/s", "higher"),
    ("total_toks_per_s", "Total tok/s", "tok/s", "higher"),
    ("error_rate", "Error rate", "%", "lower"),
]


def get_nested(data: dict, key: str):
    """Get a nested value like 'ttft_ms.p50' from a dict."""
    parts = key.split(".")
    val = data
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def load_artifact(path: Path) -> dict:
    """Load and validate a benchmark artifact."""
    with open(path) as f:
        data = json.load(f)
    if "metrics" not in data:
        raise ValueError(f"{path}: not a valid benchmark artifact (missing 'metrics')")
    return data


def artifact_label(data: dict) -> str:
    """Generate a short label for an artifact."""
    engine = data.get("engine", {}).get("name", "?")
    version = data.get("engine", {}).get("version", "")
    workload = data.get("workload", {}).get("catalog_id", "custom")
    spec = data.get("engine", {}).get("speculative_decoding")
    tag = f"+spec" if spec else ""
    return f"{engine} {version}{tag} / {workload}"


def compare_two(a_path: Path, b_path: Path):
    """Compare two artifacts and show deltas."""
    a = load_artifact(a_path)
    b = load_artifact(b_path)

    a_metrics = a["metrics"]
    b_metrics = b["metrics"]

    print(f"{'Metric':<20} {'A':>12} {'B':>12} {'Delta':>10} {'Change':>8}")
    print("-" * 65)

    for key, label, unit, direction in CORE_METRICS:
        val_a = get_nested(a_metrics, key)
        val_b = get_nested(b_metrics, key)

        if val_a is None or val_b is None:
            continue

        if val_a == 0:
            delta_pct = 0
        else:
            delta_pct = ((val_b - val_a) / abs(val_a)) * 100

        # Determine if change is good or bad
        if direction == "lower":
            is_better = val_b < val_a
        else:
            is_better = val_b > val_a

        indicator = "+" if is_better else "-" if not is_better and delta_pct != 0 else "="

        print(f"{label:<20} {val_a:>10.1f}{unit[-1]:>2} {val_b:>10.1f}{unit[-1]:>2} {delta_pct:>+8.1f}% {indicator:>3}")

    # SLO comparison
    slo_a = a.get("slo", {}).get("overall_pass")
    slo_b = b.get("slo", {}).get("overall_pass")
    if slo_a is not None or slo_b is not None:
        print(f"\n{'SLO pass':<20} {str(slo_a):>12} {str(slo_b):>12}")

    print(f"\nA: {a_path.name}")
    print(f"   {artifact_label(a)}")
    print(f"B: {b_path.name}")
    print(f"   {artifact_label(b)}")


def compare_series(paths: list[Path]):
    """Compare a series of artifacts (incremental optimization layers)."""
    artifacts = [(p, load_artifact(p)) for p in paths]

    # Header
    labels = [p.stem[:25] for p, _ in artifacts]
    print(f"{'Metric':<18}", end="")
    for label in labels:
        print(f" {label:>14}", end="")
    print(f" {'Total Δ':>10}")
    print("-" * (18 + 15 * len(labels) + 12))

    # Each metric
    for key, label, unit, direction in CORE_METRICS:
        values = [get_nested(a["metrics"], key) for _, a in artifacts]
        if any(v is None for v in values):
            continue

        print(f"{label:<18}", end="")
        for v in values:
            print(f" {v:>12.1f}{unit[-1]:>2}", end="")

        # Total delta from first to last
        if values[0] != 0:
            total_delta = ((values[-1] - values[0]) / abs(values[0])) * 100
            print(f" {total_delta:>+8.1f}%", end="")
        print()

    # Per-step deltas
    print(f"\n{'Step improvement':<18}", end="")
    for i, (key, label, unit, direction) in enumerate(CORE_METRICS):
        if key == "output_toks_per_s":
            values = [get_nested(a["metrics"], key) for _, a in artifacts]
            if any(v is None for v in values):
                continue
            print(f"\n  {label}:", end="")
            for j in range(1, len(values)):
                if values[j - 1] != 0:
                    step_delta = ((values[j] - values[j - 1]) / abs(values[j - 1])) * 100
                    print(f" step{j}: {step_delta:>+.1f}%", end="")
            break


def regression_check(baseline_path: Path, candidate_path: Path, threshold: float) -> bool:
    """Check if candidate regresses vs baseline beyond threshold. Returns True if regression found."""
    baseline = load_artifact(baseline_path)
    candidate = load_artifact(candidate_path)

    regressions = []

    for key, label, unit, direction in CORE_METRICS:
        val_base = get_nested(baseline["metrics"], key)
        val_cand = get_nested(candidate["metrics"], key)

        if val_base is None or val_cand is None or val_base == 0:
            continue

        delta_pct = ((val_cand - val_base) / abs(val_base)) * 100

        # Check if this is a regression
        if direction == "lower" and delta_pct > threshold:
            regressions.append((label, val_base, val_cand, delta_pct))
        elif direction == "higher" and delta_pct < -threshold:
            regressions.append((label, val_base, val_cand, delta_pct))

    if regressions:
        print(f"REGRESSION DETECTED (threshold: {threshold}%)")
        print(f"{'Metric':<20} {'Baseline':>12} {'Candidate':>12} {'Delta':>10}")
        print("-" * 55)
        for label, base, cand, delta in regressions:
            print(f"{label:<20} {base:>12.1f} {cand:>12.1f} {delta:>+8.1f}%")
        return True
    else:
        print(f"No regressions detected (threshold: {threshold}%)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark artifacts")
    parser.add_argument("artifacts", nargs="*", type=Path, help="Artifact JSON files to compare")
    parser.add_argument("--series", action="store_true", help="Series comparison (incremental layers)")
    parser.add_argument("--regression", action="store_true", help="Regression detection mode")
    parser.add_argument("--baseline", type=Path, help="Baseline artifact (for --regression)")
    parser.add_argument("--candidate", type=Path, help="Candidate artifact (for --regression)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold %% (default: 5)")
    args = parser.parse_args()

    if args.regression:
        if not args.baseline or not args.candidate:
            print("Error: --regression requires --baseline and --candidate")
            sys.exit(1)
        has_regression = regression_check(args.baseline, args.candidate, args.threshold)
        sys.exit(1 if has_regression else 0)

    if not args.artifacts or len(args.artifacts) < 2:
        print("Error: need at least 2 artifact files to compare")
        parser.print_help()
        sys.exit(1)

    if args.series:
        compare_series(args.artifacts)
    else:
        compare_two(args.artifacts[0], args.artifacts[1])


if __name__ == "__main__":
    main()
