"""Validate profiler artifacts against Spec 0 acceptance criteria.

Run after a batch of profiler runs to confirm the artifacts are usable
for downstream specs. Fails loudly when:
  - any canonical event is missing
  - any stage has a negative elapsed time
  - sum-of-stages + sum-of-gaps differs from total by > 5%
  - run-to-run variance on any stage exceeds sigma/mu > 0.5

Usage:
    python profiler_validate.py results/*.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

CANONICAL = [
    "T0_pod_create",
    "T1_node_assigned",
    "T2_image_pull_start",
    "T3_image_pull_complete",
    "T4_container_created",
    "T5_container_started",
    "T6_python_alive",
    "T7_weights_load_start",
    "T8_weights_loaded",
    "T9_jit_compile_start",
    "T10_jit_compile_done",
    "T11_cuda_graphs_done",
    "T12_health_200",
    "T13_first_token",
]


def validate_one(path: Path) -> list[str]:
    art = json.loads(path.read_text())
    errors: list[str] = []
    seen = {e["name"] for e in art["events"]}
    for evt in CANONICAL:
        if evt not in seen:
            errors.append(f"{path.name}: missing event {evt}")

    for stage, body in art["stages"].items():
        if body["elapsed_s"] is None:
            errors.append(f"{path.name}: stage {stage} has no elapsed (missing endpoints)")
            continue
        if body["elapsed_s"] < 0:
            errors.append(
                f"{path.name}: stage {stage} negative duration ({body['elapsed_s']:.3f}s)"
            )

    total = art["totals"].get("pod_create_to_first_token_s")
    if total is None:
        errors.append(f"{path.name}: total pod_create_to_first_token_s missing")
        return errors

    accounted = 0.0
    for body in art["stages"].values():
        if body["elapsed_s"] is not None and body["elapsed_s"] >= 0:
            accounted += body["elapsed_s"]
    for v in art["gaps"].values():
        if v >= 0:
            accounted += v

    coverage = accounted / total if total > 0 else 0.0
    if coverage < 0.95:
        errors.append(
            f"{path.name}: stage+gap coverage {coverage:.1%} < 95% "
            f"(accounted={accounted:.1f}s total={total:.1f}s)"
        )
    return errors


def validate_batch(paths: list[Path]) -> list[str]:
    """Cross-run variance check on stages."""
    errors: list[str] = []
    if len(paths) < 2:
        return errors
    by_stage: dict[str, list[float]] = {}
    for p in paths:
        art = json.loads(p.read_text())
        for stage, body in art["stages"].items():
            if body["elapsed_s"] is not None and body["elapsed_s"] >= 0:
                by_stage.setdefault(stage, []).append(body["elapsed_s"])
    for stage, values in by_stage.items():
        if len(values) < 2:
            continue
        mu = statistics.mean(values)
        sigma = statistics.stdev(values)
        if mu > 0 and sigma / mu > 0.5:
            errors.append(
                f"batch: stage {stage} sigma/mu={sigma/mu:.2f} > 0.5 "
                f"(n={len(values)} mu={mu:.2f}s sigma={sigma:.2f}s) -- needs more samples or finer instrumentation"
            )
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", nargs="+")
    args = p.parse_args()

    paths = [Path(a) for a in args.artifacts]
    all_errors: list[str] = []
    for path in paths:
        all_errors.extend(validate_one(path))
    all_errors.extend(validate_batch(paths))

    if all_errors:
        print(f"FAIL ({len(all_errors)} issue(s)):")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print(f"OK ({len(paths)} artifact(s) validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
