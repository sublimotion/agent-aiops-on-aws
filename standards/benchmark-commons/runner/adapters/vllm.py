#!/usr/bin/env python3
"""
vLLM adapter — converts vLLM benchmark_serving.py output to common artifact format.

Maps vLLM's native JSON fields to the common artifact schema.
"""

import argparse
import json
import sys
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path

from _common import compute_iac, merge_engagement_blobs


def adapt(raw: dict, sidecar: dict, workload: dict, sidecar_path: Path | None = None) -> dict:
    """Convert vLLM benchmark output + sidecar config to common artifact."""

    # Core metrics mapping
    metrics = {
        "duration_s": raw.get("duration", 0),
        "completed": raw.get("completed", 0),
        "failed": raw.get("failed", 0),
        "error_rate": raw.get("failed", 0) / max(raw.get("completed", 1) + raw.get("failed", 0), 1),

        "ttft_ms": {
            "mean": raw.get("mean_ttft_ms", 0),
            "p50": raw.get("p50_ttft_ms", raw.get("median_ttft_ms", 0)),
            "p90": raw.get("p90_ttft_ms", 0),
            "p95": raw.get("p95_ttft_ms"),
            "p99": raw.get("p99_ttft_ms", 0),
        },
        "tpot_ms": {
            "mean": raw.get("mean_tpot_ms", 0),
            "p50": raw.get("p50_tpot_ms", raw.get("median_tpot_ms", 0)),
            "p90": raw.get("p90_tpot_ms", 0),
            "p95": raw.get("p95_tpot_ms"),
            "p99": raw.get("p99_tpot_ms", 0),
        },
        "itl_ms": {
            "mean": raw.get("mean_itl_ms", 0),
            "p50": raw.get("p50_itl_ms", raw.get("median_itl_ms", 0)),
            "p90": raw.get("p90_itl_ms", 0),
            "p95": raw.get("p95_itl_ms"),
            "p99": raw.get("p99_itl_ms", 0),
        },
        "e2e_ms": {
            "mean": raw.get("mean_e2e_latency_ms", 0),
            "p50": raw.get("p50_e2e_latency_ms", raw.get("median_e2e_latency_ms", 0)),
            "p90": raw.get("p90_e2e_latency_ms", 0),
            "p95": raw.get("p95_e2e_latency_ms"),
            "p99": raw.get("p99_e2e_latency_ms", 0),
        },

        "output_toks_per_s": raw.get("output_throughput", 0),
        "request_throughput": raw.get("request_throughput", 0),
        "total_toks_per_s": raw.get("total_throughput", raw.get("output_throughput", 0)),
        "total_input_tokens": raw.get("total_input_tokens", 0),
        "total_output_tokens": raw.get("total_output_tokens", 0),
        "max_concurrent_requests": raw.get("max_concurrency", 0),
    }

    # Build artifact
    artifact = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_tool": {
            "name": "vllm-bench-serve",
            "version": sidecar.get("engine", {}).get("version", "unknown"),
            "adapter_version": "1.0.0",
        },
        "model": sidecar.get("model", {}),
        "infrastructure": sidecar.get("infrastructure", {}),
        "engine": sidecar.get("engine", {}),
        "workload": {
            "use_case": workload.get("use_case", "custom"),
            "catalog_id": workload.get("catalog_id"),
            "dataset": workload.get("dataset", {}),
            "load": workload.get("load", {}),
            "api": workload.get("api", {"type": "chat", "streaming": True, "endpoint": "/v1/chat/completions"}),
        },
        "metrics": metrics,
    }

    # SLO evaluation
    slo_targets = sidecar.get("slo")
    if slo_targets:
        slo_results = {}
        overall_pass = True

        if "ttft_p99_ms" in slo_targets:
            actual = metrics["ttft_ms"]["p99"]
            passes = actual <= slo_targets["ttft_p99_ms"]
            slo_results["ttft_p99_ms"] = {"target": slo_targets["ttft_p99_ms"], "actual": actual, "pass": passes}
            overall_pass = overall_pass and passes

        if "tpot_p99_ms" in slo_targets:
            actual = metrics["tpot_ms"]["p99"]
            passes = actual <= slo_targets["tpot_p99_ms"]
            slo_results["tpot_p99_ms"] = {"target": slo_targets["tpot_p99_ms"], "actual": actual, "pass": passes}
            overall_pass = overall_pass and passes

        if "error_rate_max" in slo_targets:
            actual = metrics["error_rate"]
            passes = actual <= slo_targets["error_rate_max"]
            slo_results["error_rate_max"] = {"target": slo_targets["error_rate_max"], "actual": actual, "pass": passes}
            overall_pass = overall_pass and passes

        artifact["slo"] = {
            "targets": slo_targets,
            "results": slo_results,
            "overall_pass": overall_pass,
        }

    # Extensions (speculative decode stats if present)
    extensions = {}
    if raw.get("acceptance_rate") is not None:
        extensions["speculative_decode"] = {
            "acceptance_rate": raw.get("acceptance_rate"),
            "tokens_per_step": raw.get("tokens_per_step"),
            "draft_tokens": raw.get("num_spec_tokens"),
        }
    if extensions:
        artifact["extensions"] = extensions

    # Intelligence-Adjusted Cost (IAC)
    iac = compute_iac(metrics, sidecar)
    if iac:
        artifact["cost"] = iac

    # CTO engagement blobs: quality gate, power, stability, cold-start, hw errors
    if sidecar_path is not None:
        merge_engagement_blobs(artifact, sidecar, sidecar_path)

    return artifact


def main():
    parser = argparse.ArgumentParser(description="vLLM adapter — raw output to common artifact")
    parser.add_argument("--raw", required=True, type=Path, help="Raw vLLM benchmark JSON")
    parser.add_argument("--sidecar", required=True, type=Path, help="benchmark.yaml sidecar")
    parser.add_argument("--workload", required=True, type=Path, help="Workload YAML")
    parser.add_argument("--output", required=True, type=Path, help="Output artifact path")
    args = parser.parse_args()

    with open(args.raw) as f:
        raw = json.load(f)
    with open(args.sidecar) as f:
        sidecar = yaml.safe_load(f)
    with open(args.workload) as f:
        workload = yaml.safe_load(f)

    artifact = adapt(raw, sidecar, workload, sidecar_path=args.sidecar)

    with open(args.output, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Artifact written: {args.output}")
    print(f"  Schema: {artifact['schema_version']}")
    print(f"  Completed: {artifact['metrics']['completed']}")
    print(f"  Output tok/s: {artifact['metrics']['output_toks_per_s']:.1f}")
    if "slo" in artifact:
        print(f"  SLO pass: {artifact['slo']['overall_pass']}")


if __name__ == "__main__":
    main()
