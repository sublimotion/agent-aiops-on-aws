#!/usr/bin/env python3
"""
Enrichment wrapper: runs AIPerf + merges output with benchmark.yaml sidecar.

Modes:
  1. Full run:   enrich-benchmark --sidecar benchmark.yaml --target http://host:8000
  2. Enrich only: enrich-benchmark --sidecar benchmark.yaml --aiperf-output results.json
  3. Validate:   enrich-benchmark --validate artifact.json

The wrapper:
  1. Loads the sidecar (model, engine, framework, infrastructure, SLO targets)
  2. Runs AIPerf against the target endpoint (or reads existing AIPerf output)
  3. Optionally scrapes engine-internal metrics (KV cache, prefix hits) during the run
  4. Merges AIPerf output + sidecar context into an enriched artifact
  5. Evaluates SLOs (targets vs actuals)
  6. Validates the artifact against the JSON Schema
  7. Writes the enriched artifact to the output directory
"""

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_sidecar(path: str) -> dict:
    """Load and validate the benchmark.yaml sidecar."""
    with open(path) as f:
        sidecar = yaml.safe_load(f)

    required = ["model", "engine", "infrastructure"]
    for key in required:
        if key not in sidecar:
            sys.exit(f"ERROR: sidecar missing required key: {key}")

    return sidecar


def run_aiperf(target: str, workload: dict, output_dir: Path) -> Path:
    """Invoke AIPerf CLI and return path to output JSON."""
    cmd = ["aiperf", "profile"]

    # Target endpoint
    cmd.extend(["--url", target])

    # Workload parameters
    load = workload.get("load", {})
    dataset = workload.get("dataset", {})

    if load.get("type") == "concurrency-sweep":
        cmd.extend(["--concurrency-sweep"])
        levels = load.get("levels", [1, 2, 4, 8, 16, 32, 64, 128, 256])
        cmd.extend(["--concurrency-levels", ",".join(str(l) for l in levels)])
        cmd.extend(["--num-prompts", str(load.get("num_prompts_per_level", 50))])
    elif load.get("type") == "constant":
        cmd.extend(["--request-rate", str(load.get("request_rate", 2.0))])
        cmd.extend(["--duration", str(load.get("duration_s", 120))])
        if load.get("max_concurrency"):
            cmd.extend(["--max-concurrency", str(load["max_concurrency"])])
    elif load.get("type") == "poisson":
        cmd.extend(["--request-rate", str(load.get("request_rate", 2.0))])
        cmd.extend(["--distribution", "poisson"])
        cmd.extend(["--duration", str(load.get("duration_s", 120))])

    # Dataset
    if dataset.get("type") == "synthetic":
        input_tokens = dataset.get("input_tokens", {})
        output_tokens = dataset.get("output_tokens", {})
        cmd.extend(["--input-tokens-mean", str(input_tokens.get("mean", 2048))])
        if input_tokens.get("std_dev"):
            cmd.extend(["--input-tokens-stddev", str(input_tokens["std_dev"])])
        cmd.extend(["--output-tokens-mean", str(output_tokens.get("mean", 512))])
        if output_tokens.get("std_dev"):
            cmd.extend(["--output-tokens-stddev", str(output_tokens["std_dev"])])

    # API config
    api = workload.get("api", {})
    if api.get("streaming", True):
        cmd.extend(["--streaming"])
    if api.get("endpoint"):
        cmd.extend(["--endpoint", api["endpoint"]])

    # Warmup
    warmup = load.get("warmup_requests", 10)
    cmd.extend(["--warmup-requests", str(warmup)])

    # Output
    output_file = output_dir / "aiperf_raw.json"
    cmd.extend(["--output", str(output_file), "--output-format", "json"])

    print(f"Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"AIPerf stderr: {result.stderr}", file=sys.stderr)
        sys.exit(f"AIPerf failed with exit code {result.returncode}")

    return output_file


def parse_aiperf_output(path: Path) -> dict:
    """Parse AIPerf JSON output into core metrics."""
    with open(path) as f:
        raw = json.load(f)

    # Map AIPerf fields to enriched artifact metrics
    # AIPerf output structure varies by version; handle common shapes
    metrics = {}

    def extract_latency(raw_metric: dict) -> dict:
        """Extract percentile structure from AIPerf metric."""
        return {
            "mean": raw_metric.get("avg", raw_metric.get("mean")),
            "p50": raw_metric.get("p50", raw_metric.get("median")),
            "p90": raw_metric.get("p90"),
            "p95": raw_metric.get("p95"),
            "p99": raw_metric.get("p99"),
        }

    # Core latency metrics
    if "time_to_first_token" in raw:
        metrics["ttft_ms"] = extract_latency(raw["time_to_first_token"])
    if "inter_token_latency" in raw:
        metrics["itl_ms"] = extract_latency(raw["inter_token_latency"])
    if "request_latency" in raw:
        metrics["e2e_ms"] = extract_latency(raw["request_latency"])

    # Derive TPOT: (e2e - ttft) / (output_tokens - 1)
    if "e2e_ms" in metrics and "ttft_ms" in metrics:
        # Use per-request data if available, otherwise estimate from means
        if "output_sequence_length" in raw:
            avg_osl = raw["output_sequence_length"].get("avg", 512)
            if avg_osl > 1:
                tpot_mean = (metrics["e2e_ms"]["mean"] - metrics["ttft_ms"]["mean"]) / (avg_osl - 1)
                metrics["tpot_ms"] = {
                    "mean": round(tpot_mean, 2),
                    "p50": raw.get("time_per_output_token", {}).get("p50"),
                    "p90": raw.get("time_per_output_token", {}).get("p90"),
                    "p95": raw.get("time_per_output_token", {}).get("p95"),
                    "p99": raw.get("time_per_output_token", {}).get("p99"),
                }

    # Throughput
    metrics["output_toks_per_s"] = raw.get("output_token_throughput", raw.get("output_toks_per_s"))
    metrics["request_throughput"] = raw.get("request_throughput")
    metrics["total_toks_per_s"] = raw.get("total_token_throughput", raw.get("total_toks_per_s"))

    # Counts
    metrics["completed"] = raw.get("completed_requests", raw.get("completed", 0))
    metrics["failed"] = raw.get("error_request_count", raw.get("failed", 0))
    total = metrics["completed"] + metrics["failed"]
    metrics["error_rate"] = metrics["failed"] / total if total > 0 else 0.0
    metrics["duration_s"] = raw.get("duration_s", raw.get("benchmark_duration", 0))
    metrics["total_input_tokens"] = raw.get("total_input_tokens", raw.get("input_token_count", 0))
    metrics["total_output_tokens"] = raw.get("total_output_tokens", raw.get("output_token_count", 0))
    metrics["max_concurrent_requests"] = raw.get("max_concurrent_requests", raw.get("max_concurrency"))

    return metrics


def evaluate_slos(metrics: dict, slo_targets: dict) -> dict:
    """Evaluate SLO targets against actual metrics."""
    if not slo_targets:
        return None

    results = {}
    mapping = {
        "ttft_p99_ms": lambda m: m.get("ttft_ms", {}).get("p99"),
        "tpot_p99_ms": lambda m: m.get("tpot_ms", {}).get("p99"),
        "e2e_p99_ms": lambda m: m.get("e2e_ms", {}).get("p99"),
        "error_rate_max": lambda m: m.get("error_rate"),
        "throughput_floor_toks": lambda m: m.get("output_toks_per_s"),
    }

    for target_key, target_value in slo_targets.items():
        if target_key in mapping:
            actual = mapping[target_key](metrics)
            if actual is not None:
                if target_key == "throughput_floor_toks":
                    passed = actual >= target_value
                else:
                    passed = actual <= target_value
                results[target_key] = {
                    "target": target_value,
                    "actual": round(actual, 2) if isinstance(actual, float) else actual,
                    "pass": passed,
                }

    overall = all(r["pass"] for r in results.values()) if results else True

    return {
        "targets": slo_targets,
        "results": results,
        "overall_pass": overall,
    }


def build_artifact(sidecar: dict, metrics: dict, workload: dict, concurrency: int = None) -> dict:
    """Assemble the enriched artifact from sidecar + metrics."""
    artifact = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_tool": {
            "name": "aiperf",
            "version": "0.6.0",
            "enrichment_version": "1.0.0",
        },
        "model": sidecar["model"],
        "engine": sidecar["engine"],
        "infrastructure": sidecar["infrastructure"],
        "workload": workload,
        "metrics": metrics,
    }

    # Optional framework
    if "framework" in sidecar:
        artifact["framework"] = sidecar["framework"]

    # SLO evaluation
    slo_targets = sidecar.get("slo", {})
    slo_result = evaluate_slos(metrics, slo_targets)
    if slo_result:
        artifact["slo"] = slo_result

    # Extensions placeholder
    artifact["extensions"] = {}

    return artifact


def artifact_filename(sidecar: dict, workload: dict, concurrency: int = None) -> str:
    """Generate the standard filename for the artifact."""
    model = sidecar["model"]["name"].lower().replace(" ", "-").replace("/", "-")
    substrate = sidecar["infrastructure"]["substrate"]
    instance = sidecar["infrastructure"]["instance_type"].replace(".", "-")
    engine = sidecar["engine"]["name"]

    # Framework suffix
    framework = sidecar.get("framework", {})
    if framework:
        fw_name = framework.get("name", "")
        fw_mode = framework.get("config", {}).get("mode", "")
        engine_part = f"{engine}-{fw_name}"
        if fw_mode:
            engine_part += f"-{fw_mode}"
    else:
        engine_part = engine

    # Workload
    catalog_id = workload.get("catalog_id", "custom")
    if not catalog_id:
        catalog_id = workload.get("custom", {}).get("name", "custom")

    parts = [model, substrate, instance, engine_part, catalog_id]
    if concurrency:
        parts.append(f"c{concurrency}")

    return "_".join(parts) + ".json"


def validate_artifact(artifact: dict) -> bool:
    """Validate artifact against JSON Schema."""
    try:
        import jsonschema

        schema_path = Path("/etc/aiperf/schema/enriched-artifact.json")
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            jsonschema.validate(artifact, schema)
            print("Schema validation: PASS", flush=True)
            return True
        else:
            print("Schema not found, skipping validation", flush=True)
            return True
    except jsonschema.ValidationError as e:
        print(f"Schema validation: FAIL — {e.message}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AIPerf enrichment wrapper — run benchmarks and produce enriched artifacts"
    )
    parser.add_argument("--sidecar", required=True, help="Path to benchmark.yaml sidecar")
    parser.add_argument("--target", help="Endpoint URL (e.g., http://localhost:8000)")
    parser.add_argument("--aiperf-output", help="Path to existing AIPerf JSON output (skip running AIPerf)")
    parser.add_argument("--concurrency", type=int, help="Current concurrency level (for sweep artifacts)")
    parser.add_argument("--workload-index", type=int, default=0, help="Index into sidecar workloads list")
    parser.add_argument("--output-dir", default="/results", help="Output directory for artifacts")
    parser.add_argument("--scrape-metrics", action="store_true", help="Scrape engine Prometheus metrics during run")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics", help="Engine metrics endpoint")
    parser.add_argument("--validate", help="Validate an existing artifact (no benchmark run)")
    parser.add_argument("--dry-run", action="store_true", help="Print AIPerf command without executing")

    args = parser.parse_args()

    # Validate-only mode
    if args.validate:
        with open(args.validate) as f:
            artifact = json.load(f)
        valid = validate_artifact(artifact)
        sys.exit(0 if valid else 1)

    # Load sidecar
    sidecar = load_sidecar(args.sidecar)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select workload
    workloads = sidecar.get("workloads", [])
    if not workloads:
        sys.exit("ERROR: no workloads defined in sidecar")

    workload_def = workloads[args.workload_index]
    catalog_id = workload_def.get("catalog_id")

    # Load catalog workload if referenced
    workload = {}
    if catalog_id:
        catalog_path = Path(f"/etc/aiperf/workloads/{catalog_id}.yaml")
        if catalog_path.exists():
            with open(catalog_path) as f:
                workload = yaml.safe_load(f)
        # Apply overrides from sidecar
        override = workload_def.get("override", {})
        for key, value in override.items():
            if isinstance(value, dict) and key in workload:
                workload[key] = {**workload.get(key, {}), **value}
            else:
                workload[key] = value
        workload["catalog_id"] = catalog_id
    else:
        # Custom workload
        workload = workload_def.get("custom", workload_def)

    # Run AIPerf or load existing output
    if args.aiperf_output:
        aiperf_path = Path(args.aiperf_output)
    elif args.target:
        if args.dry_run:
            print("DRY RUN — would execute AIPerf with workload:", json.dumps(workload, indent=2))
            sys.exit(0)
        aiperf_path = run_aiperf(args.target, workload, output_dir)
    else:
        sys.exit("ERROR: either --target or --aiperf-output required")

    # Parse metrics
    metrics = parse_aiperf_output(aiperf_path)

    # Build artifact
    artifact = build_artifact(sidecar, metrics, workload, args.concurrency)

    # Validate
    validate_artifact(artifact)

    # Write artifact
    filename = artifact_filename(sidecar, workload, args.concurrency)
    artifact_path = output_dir / filename
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Enriched artifact: {artifact_path}", flush=True)
    print(f"SLO: {'PASS' if artifact.get('slo', {}).get('overall_pass', True) else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
