#!/usr/bin/env python3
"""Post-run benchmark validity gate.

The observability smoke test answers "can we measure?". This validator answers
"what claim is this artifact allowed to support?" It reads an enriched artifact
and the workload card that produced it, then writes a `validity` block with a
stable classification:

  - valid_controlled_baseline
  - valid_production_representative
  - valid_smoke_only
  - invalid_or_incomplete

Synthetic workloads can be valid controlled baselines. Production-representative
claims require a real/trace workload and the card's required metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CLASS_INVALID = "invalid_or_incomplete"
CLASS_SMOKE = "valid_smoke_only"
CLASS_BASELINE = "valid_controlled_baseline"
CLASS_PRODUCTION = "valid_production_representative"

PRODUCTION_DATASET_TYPES = {"real", "trace-replay"}
SYNTHETIC_DATASET_TYPES = {
    "synthetic",
    "synthetic-agentic",
    "synthetic-multiturn",
    "synthetic-rag",
    "synthetic-shared-prefix",
}


@dataclass
class Validity:
    classification: str = CLASS_BASELINE
    production_representative: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    required_next: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.classification != CLASS_INVALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "production_representative": self.production_representative,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "failures": self.failures,
            "required_next": self.required_next,
        }


def _get(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _non_null(obj: dict[str, Any], path: str) -> bool:
    return _get(obj, path) is not None


def _any_non_null(mapping: Any) -> bool:
    if not isinstance(mapping, dict):
        return False
    return any(v is not None for v in mapping.values())


def _metric_present(artifact: dict[str, Any], metric: str) -> bool:
    """Map workload-card required metric names to known artifact locations."""
    aliases = {
        "prefix_cache_hit_rate": [
            "extensions.cache_stats.prefix_hit_rate",
            "extensions.cache_stats.prefix_cache_hit_rate",
            "extensions.cache_stats.prefix_cache_hit_rate_mean",
        ],
        "prefix_hit_rate": [
            "extensions.cache_stats.prefix_hit_rate",
            "extensions.cache_stats.prefix_cache_hit_rate",
            "extensions.cache_stats.prefix_cache_hit_rate_mean",
        ],
        "kv_utilization_pct": [
            "extensions.cache_stats.kv_utilization_pct",
            "extensions.cache_stats.kv_utilization_pct_mean",
            "extensions.cache_stats.kv_cache_usage_pct_mean",
        ],
        "cold_vs_warm_ttft_ratio": [
            "extensions.cache_stats.cold_vs_warm_ttft_ratio",
            "extensions.cache_stats.ttft_cold_warm_ratio",
        ],
        "preemption_count": [
            "extensions.cache_stats.preemption_count",
            "extensions.cache_stats.num_preemptions",
        ],
        "input_length_p50_p99": [
            "extensions.distribution.input_length_p50_p99",
            "extensions.distribution.input_tokens.p50",
            "workload.dataset.observed_distribution.input_tokens.p50",
        ],
        "output_length_p50_p99": [
            "extensions.distribution.output_length_p50_p99",
            "extensions.distribution.output_tokens.p50",
            "workload.dataset.observed_distribution.output_tokens.p50",
        ],
        "per_turn_ttft_distribution": [
            "extensions.distribution.per_turn_ttft_distribution",
            "extensions.per_turn_ttft_distribution",
        ],
        "ttft_histogram_by_concurrency": [
            "extensions.distribution.ttft_histogram_by_concurrency",
        ],
        "input_tokens_histogram": [
            "extensions.distribution.input_tokens_histogram",
        ],
        "output_tokens_histogram": [
            "extensions.distribution.output_tokens_histogram",
        ],
    }
    for path in aliases.get(metric, [metric]):
        if _non_null(artifact, path):
            return True
    return False


def _required_metrics(card: dict[str, Any]) -> list[str]:
    required: list[str] = []
    validation = card.get("validation") or {}
    required.extend(validation.get("required_metrics") or [])

    extensions = card.get("extensions") or {}
    cache_stats = extensions.get("cache_stats") or {}
    for name, requirement in cache_stats.items():
        if str(requirement).lower() == "required":
            required.append(name)
    distribution = extensions.get("distribution") or {}
    for name, requirement in distribution.items():
        if str(requirement).lower() == "required":
            required.append(name)
    return sorted(set(str(r) for r in required))


def _artifact_dataset_type(artifact: dict[str, Any], card: dict[str, Any]) -> str:
    return str(
        _get(artifact, "workload.dataset.type")
        or _get(card, "dataset.type")
        or "unknown"
    )


def _is_smoke(card: dict[str, Any], artifact: dict[str, Any]) -> bool:
    text = " ".join([
        str(card.get("catalog_id", "")),
        str(card.get("use_case", "")),
        str(card.get("description", "")),
        str(_get(artifact, "workload.use_case") or ""),
    ]).lower()
    return "smoke" in text


def validate_artifact(
    artifact: dict[str, Any],
    card: dict[str, Any],
    *,
    claim: str = "auto",
    strict_production: bool = False,
) -> Validity:
    validity = Validity()

    dataset_type = _artifact_dataset_type(artifact, card)
    required_next: set[str] = set()

    required_top = [
        "schema_version",
        "artifact_id",
        "created_at",
        "source_tool.name",
        "model.id",
        "engine.name",
        "infrastructure.instance_type",
        "workload.catalog_id",
    ]
    for path in required_top:
        if not _non_null(artifact, path):
            validity.failures.append(f"missing {path}")

    metrics = artifact.get("metrics") or {}
    for path in (
        "metrics.completed",
        "metrics.failed",
        "metrics.error_rate",
        "metrics.duration_s",
        "metrics.output_toks_per_s",
        "metrics.request_throughput",
    ):
        if not _non_null(artifact, path):
            validity.failures.append(f"missing {path}")

    for family in ("ttft_ms", "tpot_ms", "e2e_ms"):
        for pct in ("p50", "p99"):
            if not _non_null(metrics, f"{family}.{pct}"):
                validity.failures.append(f"missing metrics.{family}.{pct}")

    completed = metrics.get("completed")
    if isinstance(completed, int) and completed < 10:
        validity.warnings.append(
            f"only {completed} completed requests; treat as smoke unless repeated"
        )

    ttft_p50 = _get(metrics, "ttft_ms.p50")
    ttft_p99 = _get(metrics, "ttft_ms.p99")
    if isinstance(ttft_p50, (int, float)) and ttft_p50 > 0 and isinstance(ttft_p99, (int, float)):
        ratio = ttft_p99 / ttft_p50
        if ratio > 3:
            validity.warnings.append(
                f"ttft p99/p50 ratio {ratio:.2f} > 3; run may be unstable or saturated"
            )

    source_tool = _get(artifact, "source_tool.name")
    extensions = artifact.get("extensions") or {}
    reconciliation = extensions.get("reconciliation") or {}
    if source_tool == "bench-standard.py":
        if reconciliation.get("reconciled") is not True:
            validity.failures.append("Prometheus/client request reconciliation did not pass")
        if not _any_non_null(extensions.get("gpu_telemetry")):
            validity.failures.append("missing DCGM/GPU telemetry from Prometheus")
    else:
        validity.warnings.append(
            "artifact was not produced by bench-standard.py; Prometheus/DCGM reconciliation is not enforced"
        )
        required_next.add("use bench-standard.py or attach reconciliation and GPU telemetry")

    missing_required = []
    for metric in _required_metrics(card):
        if not _metric_present(artifact, metric):
            missing_required.append(metric)
    for metric in missing_required:
        validity.failures.append(f"missing card-required metric: {metric}")

    if dataset_type in PRODUCTION_DATASET_TYPES:
        validity.reasons.append(f"dataset.type={dataset_type} can support production-representative claims")
    elif dataset_type in SYNTHETIC_DATASET_TYPES:
        validity.reasons.append(f"dataset.type={dataset_type} supports controlled baseline claims only")
        required_next.add("run sharegpt-production-mix or production-mix before production claim")
    else:
        validity.warnings.append(f"unrecognized dataset.type={dataset_type}")
        required_next.add("document dataset provenance")

    wants_production = claim == "production" or (
        claim == "auto" and dataset_type in PRODUCTION_DATASET_TYPES
    )
    if wants_production and dataset_type not in PRODUCTION_DATASET_TYPES:
        validity.failures.append(
            f"production claim requires dataset.type in {sorted(PRODUCTION_DATASET_TYPES)}, got {dataset_type}"
        )

    if validity.failures:
        if wants_production or strict_production:
            validity.classification = CLASS_INVALID
            validity.production_representative = False
        elif _is_smoke(card, artifact):
            validity.classification = CLASS_SMOKE
            validity.production_representative = False
        else:
            validity.classification = CLASS_BASELINE
            validity.production_representative = False
    elif wants_production:
        validity.classification = CLASS_PRODUCTION
        validity.production_representative = True
    elif _is_smoke(card, artifact):
        validity.classification = CLASS_SMOKE
        validity.production_representative = False
    else:
        validity.classification = CLASS_BASELINE
        validity.production_representative = False

    validity.required_next = sorted(required_next)
    return validity


def _print_summary(path: Path, validity: Validity) -> None:
    print(f"VALIDITY {path.name}: {validity.classification}")
    print(f"  production_representative: {str(validity.production_representative).lower()}")
    for msg in validity.failures:
        print(f"  FAIL: {msg}")
    for msg in validity.warnings:
        print(f"  WARN: {msg}")
    for msg in validity.required_next:
        print(f"  NEXT: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate benchmark artifact claim strength")
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--workload", required=True, type=Path, help="Workload card YAML")
    ap.add_argument("--claim", choices=("auto", "baseline", "production", "smoke"), default="auto")
    ap.add_argument("--strict-production", action="store_true",
                    help="Exit non-zero if a production claim is incomplete or unsupported")
    ap.add_argument("--update", action="store_true", help="Write the validity block back into each artifact")
    args = ap.parse_args()

    card = yaml.safe_load(args.workload.read_text()) or {}
    failed = 0
    for path in args.artifacts:
        artifact = json.loads(path.read_text())
        validity = validate_artifact(
            artifact,
            card,
            claim=args.claim,
            strict_production=args.strict_production,
        )
        artifact["validity"] = validity.as_dict()
        if args.update:
            path.write_text(json.dumps(artifact, indent=2) + "\n")
        _print_summary(path, validity)
        if args.strict_production and validity.classification == CLASS_INVALID:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
