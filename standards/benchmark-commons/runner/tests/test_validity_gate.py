#!/usr/bin/env python3
"""Tests for the post-run benchmark validity gate."""
from __future__ import annotations

import os
import sys
import unittest

RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RUNNER_DIR)
ARTIFACT_DIR = os.path.dirname(RUNNER_DIR)
WORKLOAD_DIR = os.path.join(ARTIFACT_DIR, "workloads")

import yaml  # noqa: E402
from validate_run import (  # noqa: E402
    CLASS_BASELINE,
    CLASS_INVALID,
    CLASS_PRODUCTION,
    validate_artifact,
)


def load_card(name: str) -> dict:
    with open(os.path.join(WORKLOAD_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def base_artifact(dataset_type: str = "synthetic", catalog_id: str = "concurrency-sweep") -> dict:
    return {
        "schema_version": "1.0.0",
        "artifact_id": "00000000-0000-4000-8000-000000000000",
        "created_at": "2026-06-02T00:00:00Z",
        "source_tool": {
            "name": "bench-standard.py",
            "version": "1.0.0",
            "enrichment_version": "1.0.0",
        },
        "model": {"name": "fixture", "id": "org/fixture"},
        "engine": {"name": "vllm"},
        "infrastructure": {"substrate": "local", "instance_type": "p6-b300.48xlarge"},
        "workload": {
            "use_case": catalog_id,
            "catalog_id": catalog_id,
            "dataset": {"type": dataset_type},
            "load": {"warmup_requests": 30},
        },
        "metrics": {
            "duration_s": 120.0,
            "completed": 100,
            "failed": 0,
            "error_rate": 0.0,
            "ttft_ms": {"mean": 100, "p50": 90, "p90": 150, "p95": 175, "p99": 200},
            "tpot_ms": {"mean": 10, "p50": 9, "p90": 13, "p95": 14, "p99": 15},
            "itl_ms": {"mean": 10, "p50": 9, "p90": 13, "p95": 14, "p99": 15},
            "e2e_ms": {"mean": 1000, "p50": 900, "p90": 1300, "p95": 1400, "p99": 1500},
            "output_toks_per_s": 1200.0,
            "request_throughput": 0.83,
        },
        "extensions": {
            "reconciliation": {"client_ok": 100, "prom_success": 100, "reconciled": True},
            "gpu_telemetry": {"gpu_util_pct_mean": 78.0},
            "cache_stats": {
                "kv_utilization_pct_mean": 55.0,
                "prefix_hit_rate": 0.25,
            },
        },
    }


class TestValidityGate(unittest.TestCase):
    def test_synthetic_card_is_controlled_baseline(self):
        artifact = base_artifact()
        validity = validate_artifact(artifact, load_card("concurrency-sweep"))
        self.assertEqual(validity.classification, CLASS_BASELINE)
        self.assertFalse(validity.production_representative)
        self.assertIn("run sharegpt-production-mix", " ".join(validity.required_next))

    def test_synthetic_cannot_support_production_claim(self):
        artifact = base_artifact()
        validity = validate_artifact(
            artifact,
            load_card("concurrency-sweep"),
            claim="production",
            strict_production=True,
        )
        self.assertEqual(validity.classification, CLASS_INVALID)
        self.assertTrue(any("production claim requires" in f for f in validity.failures))

    def test_production_mix_can_be_production_representative(self):
        artifact = base_artifact(dataset_type="trace-replay", catalog_id="production-mix")
        artifact["extensions"]["distribution"] = {
            "input_tokens_histogram": {"p50": 240, "p99": 7800},
            "output_tokens_histogram": {"p50": 180, "p99": 2400},
            "ttft_histogram_by_concurrency": {"c64": {"p50": 200, "p99": 900}},
        }
        validity = validate_artifact(
            artifact,
            load_card("production-mix"),
            claim="production",
            strict_production=True,
        )
        self.assertEqual(validity.classification, CLASS_PRODUCTION)
        self.assertTrue(validity.production_representative)
        self.assertEqual(validity.failures, [])

    def test_missing_required_metric_invalidates_production_claim(self):
        artifact = base_artifact(dataset_type="trace-replay", catalog_id="production-mix")
        validity = validate_artifact(
            artifact,
            load_card("production-mix"),
            claim="production",
            strict_production=True,
        )
        self.assertEqual(validity.classification, CLASS_INVALID)
        self.assertTrue(any("input_tokens_histogram" in f for f in validity.failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
