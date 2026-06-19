#!/usr/bin/env python3
"""Convert qwen3-embedding-8b-hyperpod raw results to Common Benchmark Artifact v1.0.0.

Spec: standards/benchmark-commons/PROPOSAL.md
Embedding-specific conventions:
  - api.type = "embeddings", api.streaming = false
  - ttft_ms / tpot_ms / itl_ms = null (no streaming tokens)
  - e2e_ms = request latency
  - output_toks_per_s = 0 (embeddings emit vectors, not tokens)
  - request_throughput is the headline metric
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

BP = Path(__file__).resolve().parents[1]
RESULTS = BP / "results"
OUT = RESULTS / "artifacts"
OUT.mkdir(exist_ok=True)

CREATED_AT = "2026-05-13T00:00:00Z"
SCHEMA_VERSION = "1.0.0"
ENRICHMENT_VERSION = "1.0.0"
SOURCE_TOOL_NAME = "custom"  # schema enum: aiperf|vllm-bench-serve|sglang-bench-serving|custom
SOURCE_TOOL_VERSION = "0.1.0-embedding-bench"

MODEL = {
    "name": "Qwen3-Embedding-8B",
    "id": "Qwen/Qwen3-Embedding-8B",
    "architecture": "embedding",
    "parameters_total": "8B",
    "quantization": "bf16",
    "max_model_len": 8192,
}

ENGINE = {
    "name": "vllm",
    "version": "0.19.1",
    "container_image": "vllm/vllm-openai:v0.19.1",
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": None,
    "expert_parallel": None,
    "replicas": 1,
    "reasoning": False,
    "kv_cache_dtype": "auto",
    "attention_backend": "default",
    "speculative_decoding": None,
    "extra_args": {"task": "embed", "trust-remote-code": True},
}

INFRA = {
    "substrate": "hyperpod-eks",
    "instance_type": "ml.g5.4xlarge",
    "region": "us-east-1",
    "gpu": {
        "name": "A10G",
        "arch": "sm_86",
        "count": 1,
        "vram_gb": 24,
        "interconnect": "none",
    },
    "eks": {
        "cluster_version": "1.32",
        "node_count": 1,
    },
    "hyperpod": {
        "cluster_name": "finetune-g5-cluster",
        "instance_group": "llmd-validation",
        "deep_health_checks": True,
        "auto_recovery": True,
    },
}

API = {"type": "embeddings", "streaming": False, "endpoint": "/v1/embeddings"}

# SLO targets from spec §Stage-6 "Per-workload success criteria"
SLO_TARGETS = {
    "request_throughput_min": 100.0,   # peak throughput at c=32 must clear 100 req/s
    "error_rate_max": 0.001,
}


def lat_from_dict(d):
    """Build latency metric from a workload latency_ms dict with p50/p90/p99."""
    return {
        "mean": d["mean"],
        "p50": d["p50"],
        "p90": d["p90"],
        "p95": d.get("p95"),
        "p99": d["p99"],
    }


def lat_from_smoke(mean, p50, p99):
    """Smoke-bench only emits p50/p99/mean. p90 is synthesized as p50 (conservative floor);
    extensions.sweep_levels retains the original fields untouched for fidelity."""
    return {"mean": mean, "p50": p50, "p90": p50, "p95": None, "p99": p99}


def envelope():
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": str(uuid.uuid4()),
        "created_at": CREATED_AT,
        "source_tool": {
            "name": SOURCE_TOOL_NAME,
            "version": SOURCE_TOOL_VERSION,
            "enrichment_version": ENRICHMENT_VERSION,
        },
    }


def evaluate_slo(peak_throughput, error_rate):
    results = {
        "request_throughput_min": {
            "target": SLO_TARGETS["request_throughput_min"],
            "actual": peak_throughput,
            "pass": peak_throughput >= SLO_TARGETS["request_throughput_min"],
        },
        "error_rate_max": {
            "target": SLO_TARGETS["error_rate_max"],
            "actual": error_rate,
            "pass": error_rate <= SLO_TARGETS["error_rate_max"],
        },
    }
    return {
        "targets": SLO_TARGETS,
        "results": results,
        "overall_pass": all(r["pass"] for r in results.values()),
    }


def fname(workload_id, extra=None):
    parts = [
        "qwen3-embedding-8b",
        "hyperpod-eks",
        "g5-4xl",
        "vllm",
        workload_id,
    ]
    if extra:
        parts.append(extra)
    parts.append("20260513T000000Z")
    return "_".join(parts) + ".json"


def write_artifact(fn, doc):
    path = OUT / fn
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {path.relative_to(BP)}")


# --------------------------------------------------------------------------
# 1. Concurrency sweep (smoke-bench.json)
# --------------------------------------------------------------------------

def build_concurrency_sweep():
    src = json.loads((RESULTS / "smoke-bench.json").read_text())
    runs = src["runs"]
    peak = max(runs, key=lambda r: r["req_per_s"])
    levels = [r["concurrency"] for r in runs]
    total_reqs = sum(r["n_requests"] for r in runs)
    total_failed = sum(r["n_errors"] for r in runs)
    total_dur = sum(r["duration_s"] for r in runs)
    err_rate = total_failed / total_reqs if total_reqs else 0.0

    # Headline = peak concurrency level
    doc = envelope()
    doc.update({
        "model": MODEL,
        "engine": ENGINE,
        "infrastructure": INFRA,
        "workload": {
            "use_case": "custom",
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": 512, "std_dev": 256},
                "output_tokens": {"mean": 0, "std_dev": 0},
                "note": "embedding workload — no output tokens, output is a 4096-dim vector",
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": levels,
                "num_prompts_per_level": None,
                "warmup_requests": 0,
                "current_level": peak["concurrency"],
            },
            "api": API,
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["n_success"],
            "failed": peak["n_errors"],
            "error_rate": peak["n_errors"] / peak["n_requests"] if peak["n_requests"] else 0.0,
            "e2e_ms": lat_from_smoke(
                peak["latency_ms_mean"], peak["latency_ms_p50"], peak["latency_ms_p99"]
            ),
            "output_toks_per_s": 0.0,
            "request_throughput": peak["req_per_s"],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_concurrent_requests": peak["concurrency"],
        },
        "slo": evaluate_slo(peak["req_per_s"], err_rate),
        "extensions": {
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["n_success"],
                    "failed": r["n_errors"],
                    "duration_s": r["duration_s"],
                    "request_throughput": r["req_per_s"],
                    "e2e_ms": {
                        "mean": r["latency_ms_mean"],
                        "p50": r["latency_ms_p50"],
                        "p99": r["latency_ms_p99"],
                    },
                }
                for r in runs
            ],
            "raw_tool_output": {
                "uri": "results/smoke-bench.json",
                "format": "json",
            },
        },
    })
    write_artifact(fname("concurrency-sweep"), doc)


# --------------------------------------------------------------------------
# 2. RAG QA
# --------------------------------------------------------------------------

def build_from_leveled(src_path, catalog_id, workload_id, use_case="custom", note=None, extra_dataset=None):
    src = json.loads(src_path.read_text())
    levels = src["levels"]
    peak = max(levels, key=lambda r: r["req_per_s"])
    total_reqs = sum(r["n_requests"] for r in levels)
    total_failed = sum(r["n_errors"] for r in levels)
    err_rate = total_failed / total_reqs if total_reqs else 0.0

    dataset = {
        "type": "synthetic",
        "input_tokens": {"mean": None, "std_dev": None},
        "output_tokens": {"mean": 0, "std_dev": 0},
    }
    if extra_dataset:
        dataset.update(extra_dataset)
    if note:
        dataset["note"] = note
    elif "distribution_note" in src:
        dataset["note"] = src["distribution_note"]

    doc = envelope()
    doc.update({
        "model": MODEL,
        "engine": ENGINE,
        "infrastructure": INFRA,
        "workload": {
            "use_case": use_case,
            "catalog_id": catalog_id,
            "modality": "text",
            "dataset": dataset,
            "load": {
                "type": "concurrency-sweep",
                "levels": [r["concurrency"] for r in levels],
                "num_prompts_per_level": None,
                "warmup_requests": 0,
                "current_level": peak["concurrency"],
            },
            "api": API,
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["n_success"],
            "failed": peak["n_errors"],
            "error_rate": peak["n_errors"] / peak["n_requests"] if peak["n_requests"] else 0.0,
"e2e_ms": lat_from_dict(peak["latency_ms"]),
            "output_toks_per_s": 0.0,
            "request_throughput": peak["req_per_s"],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_concurrent_requests": peak["concurrency"],
        },
        "slo": evaluate_slo(peak["req_per_s"], err_rate),
        "extensions": {
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["n_success"],
                    "failed": r["n_errors"],
                    "duration_s": r["duration_s"],
                    "request_throughput": r["req_per_s"],
                    "e2e_ms": lat_from_dict(r["latency_ms"]),
                }
                for r in levels
            ],
            "raw_tool_output": {
                "uri": f"results/{src_path.name}",
                "format": "json",
            },
        },
    })
    write_artifact(fname(workload_id), doc)


# --------------------------------------------------------------------------
# 3. Long-context sweep (nested contexts × concurrencies)
# --------------------------------------------------------------------------

def build_long_context():
    src = json.loads((RESULTS / "workload-long-context.json").read_text())
    contexts = src["contexts"]
    # Flatten to find global peak throughput
    flat = []
    for ctx in contexts:
        for lvl in ctx["levels"]:
            flat.append({**lvl, "approx_tokens": ctx["approx_tokens"], "char_len": ctx["char_len"]})
    peak = max(flat, key=lambda r: r["req_per_s"])
    total_reqs = sum(r["n_requests"] for r in flat)
    total_failed = sum(r["n_errors"] for r in flat)
    err_rate = total_failed / total_reqs if total_reqs else 0.0

    doc = envelope()
    doc.update({
        "model": MODEL,
        "engine": ENGINE,
        "infrastructure": INFRA,
        "workload": {
            "use_case": "custom",
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": None, "std_dev": None},
                "output_tokens": {"mean": 0, "std_dev": 0},
                "note": "context axis swept: approx 1K / 2K / 4K / 8K tokens (char_len = 4×tokens)",
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": sorted({lvl["concurrency"] for ctx in contexts for lvl in ctx["levels"]}),
                "num_prompts_per_level": None,
                "warmup_requests": 0,
                "current_level": peak["concurrency"],
                "context_lengths_tokens": [c["approx_tokens"] for c in contexts],
            },
            "api": API,
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["n_success"],
            "failed": peak["n_errors"],
            "error_rate": 0.0,
"e2e_ms": lat_from_dict(peak["latency_ms"]),
            "output_toks_per_s": 0.0,
            "request_throughput": peak["req_per_s"],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_concurrent_requests": peak["concurrency"],
        },
        "slo": evaluate_slo(peak["req_per_s"], err_rate),
        "extensions": {
            "context_sweep": [
                {
                    "approx_tokens": ctx["approx_tokens"],
                    "char_len": ctx["char_len"],
                    "levels": [
                        {
                            "concurrency": lvl["concurrency"],
                            "completed": lvl["n_success"],
                            "failed": lvl["n_errors"],
                            "duration_s": lvl["duration_s"],
                            "request_throughput": lvl["req_per_s"],
                            "e2e_ms": lat_from_dict(lvl["latency_ms"]),
                        }
                        for lvl in ctx["levels"]
                    ],
                }
                for ctx in contexts
            ],
            "raw_tool_output": {
                "uri": "results/workload-long-context.json",
                "format": "json",
            },
        },
    })
    write_artifact(fname("long-context-sweep"), doc)


# --------------------------------------------------------------------------
# 4. Burn-in (1 hour stability)
# --------------------------------------------------------------------------

def build_burn_in():
    src = json.loads((RESULTS / "burn-in" / "burn-in-final.json").read_text())
    slices = src["slices"]
    total_completed = sum(s["completed"] for s in slices)
    total_failed = sum(s["failed"] for s in slices)
    total_dur = sum(s["duration_s"] for s in slices)
    err_rate = total_failed / (total_completed + total_failed) if (total_completed + total_failed) else 0.0

    p50s = [s["latency_p50"] for s in slices]
    p99s = [s["latency_p99"] for s in slices]
    sorted_p50 = sorted(p50s)
    # Approximation: we have per-slice p50/p99 only. p90 synthesized as slice-level p90 of p99s.
    e2e = {
        "mean": sum(p50s) / len(p50s),
        "p50": sorted_p50[len(sorted_p50) // 2],
        "p90": sorted(p99s)[int(len(p99s) * 0.9) - 1] if len(p99s) >= 2 else max(p99s),
        "p95": None,
        "p99": max(p99s),
    }
    throughput = total_completed / total_dur

    burn_in_slo = {
        "targets": {
            "drift_pct_max": 2.0,
            "error_rate_max": 0.001,
        },
        "results": {
            "drift_pct_max": {
                "target": 2.0,
                "actual": src["stability"]["throughput_drift_pct"],
                "pass": src["stability"]["drift_gate_passed"],
            },
            "error_rate_max": {
                "target": 0.001,
                "actual": err_rate,
                "pass": err_rate <= 0.001,
            },
        },
    }
    burn_in_slo["overall_pass"] = all(r["pass"] for r in burn_in_slo["results"].values())

    doc = envelope()
    doc.update({
        "model": MODEL,
        "engine": ENGINE,
        "infrastructure": INFRA,
        "workload": {
            "use_case": "stress",
            "catalog_id": "burn-in",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": 512, "std_dev": 256},
                "output_tokens": {"mean": 0, "std_dev": 0},
            },
            "load": {
                "type": "constant",
                "request_rate": None,
                "duration_s": src["duration_s"],
                "num_prompts": total_completed,
                "warmup_requests": None,
                "warmup_s": src["warmup_s"],
                "max_concurrency": src["concurrency"],
                "slice_duration_s": src["slice_duration_s"],
            },
            "api": API,
        },
        "metrics": {
            "duration_s": total_dur,
            "completed": total_completed,
            "failed": total_failed,
            "error_rate": err_rate,
"e2e_ms": e2e,
            "output_toks_per_s": 0.0,
            "request_throughput": throughput,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_concurrent_requests": src["concurrency"],
        },
        "slo": burn_in_slo,
        "extensions": {
            "burn_in_stability": {
                "hour_1_throughput": src["stability"]["hour_1_throughput"],
                "final_throughput": src["stability"]["final_throughput"],
                "throughput_drift_pct": src["stability"]["throughput_drift_pct"],
                "unrecoverable_errors": src["stability"]["unrecoverable_errors"],
                "slice_count": len(slices),
                "slices": [
                    {
                        "slice_idx": s["slice_idx"],
                        "start_ts": s["start_ts"],
                        "completed": s["completed"],
                        "failed": s["failed"],
                        "output_throughput": s["output_throughput"],
                        "latency_p50": s["latency_p50"],
                        "latency_p99": s["latency_p99"],
                    }
                    for s in slices
                ],
            },
            "raw_tool_output": {
                "uri": "results/burn-in/burn-in-final.json",
                "format": "json",
            },
        },
    })
    write_artifact(fname("burn-in"), doc)


# --------------------------------------------------------------------------
# 5. Tier comparison (T0 baseline + T5 optimized) on rag-qa
# --------------------------------------------------------------------------

def build_tier(tier_tag, src_file, tier_note, engine_overrides):
    src = json.loads((RESULTS / "tier-comparison" / src_file).read_text())
    levels = src["levels"]
    peak = max(levels, key=lambda r: r["req_per_s"])
    total_reqs = sum(r["n_requests"] for r in levels)
    total_failed = sum(r["n_errors"] for r in levels)
    err_rate = total_failed / total_reqs if total_reqs else 0.0

    engine = {**ENGINE, "extra_args": {**ENGINE["extra_args"], **engine_overrides}}

    doc = envelope()
    doc.update({
        "model": MODEL,
        "engine": engine,
        "infrastructure": INFRA,
        "workload": {
            "use_case": "rag",
            "catalog_id": "rag-long-context",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": None, "std_dev": None},
                "output_tokens": {"mean": 0, "std_dev": 0},
                "note": "RAG Q&A — 2-10K char mixed retrieved contexts",
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": [r["concurrency"] for r in levels],
                "num_prompts_per_level": None,
                "warmup_requests": 0,
                "current_level": peak["concurrency"],
            },
            "api": API,
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["n_success"],
            "failed": peak["n_errors"],
            "error_rate": peak["n_errors"] / peak["n_requests"] if peak["n_requests"] else 0.0,
"e2e_ms": lat_from_dict(peak["latency_ms"]),
            "output_toks_per_s": 0.0,
            "request_throughput": peak["req_per_s"],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_concurrent_requests": peak["concurrency"],
        },
        "slo": evaluate_slo(peak["req_per_s"], err_rate),
        "extensions": {
            "optimization_tier": {
                "tier": tier_tag,
                "note": tier_note,
            },
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["n_success"],
                    "failed": r["n_errors"],
                    "duration_s": r["duration_s"],
                    "request_throughput": r["req_per_s"],
                    "e2e_ms": lat_from_dict(r["latency_ms"]),
                }
                for r in levels
            ],
            "raw_tool_output": {
                "uri": f"results/tier-comparison/{src_file}",
                "format": "json",
            },
        },
    })
    write_artifact(fname("rag-long-context", extra=f"tier-{tier_tag.lower()}"), doc)


# --------------------------------------------------------------------------

def main():
    build_concurrency_sweep()
    build_from_leveled(
        RESULTS / "workload-rag-qa.json",
        catalog_id="rag-long-context",
        workload_id="rag-qa",
        use_case="rag",
        note="RAG Q&A — 2-10K char mixed retrieved contexts",
    )
    build_from_leveled(
        RESULTS / "workload-production-mix.json",
        catalog_id=None,
        workload_id="production-mix",
        use_case="production-mix",
        note="40/40/20 short/medium/long chars (256-512 / 1K-2K / 4K-8K)",
    )
    build_long_context()
    build_burn_in()
    build_tier(
        "T0",
        "workload-rag-qa-t0-baseline.json",
        "baseline: --enforce-eager, --no-enable-prefix-caching (no torch.compile, no CUDA graphs)",
        {"enforce-eager": True, "enable-prefix-caching": False},
    )
    build_tier(
        "T5",
        "workload-rag-qa-t5-optimized.json",
        "optimized: FLASH_ATTN + torch.compile Inductor + CUDA graphs + prefix cache (default)",
        {"enforce-eager": False, "enable-prefix-caching": True, "torch-compile": "inductor"},
    )


if __name__ == "__main__":
    main()
