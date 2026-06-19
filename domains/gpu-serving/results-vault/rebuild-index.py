#!/usr/bin/env python3
"""rebuild-index.py — Scan results-vault/ and build a manifest of all v1 artifacts.

One row per artifact with the most-queried fields hoisted to the top level so jq/filter
queries don't need to open every file.

Run:
  python3 domains/gpu-serving/results-vault/rebuild-index.py
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

VAULT = Path(__file__).resolve().parent


def _parse_cuda_from_image(container_image: str | None) -> str | None:
    """Infer CUDA major.minor from image tag.

    Conventions across vLLM/SGLang/NGC:
      cu118 / cu121 / cu124 -> 11.8 / 12.1 / 12.4 (split at 2 digits)
      cu130                 -> 13.0 (new 3-digit convention when major >= 10)
    """
    if not container_image:
        return None
    m = re.search(r"cu(\d{3,4})", container_image)
    if not m:
        return None
    digits = m.group(1)
    # 3-digit patterns for major >= 10: cu130 -> 13.0
    # All older patterns use 3 digits too (cu118 = 11.8) — split as (first two | last one)
    if len(digits) == 3:
        return f"{digits[:2]}.{digits[2]}"
    # 4-digit patterns (cu1230?) — split first two for major
    return f"{digits[:2]}.{digits[2:]}"


def _parse_engine_version(container_image: str | None, framework_version: str | None) -> str | None:
    """Extract engine version from image tag like 'vllm/vllm-openai:v0.8.3' or 'lmsysorg/sglang:v0.5.10-cu130'."""
    if container_image:
        # Tag after the last ':' — strip any cu/build suffix after a dash
        if ":" in container_image:
            tag = container_image.rsplit(":", 1)[-1]
            # Keep only the version portion (v0.5.10 from v0.5.10-cu130-blackwell)
            m = re.match(r"(v?\d+\.\d+(?:\.\d+)?)", tag)
            if m:
                return m.group(1)
            return tag
    return framework_version


def summarize(path: Path) -> dict:
    with open(path) as f:
        env = json.load(f)
    model = env.get("model", {}) or {}
    engine = env.get("engine", {}) or {}
    framework = env.get("framework", {}) or {}
    infra = env.get("infrastructure", {}) or {}
    gpu = (infra.get("gpu") or {})
    workload = env.get("workload", {}) or {}
    dataset = (workload.get("dataset") or {})
    load = (workload.get("load") or {})
    metrics = env.get("metrics", {}) or {}
    slo = env.get("slo", {}) or {}
    ext = env.get("extensions", {}) or {}
    source_tool = (env.get("source_tool") or {})

    spec = engine.get("speculative_decode") or {}
    cost = (ext.get("cost") or {})
    spec_stats = (ext.get("speculative_decode_stats") or {})
    gpu_tel = (ext.get("gpu_telemetry") or {})
    cache = (ext.get("cache_stats") or {})
    recon = (ext.get("reconciliation") or {})
    session_md = (ext.get("session_metadata") or {})

    container_image = engine.get("container_image")

    # Input tokens can appear under several field names
    isl = (
        (dataset.get("input_tokens") or {}).get("mean")
        if isinstance(dataset.get("input_tokens"), dict)
        else dataset.get("input_tokens")
    )
    if isl is None:
        isl = (dataset.get("input_tokens_first_turn") or {}).get("mean") if isinstance(dataset.get("input_tokens_first_turn"), dict) else None

    osl = (dataset.get("output_tokens") or {}).get("mean") if isinstance(dataset.get("output_tokens"), dict) else dataset.get("output_tokens")

    return {
        # Identity
        "file": path.name,
        "created_at": env.get("created_at"),
        "run_date": session_md.get("run_date"),
        "run_start": session_md.get("run_start"),
        "run_end": session_md.get("run_end"),
        "blueprint": session_md.get("blueprint"),
        "phase": session_md.get("phase"),

        # Producer / schema
        "schema_version": env.get("schema_version"),
        "source_tool_name": source_tool.get("name"),
        "source_tool_version": source_tool.get("version"),
        "enrichment_version": source_tool.get("enrichment_version"),

        # Model
        "model_id": model.get("id"),
        "model_name": model.get("name"),
        "model_architecture": model.get("architecture"),
        "parameters_total": model.get("parameters_total"),
        "parameters_active": model.get("parameters_active"),
        "quantization": model.get("quantization"),
        "max_model_len": model.get("max_model_len"),

        # Engine build
        "engine_name": engine.get("name"),
        "engine_config_tag": engine.get("engine_config_tag") or session_md.get("engine_config_tag"),
        "container_image": container_image,
        "base_image": engine.get("base_image"),
        "dockerfile": engine.get("dockerfile"),
        "engine_version": _parse_engine_version(container_image or engine.get("base_image"), framework.get("version")),
        "cuda_version": _parse_cuda_from_image(container_image or engine.get("base_image")),

        # Engine config
        "tensor_parallel": engine.get("tensor_parallel"),
        "pipeline_parallel": engine.get("pipeline_parallel"),
        "data_parallel": engine.get("data_parallel"),
        "expert_parallel": engine.get("expert_parallel"),
        "replicas": engine.get("replicas"),
        "reasoning_enabled": engine.get("reasoning"),
        "kv_cache_dtype": engine.get("kv_cache_dtype"),
        "attention_backend": engine.get("attention_backend"),
        "speculative_algorithm": spec.get("algorithm") or spec.get("method"),
        "speculative_num_steps": spec.get("num_steps"),
        "speculative_num_draft_tokens": spec.get("num_draft_tokens") or spec.get("num_tokens"),
        "draft_model": spec.get("draft_model"),

        # Framework
        "framework_name": framework.get("name"),
        "framework_version": framework.get("version"),

        # Infrastructure
        "substrate": infra.get("substrate"),
        "instance_type": infra.get("instance_type"),
        "region": infra.get("region"),
        "gpu_type": gpu.get("name"),
        "gpu_arch": gpu.get("arch"),
        "gpu_count": gpu.get("count"),
        "vram_gb_per_gpu": gpu.get("vram_gb"),
        "interconnect": gpu.get("interconnect"),

        # Workload
        "workload_catalog_id": workload.get("catalog_id"),
        "workload_use_case": workload.get("use_case"),
        "modality": workload.get("modality"),
        "concurrency": load.get("concurrency") or load.get("current_level"),
        "request_rate": load.get("request_rate"),
        "input_tokens_mean": isl,
        "output_tokens_mean": osl,

        # Metrics
        "duration_s": metrics.get("duration_s"),
        "completed": metrics.get("completed"),
        "failed": metrics.get("failed"),
        "error_rate": metrics.get("error_rate"),
        "agg_tok_per_s": metrics.get("output_toks_per_s"),
        "request_throughput_per_s": metrics.get("request_throughput"),
        "total_toks_per_s": metrics.get("total_toks_per_s"),

        # primary_throughput: modality-aware metric to use in charts.
        # Text generation → output tok/s. Embedding/reranker/audio → requests/s
        # (because output tokens are 0 or meaningless for those modalities).
        "primary_throughput": (
            metrics.get("output_toks_per_s")
            if (workload.get("modality") == "text"
                and (metrics.get("output_toks_per_s") or 0) > 0)
            else metrics.get("request_throughput")
        ),
        "primary_throughput_unit": (
            "tok/s"
            if (workload.get("modality") == "text"
                and (metrics.get("output_toks_per_s") or 0) > 0)
            else "req/s"
        ),
        "ttft_p50_ms": (metrics.get("ttft_ms") or {}).get("p50"),
        "ttft_p99_ms": (metrics.get("ttft_ms") or {}).get("p99"),
        "tpot_p50_ms": (metrics.get("tpot_ms") or {}).get("p50"),
        "tpot_p99_ms": (metrics.get("tpot_ms") or {}).get("p99"),
        "e2e_p50_ms": (metrics.get("e2e_ms") or {}).get("p50"),
        "e2e_p99_ms": (metrics.get("e2e_ms") or {}).get("p99"),

        # SLO / spec / cost
        "slo_overall_pass": slo.get("overall_pass"),
        "spec_accept_rate": spec_stats.get("accept_rate_mean"),
        "spec_accept_length": spec_stats.get("accept_length_mean"),
        "dollars_per_1m_output_tokens": cost.get("dollars_per_1m_output_tokens"),
        "instance_cost_per_hr": cost.get("instance_cost_per_hr"),

        # Observability (extensions)
        "prefix_hit_rate": cache.get("prefix_hit_rate"),
        "kv_utilization_pct": cache.get("kv_utilization_pct_mean"),
        "preemption_count": cache.get("preemption_count"),
        "gpu_util_pct_mean": gpu_tel.get("gpu_util_pct_mean"),
        "hbm_bw_util_pct_mean": gpu_tel.get("hbm_bw_util_pct_mean"),
        "tensor_active_pct_mean": gpu_tel.get("tensor_active_pct_mean"),
        "power_draw_w_mean": gpu_tel.get("power_draw_w_mean"),

        # Data quality
        "reconciled": recon.get("reconciled"),
    }


def main():
    files = sorted(VAULT.glob("*.json"))
    # Skip non-artifact files
    artifacts = [f for f in files if f.name not in {"index.json"}]

    rows = []
    for f in artifacts:
        try:
            rows.append(summarize(f))
        except Exception as e:
            print(f"skip {f.name}: {e}")

    index = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    (VAULT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Indexed {len(rows)} artifacts → {VAULT / 'index.json'}")


if __name__ == "__main__":
    main()
