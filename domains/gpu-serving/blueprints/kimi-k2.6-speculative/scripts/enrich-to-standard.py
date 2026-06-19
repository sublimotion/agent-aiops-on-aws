#!/usr/bin/env python3
"""enrich-to-standard.py — Repackage raw per-concurrency JSONs into the v1 benchmark-commons envelope.

Reads:
  results/phase-1b/{config}/c{N}.json
  results/phase-4/c{N}.json
  results/phase-5/{5a,5b,5c}/c{N}.json

Writes:
  results/standard/<model>_<substrate>_<hw>_<engine-config>_<workload>_c<N>.json

Filename scheme matches standards/benchmark-commons/examples/nemotron-super/.
"""
from __future__ import annotations
import glob
import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "standard"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SLUG = "kimi-k2.6"
SUBSTRATE = "ec2-spot"
HW = "p6-b300"
CREATED_AT = "2026-05-13T22:00:00Z"  # session end

MODEL_BLOCK = {
    "name": "Kimi-K2.6",
    "id": "moonshotai/Kimi-K2.6",
    "architecture": "moe-mla-fp8",
    "parameters_total": "1T",
    "parameters_active": "32B",
    "quantization": "fp8",
    "max_model_len": 131072,
}

DRAFT_BLOCK = {
    "name": "Kimi-K2.6-EAGLE3",
    "id": "lightseekorg/kimi-k2.6-eagle3",
    "architecture": "eagle3",
    "parameters_total": "~3B",
    "parameters_active": "~3B",
    "quantization": "bf16",
}

INFRA_BLOCK = {
    "substrate": "ec2-spot",
    "instance_type": "p6-b300.48xlarge",
    "region": "us-east-1",
    "gpu": {
        "name": "B300",
        "arch": "sm_103",
        "count": 8,
        "vram_gb": 275,
        "interconnect": "nvswitch-nv18",
    },
}

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.10-cu130"
SPOT_COST_PER_HR = 25.65  # measured at session time in us-east-1c

# Map from (phase_dir, config_dir) → engine-config tag + engine block overrides
# Engine-config tag goes into the filename.
CONFIGS = {
    # Phase 1b sweep — 13 configs
    ("phase-1b", "s1_d2_k1"): ("sglang-eagle3-s1d2k1", 1, 2, 1),
    ("phase-1b", "s1_d4_k1"): ("sglang-eagle3-s1d4k1", 1, 4, 1),
    ("phase-1b", "s1_d6_k1"): ("sglang-eagle3-s1d6k1", 1, 6, 1),
    ("phase-1b", "s1_d8_k1"): ("sglang-eagle3-s1d8k1", 1, 8, 1),
    ("phase-1b", "s2_d2_k1"): ("sglang-eagle3-s2d2k1", 2, 2, 1),
    ("phase-1b", "s2_d4_k1"): ("sglang-eagle3-s2d4k1", 2, 4, 1),
    ("phase-1b", "s2_d6_k1"): ("sglang-eagle3-s2d6k1", 2, 6, 1),
    ("phase-1b", "s2_d8_k1"): ("sglang-eagle3-s2d8k1", 2, 8, 1),
    ("phase-1b", "s3_d4_k1"): ("sglang-eagle3-s3d4k1", 3, 4, 1),
    ("phase-1b", "s3_d6_k1"): ("sglang-eagle3-s3d6k1", 3, 6, 1),
    ("phase-1b", "s3_d8_k1"): ("sglang-eagle3-s3d8k1", 3, 8, 1),
    ("phase-1b", "s4_d4_k1"): ("sglang-eagle3-s4d4k1", 4, 4, 1),
    ("phase-1b", "s4_d6_k1"): ("sglang-eagle3-s4d6k1", 4, 6, 1),
    ("phase-1b", "s4_d8_k1"): ("sglang-eagle3-s4d8k1", 4, 8, 1),
}

# Phase 4 is a standalone (fullstack = winner + HiCache)
PHASE4 = ("phase-4", None, "sglang-eagle3-s4d4k1-hicache200", 4, 4, 1)

# Phase 5 sub-phases
PHASE5 = {
    "5a-default-stack":    ("sglang-eagle3-s4d4-hicache-cudagraphs", 4, 4, 1, "default"),
    "5b-no-cuda-graph":    ("sglang-eagle3-s4d4-no-cudagraphs",      4, 4, 1, "no_cuda_graph"),
    "5c-tp4-dp2":          ("sglang-eagle3-s4d4-tp4dp2-hicache",     4, 4, 1, "tp4_dp2"),
}

WORKLOAD_BLOCK = {
    "use_case": "concurrency-sweep",
    "catalog_id": "concurrency-sweep",
    "modality": "text",
    "dataset": {
        "type": "synthetic",
        "input_tokens": {"mean": 512, "std_dev": 0},
        "output_tokens": {"mean": 256, "std_dev": 0},
    },
    "api": {
        "type": "completions",
        "streaming": False,
        "endpoint": "/generate",
    },
}

SLO_TARGETS = {
    "ttft_p99_ms": 500,
    "tpot_p99_ms": 30,
    "e2e_p99_ms": 15000,
    "error_rate_max": 0.001,
}


def engine_block(num_steps: int, num_draft: int, topk: int, extra: dict | None = None) -> dict:
    block = {
        "name": "sglang",
        "container_image": SGLANG_IMAGE,
        "base_image": SGLANG_IMAGE,
        "dockerfile": None,
        "tensor_parallel": 8,
        "pipeline_parallel": 1,
        "data_parallel": None,
        "expert_parallel": None,
        "replicas": 1,
        "reasoning": False,
        "kv_cache_dtype": "fp8",
        "attention_backend": "trtllm_mha",
        "speculative_decode": {
            "algorithm": "EAGLE3",
            "draft_model": DRAFT_BLOCK["id"],
            "num_steps": num_steps,
            "num_draft_tokens": num_draft,
            "eagle_topk": topk,
            "draft_attention_backend": "trtllm_mha",
        },
        "extra_args": {
            "reasoning-parser": "kimi_k2",
            "tool-call-parser": "kimi_k2",
            "trust-remote-code": True,
        },
    }
    if extra:
        block.update(extra)
    return block


def ms_per_token_to_tpot(per_req_tps: float) -> float | None:
    if per_req_tps is None or per_req_tps <= 0:
        return None
    return round(1000.0 / per_req_tps, 2)


def fill_percentile_stub(p50_s: float | None, p90_s: float | None, p99_s: float | None) -> dict:
    """Map our {p50, p90, p99} seconds into the standard ms dict.

    p95 isn't recorded — approximate with mid(p90, p99). Mean = p50.
    """
    def ms(x):
        return round(x * 1000.0, 2) if x is not None else None

    p50_ms = ms(p50_s)
    p90_ms = ms(p90_s)
    p99_ms = ms(p99_s)
    p95_ms = round((p90_ms + p99_ms) / 2.0, 2) if p90_ms is not None and p99_ms is not None else None
    return {
        "mean": p50_ms,
        "p50": p50_ms,
        "p90": p90_ms,
        "p95": p95_ms,
        "p99": p99_ms,
    }


def build_envelope(raw: dict, engine_tag: str, num_steps: int, num_draft: int, topk: int,
                   phase: str, variant: str | None = None,
                   engine_extra: dict | None = None) -> dict:
    c = raw["concurrency"]
    per_req = raw.get("per_req_tok_per_s")
    agg = raw.get("agg_tok_per_s")
    duration = raw.get("duration_s")
    ok = raw.get("requests_ok", 0)
    err = raw.get("requests_err", 0)
    total_toks = raw.get("total_tokens", 0)
    accept_rate = raw.get("spec_accept_rate_mean")
    accept_len = raw.get("spec_accept_length_mean")

    p50 = raw.get("p50_latency_s")
    p90 = raw.get("p90_latency_s")
    p99 = raw.get("p99_latency_s")

    e2e = fill_percentile_stub(p50, p90, p99)
    tpot_val = ms_per_token_to_tpot(per_req)
    tpot = {"mean": tpot_val, "p50": tpot_val, "p90": None, "p95": None, "p99": None}

    error_rate = err / (ok + err) if (ok + err) > 0 else 0.0

    # Cost
    cost_per_hr = SPOT_COST_PER_HR
    dollars_per_1m_out = None
    if agg and agg > 0:
        dollars_per_1m_out = round((cost_per_hr / agg) * (1_000_000 / 3600.0), 2)

    engine = engine_block(num_steps, num_draft, topk, engine_extra)

    envelope = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": CREATED_AT,
        "source_tool": {
            "name": "custbench-async",
            "version": "0.1.0",
            "enrichment_version": "1.0.0",
        },
        "model": MODEL_BLOCK,
        "engine": engine,
        "framework": {"name": "sglang-native", "version": "v0.5.10-cu130", "config": {}},
        "infrastructure": INFRA_BLOCK,
        "workload": {
            **WORKLOAD_BLOCK,
            "load": {
                "type": "concurrency-sweep",
                "levels": [1, 8, 32, 64, 128, 256, 512],
                "num_prompts_per_level": ok + err,
                "warmup_requests": 0,
                "current_level": c,
            },
        },
        "metrics": {
            "duration_s": duration,
            "completed": ok,
            "failed": err,
            "error_rate": round(error_rate, 6),
            "ttft_ms": {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None,
                        "note": "Not measured — custbench-async records e2e only; TTFT not separable"},
            "tpot_ms": tpot,
            "itl_ms":  tpot,
            "e2e_ms":  e2e,
            "output_toks_per_s": agg,
            "request_throughput": round(ok / duration, 3) if duration and duration > 0 else None,
            "total_toks_per_s": agg,
            "total_input_tokens": ok * raw.get("input_len", 512),
            "total_output_tokens": total_toks,
            "max_concurrent_requests": c,
        },
        "slo": {
            "targets": SLO_TARGETS,
            "results": {
                "ttft_p99_ms": {"target": SLO_TARGETS["ttft_p99_ms"], "actual": None, "pass": None},
                "tpot_p99_ms": {"target": SLO_TARGETS["tpot_p99_ms"], "actual": tpot_val,
                                "pass": tpot_val is not None and tpot_val <= SLO_TARGETS["tpot_p99_ms"]},
                "e2e_p99_ms":  {"target": SLO_TARGETS["e2e_p99_ms"],  "actual": e2e["p99"],
                                "pass": e2e["p99"] is not None and e2e["p99"] <= SLO_TARGETS["e2e_p99_ms"]},
                "error_rate_max": {"target": SLO_TARGETS["error_rate_max"], "actual": error_rate,
                                   "pass": error_rate <= SLO_TARGETS["error_rate_max"]},
            },
            "overall_pass": None,  # TTFT unknown — cannot certify
        },
        "extensions": {
            "speculative_decode_stats": {
                "accept_rate_mean": accept_rate,
                "accept_length_mean": accept_len,
                "effective_tokens_per_step": round((accept_rate or 0) * (accept_len or 0), 3)
                    if accept_rate is not None and accept_len is not None else None,
            },
            "cost": {
                "instance_cost_per_hr": cost_per_hr,
                "dollars_per_1m_output_tokens": dollars_per_1m_out,
                "formula": f"({cost_per_hr} / {agg}) * (1000000 / 3600)",
            },
            "session_metadata": {
                "phase": phase,
                "variant": variant,
                "engine_config_tag": engine_tag,
                "blueprint": "domains/gpu-serving/blueprints/kimi-k2.6-speculative",
                "run_date": "2026-05-13",
            },
        },
    }
    # Mark overall_pass false if any known SLO failed (TPOT/E2E/error_rate)
    env_slo = envelope["slo"]["results"]
    known_pass = [env_slo["tpot_p99_ms"]["pass"], env_slo["e2e_p99_ms"]["pass"], env_slo["error_rate_max"]["pass"]]
    envelope["slo"]["overall_pass"] = all(p for p in known_pass if p is not None)
    return envelope


def fname(engine_tag: str, c: int) -> str:
    return f"{MODEL_SLUG}_{SUBSTRATE}_{HW}_{engine_tag}_concurrency-sweep_c{c}.json"


def emit(raw_path: Path, engine_tag: str, num_steps: int, num_draft: int, topk: int,
         phase: str, variant: str | None = None, engine_extra: dict | None = None):
    with open(raw_path) as f:
        raw = json.load(f)
    env = build_envelope(raw, engine_tag, num_steps, num_draft, topk, phase, variant, engine_extra)
    out_path = OUT / fname(engine_tag, raw["concurrency"])
    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    return out_path


def main():
    written = 0

    # Phase 1b
    for (phase, cfg), (tag, s, d, k) in CONFIGS.items():
        for jf in sorted((RESULTS / phase / cfg).glob("c*.json")):
            emit(jf, tag, s, d, k, "phase-1b")
            written += 1

    # Phase 4 (fullstack)
    tag = PHASE4[2]
    engine_extra = {"hicache_gb_per_rank": 200}
    for jf in sorted((RESULTS / "phase-4").glob("c*.json")):
        emit(jf, tag, PHASE4[3], PHASE4[4], PHASE4[5], "phase-4",
             engine_extra={"extra_args": {"enable-hierarchical-cache": True, "hicache-size": 200,
                                          "reasoning-parser": "kimi_k2", "tool-call-parser": "kimi_k2",
                                          "trust-remote-code": True}})
        written += 1

    # Phase 5
    for variant_dir, (tag, s, d, k, variant) in PHASE5.items():
        base = RESULTS / "phase-5" / variant_dir
        if not base.exists():
            continue
        extra = {"extra_args": {"enable-hierarchical-cache": True, "hicache-size": 200,
                                "reasoning-parser": "kimi_k2", "tool-call-parser": "kimi_k2",
                                "trust-remote-code": True}}
        if variant == "no_cuda_graph":
            extra["extra_args"]["disable-cuda-graph"] = True
        if variant == "tp4_dp2":
            extra["tensor_parallel"] = 4
            extra["data_parallel"] = 2
        for jf in sorted(base.glob("c*.json")):
            emit(jf, tag, s, d, k, "phase-5", variant, engine_extra=extra)
            written += 1

    print(f"Wrote {written} standard-format artifacts to {OUT}")


if __name__ == "__main__":
    main()
