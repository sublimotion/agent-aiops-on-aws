#!/usr/bin/env python3
"""Enrich qwen3-235b-speculative raw bench outputs to v1 envelopes.

Mirrors the kimi-k2.6-speculative enricher pattern — one v1 envelope per
(config, concurrency) tuple. Emitted to results/standard/.
"""
from __future__ import annotations
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
OUT = RESULTS / "standard"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SLUG = "qwen3-235b-a22b-fp8"
SUBSTRATE = "eks"
HW = "p6-b300"
SPOT_PER_HR = 25.47
CREATED_AT = "2026-05-14T18:00:00Z"

MODEL_BLOCK = {
    "name": "Qwen3-235B-A22B-FP8",
    "id": "Qwen/Qwen3-235B-A22B-FP8",
    "architecture": "moe",
    "parameters_total": "235B",
    "parameters_active": "22B",
    "quantization": "fp8",
    "max_model_len": 40960,
}

DRAFT_BLOCK = {
    "name": "Qwen3-235B-A22B-EAGLE3",
    "id": "lmsys/Qwen3-235B-A22B-EAGLE3",
    "architecture": "eagle3",
    "parameters_total": "1B",
    "quantization": "bf16",
}

INFRA_BLOCK = {
    "substrate": SUBSTRATE,
    "instance_type": "p6-b300.48xlarge",
    "region": "us-west-2",
    "gpu": {
        "name": "B300",
        "arch": "sm_103",
        "count": 8,
        "vram_gb": 275,
        "interconnect": "nvswitch-nv18",
    },
}

SLO_TARGETS = {
    "ttft_p99_ms": 500,
    "tpot_p99_ms": 30,
    "e2e_p99_ms": 15000,
    "error_rate_max": 0.001,
}


def engine_block(tag, num_steps, num_draft, topk, tp=4, dp=None, extra_args=None):
    block = {
        "name": "sglang",
        "container_image": "lmsysorg/sglang:v0.5.10-cu130",
        "base_image": "lmsysorg/sglang:v0.5.10-cu130",
        "dockerfile": None,
        "tensor_parallel": tp,
        "pipeline_parallel": 1,
        "data_parallel": dp,
        "expert_parallel": None,
        "replicas": 1,
        "reasoning": True,
        "kv_cache_dtype": "auto",
        "attention_backend": "trtllm_mha",
        "speculative_decode": {
            "algorithm": "EAGLE3",
            "draft_model": DRAFT_BLOCK["id"],
            "num_steps": num_steps,
            "num_draft_tokens": num_draft,
            "eagle_topk": topk,
            "draft_attention_backend": "trtllm_mha",
        },
        "engine_config_tag": tag,
        "extra_args": {
            "tool-call-parser": "qwen3_coder",
            "reasoning-parser": "qwen3-thinking",
            "trust-remote-code": True,
            "context-length": 40960,
            "mem-fraction-static": 0.90,
            "enable-metrics": True,
            **(extra_args or {}),
        },
    }
    return block


def perc(p50, p90, p99):
    """Our bench has only p50/p90/p99 seconds. Convert to ms. p95 = mid(p90, p99)."""
    def m(x):
        return round(x * 1000, 2) if x is not None else None
    p50m, p90m, p99m = m(p50), m(p90), m(p99)
    p95m = round((p90m + p99m) / 2, 2) if p90m is not None and p99m is not None else None
    return {"mean": p50m, "p50": p50m, "p90": p90m, "p95": p95m, "p99": p99m}


def build_envelope(raw: dict, tag: str, num_steps: int, num_draft: int, topk: int,
                   phase: str, tp=4, dp=None, extra_args=None) -> dict:
    c = raw["concurrency"]
    per_req = raw.get("per_req_tok_per_s")
    agg = raw.get("agg_tok_per_s", 0)
    duration = raw.get("duration_s")
    ok = raw.get("requests_ok", 0)
    err = raw.get("requests_err", 0)
    total_tokens = raw.get("total_tokens", 0)
    accept_rate = raw.get("spec_accept_rate_mean")
    accept_len = raw.get("spec_accept_length_mean")

    error_rate = err / (ok + err) if (ok + err) > 0 else 0.0

    e2e_ms = perc(raw.get("p50_latency_s"), raw.get("p90_latency_s"), raw.get("p99_latency_s"))
    # TPOT from per_req_tok_per_s
    tpot = 1000 / per_req if per_req and per_req > 0 else None
    tpot_ms = {"mean": round(tpot, 2) if tpot else None,
               "p50": round(tpot, 2) if tpot else None,
               "p90": None, "p95": None, "p99": None}
    # TTFT not captured by bench-one.py directly — leave null with honest note
    ttft_ms = {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}

    dollars_per_1m = None
    if agg and agg > 0:
        dollars_per_1m = round((SPOT_PER_HR / agg) * (1_000_000 / 3600), 2)

    engine = engine_block(tag, num_steps, num_draft, topk, tp, dp, extra_args)

    slo_row = lambda tgt, actual: {"target": tgt, "actual": actual,
                                   "pass": None if actual is None else actual <= tgt}

    env = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": CREATED_AT,
        "source_tool": {"name": "custbench-async", "version": "0.2.0", "enrichment_version": "1.0.0"},
        "model": MODEL_BLOCK,
        "engine": engine,
        "framework": {"name": "sglang-native", "version": "v0.5.10-cu130", "config": {}},
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "concurrency-sweep",
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": raw.get("input_len", 512), "std_dev": 0},
                "output_tokens": {"mean": raw.get("output_len_target", 256), "std_dev": 0},
            },
            "load": {"type": "concurrency", "concurrency": c, "num_prompts": ok + err, "warmup_requests": 0},
            "api": {"type": "completions", "streaming": False, "endpoint": "/generate"},
        },
        "metrics": {
            "duration_s": duration,
            "completed": ok,
            "failed": err,
            "error_rate": round(error_rate, 6),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "itl_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "output_toks_per_s": agg,
            "request_throughput": round(ok / duration, 3) if duration and duration > 0 else None,
            "total_toks_per_s": agg,
            "total_input_tokens": ok * raw.get("input_len", 512),
            "total_output_tokens": total_tokens,
            "max_concurrent_requests": c,
        },
        "slo": {
            "targets": SLO_TARGETS,
            "results": {
                "ttft_p99_ms": slo_row(SLO_TARGETS["ttft_p99_ms"], None),
                "tpot_p99_ms": slo_row(SLO_TARGETS["tpot_p99_ms"], tpot_ms["p50"]),
                "e2e_p99_ms":  slo_row(SLO_TARGETS["e2e_p99_ms"],  e2e_ms["p99"]),
                "error_rate_max": slo_row(SLO_TARGETS["error_rate_max"], round(error_rate, 6)),
            },
            "overall_pass": None,
        },
        "extensions": {
            "speculative_decode_stats": {
                "accept_rate_mean": accept_rate,
                "accept_length_mean": accept_len,
                "effective_tokens_per_step": round((accept_rate or 0) * (accept_len or 0), 3)
                    if accept_rate is not None and accept_len is not None else None,
            },
            "cost": {
                "instance_cost_per_hr": SPOT_PER_HR,
                "dollars_per_1m_output_tokens": dollars_per_1m,
                "formula": f"({SPOT_PER_HR} / {agg}) * (1000000 / 3600)",
            },
            "session_metadata": {
                "phase": phase,
                "engine_config_tag": tag,
                "blueprint": "domains/gpu-serving/blueprints/qwen3-235b-speculative",
                "run_date": "2026-05-14",
                "enrichment_note": "TTFT not captured by custbench-async non-streaming client. Prometheus scrape available but not yet queried — future enrichment pass should populate ttft_ms from sglang:time_to_first_token_seconds_bucket.",
            },
        },
    }
    passes = [r["pass"] for r in env["slo"]["results"].values() if r["pass"] is not None]
    env["slo"]["overall_pass"] = all(passes) if passes else None
    return env


def fname(tag: str, c: int) -> str:
    return f"{MODEL_SLUG}_{SUBSTRATE}_{HW}_{tag}_concurrency-sweep_c{c}.json"


def main():
    written = 0

    # Phase 1 defaults
    for jf in sorted((RAW / "phase-1").glob("c*.json")):
        raw = json.loads(jf.read_text())
        env = build_envelope(raw, "sglang-eagle3-s3d4k1-default", 3, 4, 1, "phase-1")
        (OUT / fname("sglang-eagle3-s3d4k1-default", raw["concurrency"])).write_text(json.dumps(env, indent=2))
        written += 1

    # Phase 1b sweep
    for cfg_dir in sorted((RAW / "phase-1b").glob("*")):
        if not cfg_dir.is_dir(): continue
        m = re.match(r"s(\d+)d(\d+)k(\d+)", cfg_dir.name)
        if not m: continue
        s, d, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
        tag = f"sglang-eagle3-s{s}d{d}k{k}"
        for jf in sorted(cfg_dir.glob("c*.json")):
            raw = json.loads(jf.read_text())
            env = build_envelope(raw, tag, s, d, k, "phase-1b")
            (OUT / fname(tag, raw["concurrency"])).write_text(json.dumps(env, indent=2))
            written += 1

    # Phase 4 fullstack (s4d4k1 + HiCache)
    for jf in sorted((RAW / "phase-4").glob("c*.json")):
        raw = json.loads(jf.read_text())
        env = build_envelope(raw, "sglang-eagle3-s4d4k1-hicache200", 4, 4, 1, "phase-4",
                             extra_args={"enable-hierarchical-cache": True, "hicache-size": 200})
        (OUT / fname("sglang-eagle3-s4d4k1-hicache200", raw["concurrency"])).write_text(json.dumps(env, indent=2))
        written += 1

    # Phase 5
    p5 = RAW / "phase-5"
    for variant_dir in sorted(p5.glob("*")):
        if not variant_dir.is_dir(): continue
        variant = variant_dir.name
        if variant == "5a-default-stack":
            tag = "sglang-eagle3-s4d4-hicache-cudagraphs"
            extra = {"enable-hierarchical-cache": True, "hicache-size": 200}
            tp, dp = 4, None
        elif variant == "5b-no-cuda-graph":
            tag = "sglang-eagle3-s4d4-no-cudagraphs"
            extra = {"enable-hierarchical-cache": True, "hicache-size": 200, "disable-cuda-graph": True}
            tp, dp = 4, None
        elif variant == "5c-tp2-dp2":
            tag = "sglang-eagle3-s4d4-tp2dp2-hicache"
            extra = {"enable-hierarchical-cache": True, "hicache-size": 200}
            tp, dp = 2, 2
        else:
            continue
        for jf in sorted(variant_dir.glob("c*.json")):
            raw = json.loads(jf.read_text())
            env = build_envelope(raw, tag, 4, 4, 1, "phase-5", tp=tp, dp=dp, extra_args=extra)
            (OUT / fname(tag, raw["concurrency"])).write_text(json.dumps(env, indent=2))
            written += 1

    print(f"Wrote {written} v1 envelopes to {OUT}")


if __name__ == "__main__":
    main()
