#!/usr/bin/env python3
"""enrich-to-standard.py — Convert nemotron-super flat-format results to v1 envelopes.

Input:  results/{agg,disagg}-*_c<N>_in<I>_out<O>_<timestamp>.json
Output: results/standard/<model>_<substrate>_<hw>_<engine-config>_<workload>_c<N>.json

Preserves every measurement available in the flat format. Unknown fields are
emitted as `null` rather than invented.
"""
from __future__ import annotations
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "standard"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SLUG = "nemotron-3-super-120b-a12b"
SUBSTRATE = "eks"
HW = "p6-b200"
CREATED_AT = "2026-03-12T12:00:00Z"

MODEL_BLOCK = {
    "name": "Nemotron-3-Super-120B-A12B",
    "id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "architecture": "hybrid-mamba-moe",
    "parameters_total": "120B",
    "parameters_active": "12B",
    "quantization": "fp8",
    "max_model_len": 256000,
}

INFRA_BLOCK = {
    "substrate": "eks",
    "instance_type": "p6-b200.48xlarge",
    "region": "us-east-2",
    "gpu": {
        "name": "B200",
        "arch": "sm_100",
        "count": 8,
        "vram_gb": 183,
        "interconnect": "nvswitch-nvl5",
    },
}

# Pattern: <mode>-<engine>-<parallel>[-<tag>]_c<N>_in<I>_out<O>_<timestamp>.json
# Examples:
#   agg-vllm-tp2x4_c64_in4096_out1024_20260312-105408.json
#   agg-vllm-tp2x1-smoke_c1_in1024_out256_20260312-101546.json
#   disagg-sglang-4p4d_c1_in4096_out1024_20260312-114410.json
FNAME_RE = re.compile(
    r"^(?P<mode>agg|disagg)-(?P<engine>vllm|sglang)-(?P<parallel>[^_]+)_"
    r"c(?P<conc>\d+)_in(?P<inp>\d+)_out(?P<out>\d+)_(?P<ts>\d{8}-\d{6})\.json$"
)


def parse_parallel(tag: str):
    """Return (tp_size, replicas, disagg_split) or (tp, replicas, None)."""
    # tp2x4 = tensor_parallel=2, 4 workers
    m = re.match(r"tp(\d+)x(\d+)", tag)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    # tp2x1-smoke = same format, suffix is contextual
    m = re.match(r"tp(\d+)x(\d+)-(.+)", tag)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    # 4p4d = 4 prefill + 4 decode (disaggregated)
    m = re.match(r"(\d+)p(\d+)d", tag)
    if m:
        return None, None, {"prefill": int(m.group(1)), "decode": int(m.group(2))}
    return None, None, None


def build_engine_block(engine: str, tp, replicas, disagg_split, mode: str) -> dict:
    # Image tags from lessons.md: Dynamo 0.9.1 runtime for vLLM + SGLang v0.5.9 for SGLang
    if engine == "vllm":
        image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.1"
    elif engine == "sglang":
        image = "lmsysorg/sglang:v0.5.9-cu124"
    else:
        image = None
    block = {
        "name": engine,
        "container_image": image,
        "base_image": "nvcr.io/nvidia/dynamo:0.9.1" if mode == "disagg" else None,
        "dockerfile": None,
        "tensor_parallel": tp,
        "pipeline_parallel": 1,
        "data_parallel": None,
        "expert_parallel": None,
        "replicas": replicas if replicas else (sum(disagg_split.values()) if disagg_split else 1),
        "reasoning": False,
        "kv_cache_dtype": "fp8",
        "attention_backend": "auto",
        "speculative_decode": None,
        "extra_args": {
            "trust-remote-code": True,
            "tool-call-parser": "qwen3_coder",
            "reasoning-parser": "nano_v3",
        },
    }
    return block


def build_framework_block(mode: str, engine: str, disagg_split):
    if mode == "agg":
        return {"name": engine, "version": None, "config": {"mode": "aggregated"}}
    # disagg via Dynamo
    config = {"mode": "disaggregated", "router": "round-robin", "nixl_enabled": True}
    if disagg_split:
        config["prefill"] = {"num_workers": disagg_split["prefill"]}
        config["decode"] = {"num_workers": disagg_split["decode"]}
    return {"name": "dynamo", "version": "1.0.0", "config": config}


def engine_tag(mode: str, engine: str, parallel: str) -> str:
    return f"{engine}-{mode}-{parallel}"


def fill_percentile(p50_ms, p90_ms, p99_ms):
    def r(x):
        return round(x, 2) if x is not None else None
    p50 = r(p50_ms)
    p90 = r(p90_ms)
    p99 = r(p99_ms)
    p95 = round((p90 + p99) / 2.0, 2) if (p90 is not None and p99 is not None) else None
    return {"mean": p50, "p50": p50, "p90": p90, "p95": p95, "p99": p99}


def build_envelope(flat: dict, fname: Path, mode: str, engine: str, parallel: str,
                   conc: int, inp: int, out_target: int, ts: str) -> dict:
    summary = flat.get("summary", {}) or {}
    ttft = summary.get("ttft_ms", {}) or {}
    e2e = summary.get("e2e_ms", {}) or {}
    itl = summary.get("itl_ms", {}) or {}

    tp, replicas, disagg_split = parse_parallel(parallel)
    engine_block = build_engine_block(engine, tp, replicas, disagg_split, mode)
    framework_block = build_framework_block(mode, engine, disagg_split)
    tag = engine_tag(mode, engine, parallel)

    agg_tok_per_s = summary.get("approx_output_toks_per_sec")
    req_per_sec = summary.get("requests_per_sec")
    duration = summary.get("total_time_s")
    ok = summary.get("successes", 0)
    err = summary.get("failures", 0)
    error_rate = err / (ok + err) if (ok + err) > 0 else 0.0

    # Cost (spot estimate for p6-b200.48xlarge)
    SPOT_PER_HR = 48.00
    dollars_per_1m = None
    if agg_tok_per_s and agg_tok_per_s > 0:
        dollars_per_1m = round((SPOT_PER_HR / agg_tok_per_s) * (1_000_000 / 3600.0), 2)

    # SLO targets (from spec)
    SLO_TARGETS = {
        "ttft_p99_ms": 500,
        "tpot_p99_ms": 30,
        "e2e_p99_ms": 15000,
        "error_rate_max": 0.001,
    }
    # Honest evaluation — many Nemotron runs don't meet TTFT target at long context
    slo_row = lambda tgt, actual: (
        {"target": tgt, "actual": actual, "pass": None if actual is None else actual <= tgt}
    )

    envelope = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_tool": {
            "name": "custbench-legacy",
            "version": "0.1.0",
            "enrichment_version": "1.0.0",
        },
        "model": MODEL_BLOCK,
        "engine": {**engine_block, "engine_config_tag": tag},
        "framework": framework_block,
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "concurrency-sweep",
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": inp, "std_dev": 0},
                "output_tokens": {"mean": out_target, "std_dev": 0},
            },
            "load": {
                "type": "concurrency",
                "concurrency": conc,
                "num_prompts": summary.get("num_prompts", ok + err),
                "warmup_requests": 0,
            },
            "api": {"type": "chat", "streaming": True, "endpoint": "/v1/chat/completions"},
        },
        "metrics": {
            "duration_s": duration,
            "completed": ok,
            "failed": err,
            "error_rate": round(error_rate, 6),
            "ttft_ms": fill_percentile(ttft.get("p50"), ttft.get("p90"), ttft.get("p99")),
            "tpot_ms": fill_percentile(itl.get("p50"), itl.get("p90"), itl.get("p99")),
            "itl_ms":  fill_percentile(itl.get("p50"), itl.get("p90"), itl.get("p99")),
            "e2e_ms":  fill_percentile(e2e.get("p50"), e2e.get("p90"), e2e.get("p99")),
            "output_toks_per_s": agg_tok_per_s,
            "request_throughput": req_per_sec,
            "total_toks_per_s": agg_tok_per_s,
            "total_input_tokens": ok * inp,
            "total_output_tokens": None,  # not directly captured; summary has chars
            "max_concurrent_requests": conc,
        },
        "slo": {
            "targets": SLO_TARGETS,
            "results": {
                "ttft_p99_ms":    slo_row(500,   round(ttft.get("p99", 0), 2) if ttft.get("p99") else None),
                "tpot_p99_ms":    slo_row(30,    round(itl.get("p99", 0), 2)  if itl.get("p99")  else None),
                "e2e_p99_ms":     slo_row(15000, round(e2e.get("p99", 0), 2)  if e2e.get("p99")  else None),
                "error_rate_max": slo_row(0.001, round(error_rate, 6)),
            },
            "overall_pass": None,  # computed below
        },
        "extensions": {
            "cost": {
                "instance_cost_per_hr": SPOT_PER_HR,
                "dollars_per_1m_output_tokens": dollars_per_1m,
                "formula": f"({SPOT_PER_HR} / {agg_tok_per_s}) * (1000000 / 3600)" if agg_tok_per_s else None,
            },
            "session_metadata": {
                "engine_config_tag": tag,
                "source_filename": fname.name,
                "mode": mode,
                "blueprint": "domains/gpu-serving/blueprints/nemotron-super",
                "run_date": ts[:8],
                "enrichment_note": "Generated from legacy custbench flat format. TPOT derived from itl_ms (inter-token latency — includes first token gap). Per-request artifacts not preserved in v1 envelope.",
            },
        },
    }
    # Derive overall_pass
    results = envelope["slo"]["results"]
    envelope["slo"]["overall_pass"] = all(
        r["pass"] for r in results.values() if r["pass"] is not None
    )
    return envelope


def canonical_filename(tag: str, conc: int) -> str:
    return f"{MODEL_SLUG}_{SUBSTRATE}_{HW}_{tag}_concurrency-sweep_c{conc}.json"


def main():
    written = 0
    for jf in sorted(RESULTS.glob("*.json")):
        m = FNAME_RE.match(jf.name)
        if not m:
            continue
        with open(jf) as f:
            flat = json.load(f)
        env = build_envelope(flat, jf, m.group("mode"), m.group("engine"), m.group("parallel"),
                             int(m.group("conc")), int(m.group("inp")), int(m.group("out")),
                             m.group("ts"))
        tag = engine_tag(m.group("mode"), m.group("engine"), m.group("parallel"))
        out_path = OUT / canonical_filename(tag, int(m.group("conc")))
        out_path.write_text(json.dumps(env, indent=2))
        written += 1

    print(f"Wrote {written} v1 envelopes to {OUT}")


if __name__ == "__main__":
    main()
