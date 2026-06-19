#!/usr/bin/env python3
"""enrich-to-standard.py — Convert qwen3-235b-b300 legacy sweep results to v1 envelopes.

Input:  results/concurrency_sweep_{tp4,tp2dp4}.json (each has a `results` array)
Output: results/standard/<model>_<substrate>_<hw>_<engine-config>_<workload>_c<N>.json
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "standard"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SLUG = "qwen3-235b-a22b-fp8"
SUBSTRATE = "ec2-spot"
HW = "p6-b300"
SPOT_PER_HR = 16.47  # from lessons.md frontmatter

MODEL_BLOCK = {
    "name": "Qwen3-235B-A22B-FP8",
    "id": "Qwen/Qwen3-235B-A22B-Instruct-FP8",
    "architecture": "moe",
    "parameters_total": "235B",
    "parameters_active": "22B",
    "quantization": "fp8",
    "max_model_len": 40960,  # per lessons.md L#5 — NOT 131072
}

INFRA_BLOCK = {
    "substrate": SUBSTRATE,
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

SLO_TARGETS = {
    "ttft_p99_ms": 500,
    "tpot_p99_ms": 30,
    "e2e_p99_ms": 15000,
    "error_rate_max": 0.001,
}


def engine_block(config_tag: str, tp: int, dp: int | None, ep: bool) -> dict:
    return {
        "name": "vllm",
        "container_image": "vllm/vllm-openai:v0.19.1",
        "base_image": None,
        "dockerfile": None,
        "tensor_parallel": tp,
        "pipeline_parallel": 1,
        "data_parallel": dp,
        "expert_parallel": ep,
        "replicas": 1,
        "reasoning": False,
        "kv_cache_dtype": "auto",
        "attention_backend": "flashinfer",
        "speculative_decode": None,
        "engine_config_tag": config_tag,
        "extra_args": {
            "trust-remote-code": True,
            "tool-call-parser": "hermes",
            "reasoning-parser": "deepseek_r1",
            "enable-prefix-caching": True,
        },
    }


def perc(p50, p95, p99):
    # We only have p50/p95/p99 from legacy runs; p90 unknown, p99 known; mean unknown
    return {
        "mean": None,
        "p50": round(p50, 2) if p50 is not None else None,
        "p90": None,  # not captured in legacy format
        "p95": round(p95, 2) if p95 is not None else None,
        "p99": round(p99, 2) if p99 is not None else None,
    }


def fname(tag: str, conc: int) -> str:
    return f"{MODEL_SLUG}_{SUBSTRATE}_{HW}_{tag}_concurrency-sweep_c{conc}.json"


def build_envelope(result: dict, session: dict, tag: str, tp: int, dp: int | None, ep: bool, created_at: str) -> dict:
    c = result["concurrency"]
    ok = result.get("ok", 0)
    err = result.get("fail", 0)
    total_tokens = result.get("total_output_tokens", 0)
    duration = result.get("wall_time_s")
    agg = result.get("aggregate_tps")
    per_req = result.get("avg_tps_per_request")
    error_rate = err / (ok + err) if (ok + err) > 0 else 0.0

    dollars_per_1m = None
    if agg and agg > 0:
        dollars_per_1m = round((SPOT_PER_HR / agg) * (1_000_000 / 3600.0), 2)

    ttft = perc(result.get("ttft_p50_ms"), result.get("ttft_p95_ms"), result.get("ttft_p99_ms"))
    itl = perc(result.get("itl_p50_ms"), result.get("itl_p95_ms"), result.get("itl_p99_ms"))

    def slo(tgt, actual):
        return {"target": tgt, "actual": actual, "pass": None if actual is None else actual <= tgt}

    env = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": created_at,
        "source_tool": {"name": "custbench-legacy", "version": "0.1.0", "enrichment_version": "1.0.0"},
        "model": MODEL_BLOCK,
        "engine": engine_block(tag, tp, dp, ep),
        "framework": {"name": "vllm", "version": "v0.19.1", "config": {"mode": "aggregated"}},
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "concurrency-sweep",
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": session.get("input_tokens", 2048), "std_dev": 0},
                "output_tokens": {"mean": session.get("output_tokens", 512), "std_dev": 0},
            },
            "load": {
                "type": "concurrency",
                "concurrency": c,
                "num_prompts": ok + err,
                "warmup_requests": 0,
            },
            "api": {"type": "chat", "streaming": True, "endpoint": "/v1/chat/completions"},
        },
        "metrics": {
            "duration_s": duration,
            "completed": ok,
            "failed": err,
            "error_rate": round(error_rate, 6),
            "ttft_ms": ttft,
            "tpot_ms": perc(
                1000.0 / per_req if per_req and per_req > 0 else None,
                None, None,
            ),
            "itl_ms": itl,
            "e2e_ms": {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None,
                       "note": "E2E not captured in legacy custbench output"},
            "output_toks_per_s": agg,
            "request_throughput": round(ok / duration, 3) if duration and duration > 0 else None,
            "total_toks_per_s": agg,
            "total_input_tokens": ok * session.get("input_tokens", 2048),
            "total_output_tokens": total_tokens,
            "max_concurrent_requests": c,
        },
        "slo": {
            "targets": SLO_TARGETS,
            "results": {
                "ttft_p99_ms": slo(500, ttft["p99"]),
                "tpot_p99_ms": slo(30, round(1000.0 / per_req, 2) if per_req and per_req > 0 else None),
                "e2e_p99_ms": slo(15000, None),
                "error_rate_max": slo(0.001, round(error_rate, 6)),
            },
            "overall_pass": None,
        },
        "extensions": {
            "cost": {
                "instance_cost_per_hr": SPOT_PER_HR,
                "dollars_per_1m_output_tokens": dollars_per_1m,
                "formula": f"({SPOT_PER_HR} / {agg}) * (1000000 / 3600)" if agg else None,
            },
            "session_metadata": {
                "engine_config_tag": tag,
                "blueprint": "domains/gpu-serving/blueprints/qwen3-235b-b300",
                "phase": "baseline",
                "run_date": created_at[:10],
                "enrichment_note": "Legacy custbench format. p90 percentiles and E2E not captured. TPOT derived as 1000/avg_tps_per_request.",
            },
        },
    }
    # overall pass using what we have
    results = env["slo"]["results"]
    passes = [r["pass"] for r in results.values() if r["pass"] is not None]
    env["slo"]["overall_pass"] = all(passes) if passes else None
    return env


def main():
    written = 0
    for src_name, (tag, tp, dp, ep) in {
        "concurrency_sweep_tp4.json": ("vllm-tp4-prefix", 4, None, False),
        "concurrency_sweep_tp2dp4.json": ("vllm-tp2-dp4-ep", 2, 4, True),
    }.items():
        src = RESULTS / src_name
        if not src.exists():
            print(f"skip {src_name}")
            continue
        payload = json.loads(src.read_text())
        ts = payload.get("timestamp", "20260422-120000")
        try:
            # Format: '20260422-214939' → ISO
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
        created_at = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        session = {
            "input_tokens": payload.get("input_tokens", 2048),
            "output_tokens": payload.get("output_tokens", 512),
        }

        for r in payload.get("results", []):
            env = build_envelope(r, session, tag, tp, dp, ep, created_at)
            out = OUT / fname(tag, r["concurrency"])
            out.write_text(json.dumps(env, indent=2))
            written += 1

    print(f"Wrote {written} v1 envelopes to {OUT}")


if __name__ == "__main__":
    main()
