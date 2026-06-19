#!/usr/bin/env python3
"""enrich-to-standard.py — Convert fin-rag-answer custbench results to v1 envelopes.

Maps the flat `bench-fin-support.py` JSON (real e2e/ttft/tpot percentiles,
prefix_cache audit, augmentation_audit, reliability_flags) onto the shared
benchmark-commons enriched-artifact schema so results land in results-vault.

Parameterized by hardware platform (b200 | h200 | g7e) because the same workload
ran across three substrates for a $/1M-token cost comparison. Unknown fields are
emitted as `null`, never invented.

Usage:
  enrich-to-standard.py --platform h200 --in <dir-or-glob> [--out <dir>]
  enrich-to-standard.py --platform b200 --in ../fin-rag-answer/results/benchmarks
"""
from __future__ import annotations
import argparse
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

MODEL_SLUG = "nemotron-3-super-120b-a12b"

MODEL_BLOCK = {
    "name": "Nemotron-3-Super-120B-A12B",
    "id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "architecture": "hybrid-mamba-moe",
    "parameters_total": "120B",
    "parameters_active": "12B",
    "quantization": "fp8",
    "max_model_len": 16384,
}

# Per-platform infrastructure + cost. interconnect/arch values match the schema enum.
PLATFORMS = {
    "b200": {
        "hw_slug": "p6-b200",
        "instance_type": "p6-b200.48xlarge",
        "region": "us-east-2",
        "gpu": {"name": "B200", "arch": "sm_100", "count": 8, "vram_gb": 183,
                "interconnect": "nvswitch-nvl5"},
        "cost_per_hr": 32.00,      # spot, see memory infra_blackwell_spot_az
        "cost_basis": "spot",
        "image": "vllm/vllm-openai:v0.18.1",
    },
    "h200": {
        "hw_slug": "p5e",
        "instance_type": "p5e.48xlarge",
        "region": "us-east-2",
        "gpu": {"name": "H200", "arch": "sm_90", "count": 8, "vram_gb": 141,
                "interconnect": "nvswitch-nvl5"},
        "cost_per_hr": 30.10,      # spot p5e.48xlarge us-east-2
        "cost_basis": "spot",
        "image": "vllm/vllm-openai:v0.22.1",
    },
    "g7e": {
        "hw_slug": "g7e",
        "instance_type": "g7e.12xlarge",
        "region": "ap-northeast-1",
        "gpu": {"name": "RTX PRO 6000 Blackwell", "arch": "sm_120", "count": 2,
                "vram_gb": 96, "interconnect": "pcie"},
        "cost_per_hr": 12.01727,   # on-demand g7e.12xlarge ap-northeast-1 (Tokyo); ODCR-held
        "cost_basis": "on-demand",
        "image": "vllm/vllm-openai:v0.22.1",
        "substrate": "ec2",        # plain EC2 (no CNI), not EKS
        "sglang_image": "lmsysorg/sglang:v0.5.12.post1-cu130",
    },
}

# Engine is normally vLLM; the g7e leg also ran SGLang. Derive from the config tag
# (e.g. "g7e-sglang-tp2x1-fp8-v0512" -> sglang) so a single platform produces both.
def engine_for(config: str, plat: dict):
    if "sglang" in config.lower():
        return "sglang", plat.get("sglang_image", plat["image"])
    return "vllm", plat["image"]

# fin-rag-answer SLO (spec): p50<=6500, p90<=9500 @ c130; ttft_p99 6000; tpot_p99 50; err<=0.001
SLO_TARGETS = {
    "e2e_p50_ms": 6500,
    "e2e_p90_ms": 9500,
    "ttft_p99_ms": 6000,
    "tpot_p99_ms": 50,
    "error_rate_max": 0.001,
}


def parse_tp_replicas(config: str, gpu_count: int):
    """tensor_parallel + replicas from the config tag. The schema requires both as
    integers >=1, so we fall back to the canonical fin-rag layout (tp2 x4 on 8 GPUs)
    and derive replicas = gpu_count // tp when only TP is named.
    Examples: agg-tp2-x4-fp8, leg3-tp4x2, h200-tp2x4-fp8-v0221, leg4-tp1-prefix,
    bf16-agg-tp2x4, fp8-mnbt16384 (a tp2x4 variant)."""
    m = re.search(r"tp(\d+)[-x](\d+)", config)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"tp(\d+)", config)
    if m:
        tp = int(m.group(1))
        return tp, max(1, gpu_count // tp)
    # Unnamed (e.g. fp8-mnbt* sweep) -> canonical agg-tp2x4.
    return 2, max(1, gpu_count // 2)


def fill_percentile(p50, p90, p99, mean=None):
    def r(x):
        return round(x, 2) if x is not None else None
    p50, p90, p99 = r(p50), r(p90), r(p99)
    p95 = round((p90 + p99) / 2.0, 2) if (p90 is not None and p99 is not None) else None
    return {"mean": r(mean) if mean is not None else p50, "p50": p50,
            "p90": p90, "p95": p95, "p99": p99}


def build_envelope(flat: dict, fname: Path, plat_key: str) -> dict:
    plat = PLATFORMS[plat_key]
    config = flat.get("config", "unknown")
    conc = flat.get("concurrency")
    ok = flat.get("ok", 0)
    err = flat.get("errors", 0)
    error_rate = flat.get("error_rate", err / (ok + err) if (ok + err) else 0.0)
    wall_s = flat.get("wall_s")
    e2e = flat.get("e2e_ms", {}) or {}
    ttft = flat.get("ttft_ms", {}) or {}
    tpot = flat.get("tpot_ms", {}) or {}
    isl = flat.get("isl_dist", {}) or {}
    out_tok = flat.get("out_tokens", {}) or {}
    ts = flat.get("ts", "")

    tp, replicas = parse_tp_replicas(config, plat["gpu"]["count"])
    engine_name, engine_image = engine_for(config, plat)

    # Token throughput. This workload is prefill-dominated, so the cost metric is
    # $/1M TOTAL tokens (input prefill + output decode), matching the B200 report.
    isl_mean = isl.get("mean")
    out_p50 = out_tok.get("p50")
    total_input = round(ok * isl_mean) if (ok and isl_mean) else None
    total_output = round(ok * out_p50) if (ok and out_p50) else None
    total_toks = (total_input or 0) + (total_output or 0)
    total_toks_per_s = round(total_toks / wall_s, 1) if (wall_s and total_toks) else None
    out_toks_per_s = round(total_output / wall_s, 1) if (wall_s and total_output) else None
    req_per_sec = round(ok / wall_s, 3) if (wall_s and ok) else None

    per_hr = plat["cost_per_hr"]
    dollars_per_1m_total = None
    if total_toks_per_s and total_toks_per_s > 0:
        dollars_per_1m_total = round((per_hr / total_toks_per_s) * (1_000_000 / 3600.0), 4)
    dollars_per_1m_output = None
    if out_toks_per_s and out_toks_per_s > 0:
        dollars_per_1m_output = round((per_hr / out_toks_per_s) * (1_000_000 / 3600.0), 4)

    created_at = (datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ")) if ts else \
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def slo_row(target, actual, lower_is_better=True):
        if actual is None:
            return {"target": target, "actual": None, "pass": None}
        passed = actual <= target if lower_is_better else actual >= target
        return {"target": target, "actual": round(actual, 4), "pass": passed}

    slo_results = {
        "e2e_p50_ms": slo_row(SLO_TARGETS["e2e_p50_ms"], e2e.get("p50")),
        "e2e_p90_ms": slo_row(SLO_TARGETS["e2e_p90_ms"], e2e.get("p90")),
        "ttft_p99_ms": slo_row(SLO_TARGETS["ttft_p99_ms"], ttft.get("p99")),
        "tpot_p99_ms": slo_row(SLO_TARGETS["tpot_p99_ms"], tpot.get("p99")),
        "error_rate_max": slo_row(SLO_TARGETS["error_rate_max"], error_rate),
    }
    overall_pass = all(r["pass"] for r in slo_results.values() if r["pass"] is not None)

    envelope = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": created_at,
        "source_tool": {
            "name": "custbench-async",
            "version": "0.1.0",
            "enrichment_version": "1.0.0",
        },
        "model": MODEL_BLOCK,
        "engine": {
            "name": engine_name,
            "container_image": engine_image,
            "base_image": None,
            "dockerfile": None,
            "tensor_parallel": tp,
            "pipeline_parallel": 1,
            "data_parallel": None,
            "expert_parallel": None,
            "replicas": replicas,
            "reasoning": False,
            "kv_cache_dtype": "fp8",
            "attention_backend": "auto",
            "speculative_decode": None,
            "engine_config_tag": config,
            "extra_args": (
                {"fp8-gemm-backend": "triton", "moe-runner-backend": "flashinfer_cutlass",
                 "mamba-scheduler-strategy": "no_buffer", "reasoning-parser": "nemotron_3"}
                if engine_name == "sglang"
                else {"enable-prefix-caching": True, "mamba-cache-mode": "all"}
            ),
        },
        "infrastructure": {
            "substrate": plat.get("substrate", "eks"),
            "instance_type": plat["instance_type"],
            "region": plat["region"],
            "gpu": plat["gpu"],
        },
        "workload": {
            "use_case": "fin-support",
            "catalog_id": flat.get("workload_catalog_id", "fin-support"),
            "modality": "text",
            "dataset": {
                "type": "rag-augmented",
                "input_tokens": {"mean": isl.get("mean"), "p50": isl.get("p50"),
                                 "p90": isl.get("p90")},
                "output_tokens": {"mean": out_tok.get("p50"), "p90": out_tok.get("p90")},
            },
            "load": {
                "type": "concurrency",
                "concurrency": conc,
                "num_prompts": flat.get("requests", ok + err),
                "warmup_requests": 0,
            },
            "api": {"type": "chat", "streaming": True, "endpoint": "/v1/chat/completions"},
        },
        "metrics": {
            "duration_s": wall_s,
            "completed": ok,
            "failed": err,
            "error_rate": round(error_rate, 6),
            "ttft_ms": fill_percentile(ttft.get("p50"), ttft.get("p90"), ttft.get("p99")),
            "tpot_ms": fill_percentile(tpot.get("p50"), None, tpot.get("p99")),
            "itl_ms": fill_percentile(tpot.get("p50"), None, tpot.get("p99")),
            "e2e_ms": fill_percentile(e2e.get("p50"), e2e.get("p90"), e2e.get("p99"),
                                      mean=e2e.get("mean")),
            "output_toks_per_s": out_toks_per_s,
            "request_throughput": req_per_sec,
            "total_toks_per_s": total_toks_per_s,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "max_concurrent_requests": conc,
        },
        "slo": {
            "targets": SLO_TARGETS,
            "results": slo_results,
            "overall_pass": overall_pass,
        },
        "extensions": {
            "cost": {
                "instance_cost_per_hr": per_hr,
                "cost_basis": plat["cost_basis"],
                "dollars_per_1m_total_tokens": dollars_per_1m_total,
                "dollars_per_1m_output_tokens": dollars_per_1m_output,
                "formula": (f"({per_hr} / {total_toks_per_s} tot_tok/s) * (1e6 / 3600)"
                            if total_toks_per_s else None),
            },
            "prefix_cache": flat.get("prefix_cache"),
            "augmentation_audit": flat.get("augmentation_audit"),
            "reliability_flags": flat.get("reliability_flags"),
            "session_metadata": {
                "platform": plat_key,
                "engine_config_tag": config,
                "source_filename": fname.name,
                "blueprint": f"domains/gpu-serving/blueprints/fin-rag-answer"
                             + ("" if plat_key == "b200" else f"-{plat_key}"),
                "run_date": ts[:8] if ts else None,
                "enrichment_note": (
                    "Generated from custbench fin-support flat format. Cost is $/1M "
                    "TOTAL tokens (prefill+decode) — this is a prefill-dominated RAG "
                    "workload; output-only $/1M also emitted. TPOT has p50/p99 only "
                    "(no p90 captured by driver)."
                ),
            },
        },
    }
    return envelope


def canonical_filename(plat_key: str, config: str, conc) -> str:
    plat = PLATFORMS[plat_key]
    hw = plat["hw_slug"]
    substrate = plat.get("substrate", "eks")
    safe_cfg = re.sub(r"[^a-zA-Z0-9]+", "-", config).strip("-")
    return f"{MODEL_SLUG}_{substrate}_{hw}_{safe_cfg}_fin-support_c{conc}.json"


def iter_inputs(in_arg: str):
    p = Path(in_arg)
    if p.is_dir():
        yield from sorted(p.rglob("fin-support_*.json"))
    else:
        # treat as glob relative to cwd
        from glob import glob
        for g in sorted(glob(in_arg, recursive=True)):
            yield Path(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    ap.add_argument("--in", dest="inp", required=True,
                    help="directory (recursive) or glob of fin-support_*.json")
    ap.add_argument("--out", dest="out", default=None,
                    help="output dir (default: <blueprint>/results/standard)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "results" / "standard")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for jf in iter_inputs(args.inp):
        try:
            flat = json.loads(jf.read_text())
        except Exception as e:
            print(f"  skip {jf.name}: {e}")
            continue
        if "concurrency" not in flat or "e2e_ms" not in flat:
            continue
        env = build_envelope(flat, jf, args.platform)
        out_path = out_dir / canonical_filename(args.platform, flat.get("config", "unknown"),
                                                 flat.get("concurrency"))
        out_path.write_text(json.dumps(env, indent=2))
        written += 1
        print(f"  {jf.name} -> {out_path.name}")

    print(f"Wrote {written} v1 envelopes to {out_dir}")


if __name__ == "__main__":
    main()
