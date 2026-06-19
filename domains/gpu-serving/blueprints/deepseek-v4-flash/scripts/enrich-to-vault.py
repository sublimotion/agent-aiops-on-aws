#!/usr/bin/env python3
"""Enrich raw `vllm bench serve` JSONs in r2/r3/T0 → v1 envelopes under results/standard/."""
from __future__ import annotations
import json, re, uuid
from pathlib import Path
from datetime import datetime, timezone

BP = Path(__file__).resolve().parent.parent
RAW_DIRS = [BP / "results" / d for d in ("T0", "r2", "r3")]
OUT = BP / "results" / "standard"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCE_HOURLY = 26.49  # p6-b300 spot us-west-2

# session metadata per directory: (session label, has-MTP-spec, container, extra args)
SESSIONS = {
    "T0": dict(tag="vllm-tp8-baseline", spec=False,
               img="vllm/vllm-openai:nightly-pr42320", date="2026-05-17"),
    "r2": dict(tag="vllm-tp8-baseline", spec=False,
               img="vllm/vllm-openai:nightly-pr42320", date="2026-05-18"),
    "r3": dict(tag="vllm-tp8-mtp",      spec=True,
               img="vllm/vllm-openai:nightly-pr42320", date="2026-05-19"),
}

QPS_RE  = re.compile(r"qps([0-9.]+)", re.I)
CTX_RE  = re.compile(r"ctx(\d+[kK]?)", re.I)
GSP_RE  = re.compile(r"gsp_(\d+[kK])", re.I)
PFX_RE  = re.compile(r"prefix_(\d+[kK])", re.I)
SINGLE_RE = re.compile(r"single_(\d+[kK])", re.I)

def parse_size(s: str) -> int:
    s = s.lower().rstrip()
    if s.endswith("k"): return int(float(s[:-1]) * 1024)
    return int(s)

def latency_block(raw, prefix):
    return {
        "mean":  raw.get(f"mean_{prefix}_ms"),
        "p50":   raw.get(f"median_{prefix}_ms"),
        "p90":   None,
        "p95":   None,
        "p99":   raw.get(f"p99_{prefix}_ms"),
    }

def derive_workload(stem: str, raw: dict):
    """Return (catalog_id, concurrency_or_None, load_dict)."""
    lower = stem.lower()
    qps = QPS_RE.search(lower)
    ctx = CTX_RE.search(lower) or GSP_RE.search(lower)
    pfx = PFX_RE.search(lower)
    single = SINGLE_RE.search(lower)

    # Map to canonical catalog_ids from standards/benchmark-commons/workloads/
    if "smoke" in lower:
        return "burn-in", None, {"type": "smoke", "num_prompts": raw.get("num_prompts", 0)}
    if single:
        return "rag-1m-context", 1, {"type": "concurrency", "concurrency": 1, "num_prompts": raw.get("num_prompts", 1)}
    if pfx:
        return "shared-prefix-multitenant", None, {"type": "shared-prefix", "prefix_tokens": parse_size(pfx.group(1))}
    if ctx and not qps:
        return "rag-long-context", None, {"type": "constant", "context_tokens": parse_size(ctx.group(1)),
                                          "num_prompts": raw.get("num_prompts", 0)}
    if qps:
        return "qps-sweep", None, {"type": "constant", "request_rate": float(qps.group(1)),
                                    "num_prompts": raw.get("num_prompts", 0)}
    if "sharegpt" in lower:
        return "sharegpt-production-mix", None, {"type": "constant", "num_prompts": raw.get("num_prompts", 0)}
    return "qps-sweep", None, {"type": "constant", "num_prompts": raw.get("num_prompts", 0)}

def filename_for(catalog_id: str, conc, qps, session: str, src_stem: str) -> str:
    sess = SESSIONS[session]
    # use full source stem as disambiguator to avoid collisions across sessions/runs
    slug = src_stem.replace("_", "-")
    parts = ["deepseek-v4-flash", "ec2-spot", "p6-b300", sess["tag"], catalog_id, slug]
    return "_".join(parts) + ".json"

def build_envelope(raw: dict, src: Path, session: str) -> dict:
    sess = SESSIONS[session]
    catalog_id, conc, load = derive_workload(src.stem, raw)
    qps = load.get("request_rate")

    # cost: $/M output tokens = ($/hr * 1e6) / (out_tok/s * 3600)
    out_tps = raw.get("output_throughput") or 0
    dpm = (INSTANCE_HOURLY * 1e6) / (out_tps * 3600) if out_tps else None

    metrics = {
        "duration_s": raw.get("duration"),
        "completed":  raw.get("completed", 0),
        "failed":     raw.get("failed", 0),
        "error_rate": (raw.get("failed", 0) / max(raw.get("completed", 0) + raw.get("failed", 0), 1)),
        "ttft_ms":    latency_block(raw, "ttft"),
        "tpot_ms":    latency_block(raw, "tpot"),
        "itl_ms":     latency_block(raw, "itl"),
        "e2e_ms":     latency_block(raw, "e2el"),
        "output_toks_per_s":  raw.get("output_throughput"),
        "request_throughput": raw.get("request_throughput"),
        "total_toks_per_s":   raw.get("total_token_throughput"),
        "total_input_tokens":  raw.get("total_input_tokens"),
        "total_output_tokens": raw.get("total_output_tokens"),
        "max_concurrent_requests": raw.get("max_concurrent_requests"),
    }
    if sess["spec"]:
        metrics["spec_accept_rate"]   = raw.get("spec_decode_acceptance_rate")
        metrics["spec_accept_length"] = raw.get("spec_decode_acceptance_length")

    spec_block = None
    if sess["spec"]:
        spec_block = {
            "algorithm": "MTP",
            "draft_model": "self",
            "num_steps": 1,
            "num_draft_tokens": 1,
        }

    env = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_tool": {"name": "vllm-bench-serve", "version": "0.21.0",
                        "enrichment_version": "1.0.0-deepseek-v4-flash"},
        "model": {
            "name": "DeepSeek-V4-Flash",
            "id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture": "moe",
            "parameters_total": "131B",
            "parameters_active": "5.5B",
            "quantization": "fp8",
            "max_model_len": 524288,
        },
        "engine": {
            "name": "vllm",
            "container_image": sess["img"],
            "base_image": sess["img"],
            "dockerfile": None,
            "tensor_parallel": 8,
            "pipeline_parallel": 1,
            "data_parallel": None,
            "expert_parallel": None,
            "replicas": 1,
            "reasoning": False,
            "kv_cache_dtype": "fp8",
            "attention_backend": "flashinfer",
            "speculative_decode": spec_block,
            "engine_config_tag": sess["tag"],
            "extra_args": {"kv-cache-dtype": "fp8", "trust-remote-code": True},
        },
        "framework": {"name": "vllm-native", "version": "0.21.0", "config": {}},
        "infrastructure": {
            "substrate": "ec2-spot",
            "instance_type": "p6-b300.48xlarge",
            "region": "us-west-2",
            "gpu": {"name": "B300", "arch": "sm_103", "count": 8,
                    "vram_gb": 275, "interconnect": "nvswitch-nv18"},
        },
        "workload": {
            "use_case": catalog_id,
            "catalog_id": catalog_id,
            "modality": "text",
            "dataset": {
                "type": "synthetic" if "sharegpt" not in src.stem.lower() else "sharegpt",
                "input_tokens": {"mean": (raw.get("total_input_tokens") or 0) // max(raw.get("completed") or 1, 1),
                                 "std_dev": 0},
                "output_tokens": {"mean": (raw.get("total_output_tokens") or 0) // max(raw.get("completed") or 1, 1),
                                  "std_dev": 0},
            },
            "load": load,
            "api": {"type": "completions", "streaming": True, "endpoint": "/v1/completions"},
        },
        "metrics": metrics,
        "extensions": {
            "normalized_from": str(src.relative_to(BP)),
            "session_date": sess["date"],
            "cost": {
                "instance_cost_per_hr": INSTANCE_HOURLY,
                "dollars_per_1m_output_tokens": dpm,
            },
        },
    }
    return env, filename_for(catalog_id, conc, qps, session, src.stem)

def main():
    written = 0
    for sess_dir in RAW_DIRS:
        if not sess_dir.is_dir(): continue
        session = sess_dir.name
        for src in sorted(sess_dir.glob("*.json")):
            try:
                raw = json.load(open(src))
            except Exception as e:
                print(f"  skip parse-error {src}: {e}"); continue
            if "output_throughput" not in raw:
                print(f"  skip non-bench {src.name}"); continue
            env, fname = build_envelope(raw, src, session)
            out = OUT / fname
            json.dump(env, open(out, "w"), indent=2)
            written += 1
    print(f"wrote {written} envelopes → {OUT}")

if __name__ == "__main__":
    main()
