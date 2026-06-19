#!/usr/bin/env python3
"""bench-standard.py — Prometheus-first benchmark driver. Emits v1 benchmark-commons envelope.

Drives requests against SGLang/vLLM, then queries Prometheus for TTFT/TPOT/E2E histograms and DCGM
GPU metrics covering the run window. Produces ONE v1 envelope JSON per (config, concurrency).

Why this is the standard:
- Client-side timing drops TTFT and TPOT (non-streaming). Prometheus has both.
- Engines expose prefix cache hit rate and queue wait — client cannot see those.
- DCGM exposes HBM BW util, tensor-core util, SM occupancy — essential for roofline validation.
- Reconciliation step checks client vs Prometheus request counts agree (detects silent failures).

This driver is the SOURCE OF TRUTH; blueprint runners should invoke it, not reimplement it.
See SKILL.md §Always Standard Format for the mandate.

Usage:
  python3 bench-standard.py \\
      --endpoint http://localhost:30000 \\
      --engine sglang \\
      --concurrency 64 \\
      --input-len 512 --output-len 256 \\
      --requests 64 \\
      --sidecar /path/to/blueprint/benchmark.yaml \\
      --engine-tag sglang-eagle3-s4d4k1-hicache200 \\
      --workload-catalog-id concurrency-sweep \\
      --prometheus-url http://localhost:9090 \\
      --out-dir /path/to/blueprint/results/standard/
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import aiohttp  # noqa
except ImportError:
    print("ERROR: aiohttp required. pip install aiohttp", file=sys.stderr)
    sys.exit(2)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(2)


# --------- Prometheus helpers ---------

async def prom_query(session: aiohttp.ClientSession, prom_url: str, query: str, time_s: Optional[float] = None) -> Optional[float]:
    """Return a single scalar value from an instant query, or None on miss/error."""
    params = {"query": query}
    if time_s is not None:
        params["time"] = str(time_s)
    try:
        async with session.get(f"{prom_url}/api/v1/query", params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
    except Exception as e:
        print(f"WARN: prom query failed: {query!r}: {e}", file=sys.stderr)
        return None
    if data.get("status") != "success":
        return None
    result = data.get("data", {}).get("result", [])
    if not result:
        return None
    try:
        return float(result[0]["value"][1])
    except Exception:
        return None


async def prom_counter_delta(session, prom_url, query, t0, t1):
    v0 = await prom_query(session, prom_url, query, t0)
    v1 = await prom_query(session, prom_url, query, t1)
    if v0 is None or v1 is None:
        return None
    return v1 - v0


async def capture_prom_metrics(prom_url: str, engine: str, t_start: float, t_end: float) -> dict:
    """Query Prometheus histograms + DCGM for the run window. Returns dict ready for v1 envelope."""
    window = f"{max(5, int(t_end - t_start))}s"

    def hist(metric):
        return {
            "p50": f"histogram_quantile(0.50, sum(rate({metric}[{window}])) by (le))",
            "p90": f"histogram_quantile(0.90, sum(rate({metric}[{window}])) by (le))",
            "p95": f"histogram_quantile(0.95, sum(rate({metric}[{window}])) by (le))",
            "p99": f"histogram_quantile(0.99, sum(rate({metric}[{window}])) by (le))",
            "mean": f"sum(rate({metric.replace('_bucket','_sum')}[{window}])) / sum(rate({metric.replace('_bucket','_count')}[{window}]))",
        }

    # Metric name mapping — add new engines here
    METRICS = {
        "vllm": {
            "ttft": "vllm:time_to_first_token_seconds_bucket",
            "tpot": "vllm:time_per_output_token_seconds_bucket",
            "e2e":  "vllm:e2e_request_latency_seconds_bucket",
            "requests_success": 'vllm:request_success_total',
            "requests_error": 'vllm:request_error_total',
            "kv_cache_usage": "vllm:kv_cache_usage_perc",
            "prefix_hits": "vllm:prefix_cache_hits",
            "prefix_queries": "vllm:prefix_cache_queries",
            "num_preemptions": "vllm:num_preemptions_total",
        },
        "sglang": {
            "ttft": "sglang:time_to_first_token_seconds_bucket",
            "tpot": "sglang:time_per_output_token_seconds_bucket",
            "e2e":  "sglang:e2e_request_latency_seconds_bucket",
            "requests_success": "sglang:num_requests_success_total",
            "requests_error": "sglang:num_requests_error_total",
            "kv_cache_usage": "sglang:token_usage_ratio",
            "prefix_hits": "sglang:prefix_cache_hit_rate",
            "prefix_queries": None,
            "num_preemptions": None,
        },
    }.get(engine, {})

    out = {"ttft_ms": {}, "tpot_ms": {}, "itl_ms": {}, "e2e_ms": {}}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Histograms → convert seconds to ms
        for field, metric_key in (("ttft_ms", "ttft"), ("tpot_ms", "tpot"), ("e2e_ms", "e2e")):
            m = METRICS.get(metric_key)
            if not m:
                out[field] = {k: None for k in ("mean", "p50", "p90", "p95", "p99")}
                continue
            qs = hist(m)
            vals = {}
            for k, q in qs.items():
                v = await prom_query(session, prom_url, q, t_end)
                vals[k] = round(v * 1000, 2) if v is not None else None
            out[field] = vals
        # ITL ≈ TPOT for non-streaming benchmarks
        out["itl_ms"] = dict(out["tpot_ms"])

        # Counters over window
        out["prom_requests_success"] = await prom_counter_delta(session, prom_url, METRICS.get("requests_success") or "vector(0)", t_start, t_end)
        out["prom_requests_error"]   = await prom_counter_delta(session, prom_url, METRICS.get("requests_error") or "vector(0)", t_start, t_end)
        out["prom_preemptions"]      = await prom_counter_delta(session, prom_url, METRICS.get("num_preemptions") or "vector(0)", t_start, t_end)

        # KV cache / prefix cache
        out["kv_cache_usage_pct_mean"] = await prom_query(session, prom_url, f"avg_over_time({METRICS.get('kv_cache_usage') or 'vector(0)'}[{window}])", t_end)
        if METRICS.get("prefix_hits") and METRICS.get("prefix_queries"):
            hits = await prom_counter_delta(session, prom_url, METRICS["prefix_hits"], t_start, t_end)
            queries = await prom_counter_delta(session, prom_url, METRICS["prefix_queries"], t_start, t_end)
            out["prefix_cache_hit_rate"] = (hits / queries) if (queries and queries > 0) else None
        else:
            out["prefix_cache_hit_rate"] = None

        # DCGM — per-GPU, averaged across all GPUs
        dcgm = {}
        for label, metric in [
            ("gpu_util_pct_mean",  "avg(DCGM_FI_DEV_GPU_UTIL)"),
            ("gpu_util_pct_max",   "max(max_over_time(DCGM_FI_DEV_GPU_UTIL[" + window + "]))"),
            ("hbm_bw_util_pct_mean", "avg(avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE[" + window + "])) * 100"),
            ("hbm_bw_util_pct_max",  "max(max_over_time(DCGM_FI_PROF_DRAM_ACTIVE[" + window + "])) * 100"),
            ("sm_active_pct_mean", "avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE[" + window + "])) * 100"),
            ("tensor_active_pct_mean", "avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE[" + window + "])) * 100"),
            ("power_draw_w_mean",  "avg(avg_over_time(DCGM_FI_DEV_POWER_USAGE[" + window + "]))"),
            ("power_draw_w_max",   "max(max_over_time(DCGM_FI_DEV_POWER_USAGE[" + window + "]))"),
            ("temp_c_max",         "max(max_over_time(DCGM_FI_DEV_GPU_TEMP[" + window + "]))"),
            ("memory_used_mib_mean", "avg(avg_over_time(DCGM_FI_DEV_FB_USED[" + window + "]))"),
            ("xid_errors",         "sum(increase(DCGM_FI_DEV_XID_ERRORS[" + window + "]))"),
        ]:
            dcgm[label] = await prom_query(session, prom_url, metric, t_end)
        out["dcgm"] = dcgm

    return out


# --------- Client driver ---------

async def issue_request_sglang(session, endpoint, prompt, output_len):
    body = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}}
    t0 = time.time()
    async with session.post(f"{endpoint}/generate", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
    t1 = time.time()
    meta = data.get("meta_info", {})
    tokens = meta.get("completion_tokens") or output_len
    return {"duration_s": t1 - t0, "tokens": tokens,
            "spec_accept_rate": meta.get("spec_accept_rate"),
            "spec_accept_length": meta.get("spec_accept_length")}


async def issue_request_vllm(session, endpoint, prompt, output_len):
    body = {"model": "model", "prompt": prompt, "max_tokens": output_len, "temperature": 0.0}
    t0 = time.time()
    async with session.post(f"{endpoint}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
    t1 = time.time()
    tokens = data.get("usage", {}).get("completion_tokens") or output_len
    return {"duration_s": t1 - t0, "tokens": tokens, "spec_accept_rate": None, "spec_accept_length": None}


async def run_client(endpoint: str, engine: str, concurrency: int, input_len: int, output_len: int, total_requests: int) -> dict:
    issue = issue_request_sglang if engine == "sglang" else issue_request_vllm
    prompt = " ".join(["hello"] * input_len)

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    results, errors = [], 0

    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def gated(i):
            nonlocal errors
            async with sem:
                try:
                    return await issue(session, endpoint, prompt, output_len)
                except Exception as e:
                    errors += 1
                    return None

        t_start = time.time()
        results = await asyncio.gather(*[gated(i) for i in range(total_requests)])
        t_end = time.time()

    ok = [r for r in results if r is not None]
    total_tokens = sum(r["tokens"] for r in ok)
    durations = [r["tokens"] / r["duration_s"] for r in ok if r["duration_s"] > 0]
    spec_accepts = [r["spec_accept_rate"] for r in ok if r.get("spec_accept_rate") is not None]
    spec_lens = [r["spec_accept_length"] for r in ok if r.get("spec_accept_length") is not None]

    return {
        "t_start": t_start,
        "t_end": t_end,
        "duration_s": round(t_end - t_start, 3),
        "ok": len(ok),
        "err": errors,
        "total_tokens": total_tokens,
        "agg_tok_per_s": round(total_tokens / (t_end - t_start), 1) if t_end > t_start else 0.0,
        "per_req_tok_per_s_mean": round(sum(durations) / len(durations), 1) if durations else None,
        "spec_accept_rate_mean": round(sum(spec_accepts) / len(spec_accepts), 3) if spec_accepts else None,
        "spec_accept_length_mean": round(sum(spec_lens) / len(spec_lens), 3) if spec_lens else None,
    }


# --------- Envelope builder ---------

def build_envelope(sidecar: dict, engine: str, engine_tag: str, workload_catalog_id: str,
                   client: dict, prom: dict, concurrency: int, input_len: int, output_len: int,
                   total_requests: int, extra: Optional[dict] = None) -> dict:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_block = sidecar.get("model", {})
    engine_block_base = sidecar.get("engine", {})
    infra_block = sidecar.get("infrastructure", {})
    slo_targets = sidecar.get("slo", {})

    # Reconciliation — client ok count should ~match Prometheus success counter (5% tolerance)
    recon = {
        "client_ok": client["ok"],
        "client_err": client["err"],
        "prom_success": prom.get("prom_requests_success"),
        "prom_error": prom.get("prom_requests_error"),
    }
    if prom.get("prom_requests_success") is not None and client["ok"] > 0:
        diff_pct = abs(prom["prom_requests_success"] - client["ok"]) / client["ok"] * 100
        recon["success_diff_pct"] = round(diff_pct, 2)
        recon["reconciled"] = diff_pct <= 5.0
    else:
        recon["reconciled"] = None

    # SLO evaluation
    def slo_row(target_key, actual):
        tgt = slo_targets.get(target_key)
        if tgt is None or actual is None:
            return {"target": tgt, "actual": actual, "pass": None}
        return {"target": tgt, "actual": actual, "pass": actual <= tgt}

    error_rate = client["err"] / (client["ok"] + client["err"]) if (client["ok"] + client["err"]) > 0 else 0.0

    slo_results = {
        "ttft_p99_ms":    slo_row("ttft_p99_ms",    prom["ttft_ms"].get("p99")),
        "tpot_p99_ms":    slo_row("tpot_p99_ms",    prom["tpot_ms"].get("p99")),
        "e2e_p99_ms":     slo_row("e2e_p99_ms",     prom["e2e_ms"].get("p99")),
        "error_rate_max": slo_row("error_rate_max", round(error_rate, 6)),
    }
    overall_pass = all(r["pass"] for r in slo_results.values() if r["pass"] is not None)
    overall_pass = overall_pass and all(r["pass"] is not None for r in slo_results.values())

    # Cost
    cost_per_hr = (sidecar.get("cost", {}) or {}).get("spot_price_per_hr")
    dollars_per_1m = None
    if cost_per_hr and client["agg_tok_per_s"] > 0:
        dollars_per_1m = round((cost_per_hr / client["agg_tok_per_s"]) * (1_000_000 / 3600.0), 2)

    envelope = {
        "schema_version": "1.0.0",
        "artifact_id": str(uuid.uuid4()),
        "created_at": created_at,
        "source_tool": {
            "name": "bench-standard.py",
            "version": "1.0.0",
            "enrichment_version": "1.0.0",
        },
        "model": model_block,
        "engine": {
            **engine_block_base,
            "engine_config_tag": engine_tag,
        },
        "framework": sidecar.get("framework", {}),
        "infrastructure": infra_block,
        "workload": {
            "use_case": workload_catalog_id,
            "catalog_id": workload_catalog_id,
            "modality": "text",
            "dataset": {
                "type": "synthetic",
                "input_tokens": {"mean": input_len, "std_dev": 0},
                "output_tokens": {"mean": output_len, "std_dev": 0},
            },
            "load": {
                "type": "concurrency",
                "concurrency": concurrency,
                "num_prompts": total_requests,
                "warmup_requests": 0,
            },
            "api": {"type": "completions", "streaming": False,
                    "endpoint": "/generate" if engine == "sglang" else "/v1/completions"},
        },
        "metrics": {
            "duration_s": client["duration_s"],
            "completed": client["ok"],
            "failed": client["err"],
            "error_rate": round(error_rate, 6),
            "ttft_ms": prom["ttft_ms"],
            "tpot_ms": prom["tpot_ms"],
            "itl_ms":  prom["itl_ms"],
            "e2e_ms":  prom["e2e_ms"],
            "output_toks_per_s": client["agg_tok_per_s"],
            "request_throughput": round(client["ok"] / client["duration_s"], 3) if client["duration_s"] > 0 else None,
            "total_toks_per_s": client["agg_tok_per_s"],
            "total_input_tokens": client["ok"] * input_len,
            "total_output_tokens": client["total_tokens"],
            "max_concurrent_requests": concurrency,
        },
        "slo": {
            "targets": slo_targets,
            "results": slo_results,
            "overall_pass": overall_pass,
        },
        "extensions": {
            "gpu_telemetry": prom.get("dcgm", {}),
            "cache_stats": {
                "kv_utilization_pct_mean": prom.get("kv_cache_usage_pct_mean"),
                "prefix_hit_rate": prom.get("prefix_cache_hit_rate"),
                "preemption_count": prom.get("prom_preemptions"),
            },
            "speculative_decode_stats": {
                "accept_rate_mean":   client.get("spec_accept_rate_mean"),
                "accept_length_mean": client.get("spec_accept_length_mean"),
                "effective_tokens_per_step": (
                    round(client["spec_accept_rate_mean"] * client["spec_accept_length_mean"], 3)
                    if client.get("spec_accept_rate_mean") is not None and client.get("spec_accept_length_mean") is not None
                    else None
                ),
            },
            "cost": {
                "instance_cost_per_hr": cost_per_hr,
                "dollars_per_1m_output_tokens": dollars_per_1m,
                "formula": f"({cost_per_hr} / {client['agg_tok_per_s']}) * (1000000 / 3600)" if cost_per_hr else None,
            },
            "reconciliation": recon,
            "session_metadata": {
                "engine_config_tag": engine_tag,
                "run_start": datetime.fromtimestamp(client["t_start"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "run_end":   datetime.fromtimestamp(client["t_end"],   tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **(extra or {}),
            },
        },
    }
    return envelope


def canonical_filename(model_id: str, substrate: str, hw: str, engine_tag: str,
                       workload_catalog_id: str, concurrency: int) -> str:
    model_slug = model_id.split("/")[-1].lower()
    return f"{model_slug}_{substrate}_{hw}_{engine_tag}_{workload_catalog_id}_c{concurrency}.json"


# --------- Main ---------

def main():
    ap = argparse.ArgumentParser(description="Prometheus-first bench driver (emits v1 envelope).")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--requests", type=int, default=None, help="Total requests (default = 4×concurrency)")
    ap.add_argument("--sidecar", required=True, help="Path to blueprint's benchmark.yaml")
    ap.add_argument("--engine-tag", required=True, help="Engine config tag for filename")
    ap.add_argument("--workload-catalog-id", default="concurrency-sweep")
    ap.add_argument("--prometheus-url", default="http://localhost:9090")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--settle-seconds", type=int, default=3, help="Sleep between run-end and Prom query to flush buckets")
    args = ap.parse_args()

    sidecar = yaml.safe_load(Path(args.sidecar).read_text())
    infra = sidecar.get("infrastructure", {})
    model_id = (sidecar.get("model", {}) or {}).get("id", "unknown")
    substrate = infra.get("substrate", "unknown")
    hw = infra.get("instance_type", "unknown").replace(".", "-")

    total_requests = args.requests or max(4, args.concurrency * 4)
    print(f"[run] engine={args.engine} conc={args.concurrency} input={args.input_len} output={args.output_len} total={total_requests}")

    # Drive load
    client = asyncio.run(run_client(args.endpoint, args.engine, args.concurrency,
                                    args.input_len, args.output_len, total_requests))
    print(f"[client] ok={client['ok']} err={client['err']} agg={client['agg_tok_per_s']} duration={client['duration_s']}s")

    # Let engine histogram buckets flush
    time.sleep(args.settle_seconds)

    # Query Prometheus for the run window
    prom = asyncio.run(capture_prom_metrics(args.prometheus_url, args.engine,
                                             client["t_start"], client["t_end"] + args.settle_seconds))
    print(f"[prom] ttft_p99={prom['ttft_ms'].get('p99')}ms  tpot_p99={prom['tpot_ms'].get('p99')}ms  e2e_p99={prom['e2e_ms'].get('p99')}ms")
    print(f"[dcgm] gpu_util={prom.get('dcgm',{}).get('gpu_util_pct_mean')}%  hbm_bw={prom.get('dcgm',{}).get('hbm_bw_util_pct_mean')}%")

    envelope = build_envelope(sidecar, args.engine, args.engine_tag, args.workload_catalog_id,
                              client, prom, args.concurrency, args.input_len, args.output_len, total_requests)

    # Reconciliation warning
    recon = envelope["extensions"]["reconciliation"]
    if recon.get("reconciled") is False:
        print(f"WARN: client/prom request count mismatch — client={recon['client_ok']} prom={recon['prom_success']} diff={recon.get('success_diff_pct')}%", file=sys.stderr)

    # Emit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = canonical_filename(model_id, substrate, hw, args.engine_tag, args.workload_catalog_id, args.concurrency)
    out_path = out_dir / fname
    out_path.write_text(json.dumps(envelope, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
