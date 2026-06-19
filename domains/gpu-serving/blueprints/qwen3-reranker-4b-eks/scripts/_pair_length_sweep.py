#!/usr/bin/env python3
"""Pair-length axis sweep for Qwen3-Reranker-4B.

Concurrency fixed at c=4 (iteration 1's peak-throughput level). Sweep
pair_length ∈ [512, 1024, 2048, 4096]. Per length: 10 warmup + 50 steady.
k=50 candidates/request. pair_length=4096 = model max_model_len, so a pair
that exceeds that will 400 from vLLM; treat failures as data.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import (
    ENDPOINT, MODEL_ID, MODEL_BLOCK, ENGINE_BLOCK, INFRA_BLOCK,
    build_corpus, build_request_body, compute_percentiles,
    envelope, write_artifact,
)

PAIR_LENGTHS = [512, 1024, 2048, 4096]
CONCURRENCY = 4
WARMUP = 10
STEADY = 50
K_CANDIDATES = 50


async def fire_one(session: aiohttp.ClientSession, body: dict):
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{ENDPOINT}/v1/score",
            json=body,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                await resp.read()
                return None
            data = await resp.json()
    except Exception:
        return None
    t1 = time.perf_counter()
    usage = data.get("usage", {}) or {}
    scores = data.get("data", []) or []
    return (
        (t1 - t0) * 1000.0,
        len(scores),
        int(usage.get("prompt_tokens", 0)),
    )


async def run_length(body: dict, concurrency: int, n_warmup: int, n_steady: int) -> dict:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def one(_idx: int):
            async with sem:
                return await fire_one(session, body)

        warm = [asyncio.create_task(one(i)) for i in range(n_warmup)]
        await asyncio.gather(*warm)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(one(i)) for i in range(n_steady)]
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    duration = t_end - t_start
    e2e_all: list[float] = []
    total_pairs = 0
    total_prompt = 0
    failed = 0
    for r in results:
        if r is None:
            failed += 1
            continue
        e2e_ms, pairs, prom = r
        e2e_all.append(e2e_ms)
        total_pairs += pairs
        total_prompt += prom
    completed = len(results) - failed
    return {
        "num_requests": n_steady,
        "completed": completed,
        "failed": failed,
        "duration_s": duration,
        "e2e_ms": compute_percentiles(e2e_all),
        "request_throughput": (completed / duration) if duration > 0 else 0.0,
        "pairs_per_s": (total_pairs / duration) if duration > 0 else 0.0,
        "total_pairs_scored": total_pairs,
        "total_input_tokens": total_prompt,
    }


async def main_async(out_path: Path) -> None:
    lengths_out = []
    for L in PAIR_LENGTHS:
        corpus = build_corpus(k=K_CANDIDATES, pair_length=L, seed=42)
        body = build_request_body(corpus)
        print(
            f"[sweep] pair_length={L} c={CONCURRENCY} warmup={WARMUP} steady={STEADY}",
            flush=True,
        )
        r = await run_length(body, CONCURRENCY, WARMUP, STEADY)
        r["pair_length"] = L
        print(
            f"  L={L}: completed={r['completed']}/{r['num_requests']} "
            f"failed={r['failed']} dur={r['duration_s']:.2f}s "
            f"rps={r['request_throughput']:.2f} pairs/s={r['pairs_per_s']:.1f} "
            f"e2e_p50={r['e2e_ms']['p50']:.0f}ms p99={r['e2e_ms']['p99']:.0f}ms",
            flush=True,
        )
        lengths_out.append(r)

    # pick best by pairs_per_s (with some completions)
    viable = [r for r in lengths_out if r["completed"] > 0]
    peak = max(viable, key=lambda r: r["pairs_per_s"]) if viable else lengths_out[0]

    total_reqs = sum(r["num_requests"] for r in lengths_out)
    total_failed = sum(r["failed"] for r in lengths_out)
    err_rate = total_failed / total_reqs if total_reqs else 0.0
    mean_in_tokens = (
        peak["total_input_tokens"] / peak["completed"] if peak["completed"] else 0
    )

    doc = envelope()
    doc.update({
        "model": MODEL_BLOCK,
        "engine": ENGINE_BLOCK,
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "reranker",
            "catalog_id": "pair-length-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic-lorem-corpus",
                "source": "scripts/_common.py::build_corpus",
                "input_tokens": {"mean": mean_in_tokens},
                "output_tokens": {"mean": 0},
            },
            "load": {
                "type": "pair-length-sweep",
                "pair_length_axis": PAIR_LENGTHS,
                "max_concurrency": CONCURRENCY,
                "num_prompts_per_length": STEADY,
                "warmup_requests": WARMUP,
                "current_pair_length": peak["pair_length"],
            },
            "api": {
                "type": "score",
                "streaming": False,
                "endpoint": "/v1/score",
                "prompt_template": "query + k=50 candidates",
            },
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["completed"],
            "failed": peak["failed"],
            "error_rate": err_rate,
            "e2e_ms": peak["e2e_ms"],
            "output_toks_per_s": 0.0,
            "request_throughput": peak["request_throughput"],
            "total_input_tokens": peak["total_input_tokens"],
            "total_output_tokens": 0,
            "max_concurrent_requests": CONCURRENCY,
        },
        "extensions": {
            "reranker": {
                "k_candidates": K_CANDIDATES,
                "pair_length_target": peak["pair_length"],
                "pairs_per_s": peak["pairs_per_s"],
                "total_pairs_scored": peak["total_pairs_scored"],
                "pair_length_sweep": [
                    {
                        "pair_length": r["pair_length"],
                        "completed": r["completed"],
                        "failed": r["failed"],
                        "duration_s": r["duration_s"],
                        "e2e_ms": r["e2e_ms"],
                        "request_throughput": r["request_throughput"],
                        "pairs_per_s": r["pairs_per_s"],
                        "total_input_tokens": r["total_input_tokens"],
                        "total_pairs_scored": r["total_pairs_scored"],
                    }
                    for r in lengths_out
                ],
            },
            "substrate_caveat": (
                "Measured on g6e.2xlarge (L40S 48GB) — spec-preferred is g6.xlarge "
                "(L4 24GB). Per-stream latency is an upper bound for L40S; cost "
                "claims should NOT be projected to the L4 row from this artifact."
            ),
            "notes": (
                "Iteration 2 pair-length axis sweep for Qwen3-Reranker-4B on vLLM 0.19.1. "
                f"Concurrency fixed at c={CONCURRENCY} (iter-1 saturation point). "
                f"Pair-length axis {PAIR_LENGTHS}; k={K_CANDIDATES} candidates/request. "
                "Headline metrics from pair_length with best pairs/s; per-length "
                "breakdown in extensions.reranker.pair_length_sweep."
            ),
        },
    })
    write_artifact(out_path, doc)
    print(f"ARTIFACT: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.out))


if __name__ == "__main__":
    main()
