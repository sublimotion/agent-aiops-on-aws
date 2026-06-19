#!/usr/bin/env python3
"""Concurrency sweep for Qwen3-Reranker-4B.

Levels [1, 4, 16, 64]; 10 warmup + 50 steady per level; true concurrency via
asyncio+aiohttp. Every request is (fixed query + k=50 candidates) hitting
/v1/score, so one request → k pair scores. Pair length fixed at 1024.

Iteration 1 scope: validate plumbing + emit one schema-valid artifact.
Spec asks for 256 concurrency and a pair-length axis — deferred to later runs.

Headline: peak request_throughput level's percentiles. Pairs-per-second lives
at extensions.reranker.pairs_per_s (derived: request_throughput * k).
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

LEVELS = [1, 4, 16, 64]
WARMUP = 10
STEADY = 50
K_CANDIDATES = 50
PAIR_LENGTH = 1024


async def fire_one(session: aiohttp.ClientSession, body: dict):
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{ENDPOINT}/v1/score",
            json=body,
            timeout=aiohttp.ClientTimeout(total=180),
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
        len(scores),                               # pairs_scored
        int(usage.get("prompt_tokens", 0)),
    )


async def run_level(body_template: dict, concurrency: int, n_warmup: int, n_steady: int) -> dict:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def one(_idx: int):
            async with sem:
                return await fire_one(session, body_template)

        warm_tasks = [asyncio.create_task(one(i)) for i in range(n_warmup)]
        await asyncio.gather(*warm_tasks)

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
        "concurrency": concurrency,
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
    corpus = build_corpus(k=K_CANDIDATES, pair_length=PAIR_LENGTH, seed=42)
    body = build_request_body(corpus)
    print(
        f"[corpus] query_words={len(corpus.query.split())} k={len(corpus.candidates)} "
        f"pair_length_target={corpus.pair_length_target}",
        flush=True,
    )

    levels_out = []
    for c in LEVELS:
        print(f"[sweep] level c={c} warmup={WARMUP} steady={STEADY}", flush=True)
        r = await run_level(body, c, WARMUP, STEADY)
        print(
            f"  c={c}: completed={r['completed']}/{r['num_requests']} "
            f"dur={r['duration_s']:.2f}s "
            f"rps={r['request_throughput']:.2f} "
            f"pairs/s={r['pairs_per_s']:.1f} "
            f"e2e_p50={r['e2e_ms']['p50']:.0f}ms "
            f"e2e_p99={r['e2e_ms']['p99']:.0f}ms",
            flush=True,
        )
        levels_out.append(r)

    peak = max(levels_out, key=lambda r: r["request_throughput"])
    total_reqs = sum(r["num_requests"] for r in levels_out)
    total_failed = sum(r["failed"] for r in levels_out)
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
            "catalog_id": "concurrency-sweep",
            "modality": "text",
            "dataset": {
                "type": "synthetic-lorem-corpus",
                "source": "scripts/_common.py::build_corpus",
                "input_tokens": {"mean": mean_in_tokens},
                "output_tokens": {"mean": 0},
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": LEVELS,
                "num_prompts_per_level": STEADY,
                "warmup_requests": WARMUP,
                "current_level": peak["concurrency"],
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
            "output_toks_per_s": 0.0,  # classifier: no generated tokens
            "request_throughput": peak["request_throughput"],
            "total_input_tokens": peak["total_input_tokens"],
            "total_output_tokens": 0,
            "max_concurrent_requests": peak["concurrency"],
        },
        "extensions": {
            "reranker": {
                "k_candidates": K_CANDIDATES,
                "pair_length_target": PAIR_LENGTH,
                "pairs_per_s": peak["pairs_per_s"],
                "total_pairs_scored": peak["total_pairs_scored"],
            },
            "substrate_caveat": (
                "Measured on g6e.2xlarge (L40S 48GB) — spec-preferred is g6.xlarge "
                "(L4 24GB). Per-stream latency is an upper bound for L40S; cost "
                "claims should NOT be projected to the L4 row from this artifact."
            ),
            "notes": (
                "Iteration 1 concurrency sweep for Qwen3-Reranker-4B on vLLM 0.19.1 "
                "(--runner pooling --convert classify + hf_overrides). "
                "Levels [1,4,16,64]; k=50 candidates/req; pair_length=1024. "
                "256 concurrency + pair-length axis deferred. "
                "Headline metrics are from the peak-throughput level; per-level "
                "aggregates in extensions.sweep_levels."
            ),
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["completed"],
                    "failed": r["failed"],
                    "duration_s": r["duration_s"],
                    "request_throughput": r["request_throughput"],
                    "pairs_per_s": r["pairs_per_s"],
                    "e2e_ms": r["e2e_ms"],
                    "total_input_tokens": r["total_input_tokens"],
                    "total_pairs_scored": r["total_pairs_scored"],
                }
                for r in levels_out
            ],
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
