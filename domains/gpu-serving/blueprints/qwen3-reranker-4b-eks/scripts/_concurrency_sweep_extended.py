#!/usr/bin/env python3
"""Extended concurrency sweep for Qwen3-Reranker-4B (iteration 2).

Adds c=256 to iteration-1 ladder: [1, 4, 16, 64, 256]. Pair_length=1024, k=50.
If c=256 errors out (queue rejection, context overflow, OOM, timeout), we
capture failure counts + error-type metadata rather than crashing the run.
Headline metrics come from the peak pairs/s level.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import (
    ENDPOINT, MODEL_ID, MODEL_BLOCK, ENGINE_BLOCK, INFRA_BLOCK,
    build_corpus, build_request_body, compute_percentiles,
    envelope, write_artifact,
)

LEVELS = [1, 4, 16, 64, 256]
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
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            text_preview = ""
            if resp.status != 200:
                raw = await resp.read()
                text_preview = raw[:200].decode("utf-8", "replace")
                return {"err": f"http_{resp.status}", "detail": text_preview}
            data = await resp.json()
    except asyncio.TimeoutError:
        return {"err": "timeout"}
    except aiohttp.ClientError as e:
        return {"err": f"client_{type(e).__name__}", "detail": str(e)[:160]}
    except Exception as e:
        return {"err": f"exc_{type(e).__name__}", "detail": str(e)[:160]}
    t1 = time.perf_counter()
    usage = data.get("usage", {}) or {}
    scores = data.get("data", []) or []
    return {
        "e2e_ms": (t1 - t0) * 1000.0,
        "pairs": len(scores),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
    }


async def run_level(body: dict, concurrency: int, n_warmup: int, n_steady: int) -> dict:
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
    err_counter: Counter = Counter()
    err_samples: list[str] = []
    failed = 0
    for r in results:
        if r is None or "err" in r:
            failed += 1
            if r:
                err_counter[r.get("err", "unknown")] += 1
                if r.get("detail") and len(err_samples) < 3:
                    err_samples.append(f"{r['err']}: {r['detail']}")
            continue
        e2e_all.append(r["e2e_ms"])
        total_pairs += r["pairs"]
        total_prompt += r["prompt_tokens"]
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
        "error_breakdown": dict(err_counter),
        "error_samples": err_samples,
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
    saturation_notes: list[str] = []
    for c in LEVELS:
        print(f"[sweep] level c={c} warmup={WARMUP} steady={STEADY}", flush=True)
        r = await run_level(body, c, WARMUP, STEADY)
        print(
            f"  c={c}: completed={r['completed']}/{r['num_requests']} "
            f"failed={r['failed']} dur={r['duration_s']:.2f}s "
            f"rps={r['request_throughput']:.2f} pairs/s={r['pairs_per_s']:.1f} "
            f"e2e_p50={r['e2e_ms']['p50']:.0f}ms p99={r['e2e_ms']['p99']:.0f}ms "
            f"errs={r['error_breakdown']}",
            flush=True,
        )
        if r["failed"] > 0:
            saturation_notes.append(
                f"c={c}: {r['failed']}/{r['num_requests']} failed; "
                f"breakdown={r['error_breakdown']}; samples={r['error_samples']}"
            )
        levels_out.append(r)

    viable = [r for r in levels_out if r["completed"] > 0]
    peak = max(viable, key=lambda r: r["pairs_per_s"]) if viable else levels_out[0]
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
            "catalog_id": "concurrency-sweep-extended",
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
            "output_toks_per_s": 0.0,
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
                "Iteration 2 extended concurrency sweep for Qwen3-Reranker-4B on vLLM 0.19.1. "
                "Levels [1,4,16,64,256]; k=50 candidates/req; pair_length=1024. "
                "c=256 added to probe saturation / failure modes. Headline is peak pairs/s level; "
                "per-level aggregates in extensions.sweep_levels; failure breakdown in "
                "extensions.saturation_notes."
            ),
            "saturation_notes": saturation_notes,
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
                    "error_breakdown": r["error_breakdown"],
                    "error_samples": r["error_samples"],
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
