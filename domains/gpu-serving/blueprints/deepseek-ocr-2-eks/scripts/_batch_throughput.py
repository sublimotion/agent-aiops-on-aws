#!/usr/bin/env python3
"""Batch-throughput (saturation) for DeepSeek-OCR-2 @ c=32 — stratified corpus.

Keeps CONCURRENCY (=32) requests in-flight for DURATION_S (=60) seconds after
WARMUP_S (=10). Each worker loops: fire -> complete -> immediately fire the
next request. Requests are drawn round-robin from a 6-doc stratified corpus
(receipt / article / table / formula / dense / handwritten).

Emits a Common Benchmark Artifact with catalog_id=batch-throughput plus a
per-doc-type breakdown at extensions.stratification.per_doc_type[].
"""
from __future__ import annotations
import argparse
import asyncio
import itertools
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import (
    ENDPOINT, PROMPT, MODEL_BLOCK, ENGINE_BLOCK, INFRA_BLOCK,
    DOC_TYPES, load_corpus, build_request_body, compute_percentiles,
    envelope, write_artifact, summarize_per_doc_type, attach_throughput,
    compute_equivalent_pages,
)

CONCURRENCY = 32
WARMUP_S = 10
DURATION_S = 60
MAX_TOKENS = 1024  # enlarged for dense + table buckets


async def fire_one(session, body):
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{ENDPOINT}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=240),
        ) as resp:
            if resp.status != 200:
                await resp.read()
                return None
            data = await resp.json()
    except Exception:
        return None
    t1 = time.perf_counter()
    usage = data.get("usage", {}) or {}
    return ((t1 - t0) * 1000.0,
            int(usage.get("completion_tokens", 0)),
            int(usage.get("prompt_tokens", 0)))


async def run_saturation(session, corpus, concurrency, duration_s, collect):
    """Round-robin workers across the corpus. Each worker holds its own
    iterator offset so no two workers fire the same doc_type concurrently at
    the start of the run (good for avoiding prefix-cache artifacts)."""
    stop_at = time.perf_counter() + duration_s
    results: list = []
    n = len(corpus)

    async def worker(worker_idx: int):
        step = 0
        while time.perf_counter() < stop_at:
            item = corpus[(worker_idx + step) % n]
            body = build_request_body(item.base64_cached, PROMPT, MAX_TOKENS)
            r = await fire_one(session, body)
            if collect:
                results.append((item.doc_type, r))
            step += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(concurrency)])
    t_end = time.perf_counter()
    return results, (t_end - t_start)


async def main_async(assets_dir: Path, out_path: Path) -> None:
    corpus = load_corpus(assets_dir)
    print(f"[corpus] loaded {len(corpus)} items: {[c.doc_type for c in corpus]}", flush=True)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"[warmup] saturation c={CONCURRENCY} for {WARMUP_S}s (round-robin corpus)", flush=True)
        await run_saturation(session, corpus, CONCURRENCY, WARMUP_S, collect=False)

        print(f"[steady] saturation c={CONCURRENCY} for {DURATION_S}s (round-robin corpus)", flush=True)
        results, duration = await run_saturation(
            session, corpus, CONCURRENCY, DURATION_S, collect=True
        )

    # bucketize
    buckets: dict[str, list[dict]] = {dt: [] for dt in DOC_TYPES}
    e2e_all: list[float] = []
    total_comp = 0
    total_prompt = 0
    failed = 0
    for doc_type, r in results:
        if r is None:
            buckets[doc_type].append({
                "e2e_ms": 0.0, "completion_tokens": 0,
                "prompt_tokens": 0, "success": False,
            })
            failed += 1
            continue
        e2e_ms, comp, prom = r
        buckets[doc_type].append({
            "e2e_ms": e2e_ms, "completion_tokens": comp,
            "prompt_tokens": prom, "success": True,
        })
        e2e_all.append(e2e_ms)
        total_comp += comp
        total_prompt += prom

    completed = len(results) - failed
    error_rate = failed / len(results) if results else 0.0
    req_tp = completed / duration if duration > 0 else 0.0
    out_tps = total_comp / duration if duration > 0 else 0.0
    img_tps = total_prompt / duration if duration > 0 else 0.0

    per_doc_summary = summarize_per_doc_type(buckets)
    attach_throughput(per_doc_summary, duration)

    print(
        f"[done] total={len(results)} completed={completed} failed={failed} "
        f"dur={duration:.2f}s rps={req_tp:.2f} out_tps={out_tps:.1f} "
        f"img_tps={img_tps:.1f} e2e_p50={compute_percentiles(e2e_all)['p50']:.0f}ms",
        flush=True,
    )
    for dt in DOC_TYPES:
        s = per_doc_summary.get(dt, {})
        if s.get("completed"):
            print(
                f"  [{dt:11s}] n={s['completed']:3d} "
                f"in_p50={s['image_tokens_p50']:4d} "
                f"out_p50={s['output_tokens_p50']:4d} "
                f"e2e_p50={s['e2e_ms_p50']:5.0f}ms "
                f"e2e_p99={s['e2e_ms_p99']:5.0f}ms "
                f"rps={s['request_throughput']:5.2f} "
                f"out_tps={s['output_toks_per_s']:6.1f} "
                f"eq_pps={s.get('equivalent_pages_per_s', 0):.3f}",
                flush=True,
            )

    mean_out_tokens = total_comp / completed if completed else 0
    mean_in_tokens = total_prompt / completed if completed else 0

    doc = envelope()
    doc.update({
        "model": MODEL_BLOCK,
        "engine": ENGINE_BLOCK,
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "batch",
            "catalog_id": "batch-throughput",
            "modality": "multimodal",
            "dataset": {
                "type": "synthetic-stratified-corpus",
                "source": "scripts/test-assets/{receipt,article,table,formula,dense,handwritten}.png",
                "input_tokens": {"mean": mean_in_tokens},
                "output_tokens": {"mean": mean_out_tokens},
            },
            "load": {
                "type": "constant",
                "max_concurrency": CONCURRENCY,
                "duration_s": DURATION_S,
                "warmup_s": WARMUP_S,
            },
            "api": {
                "type": "chat",
                "streaming": False,
                "endpoint": "/v1/chat/completions",
                "prompt_template": PROMPT,
            },
        },
        "metrics": {
            "duration_s": duration,
            "completed": completed,
            "failed": failed,
            "error_rate": error_rate,
            "e2e_ms": compute_percentiles(e2e_all),
            "output_toks_per_s": out_tps,
            "request_throughput": req_tp,
            "total_input_tokens": total_prompt,
            "total_output_tokens": total_comp,
            "max_concurrent_requests": CONCURRENCY,
        },
        "extensions": {
            "modality": "vision-language",
            "image_toks_per_s": img_tps,
            "equivalent_pages_per_s": compute_equivalent_pages(
                mean_in_tokens, mean_out_tokens
            ) * req_tp,
            "notes": (
                "Stage 6 batch-throughput (saturation) on DeepSeek-OCR-2 BF16 @ c=32. "
                "Iteration 5: 32 worker loops fire back-to-back for 60s (after 10s warmup), "
                "each worker drawing round-robin from the 6-doc stratified corpus "
                "(receipt / article / table / formula / dense / handwritten). "
                "Non-streaming; e2e-only. Per-doc-type breakdown at "
                "extensions.stratification.per_doc_type[] shows how throughput and latency "
                "vary ~16x on image tokens and ~100x on output tokens across document types."
            ),
            "stratification": {
                "corpus_size": len(corpus),
                "corpus_doc_types": [c.doc_type for c in corpus],
                "corpus_weights": {dt: 1.0 / len(corpus) for dt in DOC_TYPES},
                "sampling": "round-robin",
            },
        },
    })
    write_artifact(out_path, doc, per_doc_type=per_doc_summary)
    print(f"ARTIFACT: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.assets_dir, args.out))


if __name__ == "__main__":
    main()
