#!/usr/bin/env python3
"""Concurrency sweep for DeepSeek-OCR-2 (latency range) — stratified corpus.

Levels [1, 4, 16, 32]; 10 warmup + 50 steady per level; true concurrency via
asyncio+aiohttp. Requests are drawn round-robin from a 6-doc corpus
(receipt, article, table, formula, dense, handwritten) so throughput and
latency aggregates reflect realistic OCR workload variance.

Per level we emit:
  - aggregate stats (headline at peak)
  - per-doc-type breakdown in extensions.stratification.per_doc_type[]
The sweep-level breakdown lives in extensions.sweep_levels[] as before.
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
    ENDPOINT, MODEL_ID, PROMPT, MODEL_BLOCK, ENGINE_BLOCK, INFRA_BLOCK,
    DOC_TYPES, CorpusItem, load_corpus, build_request_body, compute_percentiles,
    envelope, write_artifact, summarize_per_doc_type, attach_throughput,
    compute_equivalent_pages,
)

LEVELS = [1, 4, 16, 32]
WARMUP = 10
STEADY = 50
MAX_TOKENS = 1024  # enlarged so dense / table buckets aren't truncated


async def fire_one(session: aiohttp.ClientSession, body: dict):
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{ENDPOINT}/v1/chat/completions",
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
    return ((t1 - t0) * 1000.0,
            int(usage.get("completion_tokens", 0)),
            int(usage.get("prompt_tokens", 0)))


async def run_level(
    corpus: list[CorpusItem],
    concurrency: int,
    n_warmup: int,
    n_steady: int,
) -> dict:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def one(idx: int):
            item = corpus[idx % len(corpus)]
            body = build_request_body(item.base64_cached, PROMPT, MAX_TOKENS)
            async with sem:
                r = await fire_one(session, body)
            return item.doc_type, r

        # warmup (round-robin across corpus)
        warm_tasks = [asyncio.create_task(one(i)) for i in range(n_warmup)]
        await asyncio.gather(*warm_tasks)

        # steady
        t_start = time.perf_counter()
        tasks = [asyncio.create_task(one(i)) for i in range(n_steady)]
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    duration = t_end - t_start

    # aggregate + per-bucket records
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

    per_doc_summary = summarize_per_doc_type(buckets)
    attach_throughput(per_doc_summary, duration)

    completed = len(results) - failed
    return {
        "concurrency": concurrency,
        "num_requests": n_steady,
        "completed": completed,
        "failed": failed,
        "duration_s": duration,
        "e2e_ms": compute_percentiles(e2e_all),
        "request_throughput": (completed / duration) if duration > 0 else 0.0,
        "output_toks_per_s": (total_comp / duration) if duration > 0 else 0.0,
        "image_toks_per_s": (total_prompt / duration) if duration > 0 else 0.0,
        "total_input_tokens": total_prompt,
        "total_output_tokens": total_comp,
        "per_doc_type": per_doc_summary,
    }


async def main_async(assets_dir: Path, out_path: Path) -> None:
    corpus = load_corpus(assets_dir)
    print(f"[corpus] loaded {len(corpus)} items: {[c.doc_type for c in corpus]}", flush=True)

    levels_out = []
    for c in LEVELS:
        print(f"[sweep] level c={c} warmup={WARMUP} steady={STEADY} (round-robin corpus)", flush=True)
        r = await run_level(corpus, c, WARMUP, STEADY)
        print(
            f"  c={c}: completed={r['completed']}/{r['num_requests']} "
            f"dur={r['duration_s']:.2f}s "
            f"rps={r['request_throughput']:.2f} "
            f"out_tps={r['output_toks_per_s']:.1f} "
            f"img_tps={r['image_toks_per_s']:.1f} "
            f"e2e_p50={r['e2e_ms']['p50']:.0f}ms "
            f"e2e_p99={r['e2e_ms']['p99']:.0f}ms",
            flush=True,
        )
        # quick per-doc-type preview
        for dt in DOC_TYPES:
            s = r["per_doc_type"].get(dt, {})
            if s.get("completed"):
                print(
                    f"    [{dt:11s}] n={s['completed']:2d} "
                    f"in_p50={s['image_tokens_p50']:4d} "
                    f"out_p50={s['output_tokens_p50']:4d} "
                    f"e2e_p50={s['e2e_ms_p50']:5.0f}ms "
                    f"e2e_p99={s['e2e_ms_p99']:5.0f}ms "
                    f"eq_pps={s.get('equivalent_pages_per_s', 0):.3f}",
                    flush=True,
                )
        levels_out.append(r)

    peak = max(levels_out, key=lambda r: r["request_throughput"])
    total_reqs = sum(r["num_requests"] for r in levels_out)
    total_failed = sum(r["failed"] for r in levels_out)
    err_rate = total_failed / total_reqs if total_reqs else 0.0

    mean_out_tokens = (
        peak["total_output_tokens"] / peak["completed"] if peak["completed"] else 0
    )
    mean_in_tokens = (
        peak["total_input_tokens"] / peak["completed"] if peak["completed"] else 0
    )

    doc = envelope()
    doc.update({
        "model": MODEL_BLOCK,
        "engine": ENGINE_BLOCK,
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "ocr",
            "catalog_id": "concurrency-sweep",
            "modality": "multimodal",
            "dataset": {
                "type": "synthetic-stratified-corpus",
                "source": "scripts/test-assets/{receipt,article,table,formula,dense,handwritten}.png",
                "input_tokens": {"mean": mean_in_tokens},
                "output_tokens": {"mean": mean_out_tokens},
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": LEVELS,
                "num_prompts_per_level": STEADY,
                "warmup_requests": WARMUP,
                "current_level": peak["concurrency"],
            },
            "api": {
                "type": "chat",
                "streaming": False,
                "endpoint": "/v1/chat/completions",
                "prompt_template": PROMPT,
            },
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["completed"],
            "failed": peak["failed"],
            "error_rate": err_rate,
            "e2e_ms": peak["e2e_ms"],
            "output_toks_per_s": peak["output_toks_per_s"],
            "request_throughput": peak["request_throughput"],
            "total_input_tokens": peak["total_input_tokens"],
            "total_output_tokens": peak["total_output_tokens"],
            "max_concurrent_requests": peak["concurrency"],
        },
        "extensions": {
            "modality": "vision-language",
            "image_toks_per_s": peak["image_toks_per_s"],
            "equivalent_pages_per_s": compute_equivalent_pages(
                mean_in_tokens, mean_out_tokens
            ) * peak["request_throughput"],
            "notes": (
                "Stage 6 concurrency sweep (latency range) on DeepSeek-OCR-2 BF16. "
                "Iteration 5: requests drawn round-robin from 6-doc stratified corpus "
                "(receipt / article / table / formula / dense / handwritten). "
                "Non-streaming, e2e-only. Headline metrics are from the peak-throughput level; "
                "per-doc-type breakdown is at the current (peak) level only — lower-concurrency "
                "levels have tiny bucket samples (~8 per doc_type) so headline per-doc stats "
                "are taken from the peak level. Per-level aggregates in extensions.sweep_levels."
            ),
            "stratification": {
                "corpus_size": len(corpus),
                "corpus_doc_types": [c.doc_type for c in corpus],
                "corpus_weights": {dt: 1.0 / len(corpus) for dt in DOC_TYPES},
                "sampling": "round-robin",
                "level_for_per_doc_type": peak["concurrency"],
            },
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["completed"],
                    "failed": r["failed"],
                    "duration_s": r["duration_s"],
                    "request_throughput": r["request_throughput"],
                    "output_toks_per_s": r["output_toks_per_s"],
                    "image_toks_per_s": r["image_toks_per_s"],
                    "e2e_ms": r["e2e_ms"],
                    "total_input_tokens": r["total_input_tokens"],
                    "total_output_tokens": r["total_output_tokens"],
                }
                for r in levels_out
            ],
        },
    })
    write_artifact(out_path, doc, per_doc_type=peak["per_doc_type"])
    print(f"ARTIFACT: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, required=True,
                    help="Directory containing receipt.png, article.png, ... handwritten.png")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.assets_dir, args.out))


if __name__ == "__main__":
    main()
