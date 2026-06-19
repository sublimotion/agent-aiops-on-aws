#!/usr/bin/env python3
"""Voxtral-Mini-3B transcription concurrency sweep — Stage 6 single artifact.

Levels [1, 4, 16]; 5 warmup + 30 steady per level. Round-robin over a
3-bucket corpus (3s / 10s / 30s) for duration spread. Non-streaming.

For each request we record:
  - wall_ms (full multipart POST round-trip)
  - audio_seconds (known from bucket)
  - word_count of returned text (proxy for output tokens; transcription
    endpoint does NOT return usage)
  - bucket label

We emit ONE Common Benchmark Artifact:
  workload.api.type = "transcription"
  workload.modality = "audio"
  workload.api.endpoint = "/v1/audio/transcriptions"
  metrics.* — headline at peak-rtfx level
  extensions.audio = {
    rtfx_p50, rtfx_p99, ttfw_ms (null), audio_seconds_processed,
    audio_minutes_per_dollar, duration_buckets[]
  }
  extensions.substrate_caveat
  extensions.sweep_levels[]
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
    AudioItem, load_audio_corpus, compute_percentiles,
    envelope, write_artifact, audio_minutes_per_dollar,
    ON_DEMAND_PRICE_PER_HR,
)

LEVELS = [1, 4, 16]
WARMUP = 5
STEADY = 30
REQ_TIMEOUT_S = 240


async def fire_one(session: aiohttp.ClientSession, item: AudioItem) -> dict | None:
    """POST one multipart transcription request. Returns dict or None on failure."""
    # We rebuild the FormData each call because aiohttp consumes the file handle.
    data = aiohttp.FormData()
    data.add_field("model", MODEL_ID)
    data.add_field(
        "file",
        item.path.read_bytes(),
        filename=item.path.name,
        content_type="audio/wav",
    )
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{ENDPOINT}/v1/audio/transcriptions",
            data=data,
            timeout=aiohttp.ClientTimeout(total=REQ_TIMEOUT_S),
        ) as resp:
            if resp.status != 200:
                await resp.read()
                return None
            payload = await resp.json()
    except Exception:
        return None
    t1 = time.perf_counter()
    text = (payload.get("text") or "").strip()
    word_count = len(text.split()) if text else 0
    return {
        "bucket": item.bucket,
        "audio_s": item.duration_s,
        "wall_ms": (t1 - t0) * 1000.0,
        "word_count": word_count,
        "text_len": len(text),
        "success": True,
    }


async def run_level(corpus: list[AudioItem], concurrency: int, n_warmup: int, n_steady: int) -> dict:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def one(idx: int):
            item = corpus[idx % len(corpus)]
            async with sem:
                r = await fire_one(session, item)
            if r is None:
                return {"bucket": item.bucket, "audio_s": item.duration_s, "success": False}
            return r

        # warmup
        await asyncio.gather(*[asyncio.create_task(one(i)) for i in range(n_warmup)])

        # steady
        t_start = time.perf_counter()
        results = await asyncio.gather(*[asyncio.create_task(one(i)) for i in range(n_steady)])
        t_end = time.perf_counter()

    duration = t_end - t_start
    ok = [r for r in results if r.get("success")]
    failed = len(results) - len(ok)

    walls = [r["wall_ms"] for r in ok]
    # rtfx per-request = audio_s / (wall_ms/1000)
    rtfx = [(r["audio_s"] / (r["wall_ms"] / 1000.0)) for r in ok if r["wall_ms"] > 0]
    audio_total = sum(r["audio_s"] for r in ok)
    word_total = sum(r["word_count"] for r in ok)

    # per-bucket breakdown
    buckets: dict[str, list[dict]] = {}
    for r in ok:
        buckets.setdefault(r["bucket"], []).append(r)
    bucket_summary = {}
    for b, recs in buckets.items():
        bw = [r["wall_ms"] for r in recs]
        br = [(r["audio_s"] / (r["wall_ms"] / 1000.0)) for r in recs if r["wall_ms"] > 0]
        bucket_summary[b] = {
            "completed": len(recs),
            "audio_seconds_per_request": recs[0]["audio_s"],
            "wall_ms_p50": compute_percentiles(bw)["p50"],
            "wall_ms_p99": compute_percentiles(bw)["p99"],
            "rtfx_p50": compute_percentiles(br)["p50"],
            "rtfx_p99": compute_percentiles(br)["p99"],
            "audio_minutes_per_dollar_p50": audio_minutes_per_dollar(
                compute_percentiles(br)["p50"]
            ),
        }

    return {
        "concurrency": concurrency,
        "num_requests": n_steady,
        "completed": len(ok),
        "failed": failed,
        "duration_s": duration,
        "wall_ms": compute_percentiles(walls),
        "rtfx_per_req": compute_percentiles(rtfx),
        "audio_seconds_processed": audio_total,
        "audio_seconds_per_wall_second": (audio_total / duration) if duration > 0 else 0.0,
        "request_throughput": (len(ok) / duration) if duration > 0 else 0.0,
        "total_words": word_total,
        "buckets": bucket_summary,
    }


async def main_async(assets_dir: Path, out_path: Path) -> None:
    corpus = load_audio_corpus(assets_dir)
    print(f"[corpus] loaded {len(corpus)} audio items: {[c.bucket for c in corpus]}", flush=True)

    levels_out = []
    for c in LEVELS:
        print(f"[sweep] c={c} warmup={WARMUP} steady={STEADY}", flush=True)
        r = await run_level(corpus, c, WARMUP, STEADY)
        print(
            f"  c={c}: completed={r['completed']}/{r['num_requests']} "
            f"dur={r['duration_s']:.2f}s "
            f"audio_s={r['audio_seconds_processed']:.1f} "
            f"agg_rtfx={r['audio_seconds_per_wall_second']:.2f} "
            f"per-req rtfx_p50={r['rtfx_per_req']['p50']:.2f} "
            f"wall_p50={r['wall_ms']['p50']:.0f}ms "
            f"wall_p99={r['wall_ms']['p99']:.0f}ms",
            flush=True,
        )
        for b, s in r["buckets"].items():
            print(
                f"    [{b:11s}] n={s['completed']:2d} "
                f"audio={s['audio_seconds_per_request']:5.1f}s "
                f"wall_p50={s['wall_ms_p50']:6.0f}ms "
                f"rtfx_p50={s['rtfx_p50']:5.2f} "
                f"min/$_p50={s['audio_minutes_per_dollar_p50']:5.2f}",
                flush=True,
            )
        levels_out.append(r)

    # Pick "peak" by aggregate audio-seconds-per-wall-second (primary speech KPI).
    peak = max(levels_out, key=lambda r: r["audio_seconds_per_wall_second"])
    total_reqs = sum(r["num_requests"] for r in levels_out)
    total_failed = sum(r["failed"] for r in levels_out)
    err_rate = (total_failed / total_reqs) if total_reqs else 0.0

    # rough output_toks_per_s proxy from words/sec
    words_per_s = (peak["total_words"] / peak["duration_s"]) if peak["duration_s"] > 0 else 0.0

    # rtfx percentiles for the peak level
    rtfx_p50 = peak["rtfx_per_req"]["p50"]
    rtfx_p99 = peak["rtfx_per_req"]["p99"]
    agg_rtfx = peak["audio_seconds_per_wall_second"]

    # duration_buckets list for extensions.audio
    duration_buckets = []
    for b, s in peak["buckets"].items():
        duration_buckets.append({
            "bucket": b,
            "audio_seconds_per_request": s["audio_seconds_per_request"],
            "completed": s["completed"],
            "wall_ms_p50": s["wall_ms_p50"],
            "wall_ms_p99": s["wall_ms_p99"],
            "rtfx_p50": s["rtfx_p50"],
            "rtfx_p99": s["rtfx_p99"],
            "audio_minutes_per_dollar_p50": s["audio_minutes_per_dollar_p50"],
        })

    doc = envelope()
    doc.update({
        "model": MODEL_BLOCK,
        "engine": ENGINE_BLOCK,
        "infrastructure": INFRA_BLOCK,
        "workload": {
            "use_case": "speech",
            "catalog_id": "transcription-sweep",
            "modality": "audio",
            "dataset": {
                "type": "synthetic-chirp-corpus",
                "source": "scripts/test-assets/{short-3s,medium-10s,long-30s}.wav",
                "buckets_seconds": [3, 10, 30],
                "sample_rate_hz": 16000,
                "channels": 1,
                "format": "wav-pcm16",
                "input_tokens": {"mean": 0},   # transcription endpoint doesn't expose usage
                "output_tokens": {"mean": peak["total_words"] / peak["completed"] if peak["completed"] else 0},
            },
            "load": {
                "type": "concurrency-sweep",
                "levels": LEVELS,
                "num_prompts_per_level": STEADY,
                "warmup_requests": WARMUP,
                "current_level": peak["concurrency"],
            },
            "api": {
                "type": "transcription",
                "streaming": False,
                "endpoint": "/v1/audio/transcriptions",
                "prompt_template": "<multipart audio file>",
            },
        },
        "metrics": {
            "duration_s": peak["duration_s"],
            "completed": peak["completed"],
            "failed": peak["failed"],
            "error_rate": err_rate,
            "e2e_ms": peak["wall_ms"],
            "output_toks_per_s": words_per_s,   # words/s proxy (no usage from endpoint)
            "request_throughput": peak["request_throughput"],
            "total_input_tokens": 0,
            "total_output_tokens": peak["total_words"],
            "max_concurrent_requests": peak["concurrency"],
        },
        "extensions": {
            "modality": "audio",
            "substrate_caveat": INFRA_BLOCK["substrate_deviation"],
            "audio": {
                "rtfx_p50": rtfx_p50,
                "rtfx_p99": rtfx_p99,
                "rtfx_aggregate": agg_rtfx,
                "ttfw_ms": None,    # non-streaming run; TTFW requires SSE
                "ttfw_note": "non-streaming; transcription endpoint returns full text in single response",
                "audio_seconds_processed": peak["audio_seconds_processed"],
                "audio_minutes_per_dollar": audio_minutes_per_dollar(agg_rtfx),
                "audio_minutes_per_dollar_per_req_p50": audio_minutes_per_dollar(rtfx_p50),
                "on_demand_price_per_hr_usd": ON_DEMAND_PRICE_PER_HR,
                "duration_buckets": duration_buckets,
            },
            "output_tokens_proxy": "word_count_of_response_text (transcription endpoint does not expose usage)",
            "notes": (
                "Stage 6 transcription concurrency sweep on Voxtral-Mini-3B BF16 (vLLM 0.19.1). "
                "Synthetic chirp corpus (3s/10s/30s) — perf-only; LibriSpeech WER gate deferred. "
                "Headline metrics from peak-aggregate-rtfx level. Per-bucket breakdown in "
                "extensions.audio.duration_buckets. ttft_ms/tpot_ms/itl_ms are null-filled per "
                "non-streaming + non-token-stream convention (transcription endpoint emits one "
                "JSON response with the full transcript)."
            ),
            "sweep_levels": [
                {
                    "concurrency": r["concurrency"],
                    "completed": r["completed"],
                    "failed": r["failed"],
                    "duration_s": r["duration_s"],
                    "audio_seconds_processed": r["audio_seconds_processed"],
                    "rtfx_aggregate": r["audio_seconds_per_wall_second"],
                    "rtfx_p50": r["rtfx_per_req"]["p50"],
                    "rtfx_p99": r["rtfx_per_req"]["p99"],
                    "wall_ms_p50": r["wall_ms"]["p50"],
                    "wall_ms_p99": r["wall_ms"]["p99"],
                    "request_throughput": r["request_throughput"],
                    "audio_minutes_per_dollar": audio_minutes_per_dollar(
                        r["audio_seconds_per_wall_second"]
                    ),
                }
                for r in levels_out
            ],
        },
    })
    write_artifact(out_path, doc)
    print(f"ARTIFACT: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, required=True,
                    help="Directory containing short-3s.wav, medium-10s.wav, long-30s.wav")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.assets_dir, args.out))


if __name__ == "__main__":
    main()
