# Voxtral-Mini-3B-2507 on g6e.2xlarge — Lessons

Substrate: EKS / g6e.2xlarge (L40S 48GB, sm_89). Spec primary is g6.xlarge (L4 24GB).
Engine: vLLM 0.19.1 (`vllm/vllm-openai:v0.19.1`).
Date: 2026-05-13.

## Field notes

### audio-deps: `mistral-common[audio]` pip install in command works; no apt-get needed
<!-- captured: 2026-05-13 | stage: 5 -->

The stock `vllm/vllm-openai:v0.19.1` image already has `libsndfile`/`libffi` available
through the Python wheels for `soundfile` (manylinux_2_28) and `soxr`. The bash-prefix
install ran clean in ~30s:
```
pip install --no-cache-dir 'mistral-common[audio]'
```
No need for the `||` fallback or apt-get of `ffmpeg libsndfile1`. Note that
`mistral-common[audio]` pulls `soundfile + soxr`, NOT `pyav`. For pure WAV/FLAC this
is sufficient; MP3 / opus / webm decode would need pyav or ffmpeg added to the image.

**Fix**: Keep the manifest's bash-prefix `pip install 'mistral-common[audio]'` as-is.
First-boot timeline: ~30s pip + ~2 min weights + ~3 min vLLM init = ~5-6 min total
to Ready on a fresh node.

### api-surface: BOTH `/v1/audio/transcriptions` and `/v1/chat/completions` (audio_url) work
<!-- captured: 2026-05-13 | stage: 5 -->

vLLM 0.19.1 registers both routes for VoxtralForConditionalGeneration. The chat-completions
content part uses `{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}`
— matching the OpenAI image_url shape, NOT the `{"type":"audio","path":...}` form
in the spec narrative. The path/data fields described in upstream HF docs are an
older API; vLLM normalized to `audio_url` to mirror its own `image_url` handling.

**Fix**: Smoke-test bundles use `audio_url` content part. If a future client breaks,
log the actual error before reverting.

### transcription-output: chirp audio yields near-empty `.text` — that's fine for perf
<!-- captured: 2026-05-13 | stage: 5 -->

POST to `/v1/audio/transcriptions` with a 5s synthetic chirp returned `{"text":"."}`
(literal period). The endpoint is functional; the model just has nothing speech-like
to transcribe. For perf-only runs we measure RTFx (audio_s / wall_s) regardless of
output content. Quality (WER on real speech) is gated behind LibriSpeech and is
deferred this session.

**Fix**: `_transcription_sweep.py` uses bucket-known audio_duration_s (not response
length) for the RTFx numerator. word_count of response text is logged as
`metrics.output_toks_per_s` proxy but is acknowledged as near-zero on synthetic input.

### chat-audio: model hallucinates plausible content from chirps
<!-- captured: 2026-05-13 | stage: 5 -->

Path 2 smoke (`/v1/chat/completions` with audio_url + "Summarize the audio" prompt)
on a 5s sweep chirp produced: *"The audio appears to be a recording of a car horn
honking, possibly in a parking lot or busy street, with a few seconds of silence
before the next honk."* This is hallucinated — the chirp has no horn, no silence.
Voxtral's audio understanding head will confabulate when given out-of-distribution
input. This is a known LLM failure mode, not a serving bug. Do NOT use chirp audio
for understanding-quality benchmarks.

**Fix**: For Path-2 quality benchmarks, use real public-domain speech samples
(LibriSpeech subset, Mozilla Common Voice clip).

### perf: RTFx scales with concurrency but per-request RTFx degrades
<!-- captured: 2026-05-13 | stage: 6 -->

Aggregate RTFx (audio-s processed / wall-s wall): c=1 → 85, c=4 → 271, c=16 → 482.
Per-request RTFx p50: c=1 → 95, c=4 → 87, c=16 → 41 (drops as queueing kicks in).
At c=16, p99 wall latency is 735ms for a 30s clip — still ~40x real-time. The peak
aggregate RTFx of 482 corresponds to **30,121 audio-min/$ on L40S** at $2.24/hr.
On L4 ($0.80/hr) this would scale linearly with cost: same throughput would yield
~84,000 audio-min/$ if the L4 hits the same RTFx (it probably won't — L40S has 3x
the FP16 TFLOPs — but the per-token cost should still be substantially lower on L4).

**Fix**: For the L4 cell, plan c=4 sweet spot (per-request RTFx still > 80) rather
than c=16 (which sacrifices per-stream latency for aggregate throughput).

### artifact: schema requires `ttft/tpot/itl` even for non-token transcription workloads
<!-- captured: 2026-05-13 | stage: 6 -->

The Common Benchmark Artifact validator rejects payloads without `ttft_ms`,
`tpot_ms`, `itl_ms` keys under `metrics.*`. For a transcription workload these
are meaningless (no token stream). `_common.write_artifact()` injects null-filled
percentile dicts to satisfy the schema. Path-1 transcription is conceptually
"one fat response", not a stream — TTFW (time-to-first-word) is the streaming
analog and lives in `extensions.audio.ttfw_ms` (null this run because non-streaming).

**Fix**: Same pattern as deepseek-ocr (vision-language non-streaming): null-fill
the streaming percentile keys, surface domain-specific metrics in `extensions.*`.
