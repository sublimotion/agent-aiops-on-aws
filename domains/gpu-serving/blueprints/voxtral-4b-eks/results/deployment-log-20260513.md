# Voxtral-Mini-3B Deployment Log — 2026-05-13

Substrate: g6e.2xlarge (L40S 48GB) — spec primary is g6.xlarge (L4 24GB).
Substrate caveat carried in all artifacts.

## Iteration 1

### Stage 5 — apply manifest

- 09:20Z: applied `k8s/vllm-voxtral-g6e-2xlarge.yaml`. Namespace `cto-voxtral-g6e-2xlarge` created. Pod scheduled on `ip-10-0-40-231.us-east-2.compute.internal` (g6e.2xlarge, AL2 GPU AMI).
- Container does `pip install mistral-common[audio]` then `vllm serve` in single bash command. Readiness probe set with `initialDelaySeconds=240, failureThreshold=60` to absorb the install + weights pull.

### Test assets

- Generated 4 deterministic chirp WAVs at 16 kHz mono PCM (3s/5s/10s/30s) under `scripts/test-assets/` via `generate_audio.py`. These are NOT real speech — perf-only proxies. Quality eval (LibriSpeech WER) deferred this session.

### Stage 5 — pod ready

- 09:27Z: pod Ready (~6m12s from apply). pip install of `mistral-common[audio]` installed `soundfile==0.13.1` + `soxr==1.1.0`; no apt-get needed.
- vLLM init logs show `Resolved architecture: VoxtralForConditionalGeneration`, `max_model_len=32768`, both `/v1/audio/transcriptions` and `/v1/audio/translations` routes registered.

### Smoke test — both API paths PASS

- Path 1 `/v1/audio/transcriptions` (multipart, 5s chirp): HTTP 200, `text="."` (expected for synthetic chirp).
- Path 2 `/v1/chat/completions` (audio_url content part): HTTP 200, coherent hallucinated response. Note: vLLM uses `audio_url` content part shape (matching `image_url`), not `{"type":"audio","path":...}` from spec narrative.
- Saved: `results/smoke-response-transcription-20260515T132706Z.json`, `results/smoke-response-understanding-20260515T132706Z.json`.

### Stage 6 — transcription concurrency sweep

- Levels: c=[1, 4, 16]; 5 warmup + 30 steady per level; round-robin 3s/10s/30s chirps (90 steady requests total).
- Result: 90/90 succeeded.

| level | dur (s) | agg RTFx | per-req RTFx_p50 | wall_p50 (ms) | wall_p99 (ms) |
|-------|---------|----------|------------------|---------------|---------------|
| c=1   | 5.05    | 85.15    | 94.64            | 96            | 353           |
| c=4   | 1.59    | 271.03   | 86.82            | 127           | 371           |
| c=16  | 0.89    | 482.02   | 40.79            | 346           | 735           |

Per-bucket p50 at peak (c=16):

| bucket  | wall_p50 | RTFx_p50 | audio-min/$ p50 |
|---------|----------|----------|-----------------|
| 3s      | 202 ms   | 8.66     | 232             |
| 10s     | 161 ms   | 53.11    | 1,423           |
| 30s     | 523 ms   | 55.27    | 1,481           |

- Aggregate audio-min/$ at peak: ~12,910 (g6e.2xlarge @ $2.24/hr).
- Sweet-spot for per-stream latency: **c=4** (per-req RTFx_p50 ≥ 87, half the queue depth of c=16).
- Artifact: `results/artifacts/voxtral-mini-3b_eks_g6e-2xl_vllm_transcription-sweep_20260515T132713Z.json`.
- Validator: **PASS** (schema 1.0.0). All required fields present; `ttft/tpot/itl` null-filled per non-streaming convention.

### Outcome

Stage 5 + Stage 6 complete in iteration 1 (no retries). Single validated artifact emitted. Substrate caveat (L40S vs spec-preferred L4) carried in `infrastructure.substrate_deviation` and `extensions.substrate_caveat`.
