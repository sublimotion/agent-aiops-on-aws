---
blueprint: voxtral-4b-eks
spec: domains/gpu-serving/specs/mistral-voxtral-4b.md
status: in_progress
last_stage: 6
last_updated: 2026-05-13T13:30:00Z
substrate: g6e.2xlarge (L40S 48GB) — alt for spec-preferred g6.xlarge (L4 24GB)
stages:
  stage_5_serve: complete
  stage_6_artifact: complete
  stage_7_audit: not_started
---

# Voxtral-Mini-3B-2507 EKS — Progress

## Iteration 1 (2026-05-13)

### Stage 5 — serving
- Manifest applied: `k8s/vllm-voxtral-g6e-2xlarge.yaml` → namespace `cto-voxtral-g6e-2xlarge`.
- Pod Ready in ~6m12s on first boot. Breakdown: ~30s pip install (`mistral-common[audio]` → soundfile + soxr), ~2m weights pull (~10 GB), ~3m vLLM init.
- No system deps required (no apt-get needed). The `||` fallback chain in the manifest was not triggered.
- Both API routes registered: `/v1/audio/transcriptions` AND `/v1/chat/completions` with `audio_url` content parts.
- Smoke test (`scripts/smoke-test.sh`) — both paths HTTP 200:
  - Path 1 (`/v1/audio/transcriptions`, multipart, 5s chirp): returned `text="."` — expected for synthetic chirp, endpoint functional.
  - Path 2 (`/v1/chat/completions`, audio_url + "Summarize the audio"): returned coherent (hallucinated) sentence — endpoint functional.
- Smoke responses: `results/smoke-response-{transcription,understanding}-20260515T132706Z.json`.

### Stage 6 — artifact
- Sweep: c=[1, 4, 16], 5 warmup + 30 steady per level, round-robin 3s/10s/30s WAV chirps.
- Outcome: 90/90 requests succeeded across all levels.
- Headline (peak = c=16): aggregate RTFx **482.02** (audio-s processed per wall-s), p99 wall **735 ms** (for the 30s bucket).
- Per-bucket RTFx_p50 at peak: 3s → 8.66, 10s → 53.11, 30s → 55.27.
- Audio-min/$ at peak (g6e.2xlarge @ $2.24/hr): aggregate ~12,910; per-bucket p50: 3s → 232, 10s → 1,423, 30s → 1,481.
- Sweet-spot finding (per-stream latency): **c=4** keeps per-request RTFx_p50 ≥ 87 (vs ≥ 41 at c=16).
- Artifact: `results/artifacts/voxtral-mini-3b_eks_g6e-2xl_vllm_transcription-sweep_20260515T132713Z.json`.
- Validator: **PASS** (Common Benchmark Artifact schema 1.0.0).

### Surprises
- Voxtral-on-vLLM uses `audio_url` content part shape (mirroring `image_url`), NOT the `{"type":"audio","path":...}` form in the spec narrative. Both API surfaces are listed under registered routes from boot.
- L40S throughput is so high (482x real-time aggregate) that even a 4B model with audio encoder is likely under-utilizing the GPU at single replica; investigate `--max-num-seqs` higher than 16 on next iteration.

### Out of scope this iteration
- LibriSpeech WER quality gate (deferred — corpus not staged).
- TTFW (time-to-first-word) measurement — requires SSE streaming on transcription endpoint; non-streaming this run.
- FP8 row, multi-language, audio understanding workload (Path 2 was smoke-only).
- Spec-preferred L4 substrate.

### Next steps (future iteration)
1. Mirror a small LibriSpeech-clean subset (~50 utterances) to enable `librispeech` quality blob on the artifact.
2. Run streaming transcription via `?stream=true` to measure TTFW (the audio analog of TTFT).
3. Compare on g6.xlarge (L4) — model fits easily; expected per-stream RTFx ~30-50 with proportionally better $/audio-min.
4. Audio understanding benchmark on Path 2 with real public-domain speech samples.
