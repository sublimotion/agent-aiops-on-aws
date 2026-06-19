# Mistral Voxtral 4B — CTO Benchmark Model #5

## Status: DRAFT (2026-05-13)

## Overview

Speech / audio model for contact-centre and voice-surveillance workloads. ~4B parameters. Exercises the streaming-audio path, LibriSpeech-gated quantization, and real-time SLO (latency per utterance, not per token).

Parent: `cto-benchmark-engagement.md`.

**Availability**: confirmed published on HF as of 2026-05-15. Primary model `mistralai/Voxtral-Mini-3B-2507`, Apache-2.0, ~9.5 GB BF16 (3B LLM + audio encoder, ~5B effective). vLLM 0.10.0+ supports `VoxtralForConditionalGeneration` natively (we run 0.19.1; supported). Mistral self-reported baselines (paper arXiv:2507.13264): LibriSpeech-clean WER 1.88%, LibriSpeech-other 4.1%, Earnings22 10.69%, RTFx 109.86.

## Components

### 1. Compute

- **Platform**: EKS 1.32 (or HyperPod if co-located with existing voice-agent spec)
- **Primary instance**: `g6.xlarge` (1× L4 24GB, Ada). 4B BF16 ~8 GB leaves headroom for streaming audio buffers and KV. L4 wins on $/audio-minute and tokens/joule for this model size (TDP 72 W vs 300 W on L40S).
- **FP8-cell instance**: `g6e.xlarge` (1× L40S 48GB, Ada). Required only if the O3 FP8 row is enabled — L4 lacks FP8 tensor cores. BF16 runs on `g6.xlarge`; INT8/INT4 skipped (WER regression typical).
- **Stand-in**: `g5.2xlarge` (1× A10G) if L4 capacity is unavailable at engagement start.
- **Region**: us-east-2

### Cost / performance rationale

On-demand price delta at engagement time (update at run start): `g6.xlarge` ~$0.80/hr vs `g6e.xlarge` ~$1.86/hr. For a 4B speech model that is WER-gated rather than throughput-gated, FP8 typically does not pay back the quality risk — expect most of the matrix to run on `g6.xlarge`, with `g6e.xlarge` reserved for the optional FP8 cell.

### 2. Model

- **Model ID**: `mistralai/Voxtral-Mini-3B-2507`
- **Modality**: audio → text (ASR + audio understanding / Q&A / translation / summarization)
- **Format**: BF16 baseline; FP8 Pareto (INT8/INT4 typically degrade WER badly)
- **Serving**: vLLM 0.19.1 with `--tokenizer_mode mistral --config_format mistral --load_format mistral`. Requires `mistral-common[audio]` (soundfile + pyav) installed in the container.
- **API surface (dual)**:
  - `/v1/audio/transcriptions` — OpenAI Whisper-compat ASR endpoint (multipart upload of audio file)
  - `/v1/chat/completions` with `{"type":"audio","path":...}` content parts — for understanding / Q&A / translation
  - SSE streaming on chat works as standard token stream; transcriptions endpoint streams text after full audio consumed (not frame-in/frame-out — that requires the `Voxtral-Mini-4B-Realtime-2602` variant + `/v1/realtime` WebSocket)
- **Audio input**: 16 kHz Mel internally; flexible input formats via `mistral_common.audio.Audio.from_file` (MP3/WAV/FLAC). Max ~30 min for transcription, ~40 min for understanding
- **Deployment card**: run `mdc get mistral-voxtral-4b --engine vllm` before deploying; the card's `tiers:` block carries canonical configs. Use `mdc tiers:refresh` to validate against latest upstream.

## Benchmark matrix

| O# | Workload card | Sidecar axes | Expected cells |
|----|---------------|--------------|----------------|
| O1 | `concurrency-sweep` | Audio chunks drive KV growth; context axis = chunk count, not text tokens | TBD |
| O2 | `cohost-isolation` | Speech role; expect latency sensitivity to LLM noisy-neighbour load | 4 topologies × 5 roles = 20 |
| O3 | `quantization-pareto` | `--quality-eval librispeech`; tolerance = 0.5pp absolute WER (LOWER is better); one sidecar per precision | 2 precisions (BF16/FP8) |
| O9 | `cold-start` | Record Mel spectrogram + encoder + decoder sub-phases separately | TBD |
| O11 | `power-efficiency` | `--load-fraction` × precision; extension reports joules/audio-minute too | 4 × 2 = 8 |

### Practitioner workloads (beyond the CTO matrix)

Most text-centric cards don't apply to a speech model. Relevant shapes:

| Workload | Card | Speech-specific notes |
|----------|------|------------------------|
| Multi-Turn Chat | `multi-turn-chat` | Applies for speech-to-speech conversational agents (contact-centre) |
| Shared System Prompt | `shared-prefix-multitenant` | ASR persona shared across tenants |
| Production Traffic Mix | `production-mix` | Trace replay of recorded calls (redacted) |
| Long Context Scaling | `concurrency-sweep` with chunk-count axis | Audio-chunk accumulation vs streaming latency |

## Quality baselines

```yaml
quality_baselines:
  librispeech:
    bf16: 0.0188        # WER on LibriSpeech test-clean (Mistral self-report; LOWER better)
    bf16_other: 0.041   # WER on LibriSpeech test-other
    tolerance: 0.005    # 0.5pp absolute WER
```

## Verification criteria

Standard template Stages 4a–7 apply. Engagement-specific additions:

- [ ] Metrics reported as (audio-seconds processed / second) in addition to token throughput
- [ ] Streaming latency per 100 ms audio frame recorded (analogous to TPOT)
- [ ] LibriSpeech gate passes with `lower_is_better` flag set in the quality blob
- [ ] Tier Stack Table filled (T0–T5) per `docs/optimization-stack.md`

## Known limitations

- Existing voice-agent DRAFT spec (`voice-agent-hyperpod.md`) uses Orpheus 3B TTS + Parakeet ASR, **not Voxtral** — this is a separate deployment.
- See Overview for the Voxtral availability caveat.

## Links

- Parent: `cto-benchmark-engagement.md`
- Related (different model): `voice-agent-hyperpod.md`
