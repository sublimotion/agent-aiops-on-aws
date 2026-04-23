# Voice Agent Pipeline on HyperPod — Reference Architecture Spec

## Status: DRAFT (2026-04-16)

## Overview

Deploy and benchmark an **end-to-end voice agent architecture** on SageMaker HyperPod (EKS) using best-in-class open-weight models. Two deployment tiers: (1) a **cascaded pipeline** (ASR + LLM + TTS) orchestrated by LiveKit Agents, and (2) a **single-model speech-to-speech** alternative (Qwen2.5-Omni-7B). Goal is to establish a reusable reference architecture for real-time voice AI on HyperPod with sub-800ms end-to-end latency.

**Why this matters:** AR codec-based TTS is converging with LLM serving — models like Orpheus 3B are structurally identical to text LLMs and benefit from vLLM/SGLang KV cache, continuous batching, and prefix caching. HyperPod's Karpenter autoscaling + scale-to-zero is uniquely suited for voice traffic (3-10x peak/trough following business hours).

**Primary goals:**
1. Validate the cascaded pipeline achieves < 800ms end-to-end latency (TTFA)
2. Benchmark Orpheus 3B TTS on vLLM (first AR TTS served via LLM inference stack)
3. Compare cascaded pipeline vs Qwen2.5-Omni-7B single-model on latency, quality, and cost
4. Measure concurrent voice session capacity per GPU
5. Produce a deployable reference architecture with LiveKit Agents orchestration

---

## Model Selection Rationale

### ASR: NVIDIA Parakeet-TDT 0.6B v3

| Attribute | Value |
|-----------|-------|
| **Model ID** | `nvidia/parakeet-tdt-0.6b-v3` |
| **Parameters** | 600M (FastConformer-TDT) |
| **License** | CC-BY-4.0 |
| **WER** | 6.34% avg (Open ASR Leaderboard), 1.93% LibriSpeech clean |
| **Languages** | 25 European languages with auto-detection |
| **Streaming** | Yes (chunked streaming inference) |
| **VRAM** | ~2 GB |
| **Latency** | RTFx 3,386x (batch 128) |
| **Serving** | NVIDIA NeMo / Triton |

**Why Parakeet over Whisper:** Best published WER among open models, native streaming (Whisper requires 30s chunking), multilingual, CC-BY-4.0 license. Whisper Large-v3-Turbo (809M, MIT) is the fallback if NeMo integration is problematic.

**Runner-up:** Whisper Large-v3-Turbo — 7.83% WER, 99 languages, MIT license, massive ecosystem. No native streaming but well-understood chunking patterns.

### TTS: Orpheus TTS 3B

| Attribute | Value |
|-----------|-------|
| **Model ID** | `canopylabs/orpheus-tts-0.1-finetune-prod` |
| **Parameters** | 3B (Llama-3.2-3B-Instruct base) |
| **License** | Apache 2.0 |
| **Architecture** | Autoregressive LLM generating SNAC codec tokens |
| **VRAM** | ~6-8 GB FP16 |
| **Streaming** | Yes (100-200ms TTFA) |
| **KV Cache** | **YES** — structurally identical to text LLM, servable via vLLM |
| **Emotion tags** | `<laugh>`, `<sigh>`, `<gasp>`, `<chuckle>`, `<cough>`, `<yawn>` |

**Why Orpheus:** Apache-2.0 license + Llama-based architecture = deploy on vLLM/SGLang with standard KV cache optimization, continuous batching, and prefix caching. This is the key technical insight — AR codec TTS benefits from the exact same serving stack as text LLMs. Emotion tags enable expressive voice agents.

**Alternatives considered:**

| Model | Params | License | AR/KV Cache | Why Not Primary |
|-------|--------|---------|-------------|-----------------|
| Fish Speech S2 Pro | 4.4B | CC-BY-NC-SA | Yes | Non-commercial license |
| Dia 1.6B (Nari Labs) | 1.6B | Apache 2.0 | **No** (NAR) | No KV cache benefit, no streaming, batch-only |
| Sesame CSM-1B | 1B | Apache 2.0 | Yes | English only, needs fine-tuning for voices, less mature |
| Kokoro 82M | 82M | Apache 2.0 | No (non-AR) | Quality ceiling for voice agents, good as lightweight fallback |
| F5-TTS | ~500M | CC-BY-NC | No (diffusion) | NC license, no KV cache benefit |
| Parler TTS 2.2B | 2.2B | Apache 2.0 | Partial | Less natural for conversation, more for controlled generation |

### LLM (Reasoning): Configurable — Default Qwen3-8B

The LLM component is swappable. Default to **Qwen3-8B** for the reference architecture (good tool calling, fits single GPU, Apache 2.0). Users can substitute any vLLM-compatible model.

| Attribute | Value |
|-----------|-------|
| **Model ID** | `Qwen/Qwen3-8B` |
| **Parameters** | 8B |
| **License** | Apache 2.0 |
| **VRAM** | ~16 GB FP16, ~8 GB FP8 |
| **Tool calling** | Yes (`--tool-call-parser qwen3_xml`) |
| **Streaming** | Yes |

### End-to-End: Qwen2.5-Omni-7B

| Attribute | Value |
|-----------|-------|
| **Model ID** | `Qwen/Qwen2.5-Omni-7B` |
| **Parameters** | 7B (Thinker-Talker architecture) |
| **License** | Apache 2.0 |
| **Architecture** | Single model: audio/text/image/video in → text + speech out |
| **VRAM** | ~31 GB BF16 (flash_attention_2), scales to ~60 GB for long inputs |
| **Streaming** | Yes (chunked input, streaming speech output) |
| **ASR WER** | 1.6-3.5% LibriSpeech (SOTA for multimodal) |
| **KV Cache** | Yes (autoregressive) |

**Why Qwen2.5-Omni:** Only mature Apache-2.0 end-to-end model with both audio input AND audio output. Single model simplicity eliminates inter-model latency. Requires HF Transformers v4.51.3+ with Qwen2.5-Omni preview branch.

**Alternatives considered:**

| Model | Params | License | Why Not Primary |
|-------|--------|---------|-----------------|
| Moshi (Kyutai) | 7B | CC-BY-4.0 | Full-duplex is unique but quality below Qwen2.5-Omni for general conversation |
| GLM-4-Voice-9B | 9B | Custom | Not truly single-model (3-component pipeline), restrictive license |
| Ultravox v0.5 | 8-70B | MIT | Audio-in only, no audio output — needs separate TTS |

---

## Components

### 1. Compute

- **Platform**: SageMaker HyperPod with EKS orchestrator
- **Region**: TBD (us-east-2 or us-west-2, dependent on FTP availability)

#### Instance Strategy

| Tier | Instance | GPUs | VRAM | Use Case |
|------|----------|------|------|----------|
| **Tier A: Pipeline** | ml.g6e.12xlarge | 4x L40S | 4x 48 GB = 192 GB | ASR + TTS co-located (1 GPU), LLM dedicated (1 GPU), 2 spare |
| **Tier B: Pipeline** | ml.g5.12xlarge | 4x A10G | 4x 24 GB = 96 GB | Budget option: ASR+TTS (1 GPU), LLM-8B FP8 (1 GPU) |
| **Tier C: E2E** | ml.g6e.4xlarge | 1x L40S | 48 GB | Qwen2.5-Omni-7B BF16 (31 GB) |
| **Tier D: Scale** | ml.p5e.48xlarge | 8x H200 | 8x 141 GB | Multi-session scaling benchmark, 200+ concurrent |

**Primary recommendation: Tier A (ml.g6e.12xlarge)** — most practical for the reference architecture. L40S has 48 GB VRAM, enough headroom for ASR (2 GB) + TTS (8 GB) + LLM (16 GB) with room for KV cache.

- **Scaling**: Karpenter with min 1 (warm replica, non-negotiable for voice), max based on concurrent session target
- **Scale-to-zero**: Enabled for off-hours with 1 warm replica exception

### 1a. GPU Pre-Flight Validation

| Check | Expected Result |
|-------|-----------------|
| GPU count | Per instance type (1x or 4x L40S) |
| GPU driver | 535+ (CUDA 12.1+) |
| ECC errors | 0 uncorrected |
| GPU thermals | < 85C idle |
| VRAM available | Per spec above |

Note: No multi-GPU TP required for any voice model. NCCL pre-flight only needed if using TP for the LLM component on Tier D.

### 2. Models — Serving Configuration

#### 2a. Parakeet ASR (NeMo/Triton)

```bash
# Triton Inference Server with NeMo ASR
# Model repository structure:
# models/
#   parakeet-tdt/
#     1/
#       model.nemo
#     config.pbtxt

tritonserver --model-repository=/models \
  --backend-config=nemo,config_path=/models/parakeet-tdt/1/model_config.yaml \
  --http-port 8001 \
  --grpc-port 8004 \
  --metrics-port 8005
```

**Streaming protocol**: gRPC bidirectional streaming. Client sends 80ms audio chunks, server emits partial transcripts.

**Fallback (Whisper Turbo):**

```bash
# faster-whisper with CTranslate2 backend
pip install faster-whisper
# Serve via custom FastAPI WebSocket endpoint
# ~2 GB VRAM, INT8 quantization, chunked 30s processing
```

#### 2b. Orpheus TTS on vLLM

```bash
vllm serve canopylabs/orpheus-tts-0.1-finetune-prod \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8002
```

**Key insight**: Orpheus generates SNAC codec tokens autoregressively, identical to text LLM token generation. vLLM's continuous batching, paged KV cache, and prefix caching all apply directly. The SNAC decoder (codec → waveform) runs on CPU or a small GPU allocation.

**Streaming TTS flow**:
1. vLLM generates codec tokens in streaming mode
2. Codec tokens are decoded to audio chunks (50-200ms) via SNAC decoder
3. Audio chunks streamed to client via WebSocket

**Alternative (llama.cpp)**: Orpheus also runs on llama.cpp for CPU-only or edge deployment.

#### 2c. Qwen3-8B LLM on vLLM

```bash
vllm serve Qwen/Qwen3-8B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
```

#### 2d. Qwen2.5-Omni-7B (E2E Alternative)

```bash
# Requires HF Transformers with Qwen2.5-Omni support
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate flash-attn

# Custom serving script (not yet supported by vLLM)
python serve_qwen_omni.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --dtype bfloat16 \
  --flash-attention2 \
  --host 0.0.0.0 \
  --port 8000
```

**Note**: Qwen2.5-Omni is NOT yet supported by vLLM. Requires custom HF Transformers serving. This is a known limitation — vLLM support expected in future releases given the model's popularity.

### 3. Orchestration — LiveKit Agents

```python
# LiveKit Agent with custom STT/TTS plugins pointing to local models
from livekit.agents import Agent, AgentSession
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero  # VAD

agent = VoicePipelineAgent(
    vad=silero.VAD.load(),
    stt=CustomParakeetSTT(url="localhost:8004"),  # gRPC to Triton
    llm=CustomVLLMLLM(url="http://localhost:8000/v1"),  # OpenAI-compatible
    tts=CustomOrpheusTTS(url="http://localhost:8002/v1"),  # vLLM + SNAC decoder
)
```

**Alternative**: Pipecat (Daily) — broader plugin ecosystem, similar architecture. Use if LiveKit WebRTC transport isn't needed.

### 4. Networking

- **VPC**: Standard HyperPod VPC with private subnets
- **Access**: SSM Session Manager for node access, NLB for WebSocket ingress
- **Endpoints**: S3, ECR, STS, CloudWatch Logs, SSM
- **WebSocket**: NLB with TCP listener (not ALB — ALB adds latency for WebSocket)
- **Protocols**: WebSocket (full-duplex voice), gRPC (ASR streaming), HTTP (health checks)

### 5. Storage

- **Model Storage**: S3 → NVMe staging (same pattern as other HyperPod blueprints)
  - Parakeet: ~1.2 GB
  - Orpheus 3B: ~6 GB
  - Qwen3-8B: ~16 GB (FP16) or ~8 GB (FP8)
  - Qwen2.5-Omni-7B: ~14 GB
  - **Total**: ~37 GB (all models)
- **NVMe caching**: Fast reload from instance store on cold start

---

## Architecture Diagrams

### Tier 1: Cascaded Pipeline

```
                    WebSocket (full-duplex)
                         |
                   [LiveKit Agent]
                   /      |      \
            [Silero VAD]  |       |
                  |       |       |
          Audio chunks    |    Audio out
                  |       |       |
         [Parakeet ASR]   |  [Orpheus TTS]     ← vLLM (KV cache, batching)
         (Triton gRPC)    |  (vLLM :8002)
              |           |       ^
           Text           |    Codec tokens → SNAC decoder → PCM audio
              |           |
         [Qwen3-8B LLM]  |
         (vLLM :8000)-----+
              |
         Tool calls / reasoning
```

**Latency budget** (target < 800ms e2e):

| Component | Target | Notes |
|-----------|--------|-------|
| VAD + audio chunk | 80-100ms | Silero VAD, 80ms frame |
| ASR (Parakeet streaming) | 80-150ms | First partial transcript |
| LLM TTFT | 100-200ms | Qwen3-8B, prefix-cached system prompt |
| TTS TTFA | 100-200ms | Orpheus first codec tokens → SNAC decode |
| Network overhead | 20-50ms | Intra-pod, NVMe-backed |
| **Total** | **380-700ms** | Within 800ms budget |

### Tier 2: Single-Model E2E

```
          WebSocket (full-duplex)
                  |
          [LiveKit Agent]
                  |
         [Qwen2.5-Omni-7B]
         (HF Transformers :8000)
                  |
          Text + Speech out
```

---

## Benchmark Design

### Phase 0: Model Validation (1 hour)

| Step | Test | Expected |
|------|------|----------|
| 0a | Parakeet ASR health + test transcription | Correct transcript, < 200ms |
| 0b | Orpheus TTS health + test synthesis | Coherent audio, < 300ms TTFA |
| 0c | Qwen3-8B LLM health + tool call | Correct response |
| 0d | Qwen2.5-Omni E2E test | Audio in → audio + text out |
| 0e | Pipeline integration (ASR → LLM → TTS) | Full round-trip < 2s |

### Phase 1: Latency Profiling (2 hours)

**P1a: Component Latency Isolation**

| Component | Metric | Concurrency 1 | Concurrency 10 | Concurrency 50 |
|-----------|--------|---------------|----------------|-----------------|
| Parakeet ASR | Transcript latency (ms) | TBD | TBD | TBD |
| Orpheus TTS | TTFA (ms) | TBD | TBD | TBD |
| Orpheus TTS | RTF | TBD | TBD | TBD |
| Qwen3-8B | TTFT (ms) | TBD | TBD | TBD |
| Pipeline E2E | Total latency (ms) | TBD | TBD | TBD |
| Qwen2.5-Omni E2E | Total latency (ms) | TBD | TBD | TBD |

**P1b: Orpheus TTS on vLLM — Throughput Sweep**

Treat Orpheus as an LLM workload and run standard vLLM benchmarks:

```bash
for QPS in 0.5 1.0 2.0 4.0 8.0 16.0; do
  vllm bench serve \
    --model canopylabs/orpheus-tts-0.1-finetune-prod \
    --base-url http://localhost:8002 \
    --dataset-name random \
    --random-input-len 128 --random-output-len 1024 \
    --num-prompts 100 --request-rate $QPS \
    --save-result --result-dir /results \
    --result-filename "orpheus_qps${QPS}.json"
done
```

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | Throughput (tok/s) | RTF |
|-----|--------------|--------------|-------------------|-----|
| 0.5 | TBD | TBD | TBD | TBD |
| 1.0 | TBD | TBD | TBD | TBD |
| 2.0 | TBD | TBD | TBD | TBD |
| 4.0 | TBD | TBD | TBD | TBD |
| 8.0 | TBD | TBD | TBD | TBD |
| 16.0 | TBD | TBD | TBD | TBD |

**P1c: KV Cache / Prefix Caching for TTS**

Test prefix caching benefit when multiple TTS requests share system prompt + voice conditioning:

| Prefix Len | TTFA cold (ms) | TTFA warm (ms) | Cache Speedup |
|-----------|----------------|----------------|--------------|
| Voice prompt 3s (~64 tokens) | TBD | TBD | TBD |
| Voice prompt 10s (~200 tokens) | TBD | TBD | TBD |

### Phase 2: Concurrent Session Scaling (2 hours)

Simulate N concurrent voice sessions (each: continuous ASR + periodic LLM + streaming TTS):

| Concurrent Sessions | GPU Util (%) | P95 E2E Latency (ms) | Failures | GPU |
|--------------------|-------------|----------------------|----------|-----|
| 1 | TBD | TBD | TBD | 1x L40S |
| 5 | TBD | TBD | TBD | 1x L40S |
| 10 | TBD | TBD | TBD | 1x L40S |
| 20 | TBD | TBD | TBD | 2x L40S |
| 50 | TBD | TBD | TBD | 4x L40S |

### Phase 3: Quality Evaluation (1 hour)

**P3a: ASR Quality** — LibriSpeech test-clean/test-other WER comparison:

| Model | WER (clean) | WER (other) |
|-------|-------------|-------------|
| Parakeet-TDT v3 | TBD | TBD |
| Whisper Turbo (fallback) | TBD | TBD |

**P3b: TTS Quality** — MOS estimation (UTMOS or similar):

| Model | MOS (estimated) | Naturalness | Expressiveness |
|-------|----------------|-------------|----------------|
| Orpheus 3B | TBD | TBD | TBD |
| Qwen2.5-Omni TTS | TBD | TBD | TBD |

**P3c: Voice Agent Conversation Quality** — 10 scripted multi-turn conversations:

| Scenario | Pipeline Score | E2E Score |
|----------|---------------|-----------|
| Customer support (simple) | TBD | TBD |
| Technical troubleshooting | TBD | TBD |
| Appointment scheduling | TBD | TBD |
| Information retrieval | TBD | TBD |

---

## Success Criteria

| Metric | Target | Phase |
|--------|--------|-------|
| Pipeline E2E latency (P50) | < 800ms | P1a |
| Orpheus TTS TTFA | < 200ms | P1a |
| Parakeet ASR first transcript | < 150ms | P1a |
| Orpheus vLLM throughput at QPS 4.0 | > 500 tok/s | P1b |
| KV cache speedup for voice conditioning | >= 1.5x | P1c |
| Concurrent sessions per L40S (< 1s E2E P95) | >= 10 | P2 |
| Parakeet WER (LibriSpeech clean) | < 3% | P3a |
| Orpheus MOS | >= 3.5 | P3b |
| Pipeline vs E2E quality parity | Within 10% | P3c |

---

## Non-Requirements

- Training or fine-tuning any voice model (inference-only evaluation)
- Full-duplex / interruption handling (Moshi-style, future work)
- Telephony integration (SIP/PSTN — focus on WebSocket)
- Multi-region deployment
- Production monitoring beyond basic Prometheus
- Speaker verification / voice cloning
- Video understanding (Qwen2.5-Omni supports it but out of scope)

---

## Security Requirements

- **Encryption**: S3 SSE-S3, EBS KMS encryption
- **Network**: Private subnets only, no public IPs on GPU nodes
- **VPC Endpoints**: All AWS service access via endpoints
- **Node Access**: SSM Session Manager only
- **Audio data**: No persistent storage of audio streams (in-memory processing only)
- **IAM**: Least privilege roles

---

## Cost Considerations

| Instance | On-Demand $/hr | Sessions/GPU (est.) | $/1000 sessions/hr |
|----------|---------------|--------------------|--------------------|
| ml.g6e.12xlarge (4x L40S) | ~$15 | ~40 | ~$0.38 |
| ml.g5.12xlarge (4x A10G) | ~$7 | ~20 | ~$0.35 |
| ml.g6e.4xlarge (1x L40S, E2E) | ~$5 | ~10 | ~$0.50 |

**Scale-to-zero savings**: Voice traffic follows business hours. With Karpenter scale-to-zero (keeping 1 warm replica), off-hours cost drops to 1 instance instead of N. Estimated 40-60% cost reduction vs always-on.

---

## Known Limitations

1. **Orpheus on vLLM is uncharted**: No known production deployment of AR TTS via vLLM. May require custom output processing to handle SNAC codec token format.
2. **Qwen2.5-Omni not in vLLM**: Must use HF Transformers serving — no continuous batching, no paged KV cache. Scaling limited.
3. **Parakeet requires NeMo/Triton**: Different serving stack from vLLM. Adds operational complexity vs all-vLLM.
4. **SNAC decoder latency**: Codec token → waveform conversion adds ~10-30ms per chunk. Must run on CPU alongside GPU inference.
5. **Voice quality subjective**: MOS estimation is approximate. Real quality assessment requires human evaluation.

Check `mdc prs orpheus` and `mdc prs parakeet` for upstream PRs that may affect deployment.

---

## Verification Criteria

### Stage 4a — GPU Health

- [ ] All GPUs report ECC enabled, 0 uncorrectable errors
- [ ] No pending row remaps
- [ ] GPU thermals < 85C idle
- [ ] VRAM available matches instance spec

### Stage 5 — Serving Stack

- [ ] Parakeet ASR health endpoint responds (Triton `/v2/health/ready` returns 200)
- [ ] Orpheus TTS health endpoint responds (vLLM `/health` returns 200)
- [ ] LLM health endpoint responds (vLLM `/health` returns 200)
- [ ] Test transcription returns correct text
- [ ] Test synthesis returns playable audio
- [ ] Pipeline round-trip completes (audio in → audio out)

### Stage 6 — Benchmark

- [ ] Pipeline E2E P50 < 800ms at concurrency 1
- [ ] Orpheus TTFA < 200ms at concurrency 1
- [ ] No OOM at 10 concurrent sessions
- [ ] No request timeouts during benchmark

### Stage 7 — Readiness Audit

- [ ] All readiness audit categories pass
- [ ] All verification criteria above checked and recorded
- [ ] Deployment card recommendations followed or overridden with justification

---

> **Note**: Operational artifacts (lessons, results, deployment notes)
> belong in the blueprint directory: `blueprints/voice-agent-hyperpod/`.
