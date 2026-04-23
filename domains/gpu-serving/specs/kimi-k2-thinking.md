# Kimi K2-Thinking Serving Benchmark

## Status: DRAFT (2026-04-07)

## Overview

Deploy Moonshot AI's **Kimi K2-Thinking** (`moonshotai/Kimi-K2-Thinking`) on p5e.48xlarge (8× H200) to benchmark text-only reasoning and tool-calling performance with native INT4 QAT quantization.

K2-Thinking is the text-only reasoning variant of the Kimi K2 family. It shares the same 1T MoE + MLA architecture as K2.5 but drops the 400M MoonViT vision encoder and ships with native INT4 quantization-aware training (higher quality than K2.5's post-hoc compressed-tensors INT4).

**Why K2-Thinking over K2.5:**
- Text-only — no vision encoder overhead (we don't need multimodality)
- Native INT4 QAT — better quantization quality than compressed-tensors
- Same `kimi_k2` model_type in vLLM/SGLang — simpler config (no `--mm-encoder-tp-mode`)
- Always-on extended chain-of-thought (96-128K thinking tokens) — purpose-built for reasoning and tool orchestration

**Why H200 (p5e) over Blackwell:**

| Capability | H200 (sm_90) | B200 (sm_100) | RTX PRO 6000 (sm_120) |
|---|:---:|:---:|:---:|
| FlashMLA (decode) | Full | Sparse only | None |
| DeepGEMM MoE | Yes | Yes (CUDA 12.9+) | None |
| FP8 KV cache | Yes | Yes (post fix) | None (BF16 only) |
| INT4 Marlin | Works | Untested | Broken |
| Official support | Reference HW | Not documented | Not documented |
| Maturity | Production | Bleeding edge | Not viable |

**Evolution from kimi-k2.5 spec:**
- Original spec (Feb 2026) benchmarked K2.5 on p5e with vLLM v0.15.1 + LMCache
- LMCache is now **blocked for all MLA models** (shape mismatch bug — issues #2881, #2947, #2636)
- vLLM `kimi_k2` reasoning/tool parser now in stable (v0.19.0+), was nightly-only before
- Tool parser has known bugs (8KB argument truncation, token leakage — issue #37184)
- This spec uses the text-only model, drops vision, and targets vLLM v0.19+ or SGLang nightly

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (capacity block)
- **Primary Instance**: p5e.48xlarge (8× H200 141GB HBM3e, NVLink 4 / NVSwitch)
- **Region**: us-east-2 (capacity block availability)
- **System Nodes**: 2× m6i.large

### 1a. GPU & NCCL Pre-Flight

Standard pre-flight per template. H200 NVSwitch topology is well-proven — no known NCCL issues.

| Check | Expected |
|---|---|
| GPU count | 8× H200 |
| NVLink topology | All 8 GPUs via NVSwitch |
| NCCL all_reduce bus BW | > 450 GB/s |
| ECC errors (uncorrected) | 0 |

### 2. Model

- **Model ID**: `moonshotai/Kimi-K2-Thinking`
- **Architecture**: `kimi_k2` — MoE + MLA (Multi-head Latent Attention)
  - 1T total params, 32B active per token
  - 384 experts (8 active + 1 shared), 61 layers
  - Hidden size: 7168, MLA kv_lora_rank: 512, qk_rope_head_dim: 64
  - Vocabulary: 163,840 tokens
- **Context Length**: 256K tokens
- **Thinking**: Always-on (96-128K thinking tokens, not togglable)
- **Quantization**: Native INT4 QAT (quantization-aware training)
- **Format**: safetensors
- **License**: Check HuggingFace model card
- **Modality**: Text-only (no vision encoder)

#### Serving Engine Options

**Option A — vLLM (v0.19.0+):**

```bash
vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-prefix-caching \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests
```

No `--mm-encoder-tp-mode` needed (text-only model).

**Option B — SGLang (nightly):**

```bash
python -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2-Thinking \
  --tp 8 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --host 0.0.0.0
```

Requires: `pip install nvidia-cudnn-cu12==9.16.0.29`

#### Parallelism Strategy

| Config | TP | GPUs | Notes |
|---|---|---|---|
| `tp8-h200` | 8 | 8× H200 | Full NVSwitch, reference config from Moonshot |

Moonshot's deployment guide recommends TP16 minimum for K2-Instruct (BF16). INT4 QAT halves memory — TP8 should fit on 8× H200 (1,128 GB total VRAM).

#### Attention Backend

- **H200**: FlashMLA (full — sparse + dense decode, sparse prefill)
- **KV cache**: FP8 KV cache supported on H200
- **DeepGEMM**: ~15 min JIT cold start expected (same architecture family as GLM-5 on B200)

### 3. Networking

- **VPC**: Same as existing kimi-k2.5 blueprint infrastructure
- **Access**: Private subnets, port-forward for benchmarks
- **VPC Endpoints**: S3, ECR API, ECR DKR, STS, CloudWatch Logs

### 4. Storage

- **Model Weights**: S3 bucket, staged to NVMe or FSx at deploy time
- **KV Cache**: GPU VRAM + native vLLM/SGLang prefix caching only
  - LMCache is **blocked** for MLA models (shape mismatch bug)
  - For hierarchical caching, use SGLang's native HiCache (`--enable-hierarchical-cache`)
  - HiCache proven on GLM-5 (same MLA architecture): 71% throughput gain at 64 concurrency

#### KV Cache Strategy

| Tier | Backend | Capacity | Notes |
|---|---|---|---|
| GPU VRAM | HBM3e | ~500-600 GB (after model weights) | Primary |
| HiCache (SGLang) | NVMe/CPU | Configurable via `--hicache-size` | Proven on MLA models |
| vLLM prefix caching | GPU VRAM | Automatic | 76-80% hit rate (from K2.5 benchmarks) |
| LMCache | Any | N/A | **BLOCKED** — MLA shape mismatch |

### 5. Monitoring

- **Prometheus**: 1s scrape interval on vLLM/SGLang `/metrics`
- **Key metrics**: `kv_cache_usage_percent`, `prefix_cache_hit_rate`, `num_preemptions_total`, TTFT/ITL histograms

---

## Benchmark Design

Reuses the coding agent economics framework from the glm5-hyperpod spec, adapted for K2-Thinking's extended reasoning.

### Priority Tiers

```
P0 (must-have): Smoke test + tool-call validation + reasoning quality    ~30 min
P1 (must-have): Agent pressure testing + swarm capacity                  ~2 hrs
P2 (should-have): KV cache comparison (prefix caching vs HiCache)       ~1 hr
P3 (should-have): Economics analysis                                     ~30 min
Total budget: ~4 hrs
```

### P0: Smoke Test + Tool-Call Validation

| Step | Test | Config |
|---|---|---|
| 0a | Health check | Model loads, `/v1/chat/completions` responds |
| 0b | Basic inference | 1K input / 512 output, verify thinking tokens in response |
| 0c | BFCL tool-call accuracy | 200 scenarios, 5 categories |
| 0d | Reasoning quality | Math/code problems, compare thinking depth vs K2.5 |

**Tool-call known issues** (vLLM #37184):
- 8KB argument buffer limit — large tool arguments get truncated
- Token leakage between reasoning and tool channels
- Multiple fix PRs open, none merged as of Apr 7, 2026
- Monitor: if BFCL < 70%, try SGLang as fallback

**Gate**:
- BFCL >= 75% → proceed to P1
- BFCL 70-75% → caution, batch/swarm only
- BFCL < 70% → try SGLang, if still < 70% → STOP

### P1: Agent Swarm Pressure Testing

Same methodology as kimi-k2.5 spec — 12 realistic coding agent scenarios.

| Step | Concurrent Agents | Purpose |
|---|---|---|
| 1b-1 | 4 | Baseline |
| 1b-2 | 8 | Light load |
| 1b-3 | 16 | Moderate |
| 1b-4 | 32 | Heavy |
| 1b-5 | 48 | Stress |
| 1b-6 | 64 | Overload |
| 1b-7 | 96 | Ceiling test |

**K2-Thinking note**: Extended thinking (96-128K tokens) means each request consumes significantly more KV cache and GPU time than K2.5's togglable thinking mode. Expect lower SLO-max concurrent agents but potentially higher quality per turn.

**SLO thresholds**:
- TTFT p99 < 5000ms (higher than K2.5's 2000ms — thinking tokens take time)
- ITL p99 < 100ms
- Error rate < 1%
- No preemptions

### P2: KV Cache Comparison

| Config | Engine | Caching | Purpose |
|---|---|---|---|
| A | vLLM v0.19+ | Native prefix caching | Baseline |
| B | SGLang nightly | RadixAttention | SGLang baseline |
| C | SGLang nightly | HiCache (NVMe) | Hierarchical caching |

**Comparison metrics** at P1's SLO-max concurrency:
- TTFT p50/p99
- Throughput (tok/s)
- KV cache utilization
- Prefix cache hit rate

### P3: Economics

Same framework as glm5-hyperpod spec.

| Metric | K2-Thinking (measured) | K2.5 (from prev benchmarks) | Claude Sonnet 4.6 |
|---|---|---|---|
| BFCL accuracy | TBD | 100% | ~88% |
| SLO-max concurrent agents | TBD | N/A (not tested) | Unlimited |
| TTFT p99 at SLO-max | TBD | ~4400ms | ~500ms |
| Cost/engineer/month | TBD | TBD | ~$205 |

---

## Comparison: K2-Thinking vs K2.5 (from previous benchmarks)

Reference data from the kimi-k2.5 blueprint (Feb 2026, vLLM v0.15.1, p5e):

| Workload | K2.5 TTFT p50 | K2.5 Throughput | Notes |
|---|---|---|---|
| reasoning_math | 1943ms | 41.2 tok/s | Always used reasoning tokens |
| agentic_tool_use | 820-926ms | 27-30 tok/s | Fastest workload |
| multi_turn_qa | 1216-1565ms | 15-19 tok/s | Benefits from prefix caching |
| code_generation | 2828-4273ms | 18-30 tok/s | Highest E2E latency |
| long_context_rag | 1915-2261ms | 10-14 tok/s | Context-bound |

K2-Thinking's always-on extended thinking should produce:
- Higher TTFT (more thinking tokens before first output)
- Potentially higher quality answers (deeper reasoning)
- Lower throughput (more tokens per request)
- Higher KV cache pressure per request

---

## Success Criteria

### Model Viability

| Metric | Target | Phase |
|---|---|---|
| BFCL tool-call accuracy | >= 75% | P0 |
| Reasoning quality (vs K2.5) | Equal or better on math/code | P0 |
| SLO-max concurrent agents | >= 16 | P1 |
| HiCache throughput gain | >= 30% over prefix-only | P2 |

### Latency SLOs

| Metric | Target | Condition |
|---|---|---|
| TTFT p99 | < 5000ms | At SLO-max concurrency |
| ITL p99 | < 100ms | Streaming decode |
| E2E agent turn | < 45s | Single tool-call round-trip (includes thinking) |
| Error rate | < 1% | At SLO-max concurrency |

---

## Non-Requirements

- Multimodal inference (text-only model)
- Blackwell GPU support (not viable for MLA — see overview)
- LMCache integration (blocked for MLA models)
- Multi-node distributed inference (single p5e.48xlarge)
- BF16 inference (INT4 QAT only)
- Production autoscaling
- Multi-region deployment

---

## Security Requirements

- All storage encrypted (S3 SSE, EBS KMS)
- Private subnets, no public endpoints
- VPC endpoints for AWS service access
- IAM roles with least privilege

---

## Cost Considerations

### Benchmark Session

| Resource | Estimated Cost |
|---|---|
| Capacity block (p5e.48xlarge, 4 hrs) | ~$240-400 |
| EKS control plane | $0.10/hr |
| S3 model storage | ~$15/month |
| **Total session** | ~$250-420 |

### Comparison with K2.5 Results

The K2.5 benchmarks showed LMCache+FSx providing 15-28% throughput improvements on multi-turn and agentic workloads. Since LMCache is now blocked for MLA models, the caching comparison shifts to:
- vLLM native prefix caching (baseline — 76-80% hit rate proven)
- SGLang HiCache (hierarchical NVMe offload — 71% throughput gain on GLM-5)

If HiCache delivers similar gains on K2-Thinking, it would more than compensate for the loss of LMCache.

---

## Known Limitations

1. **Tool parser bugs** (vLLM #37184): 8KB argument truncation, token leakage between reasoning and tool channels. Multiple fix PRs open, none merged. SGLang has similar issues (#20878)
2. **Extended thinking consumes context**: 96-128K thinking tokens per request significantly reduces effective context window for user content. With 256K max context, only ~128-160K available for actual conversation
3. **LMCache blocked for MLA models**: Shape mismatch with MLA KV cache format (issues #2881, #2947, #2636). No fix timeline. HiCache or native prefix caching only
4. **DeepGEMM JIT cold start**: ~15 min on first startup (same as GLM-5). Subsequent restarts faster if CUDA cache persists
5. **K2-Thinking always-on thinking**: Cannot disable thinking mode (unlike K2.5's togglable mode). Every response includes extended reasoning. Not suitable for low-latency simple queries
6. **Model gating**: HuggingFace access may require approval from Moonshot AI
7. **INT4 QAT vs block-FP8**: K2-Instruct ships in block-FP8 (the primary distribution format). K2-Thinking's INT4 QAT may have different accuracy characteristics — validate in P0
8. **Marlin kernel PTX issue on H200**: vLLM #38619 — Marlin MoE repack PTX failure when vLLM wheel compiled with CUDA 12.9 runs on driver 12.8. Ensure matching CUDA versions
9. **FlashMLA FP8 + CUDA graphs**: vLLM #38719 — KV cache corruption from uninitialized slot_mapping during warmup. Fix merged but verify on target vLLM version

---

## Deployment Sequence

```
1. Pre-session
   ├── Download moonshotai/Kimi-K2-Thinking weights → upload to S3
   ├── Verify HuggingFace access (model may be gated)
   └── Reserve capacity block (p5e.48xlarge, us-east-2)

2. Infrastructure (reuse kimi-k2.5 blueprint Terraform where possible)
   ├── EKS cluster (v1.31+)
   ├── GPU node (p5e.48xlarge via capacity block)
   ├── System nodes (2× m6i.large)
   ├── FSx Lustre (for model staging + HiCache backend)
   └── Prometheus monitoring

3. GPU pre-flight
   └── nvidia-smi, topology, NCCL all_reduce

4. Model deployment
   ├── Deploy vLLM v0.19+ with kimi_k2 parsers
   ├── Verify thinking tokens in response
   └── If tool-call accuracy < 70%, switch to SGLang

5. Benchmark execution
   ├── P0: Smoke test + BFCL + reasoning quality     (~30 min)
   │   └── GATE: BFCL >= 75%
   ├── P1: Agent swarm pressure testing               (~2 hrs)
   ├── P2: KV cache comparison (prefix vs HiCache)    (~1 hr)
   └── P3: Economics analysis                          (~30 min)

6. Teardown
   ├── Destroy capacity block resources
   └── Keep S3 model weights for next session
```

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/kimi-k2-thinking/lessons.md`, `blueprints/kimi-k2-thinking/results/`, etc.
