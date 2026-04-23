# Kimi K2.5 Serving Benchmark

## Status: COMPLETE (2026-02-14)

## Overview

Deploy Moonshot AI's **Kimi K2.5** (`moonshotai/Kimi-K2.5`) on p5e.48xlarge (8x H100) to benchmark serving performance across reasoning, coding, agentic, multi-turn, and long-context workloads. Evaluate LMCache + FSx Lustre KV cache offloading vs native vLLM prefix caching.

K2.5 is a 1T MoE multimodal model (32B active per token) with MLA attention, 256K context, togglable reasoning mode, and a 400M MoonViT vision encoder. It achieves 76.8% on SWE-bench Verified and 85.0% on LiveCodeBench v6.

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (capacity block)
- **Primary Instance**: p5e.48xlarge (8x H100 80GB HBM3e, NVLink / NVSwitch)
- **Region**: us-east-2c
- **Capacity Block**: `cr-0950e9f1e415a9b30`
- **System Nodes**: 2x m6i.large

### 1a. GPU & NCCL Pre-Flight

Standard pre-flight per template. H100 NVSwitch topology is well-proven.

| Check | Expected |
|---|---|
| GPU count | 8x H100 |
| NVLink topology | All 8 GPUs via NVSwitch |
| NCCL all_reduce bus BW | > 450 GB/s |
| ECC errors (uncorrected) | 0 |

### 2. Model

- **Model ID**: `moonshotai/Kimi-K2.5`
- **Architecture**: `kimi_k2` — MoE + MLA (Multi-head Latent Attention)
  - 1T total params, 32B active per token
  - 384 experts (8 active + 1 shared), 61 layers
  - Hidden size: 7168, MLA kv_lora_rank: 512, qk_rope_head_dim: 64
  - Vocabulary: 160K tokens
- **Context Length**: 256K tokens
- **Thinking**: Togglable (can be enabled/disabled per request)
- **Quantization**: Compressed-tensors INT4 (4-bit Marlin MoE)
- **Format**: safetensors (64 shards)
- **Modality**: Multimodal (text + vision via 400M MoonViT)

#### Serving Configuration

```bash
vllm serve moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests
```

- **vLLM Version**: v0.15.1
- **Attention Backend**: FLASH_ATTN_MLA
- **Quantization Method**: CompressedTensorsWNA16MarlinMoEMethod
- **Model Loading**: ~25 minutes for 64 safetensor shards across 8x H100

### 3. Networking

- **VPC**: Private subnets, VPC endpoints for S3, ECR, FSx, STS, CloudWatch
- **Access**: Port-forward for benchmarks (`kubectl port-forward svc/vllm-benchmark 30080:8000`)

### 4. Storage

- **Model Weights**: S3 bucket, loaded via vLLM S3 connector
- **FSx Lustre**: ~100 TiB SCRATCH_2 (`fs-06794cdffdbce7e54`) for LMCache KV offloading
- **KV Cache**: GPU VRAM + native prefix caching (76-80% hit rate) + optional LMCache FSx offload

### 5. Monitoring

- **Prometheus**: 1s scrape interval on vLLM `/metrics`
- **Key metrics**: `kv_cache_usage_percent`, `prefix_cache_hit_rate`, `num_preemptions_total`

---

## Benchmark Design

### Workloads

| Workload | Pattern | Description |
|---|---|---|
| `reasoning_math` | Math problems | Reasoning-heavy, high output tokens |
| `code_generation` | Code tasks | Long output sequences |
| `multi_turn_qa` | Conversation | Shared prefix, benefits from caching |
| `long_context_rag` | Retrieval + QA | Long input context, short output |
| `agentic_tool_use` | Tool calling | Rapid back-and-forth, context switching |

### QPS Levels

| Level | Requests/sec |
|---|---|
| Low | 0.5 |
| Medium | 2.0 |
| High | 5.0 |

### Configurations

| Config | Description |
|---|---|
| Baseline | Native vLLM prefix caching only |
| LMCache + FSx | vLLM + LMCache with FSx Lustre KV offload |

---

## Results Summary (2026-02-14)

### Baseline Benchmark

| Workload | QPS | TTFT p50 (ms) | TTFT p99 (ms) | E2E p50 (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|
| reasoning_math | 0.5 | 1943 | 4426 | 3873 | 41.2 |
| reasoning_math | 2.0 | 1971 | 4414 | 4039 | 41.0 |
| reasoning_math | 5.0 | 2038 | 4125 | 3917 | 41.9 |
| code_generation | 0.5 | 4273 | 6195 | 7064 | 25.2 |
| code_generation | 2.0 | 4083 | 7036 | 7064 | 18.2 |
| code_generation | 5.0 | 2828 | 6440 | 7064 | 29.6 |
| multi_turn_qa | 0.5 | 1565 | 2614 | 2702 | 16.8 |
| multi_turn_qa | 2.0 | 1449 | 2586 | 2702 | 18.7 |
| multi_turn_qa | 5.0 | 1216 | 2526 | 2702 | 15.0 |
| long_context_rag | 0.5 | 1915 | 3559 | 3638 | 9.8 |
| long_context_rag | 2.0 | 2244 | 3568 | 3637 | 14.4 |
| long_context_rag | 5.0 | 2261 | 3629 | 3639 | 10.2 |
| agentic_tool_use | 0.5 | 926 | 2720 | 1258 | 29.7 |
| agentic_tool_use | 2.0 | 820 | 1975 | 1099 | 27.2 |
| agentic_tool_use | 5.0 | 889 | 2026 | 1134 | 30.1 |

100% success rate across all workloads and QPS levels. 100% reasoning token inclusion.

### LMCache + FSx Comparison

| Workload | QPS | Metric | Baseline | LMCache+FSx | Change |
|---|---|---|---|---|---|
| agentic_tool_use | medium | Throughput | 27.2 tok/s | 33.9 tok/s | **+24.6%** |
| agentic_tool_use | high | TTFT p50 | 890ms | 800ms | **-10.1%** |
| multi_turn_qa | low | TTFT p50 | 1565ms | 1317ms | **-15.8%** |
| multi_turn_qa | high | Throughput | 15.0 tok/s | 18.1 tok/s | **+20.8%** |
| long_context_rag | high | Throughput | 10.2 tok/s | 12.4 tok/s | **+21.2%** |
| code_generation | medium | Throughput | 18.2 tok/s | 23.4 tok/s | **+28.5%** |
| reasoning_math | low | TTFT p50 | 1944ms | 2129ms | +9.5% (overhead) |

LMCache benefits agentic (+24.6%), multi-turn (+20.8%), RAG (+21.2%), and code gen (+28.5%) workloads. Reasoning math sees slight overhead (compute-bound, not memory-bound).

### Long Context Stress Test

| Context Size | E2E p50 (ms) | Notes |
|---|---|---|
| ~24K tokens | 3095ms | Stable |
| ~36K tokens | 4273ms | Sub-linear scaling |
| ~48K tokens | 5522ms | Cold: 4403ms, Warm: 2480ms (1.8x speedup) |
| ~51K tokens | 2479ms (warm) | Exceeds 32K max_model_len with LMCache |

### Multi-Tenant (50 tenants, LMCache)

| Metric | Value |
|---|---|
| Requests | 300 |
| E2E p50 | 2926ms |
| E2E p99 | 10827ms |
| Cold → Warm speedup | **1.98x** |
| FSx cache size | 37 GB (2,160 files) |

### Cold Start Recovery

FSx cache persists across restarts. Post-restart cold TTFT (1039ms) matches warm TTFT (~1026-1145ms).

---

## Success Criteria

| Criteria | Target | Result | Status |
|---|---|---|---|
| Success rate | 100% | 100% | PASS |
| Reasoning parser | Working | 100% reasoning tokens | PASS |
| Prefix cache hit rate | > 70% | 76-80% | PASS |
| LMCache throughput gain | > 15% on agentic | +24.6% | PASS |
| Cold start recovery | < 2x warm latency | 1.0x (cache reused) | PASS |

---

## Key Findings

1. **Agentic tool use has lowest latency**: TTFT 820-926ms p50, best for coding agent use cases
2. **Prefix caching effective**: 76-80% hit rate, TTFT improves at higher QPS as cache warms
3. **LMCache + FSx adds 15-28% throughput** on multi-turn, agentic, RAG, and code gen workloads
4. **LMCache hurts compute-bound reasoning**: +9.5% TTFT overhead on math reasoning
5. **Sub-linear context scaling**: 2x context increase = ~1.4x latency increase
6. **FSx cache survives restarts**: Cold start recovery is essentially free

---

## Known Limitations

1. **LMCache now blocked for MLA models** (post-benchmark): Shape mismatch bug — issues #2881, #2947, #2636. Future deployments must use native prefix caching or SGLang HiCache
2. **Vision encoder overhead**: `--mm-encoder-tp-mode data` required even for text-only queries
3. **Compressed-tensors INT4**: Post-hoc quantization, lower quality than K2-Thinking's native INT4 QAT
4. **Tool parser bugs** (vLLM #37184, #38579): 8KB argument truncation, token leakage in streaming
5. **Marlin PTX issue** (#38619): Fails when vLLM CUDA 12.9 wheel runs on CUDA 12.8 driver
6. **Model loading**: ~25 min cold start for 64 shards

---

## Non-Requirements

- Multi-node distributed inference (single p5e.48xlarge)
- Production autoscaling
- Multi-region deployment
- Blackwell GPU support (MLA kernels not available)
- SGLang comparison (vLLM-only benchmark)

---

## Cost

| Resource | Cost |
|---|---|
| Capacity block (p5e.48xlarge, ~8 hrs) | ~$480-800 |
| EKS control plane | $0.10/hr |
| FSx Lustre (~100 TiB) | ~$50/session |
| **Total session** | ~$550-870 |

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/kimi-k2.5/results/execution-log.md` for full run details.
