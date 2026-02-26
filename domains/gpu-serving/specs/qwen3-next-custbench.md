# Qwen3-Next Customer Benchmark Spec

## Status: DRAFT (2026-02-25)

## Overview

Reproduce and optimize a customer's Qwen3-Next-80B-A3B-Instruct serving configuration on p5en.48xlarge (8x H200) in us-east-2c. The customer reports 3,612 tok/s throughput but poor latency (TTFT p50 = 940ms, p99 = 5.35s) at 1000 concurrent requests. This benchmark compares their exact config against our optimized config to quantify the impact of prefix caching, chunked prefill, and FP8 weight quantization.

### Parent Blueprint

This is an **addendum** to `domains/gpu-serving/specs/qwen3-next.md`. It reuses the same infrastructure modules and model but tests customer-specific configurations.

### Optimization Objective

```
Primary:   Reproduce customer workload (1000 concurrent, 10K input, 1K output)
Secondary: Quantify latency improvement from our recommended optimizations
Metric:    TTFT p50/p99, ITL p99, throughput tok/s under identical workload
```

---

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **System Nodes**: m6i.xlarge
- **GPU Nodes**: p5en.48xlarge via capacity blocks
  - 8x NVIDIA H200 (141 GB HBM3e each)
- **Region**: us-east-2
- **Availability Zone**: us-east-2c

### 2. Model

- **Model ID**: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Architecture**: Hybrid-attention MoE (80B total / 3B active)
- **Format**: FP8 quantized weights (~80 GB). All configs use the same FP8 checkpoint at `/mnt/nvme/models/qwen3-next-fp8`. The customer's config omits `--quantization fp8` but uses `--kv-cache-dtype fp8`; we assume they are also loading FP8 weights since BF16 would require ~160 GB and exceed TP=4 memory budget.
- **Context Length**: 32,768 tokens for all configs (controlled variable for fair A/B comparison)

### 3. Customer vLLM Image

The customer uses a custom vLLM image based on `vllm/vllm-openai:v0.11.0` with nightly vLLM wheels and `transformers==4.57.1` for Qwen3-Next support. MTP speculative decoding works on their version (blocked on our vLLM 0.16.0rc2).

> **Air-gap note**: The customer image (`docker/Dockerfile.vllm-customer`) installs packages from `wheels.vllm.ai/nightly` and PyPI at build time. It must be built on an internet-connected host and pushed to ECR before the benchmark session. See parent spec's "Air-Gap Deployment Requirements" section.

### 4. Networking

Same as parent blueprint (VPC, NAT, EFA).

---

## Benchmark Configs

### Config A: Customer Baseline (exact reproduction)

Reproduce customer's exact flags:

```yaml
extra_vllm_args:
  - "--gpu-memory-utilization"
  - "0.90"
  - "--max-model-len"
  - "32768"
  - --kv-cache-dtype
  - "fp8"
  - "--tokenizer-mode"
  - "auto"
  - "--tool-call-parser"
  - "hermes"
  - "--trust-remote-code"
  - --max-num-seqs
  - "512"
  - --compilation_config.pass_config.enable_fi_allreduce_fusion
  - "true"
  - --compilation_config.pass_config.enable_noop
  - "true"
  - --speculative-config
  - '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
  - --no-enable-chunked-prefill
```

**Key characteristics**: No prefix caching, no chunked prefill, no `--quantization fp8` (relies on FP8 checkpoint + `--kv-cache-dtype fp8`), MTP enabled, `--tool-call-parser hermes`, `--trust-remote-code`, max-num-seqs 512.

### Config B: Our Optimized

Apply our recommended fixes on top of the customer's base. All changes are explicitly listed below — flags not mentioned are kept identical to Config A.

**Changes from Config A**:

1. Add `--quantization fp8` — explicit FP8 weight quantization flag (ensures engine applies FP8 kernels regardless of checkpoint format)
2. Add `--enable-prefix-caching` — 58-76% TTFT reduction with shared prefixes
3. Remove `--no-enable-chunked-prefill` — allows scheduler to interleave prefill chunks with decode steps, reducing head-of-line blocking
4. Raise `--gpu-memory-utilization` from 0.90 to 0.92 — safe on H200, ~2.8 GB more KV cache per GPU
5. Switch `--tool-call-parser` from `hermes` to `qwen3_coder` — native Qwen3 parser
6. Add `--max-num-batched-tokens 32768` — explicit prefill batch size

**Kept from Config A (unchanged)**:

- `--speculative-config` (MTP with 2 draft tokens) — works on customer's vLLM version
- `--compilation_config` flags — low risk, possible minor benefit
- `--max-model-len 32768` — match customer for fair comparison
- `--max-num-seqs 512` — match customer concurrency ceiling
- `--kv-cache-dtype fp8` — match customer
- `--trust-remote-code` — kept for compatibility, not required for this model per parent spec but harmless

**Removed from Config A**:

- `--tokenizer-mode auto` — this is the vLLM default; removed for cleanliness, no behavioral change

### Config C: Our Optimized + No MTP

Same as Config B but without `--speculative-config` to isolate MTP's contribution to throughput and decode latency.

### Config D: Our Optimized + CPU KV Cache Offload

Same as Config B plus `--cpu-offload-gb 64`. At 1000 concurrent requests × 10K input tokens (10M total), the TP=4 GPU KV cache (~4.6M tokens) is ~2.2x oversubscribed — requests queue waiting for KV cache slots. Offloading 64 GB per GPU to CPU DRAM via PCIe Gen5 (~64 GB/s) nearly doubles effective KV capacity to ~9.2M tokens, reducing queue wait time at the cost of ~1-3ms per-token decode latency from PCIe transfers.

**Changes from Config B**:

1. Add `--cpu-offload-gb 64` — offload 64 GB of KV cache per GPU to host DRAM

**Rationale**: The g7e benchmarks (C4 test) showed that at 1000 concurrent × 10K input, KV cache oversubscription caused 229s mean TTFT on 4 GPUs (2.76M token capacity vs 10M demand). The customer's p5en sees the same effect at smaller scale (940ms TTFT p50). CPU offloading trades a small per-token latency penalty for dramatically reduced queuing under high concurrency.

> **Compatibility note**: `--cpu-offload-gb` is broken on vLLM 0.16+ (V1 engine). The customer's image is based on vLLM v0.11.0 nightly, which predates V1 and likely supports it. If the flag fails at startup, skip Config D and note the incompatibility.

### Config E: 2x TP=4 Replicas + CPU KV Cache Offload

Two vLLM instances on the same p5en.48xlarge — replica 0 on GPUs 0-3 (port 8000), replica 1 on GPUs 4-7 (port 8001). Both with `--cpu-offload-gb 64`. This doubles both prefill compute and KV cache capacity vs single replica:

- **Prefill compute**: 2x parallel prefill pipelines, halving queue wait
- **KV cache**: ~9.2M GPU tokens + ~18.4M with CPU offload across 8 GPUs
- **Total capacity**: Handles 1000-1500 concurrent × 10K input without severe oversubscription

Benchmark requests are round-robin'd across both ports. This is the recommended production topology from the parent benchmark report.

---

## Benchmark Workload

Match customer's exact workload:

| Parameter | Value |
|-----------|-------|
| Total requests | 1000 |
| Input tokens | 10,000 |
| Output tokens | 1,000 |
| Request rate | inf (all submitted simultaneously) |
| Dataset | random (synthetic) |

> **Note on concurrency**: Submitting 1000 requests at `--request-rate inf` delivers all requests simultaneously, matching the customer's "1000 concurrent" setup. Actual in-flight concurrency depends on `--max-num-seqs` (512) and server scheduling — requests beyond the concurrency limit queue in the engine.

Additional workloads for T4 load scaling:

| Label | Requests | Request Rate | Expected Peak Concurrency | Purpose |
|-------|----------|-------------|---------------------------|---------|
| low-load | 100 | 0.5 qps | ~10 | Latency floor measurement |
| moderate-load | 200 | 5.0 qps | ~100 | Realistic production load |
| high-load | 1000 | inf | ~512 (max-num-seqs cap) | Customer's stress test |

> **Concurrency note**: Peak concurrency ≈ `request_rate × mean_request_duration`. At 0.5 qps with ~20s mean latency, peak concurrency ≈ 10. At 5.0 qps, peak ≈ 100. At inf, all 1000 arrive instantly, capped by `--max-num-seqs 512`.

---

## Metrics

Key metrics compared across all configs:

| Metric | Definition | SLO (from parent) |
|--------|------------|-------------------|
| **TTFT p50** | Median time to first token | < 300ms at SLO QPS |
| **TTFT p99** | 99th percentile time to first token | < 1s at SLO QPS |
| **ITL p99** | 99th percentile inter-token latency | < 30ms |
| **TPOT p50** | Median time per output token | Informational |
| **Output tok/s** | Aggregate output token throughput | Maximize |
| **Total tok/s** | Aggregate total token throughput (input + output) | Informational |
| **Error rate** | Failed requests / total requests | 0% |

**KV cache metrics** (captured before/after T2b and T5 runs):

| Metric | Source | Relevance |
|--------|--------|-----------|
| `vllm:prefix_cache_hit_rate` | vLLM `/metrics` | Prefix caching effectiveness (T2b) |
| `vllm:gpu_cache_usage_perc` | vLLM `/metrics` | GPU KV cache pressure |
| `vllm:cpu_cache_usage_perc` | vLLM `/metrics` | CPU swap usage |
| `vllm:num_preemptions_total` | vLLM `/metrics` | Requests preempted under memory pressure |
| Dynamo KVBM per-tier hits/misses | Dynamo `/metrics` | Tier 0/1/2 offload activity (T5d) |
| FSx cache directory size | `du -sh /mnt/fsx/kv-cache/` | Validates disk tier is being written (T5d) |
| GPU memory per-GPU | `nvidia-smi` | Confirms constrained VRAM in T5 |

> **Note**: SLO thresholds are from the parent spec's controlled QPS sweeps. At 1000 concurrent requests, SLOs will not be met — the purpose here is to compare A vs B under identical overload conditions, not to certify SLO compliance.

Results will be reported as a comparison table showing Config A, B, and C side by side for each metric, with percentage improvement calculated as `(A - B) / A × 100`.

---

## Test Protocol

| Parameter | Value | Notes |
|-----------|-------|-------|
| Warmup requests | 30 | Sent before measurement begins |
| Runs per config | 3 | Median of 3 runs reported |
| Cooldown between runs | 60s | Allow GPU memory/cache to stabilize |
| Sampling params | temperature=0.7, top_p=0.8, top_k=20 | Match Qwen3-Next official recommendations |
| Percentiles collected | p25, p50, p75, p90, p95, p99 | Extended from parent's p50/p90/p99 |

Differs from parent spec (5 runs) — reduced to 3 runs since this is a focused A/B comparison, not a comprehensive characterization. Results are reported as median across runs.

---

## Benchmark Tiers

### T1: Customer Reproduction (Config A)

Run customer's exact config with their exact workload. Goal: match their reported numbers (3,612 tok/s, TTFT p50 940ms).

### T2: Optimized Head-to-Head (Config B vs A)

Same workload, our optimized config. Measure delta in:
- TTFT p50, p99
- ITL p99
- Throughput tok/s
- Error rate

### T2b: Prefix Sharing (Config B vs A)

Same 1000-concurrent workload but with `generated-shared-prefix` dataset: 8K shared system prompt + 128 unique question tokens per request. Tests prefix caching effectiveness with realistic shared prompts (e.g., tool definitions, system instructions). Two-way comparison:

1. **Config B** (prefix caching enabled) — prefix caching benefit
2. **Config A** (no prefix caching) — customer baseline for contrast

With random datasets (T1/T2), prefix caching has no shared prefixes to exploit — T2b isolates the real-world impact. CPU offload is omitted here because the shared prefix workload fits trivially in HBM (~318 MB KV out of ~63 GiB available).

### T3: MTP Isolation (Config C vs B)

Quantify MTP speculative decoding contribution by comparing Config B (with MTP) against Config C (without MTP) under the same workload.

### T4: Load Scaling (Config B)

Run optimized config at three load levels to find the throughput-latency Pareto frontier:
- **Low** (0.5 qps, ~10 concurrent) — latency floor
- **Moderate** (5.0 qps, ~100 concurrent) — realistic production
- **High** (inf, ~512 concurrent) — customer stress test

### T5: Simulated Memory-Constrained KV Cache Offloading

Simulates smaller-GPU scenarios (e.g., g7e with 96 GB GDDR7) by constraining `--gpu-memory-utilization` to 0.30, leaving ~22 GB KV cache per GPU (~920K tokens total). At 1000 concurrent × 10K input, this creates ~10.9x oversubscription — similar to what g7e.24xlarge experiences. Three sub-tests:

- **T5a**: Constrained HBM, no offload, random data — baseline showing severe queuing
- **T5b**: Constrained HBM + `--cpu-offload-gb 64` — does CPU offload reduce TTFT?
- **T5c**: Constrained HBM, no offload, 8K shared prefix — does prefix caching alone mitigate the pressure?

This demonstrates to the customer what happens on smaller GPUs and whether offloading/prefix caching are viable mitigations — directly applicable to g7e cost-optimization scenarios.

- **T5d**: Constrained HBM + NVIDIA Dynamo KVBM with hierarchical offload to FSx Lustre — uses `dynamo-run` with 4-tier KV cache: GPU VRAM (~22 GB) → CPU DRAM (128 GB) → FSx Lustre (500 GB) via NIXL GDS_MT backend. Dynamo's async write-back architecture offloads KV blocks without blocking the vLLM scheduler. Tests whether hierarchical offloading can absorb the 10.9x KV oversubscription. Requires building `dynamo-kvbm-qwen3` image from the kimi-k2.5 Dockerfile adapted for Qwen3-Next.

> **Scheduler gating caveat**: Previous kimi-k2.5 benchmarks showed vLLM's scheduler gates request admission before KV cache overflows, preventing Dynamo's tiered offloading from activating under normal conditions. The 0.30 gpu-memory-utilization constraint forces extreme memory pressure, which should trigger the overflow path. If T5d shows no improvement over T5a, the scheduler gating is the bottleneck — not the offloading mechanism.

### T6: 2x Replica + CPU Offload (Config E)

Same 1000-request workload, round-robin across two TP=4 replicas (ports 8000/8001). Tests whether doubling prefill compute + KV cache capacity via 2 replicas significantly reduces TTFT vs single replica (T5). This is the production-recommended topology.

### T7: Stress Test at 1500 Concurrent (Config E)

Push beyond the customer's 1000-concurrent workload to 1500 requests at inf QPS with 2x replicas + CPU offload. Tests headroom: 1500 × 10K = 15M input tokens against ~18.4M effective KV capacity (GPU + CPU offload across 8 GPUs). Determines whether the 2-replica + offload topology can handle 50% more load than the customer's current peak.

---

## Success Criteria

1. Customer config reproduction: throughput within 10% of reported 3,612 tok/s
2. Optimized config: TTFT p50 < 500ms at 1000 concurrent (vs customer's 940ms)
3. Optimized config: TTFT p99 < 3s at 1000 concurrent (vs customer's 5.35s)
4. Clear recommendation with quantified improvement percentages

## Non-Requirements

- SGLang comparison (already done in parent benchmark)
- DP+EP testing (already proven inferior in parent benchmark)
- Context scaling beyond 32K (customer workload is 10K input)
- Cost analysis (same instance, cost is identical)
- Extended context testing (covered in parent spec P2b)
