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
- **Format**: FP8 quantized weights (our config) or BF16 (customer config TBD)
- **Context Length**: 32,768 (customer cap) / 131,072 (our config)

### 3. Customer vLLM Image

The customer uses a custom vLLM image based on `vllm/vllm-openai:v0.11.0` with nightly vLLM wheels and `transformers==4.57.1` for Qwen3-Next support. MTP speculative decoding works on their version (blocked on our vLLM 0.16.0rc2).

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

**Key characteristics**: No prefix caching, no chunked prefill, MTP enabled, kv-cache-dtype fp8 (but no --quantization fp8), max-num-seqs 512.

### Config B: Our Optimized

Apply our recommended fixes on top of the customer's base:

1. Add `--quantization fp8` (weight quantization — frees ~80GB VRAM if customer was BF16)
2. Add `--enable-prefix-caching` (58-76% TTFT reduction)
3. Remove `--no-enable-chunked-prefill` (allow scheduler interleaving)
4. Raise `--gpu-memory-utilization` to 0.92
5. Switch `--tool-call-parser` to `qwen3_coder` (native parser)
6. Keep MTP enabled (since customer's vLLM version supports it)
7. Keep compilation config flags (low risk, possible minor benefit)

### Config C: Our Optimized + No MTP

Same as Config B but without `--speculative-config` to isolate MTP's contribution.

---

## Benchmark Workload

Match customer's exact workload:

| Parameter | Value |
|-----------|-------|
| Concurrent requests | 1000 |
| Input tokens | 10,000 |
| Output tokens | 1,000 |
| Total requests | 1,000 |
| Dataset | random (synthetic) |

Additional workloads for comparison:

| Label | Concurrent | Input | Output | Purpose |
|-------|-----------|-------|--------|---------|
| customer-repro | 1000 | 10,000 | 1,000 | Exact customer workload |
| moderate-load | 100 | 10,000 | 1,000 | Realistic production load |
| low-load | 10 | 10,000 | 1,000 | Latency floor measurement |

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

### T3: MTP Isolation (Config C vs B)

Quantify MTP speculative decoding contribution by comparing with and without.

### T4: Load Scaling (Config B)

Run optimized config at 10, 100, 1000 concurrent to find the throughput-latency Pareto frontier.

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
