# GLM-5 LMCache — Single Instance KV Cache Benchmark Spec

## Status: HICACHE BENCHMARKED / LMCACHE BLOCKED (2026-03-07)

## Overview

Single p6-b200.48xlarge on EKS, SGLang serving GLM-5-FP8, KV cache benchmark.

Pivoted from HyperPod (`glm5-hyperpod.md`) because Training Plans are unavailable (account not allowlisted). Uses vanilla EKS with capacity blocks instead. Deployed on B200 (not p5e as originally planned).

**Model**: GLM-5 by Zhipu AI — 744B MoE (256 routed + 1 shared expert, top-8, ~40B active), MLA + DSA attention. FP8 variant ~733 GB on disk.
**Model ID**: `zai-org/GLM-5-FP8`

## Key Decisions

1. **SGLang, not vLLM** — vLLM's sync scheduler blocks KV offloading (kimi-k2.5 lessons #18, #26). SGLang's RadixAttention + `--enable-lmcache` avoids this.
2. **HiCache instead of LMCache** — LMCache blocked on NSA/MLA (lesson #10). SGLang's native HiCache has `NSATokenToKVPoolHost` support. 2.86x peak throughput improvement.
3. **Reuse qwen3-next Terraform modules** — networking, eks-cluster, fsx-lustre.
4. **Build LMCache from source** — PyTorch ABI mismatch with prebuilt wheels (kimi-k2.5 lesson #13).
5. **B200 requires AL2023 AMI** — AL2 kernel lacks ib_umad for Fabric Manager on NVL5+.
6. **glm5-blackwell image required** — standard sglang:latest does not support glm_moe_dsa model type.
7. **Tool calling** — SGLang patched images available: `lmsysorg/sglang:glm5-blackwell-patched` (PR #19925). vLLM uses `--tool-call-parser glm47` with `vllm/vllm-openai:glm5` image.

## Serving Config

| Parameter | Value |
|-----------|-------|
| Engine | SGLang glm5-blackwell (dev) |
| TP | 8 |
| Context Length | 131072 |
| Chunked Prefill | 32768 |
| Max Running Requests | 256 |
| Mem Fraction Static | 0.90 |
| Quantization | FP8 (native) |
| Port | 30000 |
| Attention Backend | NSA (nsa_prefill: flashmla_auto, nsa_decode: flashmla_kv) |

## Benchmark Matrix

| Config | Status | Description |
|--------|--------|-------------|
| sglang-baseline | DONE | SGLang RadixAttention only, no LMCache |
| sglang-hicache-cpu | DONE | SGLang HiCache CPU offload (100 GB/rank, 800 GB total) |
| sglang-lmcache-cpu | BLOCKED | SGLang + LMCache L1 CPU offload (400 GB) |
| sglang-lmcache-gds | BLOCKED | SGLang + LMCache GDS → FSx Lustre |
| sglang-lmcache-posix | BLOCKED | SGLang + LMCache POSIX → FSx (fallback) |

**LMCache Blocker**: LMCache v0.3.15 incompatible with NSA/MLA attention (NSATokenToKVPool uses fused `kv_buffer`, LMCache expects separate `k_buffer`/`v_buffer`). Fix pending: LMCache PR #2629. See lessons.md #10.

**HiCache Success**: SGLang's native `--enable-hierarchical-cache` works with NSA/MLA via `NSATokenToKVPoolHost`. See lessons.md #13.

## Results Summary

| Metric | Baseline | HiCache | Delta |
|--------|:--------:|:-------:|:-----:|
| Single-request throughput | 48 tok/s | 48 tok/s | 0% |
| Peak throughput (64 conc) | 909 tok/s | 1,556 tok/s | +71% |
| Peak throughput (128 conc) | N/A | 2,602 tok/s | — |
| Throughput scaling | 19x (1→64) | 54x (1→128) | — |
| Prefix cache latency (σ) | < 1 ms | < 1 ms | 0% |
| Long gen (500 tok, 16 conc) | 578 tok/s | 568 tok/s | -2% |
| Error rate | 0% | 0% | 0% |
| GPU memory per GPU | 175 GB / 183 GB | 175 GB / 183 GB | 0% |
| Host KV cache per rank | — | 100 GB (25,423 pages) | — |

## Infrastructure

- **Instance**: p6-b200.48xlarge (8x NVIDIA B200 183GB HBM, NVSwitch NVL5+)
- **AMI**: amazon-eks-node-al2023-x86_64-nvidia-1.32-v20260304 (kernel 6.1)
- **Driver**: NVIDIA 580.126.09, CUDA 13.0
- **Storage**: FSx Lustre PERSISTENT_2 4800 GiB @ 500 MB/s/TiB, NVMe RAID0 28 TB
- **Container**: lmsysorg/sglang:glm5-blackwell
- **Region**: us-east-2b
- **Capacity Block**: cr-0827eef18c1c46bcd

## Blueprint

`domains/gpu-serving/blueprints/glm5-lmcache/`
