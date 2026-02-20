# Together.ai Infrastructure & Technical Differentiation

**Date**: February 2026
**Scope**: Together.ai's custom infrastructure, frameworks, and technical moat
**Sources**: Together.ai blog, GitHub (togethercomputer), documentation, research papers

---

## Executive Summary

Together.ai is technically differentiated from other neo-cloud providers because its founding team writes the infrastructure primitives that the entire ML industry consumes. FlashAttention, RedPajama, and sub-quadratic architecture research originated here. This gives Together a permanent lead in deploying optimizations -- they ship the next version in production before the paper is public.

---

## 1. Founding Team -- Research DNA as Core Moat

| Founder | Role | Key Contributions |
|---------|------|-------------------|
| **Tri Dao** | Chief Scientist | Creator of FlashAttention (all three versions) |
| **Chris Re** | Co-founder | Stanford professor, MacArthur Fellow, pioneer in sub-quadratic architectures (Hyena, Monarch Mixer, S4/state space models) |
| **Ce Zhang** | CTO | Former ETH Zurich professor, ML systems and data management |
| **Percy Liang** | Co-founder | Stanford professor, leads CRFM, created HELM benchmarks |
| **Vipul Ved Prakash** | CEO | Infrastructure veteran |

This is not a typical neo-cloud team that wraps vLLM with an API. These are the people who write the kernels and algorithms that everyone else uses.

---

## 2. FlashAttention -- The Crown Jewel

Together.ai's connection to FlashAttention through Tri Dao is their single biggest technical asset.

### FlashAttention 1 (2022)

- IO-aware exact attention using tiling to minimize GPU HBM-SRAM transfers
- 3x faster GPT-2 training
- First transformers to achieve above-chance on Path-X (16K sequences)

### FlashAttention 2 (2023)

- Sole-authored by Tri Dao
- 2x speedup over FA1, reaching 50-73% theoretical max FLOPs on A100
- Key innovations: reduced non-matmul FLOPs, better parallelization across thread blocks, optimized warp-level work distribution
- Achieved **225 TFLOPs/s per A100** (72% MFU)

### FlashAttention 3 (Hopper-Optimized)

- Targets H100 GPUs specifically
- Asynchronous overlap via warp-specialization (interleaving matmul and softmax)
- Exploits WGMMA, TMA, and native FP8
- Incoherent processing via Hadamard transforms for 2.6x lower FP8 quantization error
- Achieves **740 TFLOPs** (75% H100 utilization, up from FA2's 35%) and ~1.2 PFLOPS with FP8

FlashAttention is now baked into PyTorch, Hugging Face, and virtually every ML framework. Together.ai deploys the next version before anyone else.

---

## 3. Together Inference Engine

Together does not just run vLLM with default settings. Their inference engine is a vertically integrated system claiming up to 3-4x faster than vLLM/TGI on identical hardware.

### 3.1 Custom CUDA Kernels (Together Kernel Collection / TKC)

- Optimized FlashAttention-4 (unreleased publicly)
- Fused MoE implementations
- Architecture-aware execution paths for Blackwell
- 10% faster training, 75% faster inference via FP8 small-matrix optimization

### 3.2 ATLAS (AdapTive-LeArning Speculator System)

A dual-speculator framework for production speculative decoding:

- **Static speculator**: Heavyweight, trained on broad corpus
- **Adaptive speculator**: Lightweight, learns from live traffic in real-time
- **Confidence-aware controller**: Dynamically selects between them

Performance:
- 500 tokens/sec on DeepSeek-V3.1 (2.65x over standard decoding)
- 400% speedup over FP8 baseline when fully adapted
- Reduces RL rollout time by 60%+

### 3.3 Cache-Aware Prefill-Decode Disaggregation (CPD)

Novel three-tier serving architecture:

```
Pre-prefill nodes (cold/low-reuse prompts)
    |
Prefill nodes (warm/high-reuse, reads KV cache via RDMA)
    |
Decode nodes (latency-isolated)
```

Three-level KV cache hierarchy:
1. GPU memory (hot)
2. Host DRAM (warm)
3. Cluster-wide distributed cache via RDMA (cold-but-fast)

Result: 35-40% higher sustainable throughput vs conventional disaggregated serving.

### 3.4 Near-Lossless Quantization

- FP8, FP4 (nvfp4/mxfp4), hybrid precision
- Architecture-aware calibration with fine-grain block-wise scaling
- Selective mixed-precision on sensitive compute paths

---

## 4. Hardware & Cluster Architecture

### GPU Fleet

| GPU | Configuration | Key Specs |
|-----|--------------|-----------|
| GB200 NVL72 | 72 Blackwell GPUs per rack | 1.4 exaFLOPS, 30 TB fast memory, liquid-cooled |
| HGX B200 | 8 GPUs per node | Latest Blackwell |
| H200 | 8 GPUs per node | 1.1 TB HBM3e, 7.2 TB/s aggregate bandwidth |
| H100 | 8 GPUs per node | Production workhorse |
| A100 | 8 GPUs per node | Legacy fleet |

### Scale

- 16 to 100,000+ GPUs per cluster
- 2GW+ data center portfolio, 600MW near-term capacity
- 25+ global locations

### Interconnect

- **Intra-node**: NVLink (900 GB/s on H100, 1,800 GB/s on B200)
- **Inter-node**: InfiniBand (14.4 Tbps on GB200 NVL72 racks, 3.2 Tbps on H200/H100 nodes)

### Storage

- VAST Data and WEKA for high-performance parallel storage
- NVMe SSDs for local staging
- Up to 3 PB high-performance converged storage per cluster

### Orchestration

- Slurm for job scheduling on training clusters
- Kubernetes for containerized inference
- Custom `slurm-operator` for running Slurm on Kubernetes

---

## 5. Open-Source Contributions

### RedPajama -- Largest Public LLM Training Dataset

- **V1**: 1.2 trillion tokens reproducing LLaMA's training data
- **V2**: 30 trillion filtered/deduplicated tokens (100T+ raw) from 84 CommonCrawl dumps, 5 languages, 40+ pre-computed quality annotations
- 500+ models built on RedPajama by the community
- Apache 2.0 licensed

### Other Contributions

| Project | Description |
|---------|-------------|
| **FlashAttention** | Used by virtually every ML framework |
| **Cocktail SGD** | 117x communication reduction for distributed training |
| **OpenChatKit** (9K stars) | Open-source conversational AI toolkit |
| **Hyena / Monarch Mixer / FlashConv** | Sub-quadratic architecture alternatives to attention |
| **Together Kernel Collection (TKC)** | Optimized PyTorch kernels |
| **Sprocket** | SDK for building inference workers on Together's Dedicated Containers |
| **keep-talking** | Fast BPE tokenizer in Rust |
| **terraform-provider-together** | IaC for Together services |
| **SGLang fork** | Fork with proprietary optimizations for production serving |

---

## 6. Training & Fine-Tuning Platform

### Fine-Tuning

- LoRA and full fine-tuning
- Long-context up to 32K
- DPO (Direct Preference Optimization)
- Continued fine-tuning
- Conversational and instruction formats

### TorchForge Integration

Together runs Meta's TorchForge RL pipelines on "Instant Clusters":

- vLLM policy servers
- Monarch actor mesh + TorchStore for weight synchronization
- RDMA-based communication via InfiniBand/NVLink
- Integration with Together CodeSandbox (microVM environments) for tool-use during RL training

### Dedicated Container Inference

Job-orchestration model (not stateless request-response):

- Docker containers with custom runtimes
- Volume-mounted model weights (no re-packaging)
- Multiple priority queues for traffic policy control
- 1.4x-2.6x inference speedups for generative media workloads

---

## 7. Together AI Cloud Platform

| Tier | Description | Use Case |
|------|-------------|----------|
| **Serverless API** | Pay-per-token, 200+ models, OpenAI-compatible | Prototyping, variable workloads |
| **Dedicated Endpoints** | Single-tenant GPUs, per-minute billing | Production inference with SLAs |
| **Instant Clusters** | Self-service GPU clusters, Slurm-managed, 16-100K+ GPUs | Training, fine-tuning, RL |
| **Frontier AI Factory** | Reserved clusters with expert support | Massive-scale frontier model training |

Additional capabilities: Code Sandbox, Code Interpreter, LLM-as-Judge, structured output (JSON mode), embeddings, reranking, image generation, audio/speech-to-text, vision/multimodal.

---

## 8. Comparison with Other Neo-Clouds

| Dimension | Together.ai | CoreWeave | Lambda Labs | Fireworks/Groq |
|-----------|------------|-----------|-------------|----------------|
| **Kernel development** | Custom CUDA (FA, TKC, fused MoE) | None (uses vendor stacks) | None | Limited |
| **Research output** | FlashAttention, RedPajama, Cocktail SGD, sub-quadratic architectures | None | None | Limited |
| **Speculative decoding** | Adaptive dual-speculator (ATLAS) | N/A | N/A | Static or none |
| **KV cache management** | 3-tier RDMA hierarchy, CPD | Standard | Standard | Standard |
| **Quantization** | Architecture-aware FP4/FP8 with block-wise scaling | Standard | Standard | Custom (Groq: LPU) |
| **Training platform** | Full stack: fine-tuning, RL (TorchForge), custom clusters | GPU rental | GPU rental | Inference-only |
| **Open-source data** | RedPajama (30T tokens, 500+ downstream models) | None | None | None |
| **Hardware breadth** | GB200 NVL72 through A100, 100K+ GPU scale | H100/A100, K8s-native | H100/A100, simple rental | Custom ASICs (Groq) or GPU |
| **Storage** | VAST Data, WEKA, 3 PB/cluster | VAST Data | NFS-based (~10 Gbps) | Not published |

### Key Differentiator

The closest analogy: if the people who wrote CUDA also ran a cloud. The depth of systems knowledge -- from custom CUDA kernels through inference engine optimizations up to cluster orchestration -- creates compounding advantages at every layer of the stack.

---

## 9. Relevance to Checkpoint I/O

Together.ai's infrastructure choices relevant to checkpoint and model I/O:

1. **VAST Data + WEKA storage**: Both support GDS, providing high-throughput parallel filesystem access for checkpoint writes
2. **InfiniBand interconnect**: Full RDMA support for distributed checkpointing across nodes
3. **RDMA-based KV cache**: Their CPD architecture's cluster-wide RDMA cache demonstrates expertise in GPU-direct memory transfers that could extend to checkpoint I/O
4. **Slurm orchestration**: Standard HPC job scheduling with checkpoint/restart support built into the workflow
5. **3 PB per cluster**: Sufficient storage capacity for large-scale MoE checkpoint retention without tiering to object storage

Together's use of VAST and WEKA (both GDS-capable filesystems) combined with InfiniBand interconnect positions them well for high-performance checkpoint I/O, though they have not published specific checkpoint throughput benchmarks.
