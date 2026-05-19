# Nemotron-3-Super-120B-A12B — Dynamo on B200 Benchmark Spec

## Status: DRAFT (2026-03-12)

## Overview

Deploy NVIDIA Nemotron-3-Super-120B-A12B-FP8 on a single p6-b200.48xlarge using **NVIDIA Dynamo** as the inference orchestration layer. Dynamo provides disaggregated prefill/decode serving, KV-aware routing, and dynamic GPU scheduling — the key optimizations for this hybrid Mamba-2 + LatentMoE architecture. Target: validate Dynamo's aggregated and disaggregated serving modes with all three backends (vLLM, SGLang, TRT-LLM), measure the throughput gain from prefill/decode separation, and establish production serving baselines.

**Why Nemotron-3-Super:**
- Hybrid Mamba-2 + LatentMoE + Select Attention — 120B total / 12B active per token
- Up to 1M context length — RULER 96.5% @ 256K
- FP8 variant available (~124 GB) — fits TP=2 on B200 with massive KV cache headroom
- Strong agentic benchmarks: TauBench Retail 63.4%, GPQA 79.4%, LiveCodeBench 78.4%
- NVIDIA Nemotron Open Model License — permissive, commercial OK

**Why Dynamo:**
- **Disaggregated prefill/decode**: Separates compute-bound prefill from memory-bound decode — eliminates head-of-line blocking. Critical for hybrid Mamba models where prefill and decode have fundamentally different compute profiles
- **NIXL zero-copy GPU-to-GPU transfer**: KV cache transfer between prefill and decode workers without CPU bounce
- **Dynamic GPU scheduling**: Adapts prefill/decode worker allocation based on demand
- **KV-aware routing**: Prefix-hash-based routing reduces KV recomputation across replicas
- **Three-backend support**: vLLM, SGLang, TRT-LLM under a unified Rust frontend — enables direct framework comparison with identical routing/scheduling
- **Official Nemotron-3-Super recipes**: PR #7216 (merged Mar 11, 2026) provides validated deployment configs for all three backends. PR #6932 adds aggregated routing recipe with TRT-LLM
- **EKS deployment guide**: `examples/deployments/EKS/` provides production deployment patterns

**Why p6-b200.48xlarge:**
- 8x B200 (183 GB HBM each = 1,464 GB total VRAM) — massive headroom for disaggregated configs
- NVSwitch NVL5+ — enables NIXL zero-copy GPU-to-GPU KV transfer between prefill/decode workers
- 8 GPUs allow flexible prefill/decode splits: 2P+2D, 2P+6D, 4P+4D
- Proven B200 deployment patterns from GLM-5 blueprints

---

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **System Nodes**: m6i.xlarge (cluster workloads, Dynamo control plane)
- **GPU Nodes**: p6-b200.48xlarge via capacity blocks
  - 8x NVIDIA B200 (183 GB HBM3e each), NVSwitch NVL5+
  - NVMe instance store: ~28 TB (RAID0)
  - `gpu_desired_size=0` (manual launch via capacity block reservation)
- **Region**: us-east-2 (B200 capacity block availability)
- **AMI**: amazon-eks-node-al2023-x86_64-nvidia-1.32 (AL2023 required for Fabric Manager on NVL5+)

### 1a. GPU & NCCL Pre-Flight

Standard pre-flight per template. B200 NVSwitch topology is proven from GLM-5 deployments. NIXL requires NVSwitch for zero-copy GPU-to-GPU transfers — verify NVLink topology before disaggregated runs.

### 2. Model

- **Model ID**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`
- **Architecture**: `nemotron-3-super` — Hybrid interleaved Mamba-2 + LatentMoE + Select Attention
  - 120B total params, **12B active per token**
  - LatentMoE: tokens projected into smaller latent dimension for expert routing
  - Mamba-2 layers for efficient sequential processing
  - Select Attention layers (not every layer has full attention)
  - Multi-Token Prediction (MTP) layers with shared-weight design
  - `architectures: ["MambaForCausalLM"]`
- **Context Length**: Up to 1M tokens (256K default)
- **Format**: FP8 (float8_e4m3fn weights, BF16/MXFP8 for select layers)
- **Disk Footprint**: ~124 GB FP8
- **License**: NVIDIA Nemotron Open Model License (permissive, commercial)

### 3. Dynamo Architecture

#### Aggregated Mode (baseline)

All GPUs in a TP group handle both prefill and decode for each request.

```
              ┌──────────────────┐
              │  Dynamo Frontend  │  (Rust, OpenAI-compatible HTTP)
              │     :8000         │
              └────────┬─────────┘
                       │
              ┌────────┴─────────┐
              │   Dynamo Router   │  (KV-aware prefix-hash routing)
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────┴────┐  ┌─────┴────┐  ┌────┴────┐
    │ Worker 0│  │ Worker 1 │  │Worker N │   (backend: vLLM/SGLang/TRT-LLM)
    │  TP=2   │  │  TP=2    │  │  TP=2   │
    │ GPU 0,1 │  │ GPU 2,3  │  │ GPU N,N+1│
    └─────────┘  └──────────┘  └─────────┘
```

#### Disaggregated Mode (prefill/decode split)

Separate GPU pools for prefill (compute-bound) and decode (memory-bound). NIXL transfers KV cache between pools via NVSwitch.

```
              ┌──────────────────┐
              │  Dynamo Frontend  │
              │     :8000         │
              └────────┬─────────┘
                       │
              ┌────────┴─────────┐
              │   Dynamo Router   │  (routes to decode; decode→prefill on miss)
              └────────┬─────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────┴─────────┐      ┌──────────┴───┐
    │ Decode Pool   │      │ Prefill Pool  │
    │ Worker 0 TP=2 │◄────│ Worker 0 TP=2 │  (NIXL zero-copy KV transfer)
    │ GPU 0,1       │     │ GPU 4,5       │
    │ Worker 1 TP=2 │◄────│ Worker 1 TP=2 │
    │ GPU 2,3       │     │ GPU 6,7       │
    └───────────────┘     └───────────────┘
```

### 4. Serving Configurations

#### Dynamo + vLLM (aggregated, tp2-x4)

```bash
# Decode/aggregated worker (per GPU pair)
CUDA_VISIBLE_DEVICES=0,1 python -m dynamo.vllm \
  --model /local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --tensor-parallel-size 2 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-seqs 512 \
  --swap-space 0 \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin "/local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/super_v3_reasoning_parser.py" \
  --reasoning-parser super_v3

# Frontend
python -m dynamo.frontend --http-port 8000
```

> **vLLM limitation**: Disaggregated mode is NOT supported for Nemotron-3-Super on vLLM due to hybrid KV cache incompatibilities (Dynamo PR #7216). Aggregated only.

#### Dynamo + SGLang (disaggregated, 4P+4D)

```bash
# Decode workers (GPU 0-3, 2 workers x TP=2)
CUDA_VISIBLE_DEVICES=0,1 python -m dynamo.sglang \
  --model /local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --tp 2 --ep 1 \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3

# Prefill workers (GPU 4-7, 2 workers x TP=2)
CUDA_VISIBLE_DEVICES=4,5 python -m dynamo.sglang \
  --model /local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --tp 2 --ep 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3

# Frontend
python -m dynamo.frontend --http-port 8000
```

#### Dynamo + TRT-LLM (disaggregated, 4P+4D)

```bash
# Per Dynamo PR #7216 recipe
mpirun -n 1 --allow-run-as-root --oversubscribe \
python -m dynamo.trtllm \
  --model /local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --served-model-name nemotron-3-super \
  --backend pytorch \
  --tp_size 2 --ep_size 2 \
  --max_batch_size 128 \
  --max_num_tokens 16384 \
  --trust_remote_code \
  --reasoning_parser nano_v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options extra-llm-api-config.yml

# Frontend
python -m dynamo.frontend --http-port 8000
```

**TRT-LLM extra config** (`extra-llm-api-config.yml`):
```yaml
kv_cache_config:
  enable_block_reuse: false    # Required for Mamba hybrid cache
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

> **block_reuse: false**: Required for Mamba hybrid — standard KV block reuse (prefix caching) is incompatible with Mamba recurrent state.

> **Reasoning control**: All backends: `extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}` per request. Use `temperature=1.0, top_p=0.95` for all tasks.

> **Custom reasoning parser**: vLLM requires `super_v3_reasoning_parser.py` from HF repo. SGLang and TRT-LLM use built-in `nano_v3` parser.

### 5. Dynamo Infrastructure Components

| Component | Role | Deployment |
|-----------|------|------------|
| **Frontend** | Rust-based OpenAI-compatible HTTP server | Sidecar or system node |
| **Router** | KV-aware request routing (prefix-hash or round-robin) | Co-located with frontend |
| **Workers** | Backend inference (vLLM/SGLang/TRT-LLM) | GPU pods (TP=2 per worker) |
| **etcd** | Service discovery registry | System node |
| **NATS** | Inter-component messaging | System node |

### 6. Networking

- **VPC**: /16 CIDR with public/private subnets across 3 AZs
- **NAT Gateway**: Single (non-prod)
- **VPC Endpoints**: S3, ECR, STS, CloudWatch Logs
- **NVSwitch**: Intra-node TP communication + NIXL KV transfer for disaggregated mode
- **TCP**: Dynamo inter-component communication (frontend <-> router <-> workers)

### 7. Storage

- **FSx Lustre**: PERSISTENT_2, 4.8 TiB, 500 MB/s/TiB throughput
  - Mounted at `/fsx` — holds pre-staged model weights
  - Persists across capacity block sessions
- **NVMe Instance Store**: ~28 TB RAID0
  - Mounted at `/mnt/nvme` — final serving tier
  - ~124 GB FP8 model copies in ~15-20s from FSx
- **EBS**: gp3 for persistent volumes (Dynamo etcd, non-GPU workloads)

### 8. Container Images

| Image | Source | ECR Tag | Notes |
|-------|--------|---------|-------|
| Dynamo | `nvcr.io/nvidia/dynamo:1.0.0` (or latest) | `<ecr>/dynamo:1.0.0` | Includes frontend, router, all backends |
| vLLM (pinned) | vLLM nightly per HF model card commit | `<ecr>/vllm-nemotron:latest` | Nemotron-3-Super arch + `--trust-remote-code` |
| SGLang | `lmsysorg/sglang:v0.5.9` | `<ecr>/sglang:v0.5.9` | `nano_v3` parser included |
| TRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5` | `<ecr>/trtllm:1.3.0rc5` | PyTorch backend for hybrid arch |
| Benchmark runner | `python:3.11-slim` | `<ecr>/bench-runner:latest` | AIPerf or custom bench tools |

> **Dynamo container**: Check whether Dynamo ships a unified container with all backends bundled, or if each backend requires a separate image. PR #7216 recipes use K8s DynamoGraphDeployment CRD which may handle image selection.

---

## Benchmark Design

Benchmarks are structured around two axes: (1) **backend comparison** and (2) **aggregated vs disaggregated serving**. Priority tiers ensure standalone value at each gate.

```
Priority:  P0 (must-have) -> P1 (should-have) -> P2 (nice-to-have)
Budget:    P0 ~1.5 hr, P1 ~3 hrs, P2 ~1 hr = ~5.5 hrs total
```

### Controlled Variables

| Parameter | Fixed value | Why fixed |
|-----------|-------------|-----------|
| Model | Nemotron-3-Super-120B-A12B-FP8 | Single model under test |
| TP per worker | 2 | Minimum viable; matches Dynamo recipes |
| Quantization | FP8 | Native model format |
| Reasoning | configurable per request | `enable_thinking: True/False` |
| Temperature | 1.0 | Model-recommended |
| Top-p | 0.95 | Model-recommended |
| Routing | prefix-hash (agg) / round-robin (disagg TRT-LLM) | Per Dynamo recipe defaults |

### GPU Allocation Configs

| Config | Mode | Prefill GPUs | Decode GPUs | Workers | Total GPUs |
|--------|------|:------------:|:-----------:|:-------:|:----------:|
| `agg-tp2-x4` | Aggregated | — | — | 4x TP=2 | 8 |
| `agg-tp4-x2` | Aggregated | — | — | 2x TP=4 | 8 |
| `disagg-2p2d` | Disaggregated | 2 (1x TP=2) | 2 (1x TP=2) | 2 | 4 |
| `disagg-4p4d` | Disaggregated | 4 (2x TP=2) | 4 (2x TP=2) | 4 | 8 |
| `disagg-2p6d` | Disaggregated | 2 (1x TP=2) | 6 (3x TP=2) | 4 | 8 |

### P0: Smoke Test + Backend Comparison (MUST HAVE)

**Goal**: All three backends load the model under Dynamo. Establish aggregated baseline. ~1.5 hr.

| Step | Backend | Config | Workload | Concurrency | Reasoning |
|------|---------|--------|----------|-------------|-----------|
| 0a | vLLM | `agg-tp2-x4` | synthetic 4K/1K | 1, 16 | off |
| 0b | SGLang | `agg-tp2-x4` | synthetic 4K/1K | 1, 16 | off |
| 0c | TRT-LLM | `agg-tp2-x4` | synthetic 4K/1K | 1, 16 | off |
| 0d | winner | `agg-tp2-x4` | synthetic 4K/1K | 1 | on |

**Gate**: All three backends serve inference through Dynamo frontend. Compare:
- Output tok/s at concurrency 1 and 16
- TTFT p50/p99
- Cold start time (model load + JIT)
- Dynamo routing overhead vs direct backend serving

Select **best backend** for P1. If TRT-LLM or SGLang wins, they unlock disaggregated mode in P1.

### P1: Disaggregated Serving + Scaling (SHOULD HAVE)

**Goal**: Measure the throughput gain from prefill/decode separation. Compare GPU allocation strategies. ~3 hr.

#### P1a — Aggregated vs Disaggregated (winning backend)

| Step | Config | Workload | Concurrency | Reasoning |
|------|--------|----------|-------------|-----------|
| 1a-1 | `agg-tp2-x4` | synthetic 4K/1K | 1, 16, 64, 128 | off |
| 1a-2 | `disagg-4p4d` | synthetic 4K/1K | 1, 16, 64, 128 | off |
| 1a-3 | `disagg-2p6d` | synthetic 4K/1K | 1, 16, 64, 128 | off |

**Gate**: Quantify disaggregated speedup. At what concurrency does disagg overtake aggregated? Does 2P+6D (decode-heavy) beat 4P+4D (balanced)?

> **Hypothesis**: Disaggregated mode should shine at high concurrency where prefill head-of-line blocking hurts aggregated throughput. With only 12B active params, decode is very fast — decode-heavy (2P+6D) may outperform balanced (4P+4D).

#### P1b — Concurrency Sweep (best config from P1a)

| Step | Workload | Concurrency | Reasoning |
|------|----------|-------------|-----------|
| 1b-1 | synthetic 4K/1K | 1, 4, 16, 32, 64, 128, 256 | off |
| 1b-2 | synthetic 4K/1K | 1, 16, 64, 128 | on |

**Gate**: Find throughput ceiling. Reasoning overhead quantification (thinking on vs off).

#### P1c — Context Scaling (best config from P1a)

| Step | Config | Workload | Context | Concurrency | Reasoning |
|------|--------|----------|---------|-------------|-----------|
| 1c-1 | best | synthetic | 4K, 32K, 64K, 128K | 1 | off |
| 1c-2 | best | synthetic | 4K, 32K, 64K, 128K | 16 | off |

**Gate**: TTFT scaling curve. Disaggregated serving should decouple prefill latency from decode queue depth — verify TTFT stays stable at high concurrency even at 128K context.

#### P1d — Second-Best Backend Disaggregated (if time permits)

Run P1a steps 1a-1 and 1a-2 with the second-best backend from P0 to validate the framework comparison holds under disaggregated mode.

### P2: Agentic Workloads + Long Context (NICE TO HAVE)

**Goal**: Tool calling, reasoning quality, 256K+ context. ~1 hr.

| Step | Config | Workload | Context | Concurrency | Reasoning |
|------|--------|----------|---------|-------------|-----------|
| 2a | best | tool-call (BFCL) | 4K | 1, 8 | on |
| 2b | best | synthetic | 256K | 1 | off |
| 2c | best | multi-turn (sharegpt) | 32K | 4, 16 | on |

**Gate**: Tool calling correctness > 80% (BFCL subset). 256K context feasibility. Multi-turn reasoning quality.

### Workloads

Mapped to the [standard workload catalog](../../../standards/benchmark-commons/workloads/):

| Workload | Catalog mapping | Description | Latency Focus |
|----------|----------------|-------------|---------------|
| `synthetic 4K/1K` | `concurrency-sweep` | Controlled baseline, power-of-2 concurrency | Throughput ceiling |
| `multi-turn` | `chatbot-long` (extended) | sharegpt multi-turn → prefix reuse test | KV-aware routing effectiveness |
| `tool-call` | `coding-agent` | BFCL function calling via `qwen3_coder` parser | Functional correctness + TTFT warm |
| `long-context 128K` | Custom (`catalog_id: null`) | 128K input, single request | TTFT scaling |

**Enriched artifact output**: All benchmark results stored in `blueprints/nemotron-super/results/` using the enriched artifact schema. Each run captures engine-internal metrics (KV cache utilization, prefix hit rate) in the `extensions` block.

### Concurrency Levels

| Level | Concurrency | Purpose | Reference (from optimization guide) |
|-------|-------------|---------|------|
| Low | 1 | Single-request latency baseline | TP-optimal regime (low batch) |
| Medium | 4-16 | Interactive use | Hybrid TEP zone |
| High | 32-64 | Batch / swarm agents | EP starts to dominate |
| Stress | 128-256 | Throughput ceiling, disagg advantage zone | Pure DEP regime |

See `docs/inference-optimization-guide.md` Section 1 (Parallelism Strategy Selection) for why the bottleneck shifts at each level.

---

## Metrics

### Latency Metrics (per request)

| Metric | Unit | Percentiles |
|--------|------|-------------|
| Time to First Token (TTFT) | ms | p50, p90, p99 |
| Inter-Token Latency (ITL) | ms | p50, p90, p99 |
| End-to-End Latency (E2E) | ms | p50, p90, p99 |
| Time Per Output Token (TPOT) | ms | p50, p90, p99 |

### Throughput Metrics

| Metric | Unit |
|--------|------|
| Output tokens/second | tok/s |
| Requests/second | req/s |
| Prefill tokens/second | tok/s |

### Dynamo-Specific Metrics

| Metric | Unit | What It Measures |
|--------|------|------------------|
| Disagg speedup | ratio | `disagg_tok_s / agg_tok_s` at same concurrency |
| TTFT stability under load | ms (p99 delta) | TTFT p99 at conc=1 vs conc=128 — disagg should be flatter |
| Prefill worker utilization | % | GPU utilization on prefill-dedicated workers |
| Decode worker utilization | % | GPU utilization on decode-dedicated workers |
| NIXL transfer latency | ms | KV cache GPU-to-GPU transfer time (prefill -> decode) |
| Routing overhead | ms | Dynamo frontend + router latency vs direct backend |
| Optimal P:D ratio | ratio | Best prefill:decode GPU allocation at peak throughput |

### Engine-Internal Metrics (scraped from Prometheus `/metrics` during benchmark)

| Metric | Source | Why it matters |
|--------|--------|----------------|
| KV cache utilization % | `vllm:gpu_cache_usage_perc` / SGLang equivalent | Find concurrency ceiling before eviction |
| Prefix cache hit rate | `vllm:cache_hit_rate` | Validate KV-aware routing effectiveness |
| Running requests | `vllm:num_requests_running` | Confirm batch size matches config |
| Block eviction rate | Custom scrape (count evictions/sec) | Detect memory pressure during multi-turn |
| MTP acceptance rate | Engine-specific (if MTP enabled in future) | Validate speculative decode quality |

> These engine-internal metrics are the **key differentiator** vs InferenceX-style benchmarks (which only capture client-side latency). See `docs/inference-optimization-guide.md` Section 11 for the full KV cache benchmarking protocol.

### Cost Efficiency

| Metric | Unit | Formula |
|--------|------|---------|
| $/1M output tokens | USD | `(instance_cost_per_hr / tok_per_sec) * (1M / 3600)` |
| Disagg cost efficiency | ratio | `agg_cost_per_1M / disagg_cost_per_1M` |

---

## Success Criteria

### Latency SLOs

| Metric | Target | Context | Condition |
|--------|--------|---------|-----------|
| TTFT p99 | < 500ms | 4K | Low concurrency, any mode |
| TTFT p99 | < 2000ms | 32K | Low concurrency |
| TTFT p99 | < 5000ms | 128K | Low concurrency |
| ITL p99 | < 30ms | All short context | Streaming decode |
| E2E p99 | < 15s | 4K, 1K output | Low concurrency |

### Throughput

| Metric | Target | Condition |
|--------|--------|-----------|
| Output tok/s (single request) | > 100 tok/s | 12B active, baseline |
| Output tok/s (64 concurrent) | > 3,000 tok/s | Aggregated mode |
| Disaggregated speedup | > 1.3x | At concurrency >= 32 vs aggregated |

### Dynamo-Specific

| Metric | Target | Condition |
|--------|--------|-----------|
| All 3 backends serve via Dynamo | Pass | `agg-tp2-x4` on B200 |
| Disaggregated mode functional | Pass | SGLang or TRT-LLM, `disagg-4p4d` |
| TTFT stability under load | < 2x increase | TTFT p99 at conc=128 vs conc=1 (disagg mode) |
| Optimal P:D ratio identified | Documented | With rationale |

### Functional

| Metric | Target | Condition |
|--------|--------|-----------|
| Reasoning on/off | Correct behavior | `enable_thinking` via chat_template_kwargs |
| Tool calling | > 80% correctness | BFCL subset with qwen3_coder parser |
| 256K context | Serves without OOM | P2 |

---

## Non-Requirements

- Multi-node distributed inference (single p6-b200.48xlarge only)
- BF16 inference (FP8 only)
- 1M context benchmarking (256K max in P1, stretch to 512K in P2)
- Production autoscaling (Dynamo supports it but not benchmarked here)
- HiCache / LMCache KV offloading (Mamba hybrid cache is incompatible; KV headroom is not the bottleneck)
- llm-d / Gateway API integration (Dynamo replaces this role)
- Multi-region deployment
- Dynamo multi-node (single-node intra-GPU disaggregation only)

---

## Security Requirements

- All storage encrypted (KMS)
- Private subnets for compute
- IAM roles with least privilege (IRSA for EKS workloads)
- No public SSH access to nodes
- VPC Flow Logs enabled

---

## Known Limitations

1. **vLLM disaggregated mode NOT supported**: Hybrid KV cache (Mamba recurrent state + attention KV) is incompatible with vLLM's disaggregated KV transfer. vLLM is aggregated-only under Dynamo for this model (PR #7216)
2. **block_reuse: false required (TRT-LLM)**: Standard KV block reuse (prefix caching) is incompatible with Mamba recurrent state. TRT-LLM recipe explicitly disables it. KV-overlap scoring provides no benefit — round-robin routing used instead
3. **Custom reasoning parser (vLLM)**: Requires downloading `super_v3_reasoning_parser.py` from HF repo and passing via `--reasoning-parser-plugin`. SGLang and TRT-LLM use built-in `nano_v3`
4. **vLLM commit pinning**: HF model card references specific vLLM commit (`097eb544e9a22810c9b7a59e586b61627b308362`). May not work on stable releases
5. **Tool call parser format**: `qwen3_coder` parser outputs tool calls as XML in content field (SGLang) vs standard `tool_calls` array. Downstream consumers must handle both formats
6. **Temperature requirement**: Model requires `temperature=1.0` for all tasks — do not use greedy decoding
7. **DeepGEMM JIT on B200**: First startup ~15 min JIT compilation. Set readiness probe `initialDelaySeconds >= 900s`
8. **Capacity block termination**: p6-b200 takes ~10 min before slot frees up
9. **`--attention-backend TRITON_ATTN` (vLLM)**: Required for hybrid Mamba+Attention. FlashAttention/FlashInfer may not support select attention
10. **Dynamo etcd/NATS dependencies**: Dynamo requires etcd for service discovery and NATS for messaging. Must deploy these on system nodes before GPU workers
11. **NIXL requires NVSwitch**: Zero-copy GPU-to-GPU KV transfer in disaggregated mode requires NVSwitch interconnect. PCIe-only instances (g7e) would fall back to slower transfer paths
12. **KV-aware routing limited value**: Mamba hybrid state is not prefix-cacheable in the traditional sense. Prefix-hash routing helps only for the attention layers' KV cache, not Mamba recurrent state. Dynamo TRT-LLM recipe defaults to round-robin for this reason

---

## Terraform Variables

### Serving Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `serving_mode` | string | `"aggregated"` | `"aggregated"` or `"disaggregated"` |
| `serving_backend` | string | `"sglang"` | `"vllm"`, `"sglang"`, or `"trtllm"` |
| `tp_size` | number | `2` | Tensor parallel per worker |
| `num_prefill_workers` | number | `2` | Prefill workers (disaggregated only) |
| `num_decode_workers` | number | `2` | Decode workers (disaggregated only) |
| `num_agg_workers` | number | `4` | Workers (aggregated only) |
| `model_path` | string | `"/local/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"` | Path to staged model on NVMe |

### Infrastructure Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_name` | string | `"nem3-dynamo"` | Short name (12 chars max for IAM) |
| `enable_fsx_lustre` | bool | `true` | Deploy FSx Lustre filesystem |
| `fsx_storage_capacity` | number | `4800` | FSx capacity in GiB |
| `fsx_throughput_per_unit` | number | `500` | MB/s per TiB |
| `enable_nvme_staging` | bool | `true` | NVMe init container for model staging |
| `enable_dynamo_etcd` | bool | `true` | Deploy etcd for Dynamo service discovery |
| `enable_dynamo_nats` | bool | `true` | Deploy NATS for Dynamo messaging |

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| p6-b200.48xlarge capacity block | ~$85-140/hr | CB discount varies |
| FSx Lustre 4.8 TiB PERSISTENT_2 | ~$0.145/GB/month | Destroy between sessions |
| EKS control plane | $0.10/hr | Always running |
| m6i.xlarge system nodes | ~$0.192/hr each | 2 nodes (+ etcd/NATS) |
| **Total benchmark cost** | ~$90-145/hr | GPU dominates |

**Benchmark budget**: 5.5 hours x $90-145/hr = **$495-798 total**.

Model is compact (~124 GB FP8) — FSx->NVMe staging < 2 min. Dynamo's disaggregated mode may improve $/tok by 30%+ at high concurrency, which is the primary cost efficiency hypothesis.

---

## Analysis

### Comparison Dimensions

1. **Aggregated vs disaggregated**: Throughput gain from prefill/decode separation at various concurrency levels
2. **Backend shootout under Dynamo**: vLLM (agg only) vs SGLang (agg+disagg) vs TRT-LLM (agg+disagg) — throughput, latency, stability
3. **Prefill:Decode GPU ratio**: 4P+4D (balanced) vs 2P+6D (decode-heavy) vs 2P+2D (minimal) — find the optimal split for 12B-active hybrid model
4. **TTFT stability under load**: Does disaggregated mode keep TTFT flat as concurrency increases? (Key value proposition)
5. **Reasoning overhead**: thinking on vs off — cost of chain-of-thought in throughput
6. **Context scaling**: TTFT growth from 4K -> 128K — Mamba-2 sequential processing impact on prefill latency
7. **Agentic viability**: Tool calling correctness + reasoning quality for coding agent workloads

### Expected Deliverables

- [ ] P0: Backend comparison table under Dynamo (vLLM vs SGLang vs TRT-LLM aggregated baseline)
- [ ] P1a: Aggregated vs disaggregated throughput chart at multiple concurrency levels
- [ ] P1a: Optimal prefill:decode GPU ratio analysis (4P+4D vs 2P+6D)
- [ ] P1b: Concurrency scaling chart (tok/s vs concurrency, reasoning on/off)
- [ ] P1c: Context scaling chart (TTFT p99 vs context, with TTFT stability analysis)
- [ ] P2: Tool calling correctness report + long-context feasibility
- [ ] Full $/1M tokens comparison: aggregated vs disaggregated at operating point
- [ ] Recommended production config: backend + mode + P:D ratio with rationale

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/nemotron-super/lessons.md`, `blueprints/nemotron-super/results/`, etc.
