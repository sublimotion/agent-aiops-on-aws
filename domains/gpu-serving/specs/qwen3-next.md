# Qwen3-Next-80B Benchmark Spec

## Status: DRAFT (updated 2026-02-24)

## Overview

Deploy Qwen3-Next-80B-A3B-Instruct (MoE, 80B total / 3B active params) on a p5en.48xlarge (8× H200) instance via capacity blocks in us-east-2. Benchmark serving engine, parallelism strategy, MTP speculative decoding, and KV cache configuration to find the setup that **minimizes latency (primary SLO)** while **maximizing tokens per dollar (secondary)**.

### Optimization Objective

```
Primary:   Minimize TTFT p99 and ITL p99 (latency SLO)
Secondary: Maximize output tokens / $ (cost efficiency)
Metric:    Latency-adjusted cost = ($/hr) / (tok/s at SLO-compliant QPS)
```

The goal is to find the highest throughput (tok/s) operating point where latency stays within SLO bounds, then compare $/1M tokens across configurations.

---

## Components

### 1. Compute

- **Platform**: EKS 1.32 (latest)
- **System Nodes**: m6i.xlarge (cluster workloads)
- **GPU Nodes**: p5en.48xlarge via capacity blocks (~$41.61/hr)
  - 8× NVIDIA H200 (141 GB HBM3e each), ~2 TiB DDR5 CPU RAM
  - `gpu_desired_size=0` (manual launch via capacity block reservation)
- **Region**: us-east-2
- **Availability Zone**: us-east-2c (capacity blocks available)

### 2. Model

- **Model ID**: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Architecture**: Hybrid-attention MoE
  - 80B total params, 3B active per forward pass
  - 512 experts, 10 activated per token + 1 shared expert
  - 48 layers: 12 × (3× Gated DeltaNet-MoE → 1× Gated Attention-MoE)
  - Gated DeltaNet (linear attention) handles bulk context; Gated Attention handles precision layers
  - Multi-Token Prediction (MTP) head — native speculative decoding support
- **Format**: safetensors (BF16), pre-staged to FSx
- **Context Length**: 262,144 tokens native (extensible to 1,010,000 with YaRN)
- **Sampling parameters** (official): temperature=0.7, top_p=0.8, top_k=20

#### Parallelism Strategy on 8× H200

Three configurations to benchmark:

| Config | TP | Replicas | GPUs used | KV cache / GPU | Use case |
|--------|----|----------|-----------|----------------|----------|
| ~~`tp8-x1`~~ | ~~8~~ | ~~1~~ | ~~8~~ | ~~~130 GB~~ | ~~Latency-optimized~~ — **BLOCKED on all engines with FP8** |
| `tp4-x1` | 4 | 1 | 4 | ~120 GB | vLLM + SGLang baseline (only viable TP with FP8) |
| `dp8-ep` | 1 (DP=8+EP) | 1 | 8 | ~130 GB | Maximum aggregate tok/s — **vLLM only** |

> **Known issue: FP8 + TP=8 incompatible on ALL engines.** Both vLLM and SGLang use block-quantized FP8 with `block_size=128`. With TP=8, `moe_intermediate_size=512` gets partitioned to `64` per GPU, which is not divisible by `block_size=128`.
> - **vLLM error**: `ValueError: Weight input_size_per_partition = 64 is not divisible by weight quantization block_k = 128`
> - **SGLang error**: `ValueError: The output_size of gate's and up's weight = 64 is not divisible by weight quantization block_n = 128`
>
> TP=8 would require BF16 (no FP8), doubling memory requirements. All configs use TP=4 as baseline.

- ~~**`tp8-x1`**~~: **BLOCKED.** FP8 block_size=128 incompatible with TP=8 on both vLLM and SGLang. Would require BF16 quantization.
- **`tp4-x1`**: Single replica using 4 GPUs. Both vLLM and SGLang benchmarked for cross-engine comparison. In production, two `tp4-x1` replicas run side-by-side on the same node (CUDA_VISIBLE_DEVICES split, load balanced) to double aggregate throughput — but that topology adds no new performance data and is not benchmarked here. Per-request numbers from `tp4-x1` are identical to what each replica would produce.
- **`dp8-ep`** (vLLM only): Data-parallel=8 with expert parallelism enabled. No MTP possible. Highest aggregate output throughput. Best tokens-per-dollar at high concurrency.

#### Serving Engines

| Engine | Version | MTP Support | Prefix Cache | EP Support |
|--------|---------|-------------|-------------|------------|
| **vLLM** | ≥ 0.15.0 | `--speculative-config '{"method":"qwen3_next_mtp",...}'` | `--enable-prefix-caching` | `--enable-expert-parallel` |
| **SGLang** | ≥ 0.5.2 | `--speculative-algo NEXTN` | RadixAttention (on by default) | `--enable-expert-parallel` |

> SGLang's `Qwen/Qwen3-Next-80B-A3B-Instruct` is explicitly listed as supported in SGLang's supported models docs. DeltaNet hybrid attention is supported via SGLang's delta attention implementation (same family as Kimi Delta Attention).

### 3. Networking

- **VPC**: /16 CIDR with public/private subnets across 3 AZs
- **NAT Gateway**: Single (non-prod)
- **VPC Endpoints**: S3, ECR, FSx, STS, CloudWatch Logs
- **EFA**: Enabled for GPU interconnect (NVLink for intra-node TP communication)

### 4. Storage

- **FSx Lustre**: PERSISTENT_2, 4.8 TiB, 1000 MB/s/TiB throughput
  - Mounted at `/fsx` — holds pre-staged model weights
  - Persists across capacity block sessions
- **NVMe Instance Store**: 8× 3.84 TB SSDs (~30 TB, ~25 GB/s sequential read)
  - Mounted at `/local` — final serving tier (fastest load, ~30–60s for 80GB FP8 weights)
  - Ephemeral: init container re-copies from FSx on each pod start (~2–3 min)
- **EBS**: gp3 storage class for persistent volumes (non-GPU workloads)

### 5. Monitoring

- **Prometheus**: Scrape `/metrics` at 1s interval
- **Grafana**: Visualization dashboards
- **Key Metrics**: `vllm:kv_cache_usage_percent`, `vllm:num_preemptions_total`, `vllm:prefix_cache_hit_rate`

---

## Air-Gap Deployment Requirements

This is an air-gap deployment. No outbound internet access from the cluster. All artifacts must be pre-staged before the benchmark session begins.

### Container Images

Stock images from Docker Hub are not available at runtime. All images must be built with required dependencies baked in and pushed to private ECR before the session.

**Required custom image additions** (over the stock vLLM/SGLang base):

```dockerfile
# Must install from pre-downloaded wheel or local mirror — NOT from PyPI at runtime
# These are required for Qwen3-Next DeltaNet attention performance
RUN pip install flash-linear-attention causal-conv1d

# transformers must be the main branch version (adds qwen3_next model support)
# Pre-download and install from a local wheel or include the source
RUN pip install /wheels/transformers-main.whl  # or equivalent offline install
```

> **Note**: `--trust-remote-code` is **not required** for this model. The DeltaNet attention and tokenizer are natively supported in transformers main branch. Do not add this flag.

**Images to mirror to ECR** (build offline, then push):

| Image | Source | ECR Tag | Notes |
|-------|--------|---------|-------|
| vLLM server | `vllm/vllm-openai:v0.15.0+` | `<ecr>/vllm-qwen3next:v0.15.0` | Must include transformers-main + optional deps |
| SGLang server | `lmsys/sglang:v0.5.2+` | `<ecr>/sglang-qwen3next:v0.5.2` | Must include transformers-main + optional deps |
| Benchmark runner | `python:3.11-slim` | `<ecr>/bench-runner:latest` | vllm bench / sglang bench tools |

Use `scripts/stage-images-ecr.sh` to push images to ECR.

### Model Weights

Model weights must be copied to FSx **before** the GPU capacity block starts. From a host with internet access (outside the air-gap boundary):

```bash
huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --local-dir /fsx/models/qwen3-next-fp8/ \
  --local-dir-use-symlinks False
```

> Use the FP8 quantized variant. ~80 GB vs ~160 GB for BF16 — half the storage, native hardware acceleration on H200, minimal accuracy loss for inference.

### Model Staging Pipeline

```
[Pre-staged to FSx] ──(per-pod-start)──▶ NVMe Local ──▶ GPU VRAM
 /fsx/models/qwen3-next-fp8/              /local/models/    (engine startup)
 (persistent across sessions)             (ephemeral NVMe)   ~30-60s
```

NVMe load time: ~2–3 min copy + ~30–60s deserialization. Plan for this overhead when allocating capacity block time.

Init container pattern (unchanged from previous version):

```yaml
initContainers:
  - name: stage-model
    image: <ecr>/vllm-qwen3next:v0.15.0
    command:
      - /bin/sh
      - -c
      - |
        if [ ! -f /local/models/qwen3-next-fp8/config.json ]; then
          cp -r /fsx/models/qwen3-next-fp8/ /local/models/qwen3-next-fp8/
        fi
    volumeMounts:
      - name: fsx-volume
        mountPath: /fsx
      - name: nvme-volume
        mountPath: /local
```

---

## Serving Configuration Reference

### vLLM — Canonical Configs

#### ~~`tp8-x1` — Single replica, TP=8~~ ❌ BLOCKED

> **Not viable on vLLM.** FP8 `block_k=128` incompatibility with TP=8 — see known issue above. Use SGLang for TP=8 benchmarks, or vLLM `tp4-x1` as baseline.

#### `tp4-x1` — Single replica, TP=4 (vLLM baseline)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /local/models/qwen3-next-fp8 \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --port 8000
```

> GPUs 4–7 are idle during this benchmark run. In production, a second identical replica
> runs on GPUs 4–7 to double aggregate throughput — but per-request latency is the same
> as measured here.

#### `tp4-x1` + MTP

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /local/models/qwen3-next-fp8 \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --port 8000
```

#### `dp8-ep` — Data parallel + Expert parallel (throughput-max)

```bash
vllm serve /local/models/qwen3-next-fp8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --expert-placement-strategy round_robin \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}' \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 32768 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --api-server-count 8 \
  --port 8000
```

> `deepep_low_latency` + `round_robin` placement: demonstrated +14.6% throughput and +13.4% TPOT improvement vs linear placement on DeepSeek-R1-671B (similar MoE architecture). `deepep_high_throughput` enables dual-batch overlap for prefill-saturated workloads — test as a variant if `dp8-ep` saturates.

> **`--eplb-config` field reference:**
> - `window_size: 1000` — number of forward passes over which expert load statistics are accumulated before the rebalancer acts. Lower = faster adaptation to traffic shifts; higher = more stable, less rebalancing churn.
> - `step_interval: 3000` — rebalancing runs every N steps. Keeps overhead low; experts are not moved on every forward pass.
> - `num_redundant_experts: 2` — the 2 most-loaded experts are replicated across EP ranks so hot experts are always locally available. Each redundant expert costs ~2.4 GB/GPU; 2 experts = ~4.8 GB/GPU — already accounted for in `--gpu-memory-utilization 0.92`.

> MTP is not compatible with `--data-parallel-size`. Skip MTP for `dp8-ep` runs.

### SGLang — Canonical Configs

#### `tp8-x1` — Single replica, TP=8

```bash
python -m sglang.launch_server \
  --model-path /local/models/qwen3-next-fp8 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --served-model-name qwen3-next \
  --port 30000
```

#### `tp8-x1` + MTP (NEXTN speculative decoding)

```bash
python -m sglang.launch_server \
  --model-path /local/models/qwen3-next-fp8 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --served-model-name qwen3-next \
  --port 30000
```

#### `tp4-x1` — Single replica, TP=4 (parallelism comparison)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /local/models/qwen3-next-fp8 \
  --tp-size 4 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --served-model-name qwen3-next \
  --port 30000
```

> RadixAttention (prefix caching) is on by default in SGLang. No flag needed.
> `--dtype bfloat16` explicitly sets the compute dtype for non-quantized layers (attention, norms). Without this, SGLang may silently fall back to fp32 on some layer types in FP8 checkpoints.

---

## Benchmark Design

Given limited capacity block time (~$42/hr), runs are organized into strict priority tiers. **Stop at each gate before proceeding.** Every tier produces standalone results.

```
Priority:  P0 (must-have) → P1 (should-have) → P2 (nice-to-have)
Budget:    P0 ~1.5 hrs, P1 ~2.5 hrs, P2 ~1 hr = ~5 hrs total / ~$210
```

### Controlled Variables

All runs use the same server configuration unless a specific dimension is being swept. This ensures results are comparable across tiers.

| Flag | Fixed value | Why fixed |
|------|-------------|-----------|
| `--max-num-batched-tokens` | `32768` | Controls prefill chunk size — directly affects TTFT and ITL. This workload is expected to be prefill-bound (long code context, shared system prompts, multi-turn history). Larger chunks amortize prefill overhead and reduce TTFT at high input lengths. Fixed across all configs so comparisons are not confounded by chunking differences. |
| `--max-num-seqs` | `256` | Maximum concurrent sequences in flight. Fixed to keep batch composition stable across runs. |
| `--gpu-memory-utilization` | `0.92` | Fixed; already accounts for EPLB redundant expert overhead (~4.8 GB/GPU). |
| `--max-model-len` | `131072` (TP configs), `32768` (dp8-ep) | Shorter for dp8-ep to maximize batch size at high concurrency. |
| `--quantization` | `fp8` | All configs use the FP8 checkpoint. |
| `--enable-prefix-caching` | on (all vLLM configs) | Agentic coding workloads reuse large system prompts and code context across turns — prefix caching eliminates redundant prefill for the shared prefix. SGLang enables this by default (RadixAttention). No flag needed there. |
| `--tool-call-parser qwen3_coder` / SGLang equivalent | on (all vLLM configs) | Required for the agentic workload. Requires vLLM ≥ 0.15.0. SGLang handles tool calls natively with the model's chat template. |
| `--served-model-name qwen3-next` | on (all configs) | Fixes the model alias in the API. Benchmark clients use `qwen3-next` rather than the full checkpoint path, making commands portable across engine restarts. |
| `--dtype bfloat16` (SGLang only) | on (all SGLang configs) | Explicitly sets compute dtype for non-quantized layers (attention, norms). Without this, SGLang may silently fall back to fp32 in FP8 checkpoints. vLLM handles this via `--quantization fp8` auto-detection. |

> **`--max-num-batched-tokens` tuning note**: Set to `32768` because this workload is expected to be prefill-bound — large code inputs, 4K+ shared system prompts, multi-turn context accumulation. Larger chunks keep the GPU fed during prefill and reduce TTFT at long input lengths. If P1a or P1c shows unexpectedly high ITL burstiness (decode starved by large prefill chunks), drop to `16384` as a diagnostic step. Monitor `vllm:gpu_cache_usage_percent` — oversized prefill batches can evict KV blocks mid-sequence if cache pressure is high, though this is unlikely on H200 at ≤128K context.

### P0: Smoke Test + Engine and Parallelism Selection (MUST HAVE)

**Goal**: Both engines load and serve. Pick the winning engine and parallelism config.

| Step | Engine | Config | Workload | Context | QPS |
|------|--------|--------|----------|---------|-----|
| 0a | vLLM | `tp4-x1` (no MTP) | synthetic | 32K | 0.5 |
| 0b | SGLang | `tp8-x1` (no MTP) | synthetic | 32K | 0.5 |
| 0c | SGLang | `tp4-x1` (no MTP) | synthetic | 32K | 0.5 |

> **P0a uses TP=4 (not TP=8)** because vLLM FP8 + TP=8 is blocked by the `block_k=128` incompatibility. SGLang can serve at TP=8, so P0b tests that. P0c tests SGLang at TP=4 for a direct cross-engine comparison with P0a.

**Gate**: Model serves inference on both engines. Select engine (lower TTFT p99). Compare SGLang `tp8-x1` (P0b) vs `tp4-x1` (P0a/P0c) TTFT delta. If `tp4-x1` TTFT is within 15% of `tp8-x1`, carry both into P1 (tp4 doubles throughput capacity via 2 replicas). If SGLang fails to load (DeltaNet incompatibility), proceed with vLLM tp4 only.

### P1: MTP Gain and Core Latency Profile (SHOULD HAVE)

**Goal**: Quantify MTP speedup, validate latency SLOs, map the latency-throughput Pareto frontier. Uses winning engine and parallelism from P0.

#### P1a — MTP Impact (run immediately after P0)

| Step | Engine | Config | MTP | Workload | Context | QPS |
|------|--------|--------|-----|----------|---------|-----|
| 1a-1 | vLLM | `tp4-x1` | off | synthetic | 32K | 0.5, 2.0 |
| 1a-2 | vLLM | `tp4-x1` | on (2 tokens) | synthetic | 32K | 0.5, 2.0 |
| 1a-3 | SGLang | `tp8-x1` | off | synthetic | 32K | 0.5, 2.0 |
| 1a-4 | SGLang | `tp8-x1` | on (NEXTN) | synthetic | 32K | 0.5, 2.0 |

> Runs are grouped by framework to minimize server restarts (~390s model load each). P0a and P1a-1 share the same vLLM tp4 server. P0b and P1a-3 share the same SGLang tp8 server.

**Gate**: Measure ITL p99 delta with/without MTP. If MTP reduces ITL by > 15%, it stays on for all subsequent P1 runs. If MTP adds latency (draft+verify overhead exceeds savings), disable.

#### P1b — Context Scaling (winning config from P1a)

| Step | Workload | Context | QPS |
|------|----------|---------|-----|
| 1b-1 | synthetic | 32K, 64K, 128K | 0.5 |
| 1b-2 | rag | 32K, 64K, 128K | 0.5 |
| 1b-3 | prefix-sharing | 32K | 0.5, 2.0 |

**Gate**: TTFT p99 at each context tier. Validate TTFT degrades gracefully (DeltaNet linear attention should flatten the 64K→128K slope vs a dense model). `prefix-sharing` validates prefix cache hit rate > 75%.

#### P1c — QPS Sweep and Pareto Frontier (winning config from P1a)

| Step | Workload | Context | QPS |
|------|----------|---------|-----|
| 1c-1 | synthetic | 32K | 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 |
| 1c-2 | agentic | 32K | 0.5, 2.0, 4.0 |

**Gate**: Identify SLO-max QPS (highest QPS where TTFT p99 < 300ms at 32K). This is the operating point for cost analysis.

#### P1d — DP+EP Throughput Mode

| Step | Engine | Config | Workload | Context | QPS |
|------|--------|--------|----------|---------|-----|
| 1d-1 | vLLM | `dp8-ep` | synthetic | 32K | SLO-max from P1c, SLO-max×2 |

**Gate**: Compare aggregate tok/s vs best TP config from P1c. Compute tokens-per-dollar for each. `dp8-ep` is expected to win at high concurrency but loses on per-request TTFT.

### P2: CPU KV Offload (NICE TO HAVE)

**Rationale**: H200 provides 1,128 GB total GPU VRAM. At FP8, model weights consume ~80 GB leaving ~1,048 GB for KV cache — enormous headroom. CPU offload is expected to have low ROI at normal context lengths (≤128K) and adds latency (transfers are synchronous, not overlapped with compute). Run P2 only if capacity block time remains after P1.

> **Note**: SGLang HiCache (hierarchical cache to disk) has known stability issues as of Feb 2026 (Issue #19212: write_back crashes under load; PR #19177: incompatibility with speculative decoding). **Do not benchmark HiCache.**

| Step | Engine | Config | Workload | Context | QPS |
|------|--------|--------|----------|---------|-----|
| 2a | vLLM | `cpu-light` (10 GB) | synthetic | 32K, 128K | SLO-max, SLO-max×1.5 |

**Gate**: If `cpu-light` degrades TTFT p99 by > 10% vs baseline at identical QPS, skip further offload testing — H200 doesn't need it. If throughput improves at overload (SLO-max×1.5), note it.

### KV Cache Configurations

| Config | Prefix Cache | CPU Offload | Eviction Policy | GPU Util | Memory Request |
|--------|-------------|-------------|----------------|----------|----------------|
| `baseline` | Yes | 0 GB | — | 0.92 | 128Gi |
| `cpu-light` | Yes | 10 GB | LRU | 0.88 | 192Gi |

> `fsx-swap` and `hybrid` configs removed: FSx disk swap adds significant latency (disk I/O, non-overlapped transfer) and H200's VRAM makes it unnecessary. These belong to a future spec targeting long-context (256K+) workloads under high concurrency.

### Workloads

| Workload | Dataset / Pattern | Description | Latency Focus |
|----------|-------------------|-------------|---------------|
| `synthetic` | `random`, 1024 in / 512 out | Controlled, reproducible baseline | TTFT + ITL baseline |
| `rag` | `random`, 4096 in / 256 out | Long input, short output (retrieval pattern) | TTFT (long prefill) |
| `agentic` | `random`, 512 in / 512 out, multi-turn | Tool calls, context switching between turns | ITL (streaming decode) |
| `prefix-sharing` | `generated-shared-prefix` (4K system prompt + 128 question) / 256 out | Shared codebase context across requests | Prefix cache hit rate, TTFT after cache warm |

> `generated-shared-prefix` uses `--gsp-system-prompt-len 4096 --gsp-question-len 128 --gsp-output-len 256`. This dataset directly models a coding assistant with a shared repository context injected as system prompt — the most realistic proxy for the agentic coding use case.

### Context Tiers

| Tier | Context Length | Purpose |
|------|--------------|---------|
| Short | 32K | Baseline; fits standard attention layers; MTP most effective here |
| Medium | 64K | DeltaNet linear attention benefit range — TTFT should scale better than O(n²) |
| Long | 128K | Stress test; validate SLO holds; primary test for TP=8 advantage |

> 262K native max context is not benchmarked initially. Extend to 262K if 128K results are within SLO and time permits.

### QPS Levels

| Level | Requests/sec | Purpose |
|-------|-------------|---------|
| Low | 0.5 | SLO validation — isolate latency from queuing effects |
| Medium | 2.0 | Balanced |
| SLO-max | TBD | Determined from P1c sweep; highest QPS where TTFT p99 < 300ms |
| Overload | SLO-max × 1.5 | P2 offload testing; expected SLO breach used to show offload benefit |
| Sweep | 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 | Pareto frontier mapping (P1c) |

---

## Test Protocol

### Phases

| Phase | Duration | Purpose |
|-------|----------|---------|
| Warmup | 30 requests | Populate prefix cache, stabilize |
| Measurement | 5 runs × config | Statistical validity |
| Cooldown | 60s between configs | Clear scheduler state |

### Run Parameters

```yaml
runs_per_config: 5
warmup_requests: 30
cooldown_seconds: 60
request_timeout: 300s
max_tokens: 512
temperature: 0.7   # Official sampling params
top_p: 0.8
top_k: 20
```

---

## Metrics

### Latency Metrics (per request)

| Metric | Unit | Percentiles | What It Measures |
|--------|------|-------------|------------------|
| Time to First Token (TTFT) | ms | p50, p90, p99 | Prefill latency — time from request to first output token |
| Inter-Token Latency (ITL) | ms | p50, p90, p99 | Decode latency — time between consecutive output tokens |
| End-to-End Latency (E2E) | ms | p50, p90, p99 | Total request latency |
| Time Per Output Token (TPOT) | ms | p50, p90, p99 | E2E / output_token_count |
| Queue/Scheduling Delay | ms | p50, p90, p99 | Request arrival to prefill start |

### Engine Metrics (from Prometheus)

| Metric | Source | What It Measures |
|--------|--------|------------------|
| `vllm:avg_prompt_throughput_toks_per_s` | vLLM /metrics | Prefill throughput |
| `vllm:avg_generation_throughput_toks_per_s` | vLLM /metrics | Decode throughput |
| `vllm:num_preemptions_total` | vLLM /metrics | Memory pressure events |
| `vllm:num_requests_waiting` | vLLM /metrics | Queue depth |
| `vllm:prefix_cache_hit_rate` | vLLM /metrics | Prefix reuse rate |
| `vllm:kv_cache_usage_percent` | vLLM /metrics | GPU KV cache fill level |

### Throughput Metrics

| Metric | Unit | What It Measures |
|--------|------|------------------|
| Output tokens/second | tok/s | Aggregate decode throughput |
| Requests/second | req/s | Request-level throughput |
| Prefill tokens/second | tok/s | Input processing speed |

### Cost Efficiency Metrics

| Metric | Unit | Formula |
|--------|------|---------|
| $/1M output tokens | USD | `(instance_cost_per_hr / tok_per_sec) × (1M / 3600)` |
| SLO-max QPS | req/s | Highest QPS where TTFT p99 stays within SLO |
| Latency-adjusted cost | $/tok/s | `instance_cost_per_hr / tok_per_sec_at_slo_max_qps` |

### Model Loading Metrics

| Metric | Unit | What It Measures |
|--------|------|------------------|
| FSx→NVMe copy time | s | Init container staging duration |
| NVMe→VRAM load time | s | Engine startup model load |
| Time to first healthy response | s | Total cold start |

### Output Artifacts

```
results/
├── {engine}_{config}_{mtp}_{workload}_{context}_{qps}_{timestamp}.json
├── {engine}_{config}_{mtp}_{workload}_{context}_{qps}_summary.json
├── mtp_comparison.png          # ITL with/without MTP per engine
├── engine_comparison.png       # vLLM vs SGLang at matched config
├── pareto_frontier.png         # TTFT p99 vs tok/s across all configs
├── context_scaling.png         # TTFT vs context length per config
├── prometheus_snapshot/
└── pod-logs/
```

---

## Success Criteria

### Latency SLOs

| Metric | Target | Context | Condition |
|--------|--------|---------|-----------|
| TTFT p99 | < 300ms | 32K | All workloads, low QPS |
| TTFT p99 | < 500ms | 64K | All workloads, low QPS |
| TTFT p99 | < 1000ms | 128K | RAG workload, low QPS |
| ITL p99 | < 30ms | All | Streaming decode, low-medium QPS |
| TPOT p99 | < 50ms | All | Normalized per-token latency |
| E2E p99 | < 15s | 32K | 512-token generation |
| Queue delay p99 | < 100ms | All | At SLO-max QPS |
| Preemptions | 0 | All | At low-medium QPS |

### Cost Efficiency

| Metric | Target | Condition |
|--------|--------|-----------|
| Tokens per dollar | Maximize | At SLO-compliant operating point |
| $/1M output tokens | Report per config | At SLO-max QPS |
| Latency-adjusted cost | `($/hr) / (tok/s)` | Lower is better |

### Functional

| Metric | Target | Condition |
|--------|--------|-----------|
| vLLM loads model | Serves inference | `tp8-x1` on p5en.48xlarge |
| SGLang loads model | Serves inference, or document as blocked | `tp8-x1`; DeltaNet compat verified |
| MTP reduces ITL | Measurable improvement | vs same config without MTP |
| `tp4-x1` TTFT within 15% of `tp8-x1` | At 32K context, low QPS | Confirms 2-replica production pattern is viable |
| `dp8-ep` wins tok/$ | At high concurrency | vs best TP config at same QPS |
| Prefix cache hit rate | > 75% | `prefix-sharing` workload |
| Benchmark coverage | All P0+P1 configs produce valid JSON | Before P2 |

---

## Non-Requirements

- SGLang HiCache (active bugs as of Feb 2026 — Issues #19212, PR #19177)
- LMCache integration (future)
- FSx KV swap (unnecessary on H200; deferred)
- Multi-node distributed inference
- Production autoscaling
- Multi-region deployment
- Long-running stability tests (> 1 hour)
- 262K+ context benchmarking (extend spec if 128K results warrant it)

---

## Security Requirements

- All storage encrypted (KMS)
- Private subnets for compute
- IAM roles with least privilege (IRSA for EKS workloads)
- No public SSH access to nodes
- VPC Flow Logs enabled

---

## Known Limitations

1. **Capacity blocks require manual launch**: p5en instances are not EKS-managed; must reserve and launch via capacity block API, then join to cluster
2. **Air-gap image requirements**: Custom container images must include `transformers` (main branch), `flash-linear-attention`, and `causal-conv1d`. Must be built and pushed to ECR before the benchmark session
3. **Model pre-staging is mandatory**: Weights must be in FSx before GPU capacity block starts. Initial download (~80 GB FP8 or ~160 GB BF16) must happen from an internet-connected host
4. **FSx PERSISTENT_2 provisioning**: Takes 15–20 minutes to become available
5. **NVMe is ephemeral**: Init container must re-copy from FSx on every pod start (~2–3 min overhead per cold start)
6. **MoE memory footprint**: All 80B params must reside in GPU VRAM for expert routing despite only 3B being active per token — this is expected behavior, not a bug
7. **`tp4-x1` leaves 4 GPUs idle during benchmarking**: Intentional — per-request numbers are equivalent to what each replica produces in a production `tp4-x2` deployment. The 2-replica production pattern (CUDA_VISIBLE_DEVICES split, load balanced) is not validated in this benchmark; it requires no additional performance data beyond what `tp4-x1` provides
8. **MTP incompatible with DP+EP**: `dp8-ep` config does not support `--speculative-config`. Only test MTP with TP configs
9. **SGLang DeltaNet compatibility**: Verify at P0 step 0b. SGLang natively supports the DeltaNet family but integration with this specific model's hybrid layout needs validation. Mark as blocked if model fails to load

---

## Terraform Variables

### Serving Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `serving_engine` | string | `"vllm"` | `"vllm"` or `"sglang"` |
| `parallelism_config` | string | `"tp8-x1"` | `"tp8-x1"`, `"tp4-x1"`, `"dp8-ep"` |
| `enable_mtp` | bool | `true` | Enable MTP speculative decoding (TP configs only) |
| `model_path` | string | `"/local/models/qwen3-next-fp8"` | Path to staged model weights on NVMe |
| `kv_cache_config` | string | `"baseline"` | `"baseline"` or `"cpu-light"` |

### vLLM Settings

| Variable | Type | Default | Maps to vLLM Flag |
|----------|------|---------|-------------------|
| `vllm_gpu_memory_utilization` | number | `0.92` | `--gpu-memory-utilization` |
| `vllm_cpu_offload_gb` | number | `0` | `--cpu-offload-gb` |
| `vllm_max_model_len` | number | `131072` | `--max-model-len` |
| `vllm_max_num_seqs` | number | `256` | `--max-num-seqs` |

### Infrastructure Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_fsx_lustre` | bool | `true` | Deploy FSx Lustre filesystem |
| `fsx_storage_capacity` | number | `4800` | FSx capacity in GiB |
| `fsx_throughput_per_unit` | number | `1000` | MB/s per TiB |
| `enable_nvme_staging` | bool | `true` | Use NVMe init container for model staging |

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| p5en.48xlarge capacity block | ~$41.61/hr | Reserve in 1-hour blocks |
| FSx Lustre 4.8 TiB PERSISTENT_2 | ~$0.145/GB/month | ~$696/month if persistent; destroy between sessions |
| EKS control plane | $0.10/hr | Always running |
| m6i.xlarge system nodes | ~$0.192/hr each | 2 nodes |
| **Total benchmark cost** | ~$42/hr | GPU dominates |

Tokens-per-dollar analysis formula:

```
For each (engine × parallelism × MTP × KV config):
  1. Find SLO-max QPS (highest QPS where TTFT p99 < 300ms at 32K)
  2. Measure aggregate tok/s at SLO-max QPS
     - tp4-x1: sum both replicas
     - dp8-ep: single server aggregate
  3. Compute $/1M tokens = ($41.61/hr) / (tok/s × 3.6)
  4. Rank by $/1M tokens; prefer lower latency on ties
```

---

## Analysis

### Comparison Dimensions

1. **Engine**: vLLM vs SGLang — TTFT and ITL at matched TP + MTP config
2. **MTP gain**: ITL p99 with/without speculative decoding; does 2 speculative tokens reduce ITL?
3. **Parallelism**: `tp8-x1` vs `tp4-x1` — TTFT delta at 32K/64K/128K; throughput per-server vs aggregate
4. **Throughput ceiling**: `dp8-ep` vs best TP config — where does DP+EP overtake on tok/$?
5. **Context scaling**: TTFT growth from 32K → 64K → 128K — does DeltaNet linear attention flatten the curve?
6. **Prefix cache ROI**: Cache hit rate on `prefix-sharing`; TTFT improvement after warm vs cold

### Expected Deliverables

- [ ] P0 results: engine confirmed, parallelism baseline established
- [ ] P1a: MTP on/off comparison table (TTFT, ITL, TPOT per config)
- [ ] P1b: Context scaling chart (TTFT p99 vs context length per engine/config)
- [ ] P1c: Latency-throughput Pareto frontier (TTFT p99 vs tok/s)
- [ ] P1d: DP+EP tok/s and $/1M tokens vs best TP config
- [ ] Full $/1M tokens comparison table across all configs
- [ ] Recommended production config with rationale

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/qwen3-next/lessons.md`, `blueprints/qwen3-next/results/`, etc.
