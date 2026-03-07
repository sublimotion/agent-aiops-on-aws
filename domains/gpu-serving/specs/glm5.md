# GLM-5 Serving Benchmark Spec

## Status: DRAFT (2026-03-04)

## Overview

Deploy GLM-5 by Zhipu AI (THUDM) on AWS to benchmark serving performance with FP8 quantization and EAGLE speculative decoding. GLM-5 is a 744B-parameter MoE model (256 routed + 1 shared expert, top-8 routing, 40B active params per token) with a hybrid architecture combining MoE, Multi-Latent Attention (MLA), and DeepSeek Sparse Attention (DSA). Target: validate single-node FP8 inference on p5e.48xlarge with SGLang, measure TTFT/ITL under various concurrency levels, and establish production serving baselines.

**Why GLM-5:**
- State-of-the-art Chinese-English bilingual model with ~200K context length
- MIT license — production-ready
- FP8 variant available (756 GB vs 1.51 TB BF16) — fits single p5e node with headroom
- Speculative decoding support via EAGLE — designed to accelerate MoE decode latency
- Actively optimized by SGLang (FlashMLA, DeepGeMM FP8 kernels, EAGLE integration)

**Why p5e.48xlarge:**
- 8× H200 (141 GB each = 1,128 GB total VRAM) — sufficient for 756 GB FP8 model + ~350 GB KV cache
- Hopper architecture required — A100/A800 not supported due to DeepGeMM FP8 dependency (sm90+)
- NVLink/NVSwitch for TP=8 all-reduce efficiency
- ~3.8 TB NVMe instance store for fast model staging

---

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **System Nodes**: m6i.xlarge (cluster workloads)
- **GPU Nodes**: p5e.48xlarge via capacity blocks (~$98.32/hr on-demand, ~$59/hr with CB discount)
  - 8× NVIDIA H200 (141 GB HBM3e each), 2 TiB DDR5 CPU RAM
  - NVMe instance store: 8× 3.84 TB SSDs (~30 TB total, ~25 GB/s sequential read)
  - `gpu_desired_size=0` (manual launch via capacity block reservation)
- **Region**: us-east-1 or us-west-2 (p5e availability)
- **Availability Zone**: Check capacity block availability before provisioning

### 2. Model

- **Model ID**: `zai-org/GLM-5-FP8`
- **Architecture**: `glm_moe_dsa` — Hybrid MoE + Multi-Latent Attention + DeepSeek Sparse Attention
  - 744B total params (256 routed experts × 2.9B + 1 shared expert, top-8 routing, 40B active per token)
  - 80 transformer layers
  - Hidden size: 8,192, MLA heads: 64, MLA key-value heads: 8
  - Vocabulary: 256,000 tokens (multilingual tokenizer)
- **Context Length**: ~200K tokens native (official spec)
- **Format**: safetensors (FP8), 756 GB disk footprint
- **License**: MIT
- **Serving Engine**: SGLang v0.5.2+ (preferred)
  - FlashMLA: optimized attention for Multi-Latent Attention
  - DeepGeMM FP8 kernels (sm90+ Hopper-only)
  - EAGLE speculative decoding with 4 draft tokens
  - Custom parsers: `--tool-call-parser glm47`, `--reasoning-parser glm45`

#### Parallelism Strategy on 8× H200

| Config | TP | Replicas | GPUs used | Est. KV cache | Use case |
|--------|----|----------|-----------|--------------|----------|
| `tp8-x1` | 8 | 1 | 8 | ~350 GB | Latency-optimized, single-replica baseline |

> **FP8 requirement**: BF16 checkpoint is 1.51 TB — does not fit single-node with any practical KV cache capacity. FP8 quantization is mandatory for single-node deployment.

#### Serving Configuration

**Canonical SGLang command** (`tp8-x1` baseline):

```bash
python3 -m sglang.launch_server \
  --model-path /local/models/GLM-5-FP8 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name glm-5-fp8 \
  --port 30000
```

**With EAGLE speculative decoding**:

```bash
python3 -m sglang.launch_server \
  --model-path /local/models/GLM-5-FP8 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name glm-5-fp8 \
  --port 30000
```

> **Parser requirements**: GLM-5 uses custom tokenizer formats for tool calls (`glm47`) and chain-of-thought reasoning (`glm45`). These parsers are built into SGLang v0.5.2+ but must be explicitly enabled.

### 3. Networking

- **VPC**: /16 CIDR with public/private subnets across 3 AZs
- **NAT Gateway**: Single (non-prod)
- **VPC Endpoints**: S3, ECR, FSx, STS, CloudWatch Logs
- **EFA**: Enabled for GPU interconnect (NVLink for intra-node TP communication)

### 4. Storage

- **FSx Lustre**: PERSISTENT_2, 4.8 TiB, 1000 MB/s/TiB throughput
  - Mounted at `/fsx` — holds pre-staged model weights
  - Persists across capacity block sessions
- **NVMe Instance Store**: 8× 3.84 TB SSDs (~30 TB total)
  - Mounted at `/local` — final serving tier (~60-90s load time for 756 GB FP8)
  - Ephemeral: init container re-copies from FSx on each pod start (~8-12 min for 756 GB)
- **EBS**: gp3 storage class for persistent volumes (non-GPU workloads)

### 5. Monitoring

- **Prometheus**: Scrape `/metrics` at 1s interval
- **Grafana**: Visualization dashboards
- **Key Metrics**:
  - `sglang:kv_cache_usage_percent` (if exposed, else monitor via logs)
  - `sglang:num_running_requests`
  - `sglang:num_waiting_requests`
  - `sglang:avg_prompt_throughput_toks_per_s`
  - `sglang:avg_generation_throughput_toks_per_s`

---

## Air-Gap Deployment Requirements

This is an air-gap deployment. No outbound internet access from the cluster. All artifacts must be pre-staged before the benchmark session begins.

### Container Images

Stock images from Docker Hub are not available at runtime. All images must be built with required dependencies baked in and pushed to private ECR before the session.

**Required custom image additions** (over stock SGLang base):

```dockerfile
# Must include dependencies for DeepSeek sparse attention and GLM-5 architecture
# These are required for glm_moe_dsa model architecture support
RUN pip install flash-attn>=2.5.0 --no-build-isolation
RUN pip install triton>=2.2.0
RUN pip install flashinfer>=0.1.0

# Ensure transformers includes GLM-5 model support
# Check SGLang release notes for compatible transformers version
RUN pip install transformers>=4.44.0
```

> **Note**: GLM-5 model definition is native in transformers main branch as of 2025-12. `--trust-remote-code` is **not required**.

**Images to mirror to ECR**:

| Image | Source | ECR Tag | Notes |
|-------|--------|---------|-------|
| SGLang server | `lmsys/sglang:v0.5.2+` | `<ecr>/sglang-glm5:v0.5.2` | Must include GLM-5 architecture deps |
| Benchmark runner | `python:3.11-slim` | `<ecr>/bench-runner:latest` | custbench + dependencies |

Use `scripts/stage-images-ecr.sh` to push images to ECR.

### Model Weights

Model weights must be copied to FSx **before** the GPU capacity block starts. From a host with internet access (outside the air-gap boundary):

```bash
huggingface-cli download zai-org/GLM-5-FP8 \
  --local-dir /fsx/models/GLM-5-FP8/ \
  --local-dir-use-symlinks False
```

> **Size**: 756 GB FP8. Allow ~2-3 hours for download on a 1 Gbps connection.

### Model Staging Pipeline

```
[Pre-staged to FSx] ──(per-pod-start)──▶ NVMe Local ──▶ GPU VRAM
 /fsx/models/GLM-5-FP8/                   /local/models/    (engine startup)
 (persistent across sessions)             (ephemeral NVMe)   ~60-90s
```

NVMe staging time: ~8-12 min for 756 GB copy from FSx + ~60-90s deserialization. Plan for this overhead when allocating capacity block time.

Init container pattern:

```yaml
initContainers:
  - name: stage-model
    image: <ecr>/sglang-glm5:v0.5.2
    command:
      - /bin/sh
      - -c
      - |
        if [ ! -f /local/models/GLM-5-FP8/config.json ]; then
          echo "Copying model from FSx to NVMe..."
          time cp -r /fsx/models/GLM-5-FP8/ /local/models/GLM-5-FP8/
          echo "Model staging complete."
        else
          echo "Model already staged on NVMe."
        fi
    volumeMounts:
      - name: fsx-volume
        mountPath: /fsx
      - name: nvme-volume
        mountPath: /local
```

---

## Benchmark Design

Given limited capacity block time (~$98/hr on-demand, ~$59/hr with CB), runs are organized into strict priority tiers. **Stop at each gate before proceeding.** Every tier produces standalone results.

```
Priority:  P0 (must-have) → P1 (should-have) → P2 (nice-to-have)
Budget:    P0 ~1 hr, P1 ~2 hrs, P2 ~1 hr = ~4 hrs total / ~$236-392
```

### Controlled Variables

All runs use the same server configuration unless a specific dimension is being swept. This ensures results are comparable across tiers.

| Flag | Fixed value | Why fixed |
|------|-------------|-----------|
| `--tp-size` | `8` | All 8 GPUs used for latency-optimal TP configuration |
| `--dtype` | `bfloat16` | Compute dtype for non-quantized layers; model is FP8 |
| `--context-length` | `131072` | Upper limit for inference (full 200K requires validation) |
| `--chunked-prefill-size` | `32768` | Controls prefill batch size — balances TTFT and GPU utilization |
| `--max-running-requests` | `256` | Maximum concurrent sequences in flight |
| `--mem-fraction-static` | `0.85` | Fixed; leaves headroom for KV cache growth |
| `--tool-call-parser glm47` | on | Required for function calling workloads |
| `--reasoning-parser glm45` | on | Required for chain-of-thought (CoT) workloads |
| `--served-model-name` | `glm-5-fp8` | Fixes model alias in API |

> **Chunked prefill tuning note**: Set to `32768` because this workload is expected to be prefill-bound (long context, multi-turn conversations). Larger chunks amortize prefill overhead and reduce TTFT at long input lengths. If P1a shows unexpectedly high ITL burstiness (decode starved by large prefill chunks), drop to `16384` as a diagnostic step.

### P0: Smoke Test + Baseline Performance (MUST HAVE)

**Goal**: Model loads and serves inference. Establish baseline TTFT/ITL at low concurrency.

| Step | Engine | Config | EAGLE | Workload | Context | QPS |
|------|--------|--------|-------|----------|---------|-----|
| 0a | SGLang | `tp8-x1` | off | synthetic | 32K | 0.5 |
| 0b | SGLang | `tp8-x1` | off | synthetic | 64K | 0.5 |

**Gate**: Model serves inference. TTFT p99 < 1000ms at 32K, < 2000ms at 64K. No crashes or OOM errors.

### P1: EAGLE Speculative Decoding + Context Scaling (SHOULD HAVE)

**Goal**: Quantify EAGLE speedup on ITL, validate context scaling up to 128K.

#### P1a — EAGLE Impact (run immediately after P0)

| Step | Engine | Config | EAGLE | Workload | Context | QPS |
|------|--------|--------|-------|----------|---------|-----|
| 1a-1 | SGLang | `tp8-x1` | off | synthetic | 32K | 0.5, 2.0 |
| 1a-2 | SGLang | `tp8-x1` | on (4 tokens) | synthetic | 32K | 0.5, 2.0 |

**Gate**: Measure ITL p99 delta with/without EAGLE. If EAGLE reduces ITL by > 15%, it stays on for all subsequent P1 runs. If EAGLE adds latency (draft+verify overhead exceeds savings), disable.

#### P1b — Context Scaling (winning config from P1a)

| Step | Workload | Context | QPS |
|------|----------|---------|-----|
| 1b-1 | synthetic | 32K, 64K, 128K | 0.5 |
| 1b-2 | multi-turn | 32K, 64K | 0.5, 2.0 |

**Gate**: TTFT p99 at each context tier. Validate TTFT degrades gracefully (DSA sparse attention should flatten the 64K→128K slope vs a dense model). RadixAttention (SGLang's prefix cache) should show >75% hit rate on `multi-turn` workload.

#### P1c — QPS Sweep and Pareto Frontier (winning config from P1a)

| Step | Workload | Context | QPS |
|------|----------|---------|-----|
| 1c-1 | synthetic | 32K | 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 |
| 1c-2 | multi-turn | 32K | 0.5, 2.0, 4.0 |

**Gate**: Identify SLO-max QPS (highest QPS where TTFT p99 < 500ms at 32K, ITL p99 < 50ms). This is the operating point for cost analysis.

### P2: Long Context Stress Test (NICE TO HAVE)

**Rationale**: Full 200K context is not validated in P1. Run P2 only if capacity block time remains after P1.

| Step | Engine | Config | Workload | Context | QPS |
|------|--------|--------|----------|---------|-----|
| 2a | SGLang | `tp8-x1` (best from P1) | synthetic | 128K, 196K | 0.2 |

**Gate**: Validate TTFT p99 < 3000ms at 128K, < 5000ms at 196K. Monitor `sglang:num_preemptions_total` (if exposed). If preemptions occur, reduce `--mem-fraction-static` by 0.05 and retest.

### Workloads

| Workload | Dataset / Pattern | Description | Latency Focus |
|----------|-------------------|-------------|---------------|
| `synthetic` | `random`, 1024 in / 512 out | Controlled, reproducible baseline | TTFT + ITL baseline |
| `multi-turn` | `sharegpt`, multi-turn conversations | Shared prefix reuse (RadixAttention test) | Prefix cache hit rate, ITL streaming |
| `rag` | `random`, 4096 in / 256 out | Long input, short output (retrieval pattern) | TTFT (long prefill) |
| `tool-call` | `bfcl`, function calling | GLM-5's tool-call parser (glm47) | Functional correctness, ITL |

> **Tool-call workload**: Uses Berkeley Function Calling Leaderboard (BFCL) dataset. Requires `--tool-call-parser glm47` flag. Tests GLM-5's native function calling format compatibility with SGLang.

### Context Tiers

| Tier | Context Length | Purpose |
|------|--------------|---------|
| Short | 32K | Baseline; EAGLE most effective here |
| Medium | 64K | DSA sparse attention benefit range |
| Long | 128K | Stress test; validate SLO holds |
| Extended | 196K | P2 only; near-max native context |

> 200K native max is not benchmarked unless 196K results are within SLO and time permits.

### QPS Levels

| Level | Requests/sec | Purpose |
|-------|-------------|---------|
| Low | 0.5 | SLO validation — isolate latency from queuing effects |
| Medium | 2.0 | Balanced |
| SLO-max | TBD | Determined from P1c sweep; highest QPS where TTFT p99 < 500ms |
| Overload | SLO-max × 1.5 | P2 stress testing (optional) |
| Sweep | 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 | Pareto frontier mapping (P1c) |

---

## Test Protocol

### Phases

| Phase | Duration | Purpose |
|-------|----------|---------|
| Warmup | 30 requests | Populate RadixAttention cache, stabilize |
| Measurement | 5 runs × config | Statistical validity |
| Cooldown | 60s between configs | Clear scheduler state |

### Run Parameters

```yaml
runs_per_config: 5
warmup_requests: 30
cooldown_seconds: 60
request_timeout: 300s
max_tokens: 512
temperature: 0.7   # Typical sampling parameters
top_p: 0.9
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

### Engine Metrics (from Prometheus or logs)

| Metric | Source | What It Measures |
|--------|--------|------------------|
| `sglang:avg_prompt_throughput_toks_per_s` | SGLang /metrics | Prefill throughput |
| `sglang:avg_generation_throughput_toks_per_s` | SGLang /metrics | Decode throughput |
| `sglang:num_running_requests` | SGLang /metrics | Active request count |
| `sglang:num_waiting_requests` | SGLang /metrics | Queue depth |
| `sglang:kv_cache_usage_percent` | SGLang /metrics | GPU KV cache fill level |
| RadixAttention hit rate | SGLang logs | Prefix reuse rate (multi-turn workload) |

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

### Model Loading Metrics

| Metric | Unit | What It Measures |
|--------|------|------------------|
| FSx→NVMe copy time | s | Init container staging duration (756 GB) |
| NVMe→VRAM load time | s | Engine startup model load |
| Time to first healthy response | s | Total cold start |

### Output Artifacts

```
results/
├── {config}_{eagle}_{workload}_{context}_{qps}_{timestamp}.json
├── {config}_{eagle}_{workload}_{context}_{qps}_summary.json
├── eagle_comparison.png          # ITL with/without EAGLE
├── context_scaling.png           # TTFT vs context length
├── pareto_frontier.png           # TTFT p99 vs tok/s across all configs
├── prometheus_snapshot/
└── pod-logs/
```

---

## Success Criteria

### Latency SLOs

| Metric | Target | Context | Condition |
|--------|--------|---------|-----------|
| TTFT p99 | < 1000ms | 32K | All workloads, low QPS |
| TTFT p99 | < 2000ms | 64K | All workloads, low QPS |
| TTFT p99 | < 3000ms | 128K | Synthetic workload, low QPS |
| ITL p99 | < 50ms | All | Streaming decode, low-medium QPS |
| TPOT p99 | < 100ms | All | Normalized per-token latency |
| E2E p99 | < 30s | 32K | 512-token generation |
| Preemptions | 0 | All | At low-medium QPS |

### Cost Efficiency

| Metric | Target | Condition |
|--------|--------|-----------|
| Tokens per dollar | Maximize | At SLO-compliant operating point |
| $/1M output tokens | Report per config | At SLO-max QPS |

### Functional

| Metric | Target | Condition |
|--------|--------|-----------|
| SGLang loads model | Serves inference | `tp8-x1` on p5e.48xlarge |
| EAGLE reduces ITL | Measurable improvement | vs same config without EAGLE |
| RadixAttention hit rate | > 75% | `multi-turn` workload |
| Tool-call functional correctness | > 90% | `tool-call` workload (BFCL) |
| Benchmark coverage | All P0+P1 configs produce valid JSON | Before P2 |

---

## Non-Requirements

- Multi-node distributed inference (single p5e.48xlarge only)
- BF16 inference (FP8 only; BF16 at 1.51 TB does not fit)
- vLLM support (SGLang is the reference engine for GLM-5 due to FlashMLA + DeepGeMM optimizations)
- A100/A800 compatibility (Hopper sm90+ required for DeepGeMM FP8)
- Production autoscaling
- Multi-region deployment
- Long-running stability tests (> 1 hour)
- Full 200K context benchmarking (196K max in P2)

---

## Security Requirements

- All storage encrypted (KMS)
- Private subnets for compute
- IAM roles with least privilege (IRSA for EKS workloads)
- No public SSH access to nodes
- VPC Flow Logs enabled

---

## Known Limitations

1. **Capacity blocks require manual launch**: p5e instances are not EKS-managed; must reserve and launch via capacity block API, then join to cluster
2. **Air-gap image requirements**: Custom container images must include `flash-attn`, `triton`, `flashinfer`, and `transformers>=4.44.0`. Must be built and pushed to ECR before the benchmark session
3. **Model pre-staging is mandatory**: 756 GB FP8 weights must be in FSx before GPU capacity block starts. Initial download takes ~2-3 hours on a 1 Gbps connection
4. **FSx PERSISTENT_2 provisioning**: Takes 15–20 minutes to become available
5. **NVMe is ephemeral**: Init container must re-copy from FSx on every pod start (~8-12 min overhead per cold start for 756 GB)
6. **MoE memory footprint**: All 744B params must reside in GPU VRAM for expert routing despite only 40B being active per token — this is expected behavior, not a bug
7. **Hopper-only DeepGeMM kernels**: A100/A800 instances cannot run GLM-5 FP8 inference. Requires sm90+ (H100/H200)
8. **Sparse attention indexer OOM under high concurrency**: Known vLLM bug (Issue #19412) affecting DeepSeek sparse attention. SGLang mitigates this with a different indexer implementation, but monitor for similar issues under sustained high QPS
9. **Custom tokenizer parsers required**: `glm47` (tool calls) and `glm45` (reasoning) must be explicitly enabled with flags. Without these, function calling and CoT outputs may fail to parse correctly
10. **Multi-node PP instability**: Multi-node pipeline parallelism (PP=2+) for GLM-5 is unstable in current SGLang versions (as of v0.5.2). Only single-node TP=8 is validated in this spec

---

## Terraform Variables

### Serving Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `serving_engine` | string | `"sglang"` | `"sglang"` only (vLLM not optimized for GLM-5) |
| `parallelism_config` | string | `"tp8-x1"` | Fixed; single config for this spec |
| `enable_eagle` | bool | `true` | Enable EAGLE speculative decoding |
| `model_path` | string | `"/local/models/GLM-5-FP8"` | Path to staged model weights on NVMe |

### SGLang Settings

| Variable | Type | Default | Maps to SGLang Flag |
|----------|------|---------|---------------------|
| `sglang_tp_size` | number | `8` | `--tp-size` |
| `sglang_dtype` | string | `"bfloat16"` | `--dtype` |
| `sglang_context_length` | number | `131072` | `--context-length` |
| `sglang_chunked_prefill_size` | number | `32768` | `--chunked-prefill-size` |
| `sglang_max_running_requests` | number | `256` | `--max-running-requests` |
| `sglang_mem_fraction_static` | number | `0.85` | `--mem-fraction-static` |

### Infrastructure Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_fsx_lustre` | bool | `true` | Deploy FSx Lustre filesystem |
| `fsx_storage_capacity` | number | `4800` | FSx capacity in GiB (4.8 TiB) |
| `fsx_throughput_per_unit` | number | `1000` | MB/s per TiB |
| `enable_nvme_staging` | bool | `true` | Use NVMe init container for model staging |

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| p5e.48xlarge capacity block | ~$59-98/hr | $59/hr with CB discount, $98.32/hr on-demand |
| FSx Lustre 4.8 TiB PERSISTENT_2 | ~$0.145/GB/month | ~$696/month if persistent; destroy between sessions |
| EKS control plane | $0.10/hr | Always running |
| m6i.xlarge system nodes | ~$0.192/hr each | 2 nodes |
| **Total benchmark cost** | ~$60-99/hr | GPU dominates |

**Benchmark budget**: 4 hours × $60-99/hr = **$240-396 total** (with capacity block discount). On-demand (no reservation) would be ~$393-492.

Tokens-per-dollar analysis formula:

```
For each (EAGLE on/off × workload × context × QPS):
  1. Find SLO-max QPS (highest QPS where TTFT p99 < 500ms at 32K)
  2. Measure aggregate tok/s at SLO-max QPS
  3. Compute $/1M tokens = ($60-99/hr) / (tok/s × 3.6)
  4. Rank by $/1M tokens; prefer lower latency on ties
```

---

## Analysis

### Comparison Dimensions

1. **EAGLE gain**: ITL p99 with/without EAGLE speculative decoding; does 4 draft tokens reduce ITL?
2. **Context scaling**: TTFT growth from 32K → 64K → 128K → 196K — does DSA sparse attention flatten the curve?
3. **RadixAttention ROI**: Cache hit rate on `multi-turn`; TTFT improvement after warm vs cold
4. **Tool-call correctness**: BFCL functional correctness with `glm47` parser

### Expected Deliverables

- [ ] P0 results: model loads and serves, baseline TTFT/ITL established
- [ ] P1a: EAGLE on/off comparison table (TTFT, ITL, TPOT per config)
- [ ] P1b: Context scaling chart (TTFT p99 vs context length)
- [ ] P1c: Latency-throughput Pareto frontier (TTFT p99 vs tok/s)
- [ ] Full $/1M tokens comparison table across all configs
- [ ] Recommended production config with rationale

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/glm5/lessons.md`, `blueprints/glm5/results/`, etc.
