# Serving Best Practices: vLLM & SGLang for MoE Models on H200

Derived from research conducted Feb 2026 for the Qwen3-Next-80B benchmark.
Applies broadly to large MoE models (512+ experts, hybrid attention) on 8× H200 clusters.

---

## Model Dependencies (Hybrid Attention / DeltaNet)

### Do not use `--trust-remote-code`

Models like Qwen3-Next use DeltaNet hybrid attention that is natively supported in the HuggingFace
`transformers` main branch. `--trust-remote-code` loads arbitrary Python from HuggingFace at runtime —
a security risk and a runtime internet dependency. Use the native integration instead.

**Requirement**: `transformers` main branch (not PyPI stable). In an air-gap environment, this must
be baked into the container image as a pre-downloaded wheel:

```dockerfile
COPY wheels/transformers-main.whl /wheels/
RUN pip install /wheels/transformers-main.whl
```

### Install optional linear attention deps for DeltaNet performance

```dockerfile
RUN pip install flash-linear-attention causal-conv1d
```

Without these, DeltaNet layers fall back to a slower reference implementation. The performance
difference is most visible at 64K+ context where DeltaNet handles the bulk of sequence processing.

---

## Parallelism Strategy on 8× H200

### TP=4 with 2 replicas is the Qwen team's recommended default

Despite having 8 GPUs, the official HuggingFace model card recommends `--tensor-parallel-size 4`.
On 8 GPUs, run 2 independent replicas side by side with a load balancer in front.

**Why not TP=8?**
- MoE models have 3B active params per forward pass, not 80B. TP=8 splits a small compute graph
  across 8 GPUs — all-reduce overhead grows while the compute benefit shrinks.
- TP=8 doubles KV cache per GPU (~130 GB vs ~120 GB), but H200's 141 GB/GPU already provides
  enormous headroom for either configuration.
- TP=4 × 2 replicas delivers 2× the aggregate request/s of TP=8 × 1 at comparable per-request latency.

**When to prefer TP=8:**
- Context lengths ≥ 64K where prefill latency dominates (more GPUs split the O(n²) attention layers)
- Interactive / single-user workloads where aggregate throughput doesn't matter

**Replica GPU assignment** (Kubernetes):

```yaml
# Two pods in the same Deployment, each requesting 4 GPUs:
resources:
  limits:
    nvidia.com/gpu: 4
# CUDA_VISIBLE_DEVICES is set automatically by the device plugin.
# Place a Service or ALB in front; round-robin is fine for stateless inference.
```

### DP+EP is the throughput ceiling for MoE, not TP

For maximum aggregate output tokens/second at high concurrency:

```bash
vllm serve <model> \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --expert-placement-strategy round_robin \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}' \
  --api-server-count 8
```

- **`deepep_low_latency` + `round_robin`**: +14.6% throughput, +13.4% TPOT vs linear placement
  (measured on DeepSeek-R1-671B, a comparable MoE). Use this as the default.
- **`deepep_high_throughput`**: Enables dual-batch overlap (compute + communication pipelined).
  Better when the server is prefill-saturated (many long prompts). Switch to this if `deepep_low_latency`
  shows high GPU idle time in profiles.
- **Limitation**: DP+EP does not support MTP speculative decoding. Choose one or the other.

### EPLB config field reference

| Field | Value | Meaning |
|-------|-------|---------|
| `window_size` | 1000 | Forward passes over which expert load stats are accumulated before rebalancing. Lower = faster adaptation; higher = less churn. |
| `step_interval` | 3000 | Rebalancing runs every N steps, keeping overhead low. |
| `num_redundant_experts` | 2 | The 2 hottest experts are replicated across EP ranks so they are always locally available. Each costs ~2.4 GB/GPU; 2 = ~4.8 GB/GPU total overhead. |

For large-scale multi-node deployments (16+ GPUs), increase `num_redundant_experts` to 32 so
the most-requested experts are always locally available on each node.

---

## Multi-Token Prediction (MTP)

### How it works in each engine

MTP uses the model's dedicated speculative head to draft N additional tokens, then verifies them
in one forward pass. If accepted, N tokens are emitted at the cost of ~1 forward pass.

| Engine | Flag | Config |
|--------|------|--------|
| vLLM ≥ 0.10.2 | `--speculative-config` | `'{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` |
| SGLang ≥ 0.5.2 | `--speculative-algo NEXTN` | `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |

### When MTP helps (and when it doesn't)

MTP reduces **ITL (inter-token latency)** — the gap between emitted tokens during streaming decode.
It has the most impact when:
- Output is predictable / low-entropy (code generation, JSON, structured output)
- Batch size is small (1–4 concurrent requests); larger batches reduce per-request speculation benefit
- Context is short (32K); at 128K, prefill dominates and MTP's decode gain is less visible

MTP adds overhead when:
- High concurrency saturates GPU with verify passes, crowding out new requests
- Output entropy is high (creative writing, long-form reasoning)

**Start with 2 speculative tokens** (`num_speculative_tokens: 2`). Increasing to 4 risks more
rejected drafts without proportional latency gain.

### MTP is incompatible with DP+EP

`--data-parallel-size` and `--speculative-config` cannot be used together in vLLM. For throughput
benchmarking, run separate configs. Do not attempt to combine.

---

## KV Cache Configuration on H200

### H200 rarely needs CPU offloading at ≤128K context

With FP8 weights, Qwen3-Next-80B consumes ~80 GB total GPU VRAM across 8 GPUs (~10 GB/GPU at TP=8,
~20 GB/GPU at TP=4). Each H200 has 141 GB, leaving ~120–130 GB/GPU for KV cache. This is
**orders of magnitude more KV cache capacity than most deployments require** at ≤128K context.

CPU offloading (`--cpu-offload-gb`) in vLLM v1:
- GPU → CPU transfers are synchronous (not overlapped with compute) — each block swap adds latency
- Supports LRU and ARC eviction policies
- Only benefits workloads where KV cache pressure causes preemptions; verify via
  `vllm:num_preemptions_total` and `vllm:kv_cache_usage_percent` first

**Recommendation**: Start with `--gpu-memory-utilization 0.92` and no CPU offload. Add offload only
if Prometheus shows `kv_cache_usage_percent > 90%` under target load.

### Avoid SGLang HiCache in production (as of Feb 2026)

SGLang's hierarchical cache (GPU → CPU → disk) has active stability issues:
- Issue #19212: `write_back` policy crashes with AssertionError under load
- PR #19177: CUDA Graph + HiCache + Speculative Decoding integration errors

HiCache is an experimental feature. Do not enable it for benchmark sessions that require
reliable results. Revisit when these issues are resolved upstream.

### Prefix caching is free — always enable it

vLLM: `--enable-prefix-caching`
SGLang: RadixAttention is on by default (no flag needed)

For agentic coding workloads with shared system prompts (e.g., shared repo context), prefix
caching reduces TTFT dramatically after the first request warms the cache. Use
`generated-shared-prefix` dataset in benchmarks to quantify the actual hit rate.

---

## Benchmark Workload Selection

### Use `generated-shared-prefix` for agentic coding, not `sharegpt`

ShareGPT is a general chat dataset. For a coding assistant:

```bash
# SGLang bench
python -m sglang.bench_serving \
  --dataset-name generated-shared-prefix \
  --gsp-system-prompt-len 4096 \
  --gsp-question-len 128 \
  --gsp-output-len 256
```

4K system prompt = shared repository context. This is the most realistic proxy for IDE/agent use cases
and will exercise prefix caching in a way `random` datasets cannot.

### Benchmark profile summary for coding use cases

| Profile | Input | Output | Concurrency | What it measures |
|---------|-------|--------|-------------|-----------------|
| `short-agentic` | 512 | 512 | 32–64 | ITL under streaming; MTP benefit |
| `long-context` | 4096 | 256 | 16–32 | TTFT at long prefill; RAG / code review |
| `prefix-sharing` | 4K shared + 128 | 256 | 32–64 | Prefix cache hit rate; warm TTFT |
| `throughput-sweep` | 1024 | 512 | 8, 16, 32, 64, 128, 256 | Saturation point; tok/s ceiling |

Run `throughput-sweep` first to find the saturation concurrency. Use that concurrency for `short-agentic`
and `prefix-sharing` to measure at a realistic operating point.

---

## Air-Gap Deployment Checklist

All of the following must be complete **before** the GPU capacity block starts:

- [ ] Container images built with `transformers-main`, `flash-linear-attention`, `causal-conv1d`
- [ ] Images pushed to private ECR (use `scripts/stage-images-ecr.sh`)
- [ ] Model weights downloaded to FSx from an internet-connected host (not from the cluster)
- [ ] FSx PERSISTENT_2 filesystem provisioned and healthy (takes 15–20 min)
- [ ] Prometheus + Grafana pods running before GPU node joins
- [ ] `scripts/stage-images-ecr.sh` updated to reference the correct versioned tags (not `:latest`)

Do not pull wheels, packages, or model artifacts from the internet during the benchmark session.
Unexpected downloads during a capacity block window burn expensive GPU-hours.

---

## GPU Memory Utilization

| Scenario | `--gpu-memory-utilization` |
|----------|---------------------------|
| Safe default | 0.90 |
| Maximum KV cache (target for benchmarks) | 0.92 |
| If OOM / preemptions seen | 0.85 |

For DP+EP with EPLB and redundant experts, account for ~2.4 GB/GPU per redundant expert.
With `num_redundant_experts: 2`, subtract ~5 GB from available KV cache per GPU before sizing
`--gpu-memory-utilization`.

---

## vLLM Version Notes

| Version | Key additions relevant here |
|---------|-----------------------------|
| 0.10.2 | Qwen3-Next MTP support (`qwen3_next_mtp` method) |
| 0.15.0 | Qwen3-Coder-Next native support; `--tool-call-parser qwen3_coder` |
| Post-0.15 | `deepep_low_latency` + `round_robin` placement GA |

Pin to a specific version in container images. Do not use `:latest` in air-gap environments.

---

## SGLang Version Notes

| Version | Key additions relevant here |
|---------|-----------------------------|
| 0.5.2 | Qwen3-Next-80B-A3B-Instruct listed as supported; NEXTN speculative decoding |
| 0.5.8 | Qwen3-Coder-Next native support; `--tool-call-parser qwen3_coder` |

SGLang's RadixAttention prefix caching and Two-Batch Overlap (TBO) for EP are both on by default.
No explicit flags needed beyond the engine launch args.
