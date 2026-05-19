# Kimi K2.6 Serving Benchmark (Blackwell B300)

## Status: COMPLETE (2026-04-22)

## Overview

Deploy Moonshot AI's **Kimi K2.6** on p6-b300.48xlarge (8x B300 NVSwitch Blackwell) to benchmark INT4 QAT serving performance, with a dual-engine comparison between vLLM v0.19+ and SGLang v0.5.10.

K2.6 is the successor to K2.5 — same 1T MoE + MLA architecture (32B active per token) but ships with **FP8 and native INT4 QAT** weight formats. This unblocks SGLang for the first time in the Kimi family: K2.5's `CompressedTensorsWNA16MarlinMoE` packed format was incompatible with SGLang, but K2.6's standard formats work on both engines.

**Why this benchmark matters:**
- K2.5 was vLLM-only, INT4-packed-only, SGLang-blocked. K2.6 opens the full engine x format matrix
- SGLang HiCache (2.86x throughput on GLM-5 MLA) has never been tested on a Kimi model
- First Kimi deployment on B300 NVSwitch Blackwell — data center grade with full NVLink
- INT4 QAT (trained with quantization) should outperform K2.5's post-hoc compressed-tensors INT4
- B300's 2.15TB VRAM allows full 256K context testing (K2.5 was capped at 32K on H100)

**Evolution from K2.5:**

| Dimension | K2.5 (Feb 2026) | K2.6 (this spec) |
|---|---|---|
| Weight formats | CompressedTensors INT4 only | FP16, FP8, INT4 QAT |
| SGLang support | Blocked (packed MoE format) | Supported (v0.5.9+) |
| Hardware | p5e.48xlarge (8x H100 80GB, $98/hr) | p6-b300.48xlarge (8x B300 268GB, on-demand) |
| GPU interconnect | NVLink 4 / NVSwitch | NVLink 5 / NVSwitch |
| Total VRAM | 640 GB | 2,150 GB |
| Max testable context | 32K (memory constrained) | 256K (full native) |
| HiCache testable | No | Yes |
| vLLM version | v0.15.1 | v0.19.1 |
| SGLang version | N/A | v0.5.10 |
| Tool parser | Nightly-only | Stable (vLLM v0.19.0+, SGLang v0.5.9+) |
| Cold start | ~25 min (model load) | <3 min (SGLang fast warmup) |

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (spot)
- **EKS Cluster**: `qn-sglang-eks-cluster` (v1.32, us-west-2)
- **GPU Node**: p6-b300.48xlarge (8x B300 268GB HBM3e, NVLink 5 / NVSwitch)
- **AZ**: us-west-2b (subnet `subnet-001db6882dbb5ac72`)
- **System Nodes**: Existing cluster nodes
- **vCPUs**: 192
- **System RAM**: 4 TB
- **GPU Interconnect**: NVLink 5 / NVSwitch (full bisection bandwidth)
- **EFA**: Yes
- **NVMe**: 8x 3.8TB local NVMe SSDs (30.4TB total)

### 1a. GPU & NCCL Pre-Flight

B300 uses NVSwitch topology — mature NCCL support on sm_103. vLLM v0.19 explicitly added B300/GB300 support (allreduce fusion enabled by default).

| Check | Expected |
|---|---|
| GPU count | 8x B300 (sm_103) |
| GPU topology | All 8 GPUs via NVSwitch |
| Driver | 580.x+ |
| CUDA | 13.0+ |
| ECC errors | 0 |
| NCCL all_reduce bus BW | > 450 GB/s (NVSwitch) |
| Thermals | < 85C under idle |

### 2. Model

- **Model ID**: `moonshotai/Kimi-K2.6` (confirm HuggingFace path at launch)
- **Architecture**: `kimi_k2` — MoE + MLA (Multi-head Latent Attention)
  - 1T total params, 32B active per token
  - 384 experts (8 active + 1 shared)
  - Hidden size: 7168, MLA kv_lora_rank: 512, qk_rope_head_dim: 64
- **Context Length**: 256K tokens (max generation 262,144 tokens)
- **Thinking**: Togglable via `chat_template_kwargs`
  - Thinking mode (default): `temperature=1.0, top_p=1.0`
  - Instant mode: `temperature=0.6, top_p=0.95, chat_template_kwargs={"thinking": false}`
  - Preserve thinking: `chat_template_kwargs={"thinking": true, "preserve_thinking": true}`
- **Quantization**: INT4 QAT (quantization-aware training, ~594GB)
- **Format**: safetensors
- **Modality**: Multimodal (text + vision), but we benchmark text-only workloads

#### Memory Budget (INT4 QAT on p6-b300.48xlarge)

| Component | Per GPU | Total (8 GPUs) |
|---|---|---|
| Model weights (INT4, TP8) | ~74 GB | ~594 GB |
| KV cache available | ~186 GB | ~1,490 GB |
| CUDA/runtime overhead | ~8 GB | ~66 GB |
| **Total VRAM** | ~268 GB | **~2,150 GB** |

Ample headroom. Full 256K context feasible. `--gpu-memory-utilization 0.90` — no need to push higher.

#### Container Images

| Engine | Image | Size | CUDA |
|---|---|---|---|
| vLLM | `vllm/vllm-openai:v0.19.1-cu130` | 8.1 GB | 13.0 (sm_103) |
| SGLang | `lmsysorg/sglang:v0.5.10.post1-cu130` | 15.1 GB | 13.0 (sm_103) |

**Must use `-cu130` tags** — standard tags are CUDA 12.x and will not work on B300.

#### Serving Configuration — Track A: vLLM v0.19.1

```bash
vllm serve moonshotai/Kimi-K2.6 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
```

Per official deployment guide: `--mm-encoder-tp-mode data` and both parsers are **required**. `--reasoning-parser kimi_k2` is needed because K2.6 enables thinking mode by default. B300's 2.15TB VRAM allows `--max-model-len 131072` (half of native 256K) with room for high concurrency. Can push to 256K in P2 if KV cache budget allows.

**AOT compilation cache** (new in v0.19.0 — persist across restarts):

```bash
# Mount a PVC or hostPath for the compilation cache
export VLLM_TORCH_COMPILE_CACHE=/mnt/nvme/vllm-cache
# Triton autotuning cache (enabled by default in v0.19.0)
export TRITON_CACHE_DIR=/mnt/nvme/triton-cache
```

First cold start pays the full DeepGEMM JIT + torch.compile + CUDA graph warmup (~15 min). Subsequent restarts (pod reschedule, config change, engine swap) reuse the Mega AOT artifact and Triton autotuning cache from NVMe — expected warmup drops to ~2-3 min. Mount both paths as hostPath volumes so the cache survives pod deletions.

#### Serving Configuration — Track B: SGLang v0.5.10.post1

Install: `uv pip install "sglang>=0.5.10.post1" --prerelease=allow`

```bash
sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --tp 8 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --context-length 131072
```

Per official guide: uses `sglang serve` CLI (not `python -m sglang.launch_server`). Both parsers required when enabling tool usage.

**With HiCache (Track B2):**

```bash
sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --tp 8 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --context-length 131072 \
  --enable-hierarchical-cache \
  --hicache-size 500
```

`--hicache-size 500` = 500GB CPU offload (p6-b300 has 4TB system RAM). Must exceed device KV pool size. With ~186GB KV per GPU, set generously to allow spillover at high concurrency.

**Fast warmup** (new in v0.5.10):
```bash
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1  # <3 min cold start vs ~15 min
```

### 3. Networking

- **VPC**: `vpc-0bd6abcecded8edf6` (existing EKS VPC)
- **Access**: `kubectl port-forward` for benchmarks, or NodePort/NLB
- **VPC Endpoints**: S3, ECR (verify existing)
- **Pod networking**: EKS VPC CNI with `hostNetwork: true` for GPU pods (performance)

### 4. Storage

- **Model Weights**: S3 bucket or init container pulling to NVMe hostPath (`/mnt/nvme/models/kimi-k2.6/`)
- **NVMe**: 8x 3.8TB local SSDs (30.4TB total) — mount via hostPath at `/mnt/nvme`
- **Ephemeral Storage**: GPU pod needs 700GB+ ephemeral (model weights + runtime)
- **KV Cache Strategy**:

| Tier | Backend | Track A (vLLM) | Track B (SGLang) |
|---|---|---|---|
| GPU VRAM | HBM3e (~1,490 GB available) | Prefix caching | RadixAttention |
| CPU/NVMe | Host memory (4 TB) | CPU KV offload (v0.19) | HiCache (500GB) |
| LMCache | Any | **UNVALIDATED** — single-group MLA (see note) | **BLOCKED** — SGLang connector gap (PR #2629, OPEN) |

### 5. Monitoring

- **Prometheus**: Scrape vLLM/SGLang `/metrics` endpoint (1s interval)
- **Key metrics**: `kv_cache_usage_percent`, `prefix_cache_hit_rate`, `num_preemptions_total`, TTFT/ITL histograms
- **GPU metrics**: `nvidia-smi dmon` via DaemonSet or exec into GPU pod

---

## Benchmark Design

Uses the standardized `scripts/benchmark-serving.py` (W1-W6 workload suite) for all configurations. Results consumed by the `benchmark-analyst` agent and stored as JSON in `blueprints/kimi-k2.6/results/`.

### Benchmark Tool

```bash
python benchmark-serving.py \
  --api-url http://localhost:8000 \
  --model moonshotai/Kimi-K2.6 \
  --config <config_name> \
  --workloads w1,w2,w3,w4,w5,w6
```

Run per configuration (Track A, B1, B2). Output: one JSON file per run with nested results by workload.

### Configurations

| Config ID | Engine | Caching | Notes |
|---|---|---|---|
| `vllm-tp8-prefix` | vLLM v0.19.1 | Prefix caching | Baseline |
| `sglang-tp8-radix` | SGLang v0.5.10 | RadixAttention only | SGLang baseline |
| `sglang-tp8-hicache` | SGLang v0.5.10 | HiCache (200GB CPU) | Headline experiment |

### Priority Tiers

```
P0 (must-have): Smoke test + tool-call validation                    ~30 min
P1 (must-have): W1-W6 suite on all 3 configs                        ~1.5 hrs
P2 (must-have): Pressure test + HiCache concurrency sweep            ~1.5 hrs
P3 (should-have): K2.5 cross-reference + economics                  ~30 min
Total budget: ~4 hrs compute
```

### P0: Smoke Test + Gate

| Step | Test | Pass Criteria |
|---|---|---|
| 0a | Health check | `/v1/chat/completions` responds on both engines |
| 0b | Basic inference | 1K input / 512 output, verify correct output |
| 0c | Thinking mode | `chat_template_kwargs={"thinking": true}`, verify thinking tokens present |
| 0d | Instant mode | `chat_template_kwargs={"thinking": false}`, verify no thinking tokens |
| 0e | BFCL tool-call | 50 scenarios (same as GLM-5 BFCL suite) |

**Gate**: BFCL >= 75% on at least one engine → proceed to P1. If < 75% on both → STOP.

### P1: Standard W1-W6 Suite (All Configs)

Run the full workload suite on each of the 3 configurations:

| Workload | Pattern | Input/Output | Focus |
|---|---|---|---|
| **W1** | Multi-turn chat | 1024/128 tok, rounds 1/5/10 | TTFT + ITL baseline, prefix sharing |
| **W2** | RAG / long document | 2K/5K/10K in, 100 out | Prefix cache hit rate |
| **W3** | Agentic tool calling | 512/128 tok, multi-turn with tool pauses | TTFT degradation across turns |
| **W4** | Shared system prompt | 2K/4K prefix + 128 query / 128 out | Prefix cache effectiveness |
| **W5** | ShareGPT conversations | 512/256 tok, QPS sweep 0.5/2/4/8 | Real dialogue throughput |
| **W6** | Long context scaling | 1K/4K/8K/16K/32K/64K in, 256 out | TTFT scaling with context (B300 allows full sweep) |

**Standard QPS sweep** (applied to W5): 0.5, 1.0, 2.0, 3.0, 5.0, 8.0

**Metrics collected** (per workload, per config):

| Category | Metrics |
|---|---|
| Latency | TTFT p50/p90/p95/p99, ITL p50/p90/p95, E2E p50/p95 |
| Throughput | Output tok/s (aggregate), req/s |
| Reliability | Success rate, error count, preemptions |
| Caching | Prefix cache hit rate (%), KV cache usage (%) |
| Cost | $/1M output tokens = (instance_cost_per_hr / tok_per_sec) × (1M / 3600) |

**Key comparison targets** (from K2.5 on p5e):

| Metric | K2.5 (p5e, vLLM v0.15) | K2.6 target (B300) |
|---|---|---|
| Single-stream throughput | ~41 tok/s (reasoning) | >= 50 tok/s (B300 higher BW) |
| Peak throughput | ~41 tok/s (saturated early) | >= 500 tok/s (batch scaling) |
| TTFT p50 (W3 agentic) | 820-926 ms | < 500 ms |
| Prefix cache hit rate (W1) | 76-80% | >= 75% |

K2.5 showed almost no throughput scaling with concurrency on p5e — peak was ~41 tok/s regardless of batch size. B300 NVSwitch + newer engines should show dramatically better batching. GLM-5 (same MoE+MLA arch) scaled from 48 → 2,602 tok/s on B200, so MoE batching upside is proven. B300 has 4.7x more memory BW than H100.

### P2: Pressure Test + HiCache Deep-Dive

Two goals: (1) find the concurrency ceiling on B300, (2) first-ever SGLang HiCache benchmark on a Kimi model.

**Concurrency sweep** (both SGLang configs, at optimal QPS from P1):

| Concurrency | Purpose |
|---|---|
| 1 | Single-stream baseline |
| 8 | Light batch |
| 32 | Moderate |
| 64 | Heavy |
| 128 | Stress (GLM-5 HiCache peak was here) |
| 256 | Overload — B300 has ~1.5TB KV headroom, push it |
| 512 | Ceiling test — expect OOM or preemptions here |

B300's 1.5TB KV cache budget should support far higher concurrency than any previous benchmark. At 32K context per request, ~1.5TB supports ~45 concurrent sessions in KV alone. At 131K, ~11 sessions. The sweep determines the actual scaling curve.

**HiCache comparison** at each concurrency level:

Compare `sglang-tp8-radix` vs `sglang-tp8-hicache`. On GLM-5 (B200), HiCache delivered 2.86x at 128 concurrent. B300 has more VRAM so the crossover where HiCache matters should be at higher concurrency — the sweep will find it.

**Context scaling under load** (if time permits):

| Context | Concurrency | Purpose |
|---|---|---|
| 32K | 64 | Baseline context |
| 64K | 64 | Medium context under load |
| 131K | 32 | Long context stress |
| 256K | 8 | Full native context (may OOM at higher conc) |

### P3: K2.5 Cross-Reference + Economics

Map W1-W6 results back to K2.5's 5 workload categories:

| K2.5 Workload | Nearest W# | K2.5 Result | K2.6 Comparison |
|---|---|---|---|
| reasoning_math | W1 (single-turn) | 41.2 tok/s, TTFT 1943ms | Delta from W1 round=1 |
| agentic_tool_use | W3 | 29.7 tok/s, TTFT 926ms | Delta from W3 |
| multi_turn_qa | W1 (multi-turn) | 16.8 tok/s, TTFT 1565ms | Delta from W1 round=10 |
| code_generation | W5 | 25.2 tok/s, TTFT 4273ms | Delta from W5 |
| long_context_rag | W2 | 9.8 tok/s, TTFT 1915ms | Delta from W2 doc=10K |

Note: hardware differs (g7e vs p5e), so deltas reflect model + engine + hardware combined.

**Economics**:

| | K2.5 (p5e) | K2.6 vLLM (g7e) | K2.6 HiCache (g7e) |
|---|---|---|---|
| Instance $/hr | ~$98 | ~$35 | ~$35 |
| Peak tok/s | TBD | TBD | TBD |
| $/1M output tokens | TBD | TBD | TBD |

### Results Artifacts

```
blueprints/kimi-k2.6/results/
├── vllm-tp8-prefix_W{1-6}_{timestamp}.json
├── sglang-tp8-radix_W{1-6}_{timestamp}.json
├── sglang-tp8-hicache_W{1-6}_{timestamp}.json
├── hicache_concurrency_sweep_{timestamp}.json
├── comparison_table.md
└── progress.md
```

---

## Verification Criteria

### Stage 4a — GPU Health

- [ ] All GPUs report ECC enabled, 0 uncorrectable errors
- [ ] No pending row remaps
- [ ] GPU thermals < 85C under idle
- [ ] No Xid errors in dmesg
- [ ] NCCL test: **skip** (custom allreduce for inference)

### Stage 5 — Serving Stack

- [ ] Health endpoint responds: `curl localhost:8000/health` returns 200
- [ ] Test completion succeeds on both vLLM and SGLang
- [ ] Model loads without OOM on INT4 QAT TP8
- [ ] Startup time < 5 min (vLLM with AOT cache) / < 3 min (SGLang with fast warmup)

### Stage 6 — Benchmark

| Metric | Target | Phase |
|---|---|---|
| BFCL tool-call accuracy | >= 75% (either engine) | P0 |
| Single-stream throughput | >= 40 tok/s | P1 W1 |
| Peak throughput (vLLM) | >= 200 tok/s | P1 W5 |
| HiCache throughput gain | >= 50% over RadixAttention-only | P2 |
| TTFT p99 | < 3000 ms | P1 W1 at QPS=2.0 |
| ITL p99 | < 100 ms | P1 W5 streaming |
| Error rate | < 1% | P1 all workloads |
| No OOM at 128 conc | 0 errors | P2 concurrency sweep |
| Concurrency ceiling | >= 256 at 32K context | P2 pressure test |
| Prefix cache hit rate | >= 75% | P1 W4 |

### Stage 7 — Readiness Audit

- [ ] All verification criteria above checked and recorded
- [ ] No unresolved lessons with severity >= HIGH
- [ ] Results JSON files written for all 3 configs
- [ ] Comparison table generated

---

## Non-Requirements

- FP8 deployment (INT4 QAT preferred — better tok/s/$ with lower memory footprint)
- Multi-node distributed inference (single p6-b300.48xlarge)
- LMCache integration on SGLang (blocked — SGLang MLA connector PR #2629 OPEN; tracked in issue #3192). vLLM+LMCache on K2.6 is **unvalidated, not blocked** — K2.6 uses single-group MLA which avoids the multi-group bug addressed by LMCache PR #2951 (OPEN, targets GLM-5 / DeepSeek V3). A fresh smoke test on vLLM `dev` + LMCache `dev` is needed to confirm status before committing to this path.
- Bare-metal deployment (using EKS)
- Production autoscaling
- Multi-region deployment
- Vision/multimodal benchmarks (text-only workloads)

---

## Security Requirements

- EKS RBAC with least-privilege service accounts
- Private subnets, no public endpoints on GPU pods
- Model weights on local NVMe (ephemeral with instance)
- GPU node group scaled to 0 after benchmark session

---

## Cost Considerations

### Benchmark Session

| Resource | On-Demand | Spot |
|---|---|---|
| p6-b300.48xlarge (~5 hrs) | TBD | ~$75-80 (~$15/hr × 5 hrs) |
| NVMe storage | Included | Included |
| Model download from S3 | ~$5 | ~$5 |
| **Total session** | **TBD** | **~$80-85** |

**Spot pricing** (last 7 days, us-west-2b): $14.55-15.91/hr, stable with low variance. No reclaim events observed in price history (gradual drift, not spikes).

**Spot strategy**: Pre-stage model weights to S3. Checkpoint results to S3 between P-tiers so spot reclaim only loses the current workload run. Use mixed instance policy with on-demand fallback if spot is unavailable.

### Comparison with K2.5

| | K2.5 (p5e, Feb 2026) | K2.6 (B300, this spec) |
|---|---|---|
| Instance cost/hr | ~$98 (capacity block) | ~$15 (spot) |
| Session cost | ~$550-870 | ~$80-85 (spot) |
| Total VRAM | 640 GB | 2,150 GB |
| Max context tested | 32K | 131K+ |
| Engines tested | vLLM only | vLLM + SGLang |
| HiCache tested | No | Yes |

---

## Known Limitations

1. **B300 AMI requirement**: Must use AL2023 NVIDIA AMI (not AL2). AL2 kernel lacks `ib_umad` module required for Fabric Manager on NVL5+ (same as B200 — validated in GLM-5 deployments)
2. **Spot reclaim risk**: Spot instances can be terminated with 2 min warning. Checkpoint results to S3 between P-tiers. Model re-download (~594GB) takes ~15 min on NVMe if reclaimed
3. **Tool parser bugs**: vLLM #37184 (8KB truncation), SGLang may have similar issues. K2.5 parser accuracy fix PR #37384 still open
4. **`transformers` version**: K2.6 requires `>=4.57.1, <5.0.0` but vLLM v0.19.1 ships transformers v5.5.3. May need to pin or verify. SGLang v0.5.10 upgraded to transformers 5.3.0 — same constraint applies
5. **DeepGEMM JIT cold start**: ~15 min on first startup (same as GLM-5 on B200). Use `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` for SGLang (<3 min) and AOT cache for vLLM
6. **No vision benchmarks**: K2.6 includes MoonViT vision encoder. We skip multimodal to focus on serving throughput
7. **Same architecture as K2.5**: Confirmed by official guide — deployment methods directly reusable. Model ID: `moonshotai/Kimi-K2.6`

---

## Deployment Sequence

```
1. Pre-session
   ├── Confirm Kimi K2.6 model ID on HuggingFace
   ├── Verify INT4 QAT weights available and format
   ├── Check transformers version compatibility
   └── Pre-stage model weights to S3 (avoid re-download on spot reclaim)

2. EKS node group (spot)
   ├── Create p6-b300.48xlarge managed node group in us-west-2b
   │   └── Spot instance with on-demand fallback
   ├── AMI: AL2023 NVIDIA (required for Fabric Manager / ib_umad)
   ├── Configure NVMe mount via userData/launch template
   ├── Label: node.kubernetes.io/instance-type=p6-b300.48xlarge
   └── Taint: nvidia.com/gpu=present:NoSchedule

3. Model staging
   ├── Init container: S3 → /mnt/nvme/models/kimi-k2.6/
   └── Verify 594GB downloaded, checksums match

4. GPU pre-flight
   ├── nvidia-smi (driver, CUDA, GPU count, ECC, thermals)
   ├── nvidia-smi topo -m (verify NVSwitch)
   └── NCCL all_reduce test (> 450 GB/s)

5. Track A — vLLM v0.19.1
   ├── Deploy vLLM pod with INT4 QAT config
   ├── P0: Smoke test + BFCL
   ├── P1: W1-W6 suite
   └── Checkpoint results to S3

6. Track B — SGLang v0.5.10
   ├── B1: Deploy SGLang pod with RadixAttention only
   │   ├── P0: Smoke test + BFCL
   │   └── P1: W1-W6 suite
   ├── B2: Redeploy with HiCache enabled
   │   ├── P1: W1-W6 suite (HiCache)
   │   └── P2: Pressure test (concurrency 1 → 512)
   └── Checkpoint results to S3

7. P3: K2.5 cross-reference + economics
   └── Run K2.5 workload prompts, compare to Feb 2026 baseline

8. Teardown
   ├── Sync all results to blueprint results/ and S3
   ├── Write lessons.md
   └── Delete node group (spot auto-terminates)
```

---

## Open Questions

1. **transformers version**: K2.6 requires `>=4.57.1, <5.0.0` but vLLM v0.19.1 ships with transformers v5.5.3. May need to pin or verify compatibility shim
2. **Spot availability**: p6-b300 is new — spot market may have limited history. Check `aws ec2 describe-spot-price-history` before launch. Fall back to on-demand if spot unavailable
3. **EPLB on SGLang**: v0.5.10 added EPLB (Elastic Partition Load Balancing) rebalance for Kimi K2.5 — verify it activates for K2.6 on TP8
4. **B300 sm_103 INT4 kernels**: vLLM v0.19 added B300 support with allreduce fusion. Confirm INT4 QAT Marlin MoE kernels work on sm_103 (should — it's data center Blackwell, not desktop sm_120)

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/kimi-k2.6/results/`, `blueprints/kimi-k2.6/lessons.md`, etc.
