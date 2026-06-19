# Kimi K2.6 Speculative Decode + Dynamic MLA Routing Benchmark

## Status: DRAFT

## Overview

Follow-up to the K2.6 baseline benchmark (COMPLETE, 10,437 tok/s @ c=512). This spec targets the two highest-leverage **untested** optimizations for coding agent workloads: EAGLE3 speculative decoding and dynamic MLA/MHA short-prefill routing.

**Why these two optimizations:**

Our kernel optimization agent proved that MLA decode is 81% of compute at production concurrency and already at the hardware bandwidth ceiling. The only escape from that ceiling is:
1. **Speculative decode (EAGLE3)** — amortize multiple output tokens per decode step, reducing total MLA decode invocations
2. **Dynamic MLA/MHA routing** — bypass MLA's KV expansion for short prefills where standard MHA is faster

Both are untested on K2.6 despite being the top-priority items in `docs/inference-optimization-guide.md` Section 13 (Composition Matrix, items 4-5).

**Baseline (from K2.6 spec, COMPLETE):**

| Engine | Config | Peak tok/s | TTFT p50 (single) | $/M tokens |
|--------|--------|-----------|-------------------|-----------|
| vLLM v0.19.1 | TP8, FLASHINFER_MLA, prefix caching | 10,437 @ c=512 | 22ms | $0.43 |
| SGLang v0.5.10 | TP8, DeepGEMM | 3,400 @ c=512 | 82ms | $1.36 |
| SGLang + HiCache | TP8, 200GB/rank | 3,400 @ c=512 | 83ms | $1.36 |

**Target improvements (hypothesis):**

| Optimization | Expected Impact | Evidence |
|---|---|---|
| EAGLE3 (SGLang) | 1.5-3x single-stream tok/s, 20-50% aggregate | Community g7e data: 1,992 tok/s @ c=128 with MTP=3; FlashMoE paper: 2.1x on MoE models |
| EAGLE3 (vLLM) | Similar magnitude | Requires custom image (`voipmonitor/vllm:cu130-mtp-tuned-v3-20260423`) |
| Dynamic MLA routing | 2-3x TTFT for prefills <1K tokens | vLLM PR #35474 claims 3x TTFT improvement |
| EAGLE3 + prefix caching | Multiplicative | Caching skips prefill, spec decode accelerates decode — independent optimizations |

**Non-goal: Disaggregation.** Our data conclusively shows disagg is counterproductive for K2.6 on single-node (prefix caching already eliminates prefill, TP8 NVLink contention with KV transfer, EP dispatch overhead exceeds budget). See `docs/inference-optimization-guide.md` Section 2 anti-patterns.

---

## Components

### 1. Compute

Two hardware tracks — B300 (primary, existing baseline) and B200 (cost-optimized alternative):

| | p6-b300.48xlarge | p5e.48xlarge (B200) |
|---|---|---|
| **GPUs** | 8x B300 275GB HBM3e | 8x H200 141GB HBM3e |
| **HBM BW per GPU** | 8 TB/s | 8 TB/s |
| **Aggregate BW** | 64 TB/s (identical) | 64 TB/s (identical) |
| **FP8 Compute** | 5 PFLOPS/GPU | 4.5 PFLOPS/GPU |
| **Total VRAM** | 2,200 GB | 1,128 GB |
| **Model + Draft** | 609 GB (28% utilization) | 609 GB (54% utilization) |
| **KV cache budget** | ~1,500 GB | ~450 GB |
| **Max sessions @ 128K** | ~192 | ~58 |
| **Max sessions @ 32K** | ~768 | ~230 |
| **NVSwitch** | NVLink 5 / NVSwitch | NVLink 4 / NVSwitch |
| **Spot cost (est.)** | ~$65/hr | ~$48/hr |
| **Container tags** | `-cu130` (sm_103) | `-cu124` (sm_90) |
| **Decode throughput** | Baseline: 10,437 tok/s | Expected: ~10,000 tok/s (same BW) |

**Key insight: B200 and B300 have identical HBM bandwidth.** Since decode is BW-bound, throughput is nearly the same. B300 only wins on VRAM capacity (more concurrent sessions at long context).

**Recommendation**:
- **Agentic coding (32K effective after prefix cache, <200 users)**: B200 is sufficient and 26% cheaper
- **Long-context or high-concurrency (128K, >100 sessions)**: B300 needed for KV headroom
- **Benchmark both** in Phase 0 to confirm BW parity

**Primary track (existing infrastructure)**:
- **Platform**: EKS on EC2 (spot)
- **EKS Cluster**: `qn-sglang-eks-cluster` (v1.32, us-west-2)
- **GPU Node**: p6-b300.48xlarge
- **AZ**: us-west-2b
- **NVMe**: 28TB RAID0 at /mnt/nvme
- **Container tags**: Must use `-cu130` for sm_103

**Cost-optimized track (B200)**:
- **Platform**: EKS on EC2 (spot or capacity block)
- **GPU Node**: p5e.48xlarge
- **AZ**: us-west-2a/b (broader availability)
- **NVMe**: 8x 3.84TB NVMe RAID0 at /mnt/nvme
- **Container tags**: Standard `-cu124` or `-latest` (sm_90, mature support)
- **Advantage**: No sm_103 compatibility issues, wider engine/image support

### 1a. GPU & NCCL Pre-Flight

Same as K2.6 baseline — B300 NVSwitch topology is proven. For B200 track: H200 NVSwitch is equally mature (sm_90 has full NCCL support). No additional pre-flight needed beyond standard health check.

### 2. Model

- **Model**: `moonshotai/Kimi-K2.6` (1T MoE, 32B active, INT4 QAT)
- **Draft model**: `lightseekorg/kimi-k2.6-eagle3` (EAGLE3 architecture)
- **Model size**: ~594 GB (target) + draft model overhead (~TBD, expect 5-20 GB)
- **Context length**: 131072 (same as baseline)

### 3. Software Matrix

| Track | Engine | Version | Config | Key Flags |
|-------|--------|---------|--------|-----------|
| A | SGLang | v0.5.10+ | EAGLE3 spec decode | `--speculative-algorithm=EAGLE3 --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-draft-model-path lightseekorg/kimi-k2.6-eagle3 --speculative-draft-attention-backend trtllm_mha` |
| B | vLLM | custom | EAGLE3 spec decode | Image `voipmonitor/vllm:cu130-mtp-tuned-v3-20260423`, draft model `lightseekorg/kimi-k2.5-eagle3-mla` |
| C | vLLM | v0.20+ | Dynamic MLA routing | Cherry-pick PR #35474 or use version with merged support |
| D | Best of A/B | — | EAGLE3 + prefix caching | Validate composition |

### 4. Networking

- SSH via EKS kubectl exec or direct spot instance access
- Model weights pre-loaded on NVMe from baseline session
- Draft model download: `hf download lightseekorg/kimi-k2.6-eagle3 --local-dir /mnt/nvme/models/kimi-k26-eagle3`

### 5. Storage

- **Target model**: `/mnt/nvme/models/kimi-k26-fp8/` (reuse from baseline)
- **Draft model**: `/mnt/nvme/models/kimi-k26-eagle3/`
- **Results**: Blueprint `results/` directory

---

## Experiment Protocol

### Phase 0: Roofline Characterization

**Goal**: Establish the theoretical throughput ceiling for K2.6 on B300 and B200, quantify optimization headroom, confirm BW parity between SKUs, and identify the resource exhaustion point that would justify multi-node scaling.

#### Hardware Roofline Parameters (B300 8-GPU node)

| Resource | Per-GPU | 8-GPU Aggregate | Notes |
|----------|---------|-----------------|-------|
| HBM bandwidth | 8 TB/s | 64 TB/s (theoretical) | Practical ~85% = 54 TB/s |
| FP8 compute | 5 PFLOPS | 40 PFLOPS | DeepGEMM FP8 matmul |
| FP4 compute | 10 PFLOPS | 80 PFLOPS | INT4 QAT model weights |
| NVSwitch bisection BW | — | 1.8 TB/s | TP8 all-to-all |
| HBM capacity | 275 GB | 2,200 GB | KV cache budget = total - model - draft |

#### Model Arithmetic Intensity

For K2.6 (32B active, 384 experts, INT4 weights, MLA compressed KV):

```
Bytes per token (decode, batch=1):
  - Weight read: 32B active params × 0.5 bytes (INT4) = 16 GB
  - KV read: MLA compressed (d_c=512) × num_layers × batch = ~0.8 GB
  - Total: ~16.8 GB/token at batch=1

Bytes per token (decode, batch=N):
  - Weight read: 16 GB (amortized across batch → 16/N GB per token)
  - KV read: grows linearly with context × batch
  - Arithmetic intensity = FLOPs / Bytes = 2×32B×2 / (16GB/N + KV)

Roofline crossover (BW-bound → Compute-bound):
  - At batch=1: AI ≈ 0.008 → deeply BW-bound
  - At batch=N*: AI = machine_AI → crossover to compute-bound
  - B300: peak_FLOPS / peak_BW = 40e15 / 54e12 ≈ 740 ops/byte
  - Crossover batch ≈ 16GB × 740 / (2 × 32B × 2) ≈ 92 tokens
  - Above batch ~92: compute-bound; below: BW-bound
```

#### Microbenchmarks to Run

| Test | Tool | What it measures |
|------|------|-----------------|
| Memory bandwidth (sustained) | `nccl-tests -b 1G -e 8G -t 8` + HBM memcpy kernel | Actual vs spec BW |
| DeepGEMM FP8 matmul | Custom GEMM benchmark at MoE shapes | Actual FP8 TFLOPS |
| NVSwitch all-reduce | `nccl-tests` all_reduce at TP8 | Actual collective BW |
| Attention BW | Profile MLA decode at batch=1,8,64,256 | Per-layer time |
| MoE dispatch | Profile expert routing + matmul at batch=1,8,64,256 | Routing overhead |

#### Per-Layer Time Breakdown

Profile K2.6 decode at c=1, 64, 256, 512 using:
```bash
# nsys profile with kernel-level timing
nsys profile --trace=cuda,nvtx -o /mnt/nvme/profiles/roofline_c${CONC} \
  python -c "
import sglang as sgl
# Single decode step at target concurrency
..."

# Or use vLLM profiling:
vllm serve ... --collect-detailed-traces trace_file.json
```

Expected breakdown (hypothesis from kernel-opt-agent data):

| Component | c=1 (% time) | c=64 | c=256 | c=512 |
|-----------|-------------|------|-------|-------|
| MLA decode attention | 35% | 55% | 70% | 81% |
| MoE expert matmul | 40% | 25% | 15% | 10% |
| MoE routing/dispatch | 10% | 8% | 5% | 3% |
| NVLink all-reduce | 5% | 5% | 5% | 3% |
| Sampling + scheduling | 10% | 7% | 5% | 3% |

#### Theoretical Maximum Throughput

```
At batch=512 (compute-bound regime):
  FLOPs per token: 2 × 32B × 2 = 128 GFLOPs (forward pass, active params)
  Peak FP8: 40 PFLOPS (aggregate)
  Theoretical max: 40e15 / 128e9 = 312,500 tok/s

But! MoE routing adds overhead (~15%), scheduling (~5%), NVLink (~3%):
  Practical ceiling: 312,500 × 0.77 ≈ 240,000 tok/s

Current achieved: 10,437 tok/s @ c=512
Efficiency: 10,437 / 240,000 = 4.3%

Where does the 95.7% gap come from?
  - KV cache memory limits max batch (not running batch=312K!)
  - Actual concurrent sequences = 512, not the compute-optimal batch
  - Scheduling overhead, token sampling, Python GIL, CUDA launch latency
  - Output token generation is serial per sequence (can't batch future tokens)
```

The "real" roofline for decode with actual batch sizes:

```
At actual batch=512 (BW-bound, since 512 < compute crossover for full model):
  Actual BW: ~54 TB/s × utilization factor
  Bytes per batch decode step: 16 GB (model weights) + KV per token
  Max steps/s: 54e12 / 16e9 ≈ 3,375 steps/s
  Tokens per step (no spec decode): 512 (one per sequence)
  Theoretical max: 3,375 × 512 = 1,728,000 tok/s

  But MLA KV read adds ~0.8 GB × 512 = 410 GB per step at full context
  Adjusted: 54e12 / (16e9 + 410e9) ≈ 127 steps/s → 65,000 tok/s

Current: 10,437 tok/s → 16% of adjusted BW ceiling
Gap source: KV cache read at long contexts, scheduling, kernel launch

At short context (agentic turns, ~2K after cache hit):
  KV read: negligible vs model weights
  Theoretical: ~3,375 × 512 ≈ 1.7M tok/s
  Achieved efficiency would be ~0.6% — dominated by scheduling/launch overhead
```

#### Multi-Node Decision Framework

The roofline analysis answers: **"When does single-node resource exhaustion force multi-node?"**

| Resource Exhaustion | Symptom | When it Happens | Multi-Node Solution | K2.6 B300 Status |
|---|---|---|---|---|
| **KV cache memory** | Can't grow batch beyond N | Total KV for max_seqs × max_context > VRAM budget | Disagg P/D: offload KV to decode-only nodes | **NOT exhausted** — 2,200 GB total, model=594 GB, draft=~15 GB, KV budget=~1,500 GB → supports 256+ seqs at 128K |
| **Compute (prefill)** | TTFT grows linearly with prompt length | Single prefill saturates 8 GPUs (prompt > ~32K tokens) | Dedicated prefill node(s) | **NOT exhausted** — prefix caching eliminates 95%+ of prefill; residual turns are <2K |
| **Compute (decode)** | Throughput plateaus, can't increase c further | Scheduler queue depth growing, all GPUs at 100% | More decode replicas (independent nodes) | **APPROACHING** — at c=512 we're at ~16% of BW ceiling; headroom exists in batch efficiency |
| **Network (multi-turn state)** | KV eviction between turns causes re-prefill | Session count > KV cache slots (idle sessions evicted) | Disagg with remote KV store (Mooncake, NIXL) | **NOT exhausted** — HiCache 200 GB/rank on NVMe retains idle sessions |
| **Memory bandwidth** | Can't read weights + KV fast enough per step | Batch size grows beyond BW/model_size crossover | Not solvable by multi-node (each node has same BW) | **CEILING** — this IS the bottleneck at c=512 |

**Key Insight**: For K2.6 on B300, the binding constraint is **memory bandwidth per decode step**, NOT any capacity limit. Multi-node disaggregation does NOT help because:
1. Each decode node faces the same BW wall
2. Adding nodes just adds replicas (horizontal scale) — same per-request latency
3. KV transfer overhead between nodes actually makes it WORSE for short agentic turns

**When multi-node DOES become necessary** (not for this spec, but documenting the threshold):

```
Condition 1: Concurrent sessions > KV cache capacity
  K2.6 threshold: >256 sessions at 128K context (or >1024 at 32K)
  → Disagg with external KV store

Condition 2: QPS exceeds single-node saturation
  K2.6 threshold: >~20 req/s sustained (at 512 output tokens avg)
  → Horizontal replicas behind load balancer (not disagg)

Condition 3: Context length > 128K (future models)
  → Long-context prefill nodes (not relevant for K2.6)
```

**Decision rule for this benchmark**: If Phase 4 full-stack achieves <80% of BW roofline at target concurrency, the gap is in software (scheduling, kernels) and further single-node optimization is possible. If ≥80%, we've hit the hardware wall and only speculative decode (more tokens per BW-limited step) or horizontal replicas can improve throughput.

#### B200 vs B300 Comparison (Phase 0 deliverable)

Run the same roofline microbenchmarks on both SKUs to confirm BW parity:

| Microbenchmark | B300 (expected) | B200 (expected) | Implication |
|---|---|---|---|
| HBM sustained BW | ~54 TB/s (85% of 64) | ~54 TB/s (85% of 64) | Identical decode speed |
| DeepGEMM FP8 | ~35 PFLOPS | ~32 PFLOPS | B300 slightly faster prefill |
| NVSwitch all-reduce TP8 | ~1.8 TB/s (NVL5) | ~900 GB/s (NVL4) | B300 2x faster collective |
| K2.6 decode @ c=512 | 10,437 tok/s (measured) | ~10,000 tok/s (estimated) | <5% difference |

**If confirmed** (B200 within 5% of B300 on decode): B200 becomes the recommended SKU for agentic coding, saving ~$17/hr (26%).

**B200 limitations** (document these):
- Max 58 sessions at 128K (vs 192 on B300) — OK for <200 agentic users at 32K
- HiCache budget: ~50 GB/rank max (vs 200 GB/rank on B300) — less cold KV retention
- NVLink 4 all-reduce is 2x slower — marginal impact for TP8 decode (all-reduce is <5% of time)
- sm_90: mature CUDA support, no `-cu130` compatibility issues (actually an advantage)

#### Deliverables

- [ ] Measured HBM bandwidth (actual vs spec) — **both B300 and B200**
- [ ] Measured DeepGEMM FP8 TFLOPS (actual vs spec)
- [ ] Per-layer time breakdown at c=1, 64, 256, 512
- [ ] Roofline plot: achieved tok/s vs theoretical ceiling at each concurrency
- [ ] Efficiency % at each operating point
- [ ] Multi-node decision: confirmed NOT needed (or documented threshold)
- [ ] **B200 parity confirmed or denied**: measured tok/s within 5% of B300
- [ ] **Optimization headroom quantified**: "X% of theoretical remains; Y% is in {component}"
- [ ] **SKU recommendation**: B200 or B300 based on workload profile

---

### Phase 1: EAGLE3 on SGLang (Track A)

**Goal**: Measure speculative decode impact on K2.6 throughput and latency.

```bash
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path /mnt/nvme/models/kimi-k26-fp8 \
  --tp 8 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --speculative-algorithm=EAGLE3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path /mnt/nvme/models/kimi-k26-eagle3 \
  --speculative-draft-attention-backend trtllm_mha \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000
```

**Sweep parameters** (find optimal spec decode config):

| Parameter | Values to test | Why |
|-----------|---------------|-----|
| `speculative-num-steps` | 1, 2, 3, 4 | More steps = more draft tokens but higher rejection risk |
| `speculative-num-draft-tokens` | 2, 4, 6, 8 | Trade-off: more tokens drafted vs verification cost |
| `speculative-eagle-topk` | 1, 2, 4 | Wider search = better acceptance but more compute |

**Measurements** (at each config):
- W1-W6 workloads from baseline (direct comparison)
- Concurrency sweep: 1, 8, 32, 64, 128, 256, 512
- **Acceptance rate**: tokens accepted / tokens drafted (key efficiency metric)
- **Spec decode overhead**: VRAM consumed by draft model
- **Crossover point**: concurrency where spec decode stops helping

### Phase 2: EAGLE3 on vLLM (Track B)

Same measurements as Phase 1 but on vLLM with custom image.

```bash
# Using custom image with EAGLE3 + TRITON_MLA + CUDA graphs for Blackwell
docker run --gpus all --network host \
  voipmonitor/vllm:cu130-mtp-tuned-v3-20260423 \
  --model /mnt/nvme/models/kimi-k26-fp8 \
  --tensor-parallel-size 8 \
  --speculative-model /mnt/nvme/models/kimi-k26-eagle3 \
  --num-speculative-tokens 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

**Key question**: Does vLLM's CUDA graph + FLASHINFER_MLA advantage persist with speculative decode?

### Phase 3: Dynamic MLA/MHA Routing (Track C)

**Goal**: Measure TTFT improvement for short prefills on agentic workloads.

vLLM PR #35474 adds dynamic routing that uses standard MHA (faster kernel) for prefills <1024 tokens, switching to MLA only for longer contexts. This matches the agentic coding pattern perfectly (most turns are <1K after system prompt is cached).

**Setup**: Cherry-pick PR #35474 into vLLM v0.19.1 or use a version where it's merged.

**Measurements**:
- W3 (Agentic Tool Calling) — primary beneficiary
- W1 (Multi-Turn Chat) — second beneficiary
- TTFT sweep by input length: 128, 256, 512, 1024, 2048, 4096 tokens
- Measure MLA vs MHA routing decision accuracy

### Phase 4: Full Additive Stack (Track D)

**Goal**: Deploy ALL composable optimizations simultaneously and measure the compound effect.

From our composition matrix (`docs/inference-optimization-guide.md` Section 13), these optimizations are additive/multiplicative and should stack:

| Optimization | Layer | Why it composes |
|---|---|---|
| CUDA graphs | Decode scheduling | Eliminates launch overhead — independent of everything else |
| MoE tile config (H200/B300) | Kernel params | Already baked into engine — independent |
| Prefix caching | Prefill elimination | Skips prefill entirely — doesn't touch decode path |
| EAGLE3 speculative decode | Decode acceleration | Reduces decode steps — operates on remaining work after cache hit |
| Dynamic MLA/MHA routing | Prefill kernel selection | Faster kernel for short residual prefills after cache hit |
| HiCache (NVMe/DRAM KV offload) | KV capacity | Extends context retention between turns — independent of decode speed |

**The full stack config:**

```bash
# SGLang full stack (if SGLang wins Phases 1-2)
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path /mnt/nvme/models/kimi-k26-fp8 \
  --tp 8 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  # EAGLE3
  --speculative-algorithm=EAGLE3 \
  --speculative-num-steps <best_from_phase1> \
  --speculative-eagle-topk <best_from_phase1> \
  --speculative-num-draft-tokens <best_from_phase1> \
  --speculative-draft-model-path /mnt/nvme/models/kimi-k26-eagle3 \
  --speculative-draft-attention-backend trtllm_mha \
  # HiCache
  --enable-hierarchical-cache \
  --hicache-size 200 \
  # Standard
  --trust-remote-code \
  --host 0.0.0.0 --port 30000
```

```bash
# vLLM full stack (if vLLM wins Phases 1-2)
# Image: voipmonitor/vllm:cu130-mtp-tuned-v3-20260423 or patched v0.20+
vllm serve /mnt/nvme/models/kimi-k26-fp8 \
  --tensor-parallel-size 8 \
  # EAGLE3
  --speculative-model /mnt/nvme/models/kimi-k26-eagle3 \
  --num-speculative-tokens <best_from_phase2> \
  # Prefix caching
  --enable-prefix-caching \
  # Dynamic MLA routing (if PR #35474 available)
  --enable-dynamic-mla-routing \
  # Standard
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

**Incremental layering protocol** (isolates contribution of each optimization):

```
Step 0: Roofline characterization — establish theoretical ceiling and efficiency %
Step 1: Baseline (vLLM v0.19.1 TP8 prefix caching) — already measured: 10,437 tok/s (X% of ceiling)
Step 2: + EAGLE3 only — measure delta and new efficiency %
Step 3: + EAGLE3 + dynamic MLA routing — measure delta
Step 4: + EAGLE3 + dynamic MLA routing + HiCache — measure delta
Step 5: Full stack — confirm no regression vs Step 3/4; plot against roofline
```

This layering lets us attribute gains precisely. If any layer causes regression, we remove it from the final config.

**Validation checklist**:
1. No accuracy degradation (compare outputs to baseline on 50 coding prompts — exact tool call match)
2. No deadlocks or crashes across full concurrency sweep (1→512)
3. Each layer is net-positive or neutral (remove if negative)
4. VRAM overhead of full stack still allows c=256+ operation
5. Cold start penalty documented (draft model + HiCache init)

**What we expect the full stack to achieve:**

```
Baseline:           128 tok/s single-stream, 10,437 tok/s @ c=512
+ EAGLE3:           ~200-250 tok/s single-stream (+60-100%), aggregate TBD
+ Dynamic MLA:      TTFT 45ms → ~15-20ms for agentic turns
+ HiCache:          No throughput gain at high c, but retains KV across idle turns
= Full stack:       ~200+ tok/s single-stream, ≤20ms agentic TTFT, $0.30-0.35/M tokens
```

The compound hypothesis: prefix caching eliminates redundant prefill (103x TTFT already), EAGLE3 accelerates the remaining decode steps (1.5-2x), and dynamic MLA routing makes the residual short prefills even faster (2-3x on the already-small remaining TTFT). These operate on different parts of the request lifecycle and should multiply.

### Phase 5: Further Single-Node Frontier Expansion

**Goal**: After Phases 1-4 establish the speculative decode frontier, test additional software-level optimizations that can push further without adding hardware.

These target the **73% framework overhead** at low batch (agentic workloads) and the remaining compute slack.

#### 5A: torch.compile / Graph Fusion

**What it does**: Fuses small CUDA kernels into larger ones, eliminates Python→CUDA dispatch overhead, and enables operator fusion (e.g., RMSNorm + residual + quant in one kernel).

**Expected impact**: +10-20% at low batch (c=1-8), diminishing at high batch where kernels dominate.

```bash
# vLLM with torch.compile (available in v0.19+)
vllm serve /mnt/nvme/models/kimi-k26-fp8 \
  --tensor-parallel-size 8 \
  --compilation-config '{"level": 3, "backend": "inductor"}' \
  --enable-prefix-caching \
  --speculative-model /mnt/nvme/models/kimi-k26-eagle3 \
  --num-speculative-tokens <best> \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000

# SGLang equivalent (torch.compile enabled by default in recent versions)
# Add --disable-cuda-graph to measure compile-only benefit isolated from graphs
```

**Measurements**:
- Compare with/without torch.compile at c=1, 4, 8, 32, 128
- Measure cold start penalty (compilation adds 5-15 min to first request)
- Profile: count kernel launches per decode step (should decrease with fusion)
- Verify no accuracy regression (compiled path must match eager output)

**Risk**: torch.compile + EAGLE3 + CUDA graphs may have conflicts (graph capture fails on compiled code in some versions). Test compilation levels 1-3 independently.

#### 5B: Overlap Scheduling (Async Prefetch)

**What it does**: While GPU executes layer N's compute, the scheduler prefetches layer N+1's weights from HBM and prepares the next batch's metadata. Reduces inter-kernel idle time.

**Expected impact**: +10-30% at medium batch (c=32-128), where scheduling overhead is significant but not dominant.

```bash
# SGLang already implements overlap scheduling via:
#   --overlap-scheduler (enabled by default in v0.5.10+)
# Verify it's active:
SGLANG_LOG_LEVEL=INFO python -m sglang.launch_server ... 2>&1 | grep "overlap"

# vLLM V1 async scheduler (if available in v0.20+):
vllm serve ... --scheduler-policy async
```

**Measurements**:
- A/B test: overlap enabled vs disabled at c=8, 32, 64, 128, 256
- Profile: measure inter-kernel gap (nsys timeline should show tighter packing)
- Key metric: "GPU active %" (time GPU is running kernels / total wall time)
- Target: GPU active % from ~30% → ~45% at c=4 (agentic sweet spot)

#### 5C: Pipeline Parallelism (PP2×TP4 vs TP8)

**What it does**: Splits the model into 2 pipeline stages of 4 GPUs each (PP2), with TP4 within each stage. Reduces all-reduce from 8-way to 4-way (less NVLink traffic per step).

**Expected impact**: +5-15% on aggregate throughput. Trade-off: adds pipeline bubble overhead but halves all-reduce size.

```bash
# vLLM with PP
vllm serve /mnt/nvme/models/kimi-k26-fp8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --enable-prefix-caching \
  --speculative-model /mnt/nvme/models/kimi-k26-eagle3 \
  --num-speculative-tokens <best> \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000

# SGLang equivalent
python -m sglang.launch_server \
  --model-path /mnt/nvme/models/kimi-k26-fp8 \
  --tp 4 --dp 2 \
  ...
```

**Why this might work for K2.6**:
- K2.6 has 61 layers. PP2 = 30-31 layers per stage, TP4 within each stage
- All-reduce at TP4: 4-way across NVLink (faster than 8-way)
- NVSwitch on B300 means PP inter-stage communication is also NVLink (no PCIe penalty)
- Pipeline bubble = 1/(num_microbatches) — at c≥32, microbatch count is high enough that bubble is <5%

**Why this might NOT work**:
- MoE expert routing needs to see all layers' routing decisions for load balance
- KV cache must be split across pipeline stages (complicates prefetch caching)
- EAGLE3 draft model may not support PP (verification needs full model forward pass)
- Additional complexity for marginal gain

**Measurements**:
- PP2×TP4 vs TP8 at c=64, 128, 256, 512 (concurrency sweep)
- Measure pipeline bubble % (should be <5% at c≥64)
- Verify EAGLE3 compatibility (spec decode + PP is non-trivial)
- If EAGLE3 incompatible with PP: measure PP benefit without spec decode and compare against TP8+EAGLE3

#### 5D: NVFP4 Tensor Core for Speculative Verification

**What it does**: Uses B300's FP4 tensor cores (10 PFLOPS, 2x faster than FP8) for the speculative decode verification step. The verification is compute-bound (single forward pass verifying multiple draft tokens simultaneously) — this is exactly where FP4 compute helps.

**Expected impact**: +5-10% on effective EAGLE3 throughput by making verification faster, allowing higher `num_draft_tokens` before overhead exceeds gain.

**Prerequisite**: K2.6 weights are already INT4 QAT. The question is whether the tensor core FP4 path can execute INT4-quantized matmuls faster than the FP8 path currently used for dequant→compute.

```bash
# This requires vLLM/SGLang to expose FP4 compute path for verification
# As of May 2026, this may require:
#   1. Custom Triton kernel with fp4 accumulation
#   2. Or cutlass 3.x with sm_103 FP4 warp-specialized GEMM

# Test: measure verification step time in isolation
# Profile with nsys: look for "verify" or "score" kernels in EAGLE3 path
```

**Why this is specific to EAGLE3**:
- Normal decode is BW-bound (reading 16 GB weights) — FP4 compute speed doesn't help
- But EAGLE3 verification processes 4 draft tokens in one forward pass (4x the arithmetic intensity)
- At 4 tokens/step, arithmetic intensity crosses the roofline into compute-bound territory
- FP4 cores (10 PFLOPS) directly accelerate this compute-bound verification

**Measurements**:
- Profile EAGLE3 verification step: is it BW-bound or compute-bound?
- If compute-bound: measure with FP4 vs FP8 tensor core path
- Key metric: verification latency (ms) and how it changes with `num_draft_tokens` (2, 4, 6, 8)
- If verification becomes faster, test higher `num_draft_tokens` (crossover shifts right)

**Risk**: FP4 accumulation may introduce numerical drift in MoE routing (softmax in expert selection). Validate with 50-prompt accuracy check.

---

#### Phase 5 Composition

These optimizations target different bottlenecks and should compose:

| Opt | Targets | Composes with |
|---|---|---|
| torch.compile | Kernel launch overhead | Everything (pure software) |
| Overlap scheduling | Inter-kernel idle time | Everything (scheduler-level) |
| PP2×TP4 | All-reduce bandwidth | May conflict with EAGLE3 |
| NVFP4 verification | Spec decode compute | EAGLE3 only (makes spec decode more efficient) |

**Expected full Phase 5 stack (if all compose)**:

```
Phase 4 result:     ~17,000 tok/s @ c=512, ~250 tok/s single-stream
+ torch.compile:    +15% at low c → ~290 tok/s single-stream
+ overlap sched:    +20% at medium c → ~20,000 tok/s @ c=512
+ PP2×TP4:          +10% aggregate (if EAGLE3 compatible) → ~22,000 tok/s
+ NVFP4 verify:     +8% on spec decode efficiency → ~310 tok/s single-stream
= Phase 5 target:   ~300+ tok/s single-stream, ~20-22K tok/s @ c=512, $0.22-0.25/M tokens
```

**Incremental layering** (continues from Phase 4):

```
Step 6: Phase 4 best + torch.compile — measure delta at c=1,4,8
Step 7: + overlap scheduling — measure delta at c=32,128
Step 8: PP2×TP4 test (without EAGLE3 first, then with) — compare vs TP8+EAGLE3
Step 9: NVFP4 verification (profile only — requires kernel availability)
Step 10: Full Phase 5 stack — confirm no regression, plot against roofline
```

---

## Metrics

### Primary (must beat baseline to declare success)

| Metric | Baseline (vLLM v0.19.1) | Target |
|--------|------------------------|--------|
| Single-stream tok/s | 128 | **≥200** (EAGLE3) |
| Aggregate tok/s @ c=128 | 4,716 | **≥6,000** |
| Aggregate tok/s @ c=512 | 10,437 | **≥12,000** (stretch) |
| TTFT p50 (agentic, <1K input) | 45ms | **≤20ms** (dynamic MLA routing) |
| $/M output tokens (optimal config) | $0.43 | **≤$0.35** |

### Secondary

| Metric | Purpose |
|--------|---------|
| EAGLE3 acceptance rate | Efficiency of draft model — >70% is good |
| Spec decode VRAM overhead | Impact on max_concurrent_seqs |
| MLA/MHA routing accuracy | Does the router make correct decisions? |
| Cold start with draft model | Additional startup time |
| Error rate at c=512 with spec decode | Stability under load |

### Derived

| Metric | Formula |
|--------|---------|
| Effective tokens per decode step | accepted_tokens + 1 (verification token) |
| EAGLE3 speedup factor | baseline_tps / eagle3_tps at same concurrency |
| Crossover concurrency | Where EAGLE3 stops beating baseline |
| Composition multiplier | composed_tps / max(individual_tps) |

---

## Workloads

All workloads use the **canonical cards** under `standards/benchmark-commons/workloads/` — the runner reads the card by `catalog_id`, so parameter values live in the card, not in this spec. The spec's job is to name the customer scenario, pick the right card, and call out which optimization track the card validates.

| W# | Customer scenario | Customers | Catalog card | Why this card | Optimization it validates |
|----|-------------------|-----------|--------------|----|---|
| **W1** | **Multi-Turn Chat** — conversations with 1–10 rounds of back-and-forth | Copilots, virtual assistants, customer support | [`chatbot-short`](../../../standards/benchmark-commons/workloads/chatbot-short.yaml) | Short latency-sensitive turns; TTFT + ITL baseline | EAGLE3 (short-output turns amortize draft-model cost) |
| **W2** | **RAG Document Q&A** — documents from 2K–10K tokens injected as context | Enterprise knowledge bases, legal & medical search | [`rag-long-context`](../../../standards/benchmark-commons/workloads/rag-long-context.yaml) | Measures shared-prefix caching (103× TTFT gain on K2.6 baseline) | Prefix caching + dynamic MLA/MHA routing |
| **W3** | **Agent Tool Calling** — multi-step agents with 5–10 tool calls per session | Coding agents, research agents, workflow automation | [`coding-agent`](../../../standards/benchmark-commons/workloads/coding-agent.yaml) | 12K system prompt + 20 tool defs + 8 turns/session matches Claude Code loop | **Full stack**: caching + EAGLE3 + dynamic MLA routing |
| **W4** | **Shared System Prompt** — same prompt across 4–16 users served concurrently | Multi-tenant SaaS, ISVs serving multiple customers | [`coding-agent`](../../../standards/benchmark-commons/workloads/coding-agent.yaml) (reused, `prefix_reuse: true`) | Card's `prefix_reuse` flag shares system prompt across concurrent sessions — exactly the multi-tenant case; sweep `concurrent_sessions` to span 4–16 | EAGLE3 × prefix-cache interaction under contention |
| **W5** | **Production Traffic Mix** — real conversation data at varying concurrency | Any customer moving from PoC to production | [`qps-sweep`](../../../standards/benchmark-commons/workloads/qps-sweep.yaml) | 2K in / 512 out QPS sweep is the canonical go/no-go for sustained load | Saturation point with spec decode active |
| **W6** | **Long Context Scaling** — inputs from 1K to 16K tokens at production load | Code generation, document analysis, summarization | [`chatbot-long`](../../../standards/benchmark-commons/workloads/chatbot-long.yaml) | 32K-input prefill card; run at lower sweep points (1K/4K/16K) via concurrency-sweep overlay | Dynamic MLA/MHA routing crossover |

**Parameter overrides vs the cards**: if a K2.6-specific run needs different values (e.g. W4 wants `concurrent_sessions: [4,8,16]` rather than the card's default `4`), record the override in the blueprint sidecar `domains/gpu-serving/blueprints/kimi-k2.6-speculative/benchmark.yaml` — do **not** edit the canonical card, and do **not** inline divergent YAML here. The spec names the scenario; the card defines the workload; the sidecar captures deltas.

**Gaps to raise separately** (do not block this spec):

- W1 card does not parameterize multi-turn round count — today a round is a separate request. If round-aware evaluation is needed later, propose a card extension.
- No dedicated "shared system prompt" multi-tenant card exists; W4 currently borrows `coding-agent`'s `prefix_reuse`. If other blueprints also need this pattern, propose a new `shared-prefix` card.
- W6 uses `chatbot-long` (fixed 32K); a full 1K→64K sweep needs an additional card or a runner-level parameter sweep.

### W7 / W8 — Stressed variants (sidecar overrides)

Earlier drafts defined W7 (Claude-Code-like 15-turn coding loop) and W8 (QPS sweep extended to 32) as separate workloads. These are **not new workloads** — they are W3 and W5 run with stressed parameters to probe EAGLE3 overhead at longer sessions and higher QPS. Capture those overrides in the blueprint sidecar rather than inlining different YAML here:

- **W7 stress**: `coding-agent` with `turns_per_session: 15`, `concurrent_sessions: [1,4,8,16]`
- **W8 stress**: `qps-sweep` with `rates: [0.5, 1, 2, 4, 8, 16, 32]`

Location: `domains/gpu-serving/blueprints/kimi-k2.6-speculative/benchmark.yaml` (sidecar). If either override produces a pattern worth reusing across blueprints, promote it to a new canonical card.

---

## Success Criteria

1. **Roofline established**: Theoretical ceiling quantified, current efficiency % measured, optimization headroom identified
2. **EAGLE3 validated**: Measured acceptance rate, speedup factor, and crossover point on B300
3. **Single-stream improvement ≥1.5x**: Full stack brings single-stream from 128 to ≥192 tok/s
4. **Agentic TTFT ≤20ms**: Dynamic MLA routing + prefix caching reduces TTFT for cached agentic turns by ≥2x
5. **Full stack is net-positive**: Compound config beats best individual optimization (no negative interaction)
6. **No accuracy regression**: Outputs match baseline on 50 coding prompts (exact tool call match)
7. **Attribution table**: Each optimization's isolated contribution measured via incremental layering, plotted against roofline
8. **Multi-node decision documented**: Either "not needed (X% headroom remains)" or "crossover at Y sessions/QPS"
9. **Updated guide**: Results flow back into `docs/inference-optimization-guide.md` with measured data replacing "untested" labels
10. **Production config exported**: Final `serve.sh` / pod spec with all flags for immediate reuse

## Termination Conditions

- **Success**: ≥2 of the 5 criteria above met
- **Partial**: Characterization complete even if no improvement (documents ceilings)
- **Hard stop**: 1 capacity block session (~8 hours)
- **Known blocker**: If EAGLE3 + TP8 deadlocks (vLLM #41404), switch to SGLang-only

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| EAGLE3 + TP8 deadlock at concurrency | Medium | High | Use SGLang (more mature spec decode on MoE) |
| Draft model incompatible with B300 sm_103 | Low | High | Use `trtllm_mha` backend; Triton fallback if needed |
| Dynamic MLA routing not merged in any stable release | Medium | Medium | Cherry-pick PR #35474; isolated test |
| EAGLE3 acceptance rate too low (<50%) | Low | High | Tune num_steps/draft_tokens; try alternative draft model |
| Spec decode VRAM overhead prevents c=512 | Medium | Medium | Reduce gpu_memory_utilization; test lower max_model_len |
| CUDA graph capture fails with spec decode | Medium | Medium | Fall back to eager mode (still faster than no spec decode) |

## Verification Criteria

### Stage 4a — GPU Health

Pre-flight is minimal (reuse existing B300 node from K2.6 baseline). Verify only:

- [ ] All 8 GPUs report ECC enabled, 0 uncorrectable errors
- [ ] No pending row remaps
- [ ] GPU thermals < 85°C under idle
- [ ] NVSwitch topology confirmed (8-way full mesh)
- [ ] No Xid errors in dmesg since last deployment

### Stage 5 — Serving Stack

For each track (A/B/C/D), verify:

- [ ] Health endpoint responds: `curl -s localhost:PORT/health` returns 200
- [ ] Test completion succeeds with speculative decode active (output is coherent)
- [ ] Draft model loads without OOM alongside target model
- [ ] EAGLE3 acceptance rate > 0% on first request (draft model is functional)
- [ ] Startup time documented (target + draft model cold start)
- [ ] No `CUDA out of memory` or workspace allocation errors in logs

### Stage 6 — Benchmark

**Workload selection** from [standard workload catalog](../../../standards/benchmark-commons/workloads/):

| Workload | Catalog ID | Why | Artifact produced |
|----------|-----------|-----|-------------------|
| W1: Multi-Turn Chat | `chatbot-short` | Baseline comparison to K2.6 results; EAGLE3 primary benefit | 1 artifact per rounds setting |
| W2: RAG Document Q&A | `rag-long-context` | Dynamic MLA routing benefit at 2K tier | 1 artifact per input length |
| W3: Agent Tool Calling | `coding-agent` | Primary target use case — validates full composition | 1 artifact per config |
| W4: Shared System Prompt | `chatbot-short` (shared-prefix) | Multi-tenant prefix cache effectiveness with EAGLE3 | 1 artifact per concurrency level |
| W5: Production Traffic Mix | `qps-sweep` | Sustained load characterization, go/no-go signal | 1 artifact per QPS level |
| W6: Long Context Scaling | `chatbot-long` | MLA/MHA routing crossover validation | 1 artifact per input length |
| Concurrency sweep | `concurrency-sweep` | Find EAGLE3 crossover point | 1 artifact per concurrency level |

**Required measurements** (per the template):

- [ ] Concurrency sweep completed (1, 8, 32, 64, 128, 256, 512)
- [ ] TTFT P50 ≤ 20ms, P99 ≤ 100ms at c=4 for coding-agent (agentic cached turns)
- [ ] Single-stream throughput ≥ 192 tok/s (1.5x baseline)
- [ ] Aggregate throughput ≥ 6,000 tok/s at c=128
- [ ] No OOM at max concurrent requests = 256
- [ ] No request timeouts during any benchmark run
- [ ] Error rate < 0.1% at all concurrency levels

**KV cache validation**:

- [ ] Prefix cache hit rate measured with EAGLE3 active (confirm caching still works)
- [ ] KV cache utilization % at c=256 (with draft model VRAM overhead)
- [ ] VRAM overhead of draft model quantified (GB)

**Speculative decode metrics** (required extension):

- [ ] EAGLE3 acceptance rate at c=1, c=32, c=128
- [ ] Effective tokens per decode step at each concurrency
- [ ] Crossover concurrency documented (where EAGLE3 = baseline)

**Engine-internal metrics** (scraped from `/metrics`):

- [ ] KV cache utilization during coding-agent workload
- [ ] Running requests at saturation
- [ ] Speculative decode acceptance rate (real-time)

**Enriched artifact output**:

All results stored as **common artifacts** per `standards/benchmark-commons/PROPOSAL.md`:

```
domains/gpu-serving/blueprints/kimi-k2.6-speculative/results/
├── benchmark.yaml                                    ← sidecar config
├── kimi-k2.6_eks_b300_sglang-eagle3_coding-agent_*.json
├── kimi-k2.6_eks_b300_sglang-eagle3_concurrency-sweep_*.json
├── kimi-k2.6_eks_b300_vllm-eagle3_coding-agent_*.json
├── kimi-k2.6_eks_b300_vllm-eagle3_concurrency-sweep_*.json
├── kimi-k2.6_eks_b300_vllm-mla-routing_coding-agent_*.json
├── kimi-k2.6_eks_b300_fullstack_coding-agent_*.json
├── kimi-k2.6_eks_b300_fullstack_concurrency-sweep_*.json
└── comparison/
    └── incremental-layers.json                       ← compare.py --series output
```

**Runner invocation** (using `standards/benchmark-commons/runner/`):

```bash
# Phase 1: SGLang EAGLE3 — coding-agent workload
./run-benchmark.sh \
  --platform local \
  --endpoint http://localhost:30000 \
  --workload coding-agent \
  --tool sglang \
  --sidecar benchmark-eagle3-sglang.yaml \
  --tag "eagle3-sglang" \
  --output domains/gpu-serving/blueprints/kimi-k2.6-speculative/results/

# Phase 4: Full stack — concurrency sweep
./run-benchmark.sh \
  --platform local \
  --endpoint http://localhost:8000 \
  --workload concurrency-sweep \
  --tool vllm \
  --sidecar benchmark-fullstack.yaml \
  --tag "fullstack" \
  --output domains/gpu-serving/blueprints/kimi-k2.6-speculative/results/

# Compare incremental layers
./compare.py --series \
  ../kimi-k2.6/results/vllm_benchmark.json \
  results/*eagle3-only*.json \
  results/*eagle3+mla*.json \
  results/*fullstack*.json
```

**Publication**: After completion, publish to community repos:

```bash
./publish.py \
  --target ai-on-eks \
  --blueprint domains/gpu-serving/blueprints/kimi-k2.6-speculative/ \
  --repo ~/repos/ai-on-eks
```

### Stage 7 — Readiness Audit

- [ ] All Stage 6 measurements completed and stored as common artifacts
- [ ] Incremental layering comparison produced (attribution table)
- [ ] No unresolved HIGH-severity issues in lessons.md
- [ ] `docs/inference-optimization-guide.md` updated with measured data
- [ ] Final production config exported as `serve.sh` + benchmark.yaml sidecar
- [ ] Regression check passed: `./compare.py --regression --baseline <K2.6-baseline> --candidate <fullstack> --threshold 5`

---

## Non-Requirements

- **Not testing disaggregated P/D** — proven counterproductive for K2.6 single-node (see guide Section 2)
- **Not testing Expert Parallelism** — proven to lose on single node (kernel agent report)
- **Not testing multi-node** — single B300 node is the target deployment
- **Not modifying model weights** — serving optimization only
- **Not testing accuracy beyond tool calling** — 50-prompt validation, not full eval suite

## Estimated Cost

**Option A: B300 only (primary track)**

| Phase | Duration | Instance | Cost |
|-------|----------|----------|------|
| Phase 0: Roofline (B300 + B200) | ~2 hours | B300 ($16/hr) + B200 ($12/hr) | ~$56 |
| Phase 1: SGLang EAGLE3 sweep | ~3 hours | B300 spot ($16/hr) | ~$48 |
| Phase 2: vLLM EAGLE3 | ~2 hours | B300 spot ($16/hr) | ~$32 |
| Phase 3: Dynamic MLA routing | ~1 hour | B300 spot ($16/hr) | ~$16 |
| Phase 4: Composition | ~2 hours | B300 spot ($16/hr) | ~$32 |
| Phase 5: Further optimizations | ~3 hours | B300 spot ($16/hr) | ~$48 |
| **Total** | **~13 hours** | | **~$232** |

**Option B: Switch to B200 after Phase 0 confirms parity**

| Phase | Duration | Instance | Cost |
|-------|----------|----------|------|
| Phase 0: Roofline (both SKUs) | ~2 hours | B300 + B200 | ~$56 |
| Phases 1-5: All on B200 | ~11 hours | B200 spot ($12/hr) | ~$132 |
| **Total** | **~13 hours** | | **~$188** |

If Phase 0 confirms B200 parity (expected), Option B saves ~$44 and proves the cost-optimized path. Monthly production savings: **$12,400/yr per node** ($17/hr × 730 hrs).

## Known Limitations

1. **EAGLE3 + DCP workspace allocation failures** — active bug, avoid combining
2. **vLLM MTP deadlocks at concurrency > 1 with TP > 1** — may limit vLLM track
3. **Dynamic MLA routing may not be in any stable release** — cherry-pick required
4. **Draft model quality unknown on B300** — `lightseekorg/kimi-k2.6-eagle3` may need sm_103 validation
5. **INT4 Marlin broken on Blackwell** — only FP8 Marlin works; confirm draft model format compatible

## Relationship to Other Specs

| Spec | Relationship |
|------|-------------|
| `kimi-k2.6.md` (COMPLETE) | Direct baseline — all measurements compared against this |
| `kernel-optimization-agent.md` (DRAFT) | Established that MLA is at HW ceiling — spec decode is the escape hatch |
| `docs/inference-optimization-guide.md` | Results update Section 3 (MTP), Section 13 (Composition Matrix) |
| `glm5.md` | Reference: GLM-5 MTP gave measurable gains on B200 |
| `nemotron-super.md` | Reference: Dynamo disagg approach — contrast with our non-disagg result |

---

> **Note**: Operational artifacts (lessons learned, benchmark results, profiling data)
> belong in the blueprint directory, not in this spec.
> See `blueprints/kimi-k2.6-speculative/results/`, `blueprints/kimi-k2.6-speculative/lessons.md`.
