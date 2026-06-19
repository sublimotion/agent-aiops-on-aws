# Autoresearch Spec: Kernel Optimization Agent for MoE+MLA

## Status: DRAFT

## Overview

Build an autonomous agent that iterates on GPU kernel optimization for MoE+MLA model architectures (Kimi K2.6, DeepSeek V3/V4, GLM-5), structured around the **Verification Framework primitives** (Generator → Verifier → Selector → Telemetry → Feedback Loop) established in `learned-verifier/VERIFICATION_FRAMEWORK.md`.

**Core hypothesis**: An LLM-driven kernel optimization agent with cascaded hardware verification and a constraint database achieves ≥1.5x throughput improvement on MoE+MLA sub-kernels, and improvements transfer across models sharing the same MLA dimensions (Kimi K2.6 ↔ DeepSeek V3).

**Why now**:
- 17+ papers on agent-driven kernel optimization in 2025-2026, but none targeting MoE+MLA specifically
- Kimi K2.6 shares identical MLA dimensions with DeepSeek V3 (`kv_lora_rank=512`, `v_head_dim=128`) — kernel improvements transfer directly
- Open source megakernels (DeepGEMM Mega MoE, FlashMoE, TileKernels) provide strong baselines to optimize against
- Moonshot partnership creates a direct path for upstream contribution
- Kimi's 384-expert routing (vs DeepSeek's 256) is under-optimized — no megakernel is tuned for this config

**Engine-agnostic**: The target kernels (FlashInfer MLA, DeepGEMM, fused MoE dispatch) are shared libraries used by **both vLLM and SGLang**. Both engines are thin scheduling/batching layers on top of these kernels. An optimized MLA decode kernel or 384-expert megakernel slots into either engine — benchmark both to measure impact.

**The durable artifact is the method, not the kernel**: There is no universal optimal kernel, and there never was. What generalizes is the *algorithm* (IO-aware tiling, grouped GEMM, online softmax); the *implementation* is disposable — it re-specializes per hardware generation (FA2→FA3→Blackwell) and per attention variant (full→MLA→NSA). MoE sits at the far end of this spectrum: its problem shape is dynamic and tiny (per-expert M is data-dependent and load-imbalanced), so even a per-shape tuned config doesn't hold still. Hybrid architectures (GLM-5 NSA, Qwen3.5 linear+full, sliding-window, SSM blocks) push everything further — a single layer stack mixes primitives with different roofline positions, each wanting a different kernel, so the autotuning surface grows combinatorially. The universal thing therefore moves *up a level* to the **procedure**: profile → find the roofline-limiting layer → autotune/swap that kernel → verify → keep the winner → **stop when at the hardware ceiling**. This procedure is shape-, model-, and hardware-agnostic, and it is exactly what this agent automates (States 0→5 of `generate-candidate` plus the freeze manager). Knowing when to stop is half the method: a "negative" result (e.g. MLA decode already at >100% BW util) is as valuable as a speedup — it redirects the loop to the next layer (e.g. CUDA-graph capture, a scheduling win). Educational walkthrough: `domains/autoresearch/blueprints/kernel-optimization-agent/results/tile-tuning-explainer-20260605.html`.

**Relationship to Wafer.ai**: Wafer (YC S25) is building a commercial agentic kernel optimization service with the same pattern (profile → diagnose → patch → verify). Our experiment validates whether the same approach works with open-source tools targeting Moonshot-specific architectures. Benchmark against Wafer Pass ($0.60/$3.60 per 1M tokens for Qwen3.5) as a reference point.

---

## Verification Framework Mapping

This experiment applies the five primitives from `VERIFICATION_FRAMEWORK.md` to kernel optimization:

### Primitive 1: Generator

**What generates candidates**: LLM (Claude/GPT) produces kernel code (Triton or TileLang) given:
- Target sub-kernel specification (MoE dispatch, MLA decode, megakernel)
- Hardware constraints (SM103, 227KB shared memory, NVLink 5)
- NSight Compute profile of current bottleneck (roofline position)
- Constraint database (all prior failures + learned rules)

**Properties**:
- **Throughput**: ~4 candidates/hour (15 min per compile+verify+benchmark cycle)
- **Diversity**: Controlled via state vector (see below) — advances from tile-size tuning → algorithmic change → DSL rewrite → megakernel adaptation
- **Fix rate** (compiles + correct): Target ≥66% (matching EFA harness L3 pipeline throughput)
- **Pass rate** (improves performance): Unknown — the experiment measures this

**Verification opportunity**: The gap between fix rate and pass rate. A kernel that is correct may still be slower — the verifier must discriminate performance, not just correctness.

### Primitive 2: Verifier (Cascaded, L0-L4)

Hardware-as-judge, structured as cascaded verification with increasing cost:

| Level | What | Signal | Cost | Analogy to Learned-Verifier |
|-------|------|--------|------|----------------------------|
| **L0: Parse** | Syntax validation (Triton/TileLang compile check) | Binary pass/fail | ~1s | Behavioral features (fast, cheap) |
| **L1: Compile** | Full compilation against target env (nvcc/triton JIT) | Binary + error classification | ~10s | v009 rubric (structured signal) |
| **L2: Correctness** | 100 random inputs vs PyTorch reference, rtol check | Numeric (max deviation) | ~30s | Gold test suite |
| **L3: Kernel benchmark** | ncu profiling — TFLOPS, bandwidth, occupancy, roofline | Multi-dimensional continuous | ~2 min | SVG consensus (multi-signal) |
| **L4: E2E benchmark** | Full model serving (TPOT, TTFT, throughput at c=128/512) | Continuous, CI-gated | ~5 min | SWE-bench gold eval |

**Cascaded filtering**: A candidate that fails L0 never reaches L1. A candidate that fails L2 never wastes 5 min on L4. This is exactly the verification spectrum — hard verifier (L0-L2) gates access to expensive evaluation (L3-L4).

**Correctness tolerance**: FP8 kernels have numerical edge cases. Use statistical verification:
- rtol=1e-3 for FP16/BF16
- rtol=1e-2 for FP8 with 100-input verification + distribution check
- Reject if max deviation exceeds 5x the tolerance on any single input

**Performance CI gate** (analogous to ECE calibration gate): A candidate passes L4 only if the speedup confidence interval (5 runs, median) excludes 1.0. Prevents promoting noise as improvement.

### Primitive 3: Selector

**Top-K selection with cherry-pick evaluation**:
- Maintain a leaderboard of top-5 kernels per target (MoE dispatch, MLA decode)
- New candidates must beat the current champion on L4 to be promoted
- Cherry-pick evaluation: when upstream PRs land (Alpha-MoE, Mega MoE), evaluate each *individually* against K2.6 rather than assuming bundles are net-positive (lesson from EFA harness: bundled expert commits regressed LL performance by 10%)

**Composition selector**: Phase 3 combines the best kernel per layer. The selector tests all pairwise compositions (best MoE dispatch × best MLA decode × dynamic routing) because interactions may be non-linear.

### Primitive 4: Telemetry

**Signals extracted per candidate** (the feature vector for the feedback loop):

| Signal Category | Features | Analogy |
|----------------|----------|---------|
| **Compilation** | Success/fail, error class, compile time | Behavioral (fix rate) |
| **Correctness** | Max rtol deviation, distribution of errors, edge-case failures | Gold labels |
| **Kernel profile** | TFLOPS, memory BW utilization, SM occupancy, warp stall reasons, shared memory usage | v009 sub-scores |
| **Roofline position** | Compute-bound vs memory-bound vs latency-bound | Task difficulty |
| **E2E impact** | Δ throughput, Δ TPOT, Δ TTFT at each concurrency level | Pass/fail outcome |
| **Code features** | DSL (Triton/TileLang/CUDA), LoC, tile sizes, num_warps, num_stages, fusion degree | Patch features (size, complexity) |
| **Agent trace** | Which model generated, what strategy was used, what constraints were active | Harness metadata |

All telemetry stored as append-only JSONL (`results/telemetry.jsonl`). This dataset becomes training data for a future learned selector (Phase 4 follow-on: predict kernel quality from code features without running L3/L4).

### Primitive 5: Feedback Loop (Constraint Database)

**The compound learning mechanism** — directly parallel to learned-verifier's Phase 3 signal combination and EFA harness's constraint database:

#### Constraint Types (append-only, never deleted, can be demoted)

| Type | Example | Injection Method |
|------|---------|-----------------|
| **Hard constraint** (from L0/L1 failures) | "SM103 shared memory is 227KB, not 256KB" | Always injected into generation prompt |
| **Correctness constraint** (from L2 failures) | "FP8 tile-parallel reduction requires explicit sync before store" | Injected when target kernel touches FP8 reduction |
| **Performance constraint** (from L3 regression) | "Shared memory bank conflicts at tile_k=64 on SM103 MoE dispatch" | Injected when generating MoE dispatch with tile_k near 64 |
| **Architectural constraint** (from L4 null result) | "384-expert routing with n_group=1 causes load imbalance at tile_m<32" | Injected for all K2.6 MoE dispatch candidates |
| **Positive pattern** (from L4 promotion) | "Multi-stage pipeline (3 stages) + warp specialization gives 1.3x on MLA decode" | Injected as suggested approach when generating MLA decode |

#### Convergence Acceleration

Same mechanism as learned-verifier's forward feature selection and EFA harness's 2.3x convergence: each constraint narrows the search space so the generator produces higher-quality candidates earlier. Measure convergence speed (candidates-to-promotion) in first 20 vs last 20 iterations to quantify learning rate.

#### Freeze Manager (Plateau Detection)

Formalized from EFA harness pattern, analogous to forward feature selection termination:

```
for each target_region in [moe_dispatch, mla_decode, megakernel]:
  if last_3_promoted_candidates.improvement < 2%:
    FREEZE region
    redirect to next-highest-headroom region (from Phase 1 profiling)
  if meta_analysis detects new bottleneck in frozen region:
    UNFREEZE with accumulated constraints
```

The freeze manager prevents unbounded iteration on a plateau (EFA harness spent 120 candidates on kernel layer before redirecting to proxy, which delivered 46% additional gain).

#### State Vector (Self-Advancing Approach Progression)

Guarantees forward progress — the system never retries the same strategy:

```
STATE 0: Triton autotuning (tile sizes, warps, stages) on existing kernel
STATE 1: Triton algorithmic rewrite (different memory access pattern, fusion)
STATE 2: TileLang rewrite (different DSL, different optimization space)
STATE 3: CUDA C++ adaptation (port megakernel from DeepGEMM/FlashMoE)
STATE 4: Upstream cherry-pick (evaluate existing PRs individually on K2.6)
STATE 5: Composition (combine best of each layer)
ESCALATE: Freeze region, switch target
```

Advance to next state when: 3 consecutive candidates at current state fail to improve. Each advancement injects all constraints learned from prior states.

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (capacity block or spot)
- **Primary (Phases 1-2)**: p5en.48xlarge (8x H200 141GB HBM3e, NVLink 4 / NVSwitch, sm_90)
- **Final validation (Phase 3)**: p6-b300.48xlarge (8x B300 268GB HBM3e, NVLink 5 / NVSwitch, sm_103)
- **GPUs**: 8x NVIDIA H200 (primary) / 8x B300 (validation)

**Why p5en for Phases 1-2**: The high-value optimizations (384-expert dispatch patterns, MLA routing, fusion boundaries, constraint learning) are architecture-independent — they target model structure, not hardware instructions. FlashInfer MLA, DeepGEMM, and FlashMoE all support SM90. Profiling on H200 reveals the same pipeline bottlenecks. p5en is more available and cheaper than B300.

**Why B300 for Phase 3**: Final autotuning (tile sizes, warps, stages) is hardware-specific. SM103 has TMA, TCGEN5, 227KB shared memory, different peak BW (2.4 TB/s vs 3.35 TB/s). Production numbers for Moonshot must come from Blackwell. Triton `@autotune` re-runs on B300 to find SM103-optimal configs.

**What does NOT transfer from H200 → B300**:
- Tile size / num_warps / num_stages optimal configs (re-autotune)
- TMA-specific kernel paths (SM100+ only)
- FP4/MXFP4 quantization (Blackwell-only)
- Absolute TFLOPS/BW numbers (relative improvements should hold)

**What DOES transfer**:
- Algorithmic insights (fusion boundaries, dispatch patterns, routing logic)
- Constraint database (architecture facts, correctness rules, approach dead-ends)
- Kernel source code (Triton re-autotunes automatically; TileLang recompiles)

- **NVMe**: Local NVMe on both instance types (model weights, profiling, kernel cache)
- **Profiling tools**: NSight Compute (ncu), NSight Systems (nsys), DCGM metrics

### 2. Codebase

- **Source repositories**:
  - `github.com/deepseek-ai/DeepGEMM` (MIT) — Mega MoE megakernel baseline
  - `github.com/osayamenja/FlashMoE` (BSD-3) — Persistent MoE kernel
  - `github.com/deepseek-ai/TileKernels` (open) — TileLang MoE kernels
  - `github.com/flashinfer-ai/flashinfer` — FlashInfer MLA kernels
  - `github.com/sgl-project/sglang` — Integration target
  - `github.com/vllm-project/vllm` — Integration target
  - `github.com/ScalingIntelligence/KernelBench` — Evaluation framework
  - `github.com/ScalingIntelligence/caesar` — Multi-turn batch kernel eval

- **Fixed files** (define the metric):
  - PyTorch reference implementations (L2 ground truth)
  - Model weights (Kimi K2.6 INT4 QAT)
  - Benchmark workloads from K2.6 spec (W1-W6)
  - NSight Compute profiling scripts

- **Agent-editable files**:
  - Triton kernel implementations (`.py`)
  - TileLang kernel implementations (`.py`)
  - Kernel autotuning configs
  - Fusion strategies
  - Constraint database (`constraints.jsonl`)

- **Agent instructions**: `domains/autoresearch/blueprints/kernel-optimization-agent/program.md`

### 3. Experiment Protocol

#### Metrics

- **Primary**: End-to-end throughput (tok/s) at c=128 and c=512 on Kimi K2.6 serving
- **Secondary**: TPOT p50/p99, TTFT p50/p99, kernel TFLOPS, memory bandwidth utilization
- **Baselines** (from K2.6 benchmark):
  - vLLM v0.19.1 FLASHINFER_MLA on B300 = 10,437 tok/s @ c=512
  - SGLang v0.5.10 on B300 = 3,400 tok/s @ c=512
- **Benchmark both engines**: Measure impact on vLLM and SGLang independently.
- **Learning rate metric**: Candidates-to-promotion ratio, measured per-decile to quantify convergence acceleration.

#### Time budget
- **Per candidate cycle**: ~15 min (L0 through L4)
- **Per phase**: 1 capacity block session (~8 hours, ~32 candidates)
- **Total**: 3-4 sessions across 3 phases (~100-128 candidates)

#### Loop Structure

```
PHASE 1: Decompose + Profile (Session 1)
  [Verification Framework Phase 0 — identify verification opportunity]

  1. PROFILE K2.6 baseline (vLLM FLASHINFER_MLA) with ncu --set full
     - Top-10 kernels by wall-clock time
     - Classify: compute-bound, memory-bound, latency-bound
     - Map to pipeline stage: MoE dispatch, MLA decode, attention, comm
  2. BENCHMARK existing megakernels on K2.6 (cherry-pick evaluation):
     - DeepGEMM Mega MoE (SGLang #23167)
     - Alpha-MoE (vLLM #30078)
     - FlashMoE persistent kernel
     - Evaluate EACH individually — do not assume bundle is net-positive
  3. TEST dynamic MLA/MHA routing (vLLM #35474) on agentic workloads
  4. SEED constraint database with:
     - K2.6 architectural constraints (384 experts, n_group=1, 64 heads)
     - SM103 hardware constraints (227KB smem, TMA, TCGEN5)
     - Known failures from upstream PRs (accuracy regressions, etc.)
  5. IDENTIFY highest-headroom target for Phase 2

PHASE 2: Generate + Verify + Learn (Sessions 2-3)
  [Verification Framework Phases 1-3 — build cascaded verifier, train selector]

  for iteration in 1..N:
    1. GENERATE: LLM produces kernel candidate
       - Inject all active constraints from database
       - Use current state vector approach
    2. VERIFY (cascaded L0→L4):
       - L0: Parse/syntax
       - L1: Compile against target env
       - L2: Correctness (100 random inputs)
       - L3: Kernel benchmark (ncu profile)
       - L4: E2E benchmark (throughput at c=128/512)
    3. RECORD telemetry (all signals, regardless of pass/fail)
    4. UPDATE constraint database:
       - Classify failure stage and root cause
       - Add constraint with evidence and severity
       - Update positive patterns from promotions
    5. SELECTOR: promote if CI excludes 1.0 AND beats champion
    6. ADVANCE state vector if 3 consecutive non-improvements
    7. FREEZE region if plateau detected, redirect to next target

  Termination: freeze manager has frozen all targets OR budget exhausted

PHASE 3: Transfer + Compose (Session 4)
  [Verification Framework Phase 3 — compound signals, cross-model validation]

  1. RUN best Phase 2 kernels on DeepSeek V3 (same MLA dims, 256 experts)
  2. MEASURE transfer gap (does K2.6-tuned kernel work on DS V3?)
  3. If gap >10%: constraint database has K2.6-specific rules — test which
     constraints are portable vs architecture-specific
  4. COMPOSE: combine best kernel per layer (MoE dispatch × MLA decode × routing)
  5. BENCHMARK composed stack on both engines
  6. COMPARE against Wafer.ai Pass endpoint (if available)
  7. EXPORT: telemetry dataset for future learned selector training
```

#### Termination
- **Success**: ≥10% e2e throughput improvement with all constraints documented
- **Partial**: Comprehensive profiling + constraint database, even if <10% improvement (negative results are valuable — documents where ceiling is)
- **Hard stop**: 4 sessions (budget cap)

#### Logging
- Kernel source: `scripts/kernels/{target}/{iteration}.py`
- Profiles: `results/profiles/`
- Telemetry: `results/telemetry.jsonl` (all signals per candidate)
- Constraints: `results/constraints.jsonl` (append-only)
- Agent traces: `results/agent_traces/`
- Leaderboard: `results/leaderboard.json`

### 4. Networking

- **Access**: SSH via bastion or EKS kubectl exec
- **Model weights**: HuggingFace download to NVMe at pod startup
- **Profiling**: ncu/nsys run locally on GPU node

### 5. Storage

- **Model weights**: `/mnt/nvme/models/Kimi-K2.6/` (~594GB)
- **Kernel compilation cache**: `/mnt/nvme/kernel-cache/`
- **Profiling artifacts**: `/mnt/nvme/profiles/` (~1-5GB each)
- **Results**: Blueprint `results/` directory, synced to S3

---

## Upstream Landscape (As of May 2026)

### Already Being Addressed (DO NOT duplicate)

| Area | vLLM PRs | SGLang PRs | Notes |
|------|----------|------------|-------|
| DeepGEMM Mega MoE integration | #40833 | #23167, #24301 | Both engines actively integrating |
| MLA decode FP8 fusion | #36297, #41568, #40908 | Via FlashInfer | 3 vLLM PRs attacking different fusion points |
| RoPE+KV cache fusion for MLA | #35879 (+16% TPOT), #40392 | #24324 (NSA) | Active on both |
| DeepGEMM JIT warmup | #25619 (merged) | #23756 (merged) | **Solved** |
| SM120 (consumer Blackwell) | #40991 | #24303, #24047 | Not our target hardware |
| FlashMLA sparse prefill | #41150 | #24225 | Open on both |
| TileKernels/TileLang | DSv4 roadmap #40902 | #24178 (merged) | SGLang ahead |
| MoE fused shared expert | #39280 (+16-24% AMD) | #23597 | AMD-driven |

### Our Opportunity (Not Being Addressed)

| Gap | Why It's Open |
|-----|---------------|
| **384-expert MoE dispatch tuning** | All megakernel work targets 256 experts with 8-group routing |
| **Alpha-MoE vs Mega MoE on Kimi** | Neither tested on 384-expert layout |
| **Systematic K2.6 kernel profiling** | No holistic profiling of full K2.6 pipeline |
| **Cross-model MLA kernel transfer** | Same dims, different head count — unmeasured |
| **Dynamic MLA/MHA short-prefill routing** | vLLM #35474 untested on Kimi |
| **Agent-driven kernel iteration with constraint learning** | All upstream work is human PRs |
| **FlashMoE persistent kernel for 384-expert models** | Untested on B300 |

---

## Target Kernels (Priority Order)

### P0: 384-Expert MoE Dispatch + Megakernel Tuning
- **Current**: FlashInfer fused MoE (default)
- **Baselines**: DeepGEMM Mega MoE, Alpha-MoE (16% faster on DS V3), FlashMoE (5.1x on Qwen-30B)
- **Opportunity**: K2.6's 384 experts with `n_group=1` → top-8 from full pool → different tile sizes, batching, load balance
- **Experiment**: Benchmark all three on K2.6, profile winner, agent-optimize for 384-expert dispatch

### P1: Dynamic MLA/MHA Short-Prefill Routing
- **Upstream**: vLLM #35474 (open, ~3x TTFT for <1024 token prefills)
- **Experiment**: Cherry-pick, benchmark on W3/W4 agentic workloads
- **Low effort, high impact** — no new kernel, just routing logic

### P2: Cross-Model MLA Kernel Transfer
- **Question**: Does FlashInfer MLA tuned for DS V3 (128 heads) work identically on K2.6 (64 heads)?
- **Experiment**: Profile per-kernel TFLOPS divergence, autotune if gap >10%

### P3: Agent Loop Meta-Experiment
- **Goal**: Validate the autonomous profile→generate→verify loop itself
- **Measure**: Convergence curve, constraint database growth, learning rate acceleration
- **Comparison**: Agent-generated vs best human PR vs Wafer.ai

---

## Research Questions

### RQ1: Which MoE+MLA sub-kernel has the highest optimization headroom on K2.6?
Profile-first. Hypothesis: 384-expert dispatch is the bottleneck (Kimi-specific, not tuned by DeepGEMM).

### RQ2: Does the constraint database accelerate convergence?
Measure candidates-to-promotion in first 20 vs last 20 iterations. Target: ≥1.5x acceleration (EFA harness achieved 2.3x over 6 weeks; we have less time but more focused scope).

### RQ3: Do optimized MLA kernels transfer across models?
K2.6 (64 heads, 384 experts) vs DS V3 (128 heads, 256 experts) — same MLA latent dims. Quantify the architecture-specific vs universal constraints in the database.

### RQ4: What is the verification opportunity for kernel optimization?
Measure fix rate (L2 pass) vs pass rate (L4 improvement). This is the central question: how much of the generator's output is "correct but slow" — the gap that justifies the full verification stack.

### RQ5: Which state vector approach produces the most promotions?
Track promotion rate per state (autotuning vs algorithmic rewrite vs DSL change vs megakernel adaptation). Identifies where LLMs add value vs where search/autotuning is sufficient.

---

## Success Criteria

1. **Profiling complete**: Top-10 kernels identified, roofline-classified, mapped to pipeline stages
2. **Megakernel A/B on K2.6**: Each evaluated individually (cherry-pick discipline)
3. **≥10% e2e throughput improvement** over 10,437 tok/s baseline (any combination of techniques)
4. **384-expert dispatch gap quantified**: Performance delta vs 256-expert on same megakernel
5. **Cross-model transfer quantified**: K2.6 vs DS V3 on same kernels
6. **Constraint database seeded**: ≥30 constraints with evidence and severity
7. **Telemetry dataset exported**: ≥50 candidates with full feature vectors (training data for future learned selector)
8. **Convergence measured**: Candidates-to-promotion ratio documented per-decile

## Non-Requirements

- **Not optimizing training kernels** — inference/serving only
- **Not targeting SM120 (g7e)** — must transfer to datacenter GPUs
- **Not building a production service** — research experiment
- **Not fine-tuning the kernel-writing LLM** — use frontier models as-is; RL training is Phase 4 follow-on (parallel to learned-verifier Phase 4)
- **Not modifying model weights** — kernel-level only
- **Not implementing P/D disaggregation** — single-node for controlled measurement

## Known Limitations

1. **NSight Compute overhead**: ncu 10-100x slowdown. Profile subset, benchmark all.
2. **DeepGEMM JIT interaction**: May need to disable JIT for target layers.
3. **Correctness brittleness**: FP8 edge cases. Statistical verification with distribution check.
4. **Integration complexity**: Replacing a kernel in vLLM/SGLang requires understanding the full pipeline. May benchmark kernel-level only if integration too complex.
5. **B300 availability**: Only needed for Phase 3 (final validation + autotuning). Phases 1-2 run on p5en (H200). If B300 unavailable for Phase 3, export Triton kernels and re-autotune when capacity appears — constraint database and algorithmic insights are not lost.
6. **Small N**: ~100-128 candidates in 4 sessions. Insufficient for RL training of the selector — but sufficient for constraint database seeding and convergence measurement.

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Incorrect kernels pass L2 rtol check | Medium | High | 100 random inputs + distribution check + 2+ references |
| Kernel-level speedup doesn't translate to e2e | Medium | Medium | Phase 1 profiling identifies actual bottlenecks first |
| DeepGEMM Mega MoE already near HW limit | Medium | High | Focus on 384-expert gap, not beating DeepGEMM on 256 |
| Agent plateaus at trivial tile-size tuning | Medium | Medium | State vector forces approach advancement |
| Constraint database over-constrains (kills diversity) | Low | Medium | Constraints have severity levels; soft constraints can be overridden |
| B300 unavailable | Low | Medium | B200 fallback, kernels forward-compatible |

## Estimated Cost

| Phase | Sessions | Hours | Instance | API | Total |
|-------|----------|-------|----------|-----|-------|
| Phase 1: Decompose+Profile (p5en) | 1 | ~8 | ~$250 | ~$50 | ~$300 |
| Phase 2: Generate+Verify+Learn (p5en) | 1-2 | ~16 | ~$500 | ~$100 | ~$600 |
| Phase 3: Transfer+Compose (B300) | 1 | ~8 | ~$400 | ~$50 | ~$450 |
| **Total** | **3-4** | **~32** | **~$1,150** | **~$200** | **~$1,350** |

## Relationship to Other Specs

| Spec | Relationship |
|------|-------------|
| `learned-verifier` (VERIFICATION_FRAMEWORK.md) | **Methodology source** — primitives, cascaded verification, compound learning, Phase 0-4 progression |
| `verifier-reward.md` | Pattern reference — constraint database parallels rubric iteration (v001→v009); freeze manager parallels "11 consecutive negatives → experiment complete" |
| `kimi-k2.6.md` | Baseline benchmarks (10,437 tok/s, TTFT, TPOT) |
| `mooncake-kv-tiering.md` | Complementary — Mooncake optimizes caching/scheduling, this optimizes compute |
| `agent-harness.md` | Agent loop design — Parkinson's Law findings inform state vector timing |
| `verification-primitives.md` | Two-stage checkpoint pattern may apply (force profiling at 40% of iteration budget) |

## Phase 4 Follow-On (Not This Spec)

If Phases 1-3 produce ≥50 candidates with full telemetry:
- **Train a learned selector** (Random Forest/XGBoost) that predicts L4 outcome from L0-L3 features + code features
- **Calibrate** with Platt scaling (ECE < 0.1 required for RL safety)
- **Use as reward model** for RL fine-tuning of a kernel-writing model (TritonForge provides 18.2K training pairs as starting point)
- This mirrors learned-verifier Phase 4 (verifier rewards → generator improvement via RL)

---

> **Note**: Operational artifacts (lessons learned, experiment results, profiling data, constraint database)
> belong in the blueprint directory, not in this spec.
