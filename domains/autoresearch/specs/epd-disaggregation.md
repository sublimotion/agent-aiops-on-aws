# Autoresearch Spec: EPD Disaggregation for Multimodal & Wan 2.2 Video

## Status: DRAFT

## Overview

Evaluate **Encode–Prefill–Decode (EPD) disaggregation** as a serving architecture for heterogeneous-profile multimodal pipelines, using two instances of the same pattern: (1) a vision-language model (VLM) where the modality encoder is split from the LLM core, and (2) Wan 2.2 video generation where denoise / VAE-decode / frame-encode run as independently-scaled tiers.

**Core hypothesis**: A multimodal pipeline is a chain of stages with different rooflines. Splitting them onto independently-scaled tiers — and crossing the network only where the hand-off artifact is small (latents/embeddings, never raw pixels) — beats co-located execution at scale by eliminating head-of-line blocking on the expensive GPU, raising its utilization from the ~80% co-located ceiling toward saturation.

**Why now**:
- AWS's Synthesia/Wan 2.2 write-up demonstrated the *intra-box* version of this move (async dual-stream pipeline: 82% → 99.9% kernel utilization, 8.2% lower decode latency) but explicitly stopped short of disaggregation and of compilation. EPD disaggregation is the *at-scale* generalization.
- "EPD disaggregation" emerged as an active 2025 serving topic for VLMs (encoder pool + embedding cache), directly paralleling LLM prefill/decode (P/D) disaggregation — which this repo has already characterized (`pd_disagg_single_node`).
- This repo already has the *unnamed* prior art: `domains/gpu-serving/blueprints/ray-serve-video/` is a disaggregated multimodal pipeline in everything but name (Ray + async Kafka, per-stage `runtime_env` isolation, in-memory hand-off). This spec makes the EPD pattern explicit and measures the at-scale payoff.
- NVENC/NVDEC are separate silicon from the CUDA cores. Frame encode stealing SM cycles from denoising is pure waste — a free decouple.

**The one design rule under test**: *split where the hand-off artifact is small.* In LLM P/D the artifact is the KV cache (why MLA's compressed latent makes disagg cheap). Here:
- VLM: ship **embeddings** between encoder and LLM tiers (small), never raw media.
- Wan 2.2: ship **latents** between denoise and VAE-decode tiers (~8× downsampled, low-channel — tiny); keep VAE-decode → frame-encode co-located so only the compressed bitstream leaves the box.

**Relationship to P/D disaggregation**: EPD is the same lever one stage earlier in the pipeline. P/D splits prefill from decode (both inside the LLM); EPD splits the modality encoder (and, for diffusion, the VAE) from the autoregressive/denoising core. The crossover economics are identical: it pays at scale, loses at low volume where the network hop + orchestration overhead exceeds the utilization gain (this repo's own single-node EP/PD lesson).

---

## Two Instances of the Pattern

### Instance A: VLM EPD (image/video understanding)

```
[raw image/video] → vision encoder (ViT) → projector → LLM prefill → LLM decode
                    └──────── encoder tier ────────┘   └──── LLM tier (P/D) ────┘
                    hand-off artifact: embeddings (small)
```

| Stage | Roofline profile | Cheapest sufficient HW | Batching policy |
|-------|------------------|------------------------|-----------------|
| Vision encode (ViT) | Compute-bound, bursty | mid GPU (L4 / g6e-class) | dynamic, image-count |
| Projector | Trivial | co-locate with encoder | — |
| LLM prefill | Compute-bound | H100/B200-class | chunked |
| LLM decode | Memory-bound | H100/B200-class | continuous |

**Disaggregation wins**: vision encode is bursty and shaped differently from text decode; a 4K image's encode should not block the decode GPUs. Embedding cache = prefix caching for vision: the same image across requests is encoded once.

### Instance B: Wan 2.2 video-gen EPD (generation)

```
[prompt] → denoise (DiT, latent space, N steps) → VAE decode (latent→pixel) → frame encode (NVENC)
           └──────── denoise tier ────────┘        └──── decode+encode tier (co-located) ────┘
           hand-off artifact: latents (tiny)
```

| Stage | Roofline profile | Cheapest sufficient HW | Notes |
|-------|------------------|------------------------|-------|
| Denoise (DiT) | Tensor-core bound — the big cost | B200/g7e RTX PRO 6000-class | iterates N diffusion steps |
| VAE decode | Conv / bandwidth bound | mid GPU (L4-class) | latent → pixel frames |
| Frame encode | Dedicated ASIC | **NVENC** (not CUDA, not CPU) | mux to bitstream |

**Disaggregation wins**: the denoise GPU does *only* denoising and never stalls on VAE/encode (Synthesia measured this as the 82% util ceiling). VAE-decode → encode stay co-located so only the compressed bitstream crosses the network.

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (reuses `ray-serve-video` / `ray-serve-ft` KubeRay infrastructure where possible)
- **Denoise / LLM tier** (expensive): g7e.xlarge–12xlarge (RTX PRO 6000 Blackwell) or p5/p6 if available. See memory: g7e supports NVENC/NVDEC; `nerdctl` runtime; `/mnt/nvme`.
- **Encoder / VAE tier** (mid): g6e / g5 (L4 / A10G-class)
- **NVENC**: validate NVENC session availability on the chosen instance (consumer-Blackwell RTX PRO 6000 has rich NVENC; datacenter A100/H100 have limited or no NVENC — a key tier-placement constraint).
- **GPUs**: 2 tiers, independently scaled (start 1×expensive + 1×mid, scale the saturated tier)

### 2. Codebase

- **Reference reading** (context, not dependencies):
  - AWS Architecture Blog — "How Synthesia optimizes generative AI video inference on EC2 G7e" (intra-box async baseline, B1)
  - Anyscale — "Architecting multimodal data pipelines that scale with Ray" — useful Ray-composition reference, but **batch-oriented and offline**; this spec targets *online/low-latency serving* with tight per-request SLOs, so treat it as a topology reference only, not an architecture to copy.

- **Source repositories**:
  - `github.com/aws-samples/sample-asynchronous-video-decoding` — Synthesia async pipeline reference (intra-box baseline)
  - Wan 2.2 (HuggingFace Diffusers format, 14B) — denoise + VAE
  - A VLM (Qwen-VL or similar with separable ViT encoder) for Instance A
  - `github.com/ray-project/ray` (Ray Serve) — composition + per-stage autoscaling
  - Prior art: `domains/gpu-serving/blueprints/ray-serve-video/` (Kafka ingest, runtime_env isolation, in-memory hand-off)

- **Fixed files** (define the metric):
  - Reference outputs for correctness (bit-identical latents must produce identical frames vs co-located baseline)
  - Workload traces (request arrival pattern, image/prompt mix)
  - Baseline numbers: co-located E2E latency + GPU utilization; Synthesia intra-box async result

- **Agent-editable files**:
  - Tier topology (which stage on which instance type, replica counts)
  - Hand-off transport (Ray object store / in-memory vs network serialization)
  - Encoder embedding cache config
  - NVENC vs CUDA vs CPU encode path
  - Autoscaling policy per tier

- **Agent instructions**: `domains/autoresearch/blueprints/epd-disaggregation/program.md`

### 3. Experiment Protocol

#### Metrics
- **Primary**: Expensive-tier GPU utilization (%) and aggregate throughput (requests/s or frames/s) at fixed SLO
- **Secondary**: E2E latency p50/p99, Real-Time Factor (decode tier, Wan), per-tier utilization, hand-off artifact size + transfer latency, $/1000 units of output
- **Baselines**:
  - **B0 co-located**: all stages on the expensive GPU (Synthesia "synchronous" — ~82% util, 21.99s/video)
  - **B1 intra-box async**: Synthesia dual-stream pipeline (~99.9% util, 20.17s/video) — the tune-once ceiling without disaggregation
  - **EPD (this spec)**: disaggregated tiers — must beat B0 on throughput-per-dollar at scale, and reveal where it beats/loses vs B1

#### Time budget
- **Per configuration**: ~30 min (deploy tiers + warm caches + run trace)
- **Total**: 2 capacity-block sessions (Instance A + Instance B)

#### Loop structure

```
PHASE 1: Characterize stages (both instances)
  Profile each stage in isolation: roofline position, latency, HW it saturates.
  Confirm hand-off artifact sizes (embeddings, latents, pixels, bitstream).
  Establish B0 (co-located) and reproduce B1 (Synthesia async) as baselines.

PHASE 2: Single-node disaggregation (within one box)
  Split stages across streams/processes on ONE instance (no network hop yet).
  This is the Synthesia regime — measure the utilization ceiling and confirm
  the hand-off-artifact rule (small intermediates overlap cleanly).

PHASE 3: Multi-node disaggregation (the at-scale test)
  Split tiers across instance types. Cross the network ONLY at small artifacts.
  Sweep replica ratios to find the balance point where both tiers saturate.
  Measure the crossover: at what request rate does EPD beat co-located on $/unit?

PHASE 4: Tier-specific optimizations
  - Encoder embedding cache (VLM): hit rate, throughput delta on repeated media
  - NVENC offload (Wan): util reclaimed on denoise tier vs CUDA/CPU encode
  - Heterogeneous HW: cheapest sufficient GPU per tier (cost floor)

PHASE 5: Failure & elasticity
  - Independent tier autoscaling under bursty arrivals
  - Tier failure isolation (encoder pool dies → does decode tier survive?)
```

#### Termination
- **Success**: EPD multi-node beats co-located (B0) on throughput-per-dollar by ≥30% at the target request rate, with expensive-tier util ≥95%
- **Partial**: EPD matches B1 (intra-box async) util but adds independent scaling + cheaper-HW-per-tier cost wins
- **Negative (valuable)**: Document the crossover request rate below which co-located/intra-box wins (the network hop doesn't amortize) — directly parallels the single-node P/D lesson
- **Hard stop**: 2 sessions

#### Logging
- Tier topology + replica counts per config → blueprint `configs/`
- Per-tier utilization + E2E latency → `results/experiments.jsonl`
- Hand-off artifact sizes + transfer latencies → `results/handoff_metrics/`
- Cost breakdown per config → `results/cost_analysis.md`

### 4. Networking

- **Intra-tier**: Ray object store / shared memory (in-memory, ~10ms — proven in ray-serve-video)
- **Inter-tier**: small-artifact transport only (embeddings/latents). Measure serialization cost; this is the rule-under-test boundary.
- **Ingress**: Kafka (reuse ray-serve-video pattern) or HTTP

### 5. Storage

- **Model weights**: `/mnt/nvme/models/` per tier (Wan 2.2, VLM, VAE)
- **Embedding cache**: encoder-tier-local (DRAM + NVMe spill)
- **Output**: bitstream/frames to S3; results to blueprint `results/`

---

## Research Questions

### RQ1: Does the hand-off-artifact rule hold empirically?
Latents/embeddings are small; pixels are huge. Confirm that splitting at small-artifact boundaries adds negligible transfer overhead while splitting at a pixel boundary would not. Quantify artifact size × transfer latency at each candidate split point.

### RQ2: Where is the crossover?
At what request rate does multi-node EPD beat co-located on $/unit? Below it, the hop doesn't amortize (the single-node P/D lesson). Above it, independent saturation wins.

### RQ3: How much does NVENC offload reclaim?
Frame encode on NVENC (separate silicon) vs CUDA vs CPU. How much denoise-tier SM time is freed? Is NVENC session count a binding constraint per instance type?

### RQ4: Does the embedding cache pay off for VLMs?
Repeated media (same image/video across requests) encoded once. Measure hit rate on a realistic trace and the throughput delta — this is prefix caching for vision.

### RQ5: Can tiers scale and fail independently?
Bursty image uploads should scale the encoder pool without touching the LLM/denoise tier. An encoder-pool failure should not take down decode. Verify isolation.

### RQ6: EPD vs intra-box async — complementary or competing?
Synthesia chose intra-box overlap (B1). Is multi-node EPD strictly better at scale, or do they compose (async overlap *within* each tier + disaggregation *across* tiers)?

---

## Success Criteria

1. **Stage characterization**: each stage's roofline, saturating HW, and hand-off artifact size documented for both instances
2. **Rule validated**: small-artifact split points add <10% transfer overhead; the cost of a (counterfactual) pixel-boundary split is quantified
3. **Throughput-per-dollar**: multi-node EPD ≥30% better than co-located at target rate, expensive-tier util ≥95%
4. **Crossover documented**: the request rate below which co-located/intra-box wins
5. **NVENC win quantified**: denoise-tier util reclaimed by offloading encode to NVENC
6. **Embedding cache win quantified** (VLM): hit rate + throughput delta on a repeated-media trace
7. **Elasticity demonstrated**: independent per-tier autoscaling + tier-failure isolation

## Non-Requirements

- **Not training/fine-tuning models** — serving architecture only, weights frozen
- **Not optimizing kernels** — that's `kernel-optimization-agent`. This is the scheduling/topology layer (the complementary half).
- **Not improving model quality** — pure utilization/cost win, outputs must match co-located baseline
- **Not cross-region** — single-region, multi-instance only
- **Not implementing a custom diffusion scheduler** — use Diffusers/Wan as-is; split the pipeline, don't rewrite the model

## Known Limitations

1. **NVENC availability varies by instance type** — datacenter A100/H100 have limited/no NVENC; consumer-Blackwell RTX PRO 6000 (g7e) is rich. Tier placement is constrained by where NVENC exists.
2. **EFA is not GPUDirect RDMA** (memory: g7e EFA = CPU-bounce). Inter-tier transfer pays a CPU bounce; this is acceptable *because* the artifacts are small, but it bounds how small the crossover can get.
3. **Ray runtime_env isolation cost** — per-stage dependency isolation (PT/TF/Diffusers) has cold-start cost (ray-serve-video lessons: protobuf/cuDNN/numpy pins). Budget for it.
4. **Diffusion is iterative** — denoise runs N steps per request; the tier-balance ratio depends heavily on N (step count) and resolution. Sweep both.
5. **Correctness of disaggregated latents** — bit-identical latents must produce identical frames; serialization/transport must not perturb the latent. Verify against co-located reference.
6. **Low-volume regime loses** — like all disaggregation, EPD adds orchestration overhead that only amortizes at scale. The negative-result boundary is an expected, documented outcome, not a failure.

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NVENC unavailable on chosen denoise HW | Medium | Medium | Place encode on a tier that has NVENC; fall back to CUDA encode and measure the penalty |
| Crossover rate above realistic load | Medium | High | Then the honest result is "intra-box async (B1) is the right answer at this scale" — a valid finding; document it |
| Inter-tier serialization dominates | Low | High | Confirm artifact sizes in Phase 1 before multi-node; if large, the split point is wrong |
| Diffusion latent transport perturbs output | Low | High | Bit-exact latent check vs co-located reference in Phase 1 |
| Ray multi-framework cold start too slow | Medium | Low | Reuse ray-serve-video runtime_env pins; pre-warm replicas |
| Tier-balance ratio unstable under bursty load | Medium | Medium | Phase 5 autoscaling; measure under realistic arrival traces |

## Estimated Cost

| Phase | Sessions | Hours | Instances | Total |
|-------|----------|-------|-----------|-------|
| Instance A (VLM EPD) | 1 | ~8 | mid + expensive GPU tiers | ~$300 |
| Instance B (Wan 2.2 EPD) | 1 | ~8 | mid + g7e/expensive tiers | ~$350 |
| **Total** | **2** | **~16** | | **~$650** |

## Relationship to Other Specs

| Spec / Blueprint | Relationship |
|------------------|-------------|
| `ray-serve-video` (blueprint) | **Prior art** — disaggregated multimodal pipeline in all but name (Kafka, runtime_env isolation, in-memory hand-off). EPD makes the pattern explicit and measures at-scale payoff. |
| `kernel-optimization-agent.md` | **Complementary** — that optimizes compute *inside* the kernel; this optimizes topology *around* the pipeline. The two halves of utilization. |
| `pd_disagg_single_node` (memory) | EPD is the same lever one stage earlier; inherits the small-artifact + at-scale-crossover economics |
| `mooncake-kv-tiering.md` | Both are scheduling/data-movement layer (not kernels); shares the "split where the artifact is small" intuition (KV cache there, latents/embeddings here) |

---

> **Note**: Operational artifacts (lessons learned, experiment results, tier topology configs, cost analysis)
> belong in the blueprint directory, not in this spec.
