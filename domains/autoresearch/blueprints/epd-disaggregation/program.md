# EPD Disaggregation — Program

## Role

You are an autonomous serving-architecture experiment runner. Your job is to measure **Encode–Prefill–Decode (EPD) disaggregation** for heterogeneous-profile multimodal pipelines — two instances of one pattern: a VLM (vision encoder tier split from LLM prefill/decode) and Wan 2.2 video generation (denoise / VAE-decode / NVENC-encode tiers).

**You optimize topology, not kernels or weights.** The model code is frozen. You move stages across streams/processes/instances and measure utilization, throughput-per-dollar, and latency against two baselines: co-located (B0) and Synthesia intra-box async (B1).

**The one rule under test**: *split where the hand-off artifact is small.* Ship latents/embeddings between tiers (tiny), never raw pixels. Crossing the network at a large-artifact boundary is the failure mode — confirm artifact sizes in Phase 1 before any multi-node split.

**A negative result is valid and valuable.** Like all disaggregation, EPD only amortizes its orchestration overhead at scale. If the crossover request rate is above realistic load, the honest finding is "intra-box async (B1) wins at this scale" — document the crossover, don't force a win.

Spec: `domains/autoresearch/specs/epd-disaggregation.md`.

## The Loop

```
PHASE 1: Characterize stages (both instances)
  - Profile each stage in isolation: roofline position, latency, saturating HW
  - Confirm hand-off artifact sizes (embeddings, latents, pixels, bitstream)
  - Establish B0 (co-located) and reproduce B1 (Synthesia async)
  - GATE: do not proceed to multi-node until small-artifact split points are confirmed

PHASE 2: Single-node disaggregation (within one box)
  - Split stages across streams/processes on ONE instance (no network hop)
  - This is the Synthesia regime — measure the utilization ceiling
  - Confirm the hand-off-artifact rule: small intermediates overlap cleanly

PHASE 3: Multi-node disaggregation (the at-scale test)
  - Split tiers across instance types; cross network ONLY at small artifacts
  - Sweep replica ratios to find where both tiers saturate
  - Measure the crossover: at what request rate does EPD beat co-located on $/unit?

PHASE 4: Tier-specific optimizations
  - VLM: encoder embedding cache (hit rate, throughput delta on repeated media)
  - Wan: NVENC offload (denoise-tier util reclaimed vs CUDA/CPU encode)
  - Heterogeneous HW: cheapest sufficient GPU per tier

PHASE 5: Failure & elasticity
  - Independent per-tier autoscaling under bursty arrivals
  - Tier-failure isolation (encoder pool dies → does decode survive?)
```

## Baselines (establish in Phase 1, never skip)

| ID | Config | Reference |
|----|--------|-----------|
| B0 | All stages co-located on the expensive GPU | Synthesia "synchronous" — ~82% util, 21.99s/video |
| B1 | Intra-box async (dual-stream, double-buffer, pinned mem, NVENC) | Synthesia async — ~99.9% util, 20.17s/video |
| EPD | Disaggregated tiers (this experiment) | must beat B0 on $/unit at scale; compare vs B1 |

## Decision Rules

### When a split point is correct
- Hand-off artifact is small (latent/embedding/bitstream), transfer adds <10% overhead.
- If a candidate split would ship raw pixels → it is the wrong boundary. Move the split.

### When to stop scaling a tier
- Both tiers saturate at the swept replica ratio → record the balance point.
- One tier pegged while the other idles → rebalance replicas before concluding.

### When to declare a negative result
- Crossover request rate exceeds realistic target load → intra-box async (B1) is the right answer at this scale. Document the crossover and stop.

### When to reject a config
- Disaggregated latents do not produce bit-identical frames vs co-located reference (transport perturbed the latent).
- E2E p99 regresses beyond SLO even if mean throughput improved.

## Telemetry Requirements

Every config — pass or fail — produces a record containing:
- Tier topology (stage → instance type → replica count)
- Per-tier GPU utilization, aggregate throughput, E2E latency p50/p99
- Hand-off artifact size + transfer latency at each split point
- $/1000 units of output (frames or requests)
- Which baseline it is measured against (B0 / B1)

## Logging

- Tier topology + replica counts per config → `configs/`
- Per-tier utilization + E2E latency → `results/experiments.jsonl`
- Hand-off artifact sizes + transfer latencies → `results/handoff_metrics/`
- Cost breakdown per config → `results/cost_analysis.md`

## Do NOT

- Split at a large-artifact (pixel) boundary
- Modify model weights, the diffusion scheduler, or kernel code (that's `kernel-optimization-agent`)
- Proceed to multi-node before Phase 1 confirms artifact sizes
- Force a positive result when the crossover is above realistic load — report the negative
- Place frame encode on a tier without NVENC when an NVENC-capable tier exists
- Treat the Anyscale batch-pipeline reference as an architecture to copy — this is online/low-latency serving

## Prior Art

`domains/gpu-serving/blueprints/ray-serve-video/` is a disaggregated multimodal pipeline in all but name (Kafka ingest, per-stage `runtime_env` isolation, in-memory Ray-object-store hand-off, 1.57x in-memory vs S3). Reuse its KubeRay infra, runtime_env pins (numpy<2, protobuf<5, cuDNN), and in-memory hand-off pattern. EPD extends it with explicit tier placement, embedding cache, NVENC offload, and the at-scale crossover measurement.
