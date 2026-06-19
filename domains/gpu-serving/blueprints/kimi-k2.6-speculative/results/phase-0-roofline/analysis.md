# Phase 0 — Roofline Analysis (B300 × Kimi K2.6)

**Date**: 2026-05-13
**Hardware**: p6-b300.48xlarge — 8× B300 SXM6 275 GiB HBM3e, NVSwitch NV18

## Hardware Ceilings (per-GPU × 8)

| Resource | Per-GPU | 8-GPU Aggregate | Practical (85%) |
|---|---|---|---|
| HBM bandwidth | 8 TB/s | 64 TB/s | 54 TB/s |
| FP8 tensor core | 5 PFLOPS | 40 PFLOPS | 34 PFLOPS |
| FP4 tensor core | 10 PFLOPS | 80 PFLOPS | 68 PFLOPS |
| NVSwitch bisection | — | 1.8 TB/s | 1.5 TB/s |
| HBM capacity | 275 GB | 2,200 GB | — |

## Model Footprint

- Kimi K2.6 FP8: ~594 GB weights, 1T params total, 32B active, 384 experts
- EAGLE3 draft: ~6 GB
- Remaining VRAM for KV + activations: 2,200 − 594 − 6 = **1,600 GB budget**

## Arithmetic Intensity at Decode (batch=1)

```
Weight read:       32B active × 0.5 B/param (FP8) = 16 GB / token
KV read (MLA):     compressed d_c=512 × 61 layers ≈ 0.8 GB / token
Total bytes:       ~16.8 GB / token

FLOPs:             2 × 32B × 2 (fwd) = 128 GFLOPs / token

Arithmetic intensity = 128e9 / 16.8e9 ≈ 7.6 FLOPs/byte
Machine AI (B300):   34e15 / 54e12 ≈ 630 FLOPs/byte
```

**7.6 ≪ 630 → deeply bandwidth-bound at batch=1.** Decode won't benefit from more FLOPS; only faster/larger batches or fewer decode steps (speculative decoding) help.

## Decode Throughput Ceilings

### BW-bound regime (realistic batches ≤92)
```
At batch=B:
  Bytes/step = 16 GB (weight read, constant) + 0.8×B GB (KV read scales)
  Steps/s    = 54e12 / (16e9 + 0.8e9 × B)
  Tokens/s   = B × steps/s

B=1:    54e12 / 16.8e9 ≈ 3,214 steps/s   →   3,214 tok/s   (single-stream ceiling)
B=32:   54e12 / 41.6e9 ≈ 1,298 steps/s   →   41,538 tok/s
B=128:  54e12 / 118.4e9 ≈ 456 steps/s    →   58,378 tok/s
B=512:  54e12 / 425.6e9 ≈ 127 steps/s    →   65,000 tok/s  ← **BW ceiling at c=512**
```

### Compute-bound regime (batches >92)
Crossover at B* ≈ 92 where AI = 630. Above that:
```
Aggregate tok/s = 34 PFLOPS × 8 / 128 GFLOPs/token = **212,500 tok/s**
  (practical limit pre routing/scheduling/sampling overheads)
```

## Measured vs Ceiling (K2.6 baseline)

| Operating point | Measured | BW ceiling | Efficiency |
|---|---|---|---|
| Single-stream (c=1) | 128 tok/s | 3,214 tok/s | **4%** |
| Aggregate c=512 (no spec) | 10,437 tok/s | 65,000 tok/s | **16%** |
| Aggregate c=64 (Phase 1 EAGLE3) | 3,657 tok/s | ~30,000 tok/s (est) | **12%** |

## Gap Attribution

The 84% gap at c=512 has these sources (hypothesis, will verify with nsys in Phase 5):
- **Scheduler + Python overhead**: ~15-20%
- **KV cache fragmentation / paging**: ~10-15%
- **Kernel launch latency (many small ops per layer)**: ~15-25%
- **MoE routing + expert imbalance**: ~10%
- **Not running optimal batch (512 < compute crossover but KV memory caps batch)**: ~15-20%

## Spec Decode Ceiling

EAGLE3 amortizes weight reads across multiple generated tokens per step. If accept rate = a and draft tokens per step = k:
```
Effective tokens/decode = 1 + a×k
  Phase 1 data: a=0.56, k=2.2 acc length → eff = 1 + 1.23 = 2.23 tokens/step

But overhead:
  Extra BW per step   = draft model fwd (~0.3 GB) + verify GEMM (~0.5 GB extra) ≈ +1 GB/step
  At B=64:   (16 + 0.8×64 + 1) GB = 68 GB/step
  vs baseline (16 + 0.8×64) GB = 67.2 GB/step   (overhead is ~1.2% per step)

Predicted agg @ c=64:
  Baseline: 54e12 / 67.2e9 × 64 ≈ 51,400 tok/s   (BW ceiling)
  Actual baseline measured: ~10,000 tok/s (20% efficiency)
  EAGLE3 ceiling: 54e12 / 68e9 × 64 × 2.23 ≈ 113,300 tok/s
  Actual EAGLE3 measured: 3,657 tok/s   (3.2% efficiency)
```

**Why EAGLE3 measured so far below its ceiling at c=64:**
- The 30-40% overhead estimate from lessons.md L6 is *wall-time* overhead, not *BW* overhead — includes draft model python overhead, verification scheduling, CUDA graph recapture per batch change
- A tuned EAGLE3 (fewer num_steps, smaller draft_tokens) should close much of this gap

## Phase 1b Hypothesis

The sweep should find a config where:
- **num_steps=1 or 2** (reduces draft overhead proportionally)
- **num_draft_tokens=2 or 4** (matches accept length 2.2; beyond that is wasted)
- **eagle_topk=1** (wider search adds compute without better accept rate for K2.6)

At this target config, expect aggregate recovery to ~70-80% of non-spec baseline at c=64 (≈7,000 tok/s) with 30-50% better per-request throughput. Absolute wins would be single-stream tok/s (the real use case for coding agents).

## Multi-Node Decision

**NOT needed for K2.6 on B300.** Binding constraint is per-GPU HBM bandwidth. Adding nodes adds replicas at same per-request latency. See spec §Multi-Node Decision Framework.

## Deliverables for Phase 0 (live measurements pending)

When the node is ready, these will populate `phase-0-roofline/`:

- [ ] Measured HBM sustained BW via NCCL all_reduce + dmon
- [ ] Measured FP8 DeepGEMM TFLOPS at MoE shapes (M=128, 512, 2048, 8192)
- [ ] NVSwitch all-reduce bisection measured
- [ ] nsys trace of K2.6 decode at c=1, 64, 256, 512 — per-layer breakdown
- [ ] Plot: measured efficiency % at each operating point

The arithmetic ceilings above are what we're measuring against.
