---
name: verify-kernel
description: Run the cascaded L0-L4 verification pipeline on a kernel candidate. Each level gates the next — a candidate that fails L1 never wastes time on L4. Records telemetry at every stage regardless of pass/fail. Use after /generate-candidate produces a new kernel, or when evaluating an upstream PR cherry-pick.
---

# Verify Kernel

## Overview

Cascaded verification pipeline that progressively filters kernel candidates. Parallel to the learned-verifier cascade (RF → multiprompt → debate), but using hardware signals instead of LLM judges.

## Verification Levels

### L0: Parse (< 1s)

**Signal**: Binary pass/fail
**What**: Syntax validation. For Triton: `triton.compile()` dry run. For TileLang: `tilelang.parse()`. For CUDA: `nvcc --dry-run`.
**Reject if**: Missing imports, brace mismatch, undefined symbols, invalid decorator args.
**Record**: Error class (syntax, import, type) → feeds L0 constraints.

### L1: Compile (< 30s)

**Signal**: Binary + error classification
**What**: Full compilation against target environment.
- Triton: JIT compile with target `sm_103`, verify PTX generation
- TileLang: Full build with NVRTC
- CUDA: `nvcc -arch=sm_103 -O3`
**Reject if**: Compilation fails (ABI mismatch, unsupported op, register overflow)
**Record**: Error class (ABI, register, shared_memory, arch_compat) → feeds L1 constraints.

### L2: Correctness (< 60s)

**Signal**: Numeric (max deviation, mean deviation, edge-case failures)
**What**: Run kernel on 100 random inputs, compare against PyTorch reference.
- Generate inputs with realistic shapes (batch sizes 1, 16, 128, 512)
- For FP8: additional edge-case inputs (denormals, near-max values)
- Tolerance: rtol=1e-3 (BF16), rtol=1e-2 (FP8)
**Reject if**: Any input exceeds 5x tolerance OR >5% of inputs exceed 1x tolerance
**Record**: Max deviation, deviation distribution, failing input shapes → feeds L2 constraints.

**Statistical verification** (for FP8):
- Not just max error, but KL divergence of output distribution vs reference
- Reject if KL > 0.01 even if all individual values pass rtol

### L3: Kernel Benchmark (< 3 min)

**Signal**: Multi-dimensional continuous (TFLOPS, bandwidth, occupancy, speedup)
**What**: NSight Compute metrics on the candidate kernel in isolation.
- 5 iterations, 3 warmup
- Extract: achieved TFLOPS, memory BW, SM occupancy, warp stalls
- Compare against champion kernel on same metrics
**Reject if**: Regression on primary metric (TFLOPS for compute-bound, BW for memory-bound)
**Record**: Full ncu metric vector → telemetry.jsonl. Even rejected candidates are valuable data.

### L4: End-to-End Benchmark (< 7 min)

**Signal**: Continuous with confidence interval, CI-gated
**What**: Full model serving benchmark with the candidate kernel integrated.
- Run benchmark-serving.py at c=1, c=128, c=512
- 5 runs per concurrency level
- Extract: throughput (tok/s), TPOT p50/p99, TTFT p50/p99
**Promotion gate**: Speedup CI (95%) must exclude 1.0 AND beat current champion
**Record**: Full benchmark results → telemetry.jsonl + leaderboard update if promoted.

## Telemetry Output

Every candidate, regardless of pass/fail, produces a telemetry record:

```json
{
  "candidate_id": "moe_dispatch_042",
  "timestamp": "2026-05-15T14:30:00Z",
  "target_region": "moe_dispatch_384expert",
  "state_vector": 1,
  "dsl": "triton",
  "constraints_active": 12,
  "l0_pass": true,
  "l1_pass": true,
  "l1_compile_time_s": 8.3,
  "l2_pass": true,
  "l2_max_rtol": 0.00043,
  "l2_mean_rtol": 0.00012,
  "l3_pass": true,
  "l3_tflops": 312.5,
  "l3_bw_utilization": 0.91,
  "l3_occupancy": 0.75,
  "l3_speedup_vs_champion": 1.08,
  "l4_pass": true,
  "l4_throughput_c512": 11240,
  "l4_speedup_vs_baseline": 1.077,
  "l4_speedup_ci_lower": 1.03,
  "l4_speedup_ci_upper": 1.12,
  "promoted": true,
  "code_features": {"loc": 87, "tile_m": 64, "tile_k": 128, "num_warps": 8, "num_stages": 3}
}
```

## Usage

```
/verify-kernel <path-to-candidate>
```

Runs L0→L4 sequentially, short-circuiting on first failure. Returns verdict + telemetry.

### When NOT to Use

- For profiling (use `/profile-kernel` instead — different purpose)
- On candidates that haven't been generated yet
- During Phase 1 (profiling phase) — verification is for Phase 2+ candidates
