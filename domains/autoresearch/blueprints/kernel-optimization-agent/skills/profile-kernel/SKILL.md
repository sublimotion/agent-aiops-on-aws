---
name: profile-kernel
description: Profile a GPU kernel or full serving pipeline using NSight Compute. Produces roofline classification (compute/memory/latency-bound), identifies top-N kernels by wall-clock time, and maps each to pipeline stage (MoE dispatch, MLA decode, attention, comm). Use when starting optimization, after a new baseline is established, or when the freeze manager redirects to a new target region.
---

# Profile Kernel

## Overview

Runs NSight Compute (ncu) profiling on a target kernel or full model serving pipeline, extracts roofline position, and produces a structured bottleneck analysis that feeds into the constraint database and guides the generator.

## Usage

### Input

1. **Target**: Either a specific kernel function name OR "full pipeline" for holistic profiling
2. **Model**: Which model is being served (e.g., Kimi K2.6, DeepSeek V3)
3. **Engine**: vLLM or SGLang
4. **Concurrency**: At what load level to profile (default: c=1 for kernel isolation, c=128 for pipeline)

### Output

Structured JSON with:
```json
{
  "target": "fused_moe_dispatch",
  "roofline": "memory-bound",
  "tflops": 245.3,
  "memory_bw_utilization": 0.87,
  "sm_occupancy": 0.62,
  "shared_memory_usage_bytes": 98304,
  "warp_stall_reasons": {"memory_dependency": 0.43, "execution_dependency": 0.21},
  "wall_clock_us": 142.5,
  "fraction_of_pipeline": 0.18,
  "pipeline_stage": "moe_dispatch",
  "optimization_headroom": "high — memory-bound with 13% BW unused, occupancy limited by shared memory"
}
```

### Procedure

1. Ensure model is serving and warmed up (>10 requests completed)
2. Run `ncu --set full --target-processes all -o profile_output <benchmark_command>`
3. For full pipeline: use `--replay-mode kernel` with `--kernel-name-base` filter
4. Parse ncu output for each kernel:
   - Compute throughput (TFLOPS) vs roofline peak
   - Memory bandwidth vs peak HBM bandwidth (2.4 TB/s on B300)
   - SM occupancy and limiting factor
   - Warp stall breakdown
   - Shared memory and register pressure
5. Rank by wall-clock contribution to total serving time
6. Classify roofline position: compute-bound (>80% compute utilization), memory-bound (>80% BW utilization), latency-bound (neither)
7. Emit structured output + update constraint database with hardware facts

### Constraints

- ncu slows execution 10-100x — profile at low concurrency (c=1 or c=4)
- For e2e timing, use nsys (lighter) rather than ncu
- B300 SM103: peak 227KB shared memory, 2.4 TB/s HBM BW, ~2400 TFLOPS FP8
- Profile AFTER DeepGEMM JIT warmup completes (wait for "warmup done" in logs)

### When NOT to Use

- When you need e2e throughput numbers (use benchmark-serving.py instead)
- When the kernel has already been profiled this session and nothing changed
- During L4 benchmark runs (profiling overhead corrupts timing data)
