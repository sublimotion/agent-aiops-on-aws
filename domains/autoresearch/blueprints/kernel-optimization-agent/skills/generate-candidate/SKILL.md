---
name: generate-candidate
description: Generate an optimized kernel candidate using the current state vector approach. Injects all active constraints from the database, uses the specified DSL (Triton/TileLang/CUDA), and targets the highest-headroom region identified by profiling. Use after profiling identifies a target, or when the previous candidate fails and the state vector has advanced.
---

# Generate Candidate

## Overview

Produces a new kernel implementation targeting a specific bottleneck, guided by the constraint database, profile data, and current state vector position. The generator adapts its approach based on accumulated failure evidence.

## Usage

### Input

1. **Target region**: Which kernel/sub-kernel to optimize (e.g., `moe_dispatch_384expert`, `mla_decode_fp8`)
2. **Profile data**: Output from `/profile-kernel` — roofline position, bottleneck analysis
3. **State vector position**: Current approach level (0-5)
4. **Active constraints**: Auto-loaded from `constraints.jsonl`
5. **Champion kernel**: Current best implementation (to beat)

### State Vector Approaches

| State | Approach | Description |
|-------|----------|-------------|
| 0 | Triton autotuning | Sweep tile sizes, num_warps, num_stages on existing kernel (why a *search* and not a formula: `results/tile-tuning-explainer-20260605.html`) |
| 1 | Triton algorithmic rewrite | Different memory access pattern, loop ordering, fusion |
| 2 | TileLang rewrite | Different DSL, different optimization space, TileKernels reference |
| 3 | CUDA C++ megakernel adaptation | Port from DeepGEMM/FlashMoE, adapt to 384-expert |
| 4 | Upstream cherry-pick | Evaluate existing PRs individually on K2.6 |
| 5 | Composition | Combine best kernels across layers |

### Output

A kernel source file at `scripts/kernels/{target}/{iteration_id}.py` (or `.cu` for CUDA) with:
- Header comment documenting: target, state vector position, constraints applied, hypothesis
- The kernel implementation
- A correctness test function (calls kernel + PyTorch reference, asserts rtol)
- An autotuning config (if Triton)

### Prompt Construction

The generation prompt is assembled from:

```
[SYSTEM] You are a GPU kernel engineer optimizing for {target_hardware}.

[CONSTRAINTS — from database, always injected]
HARD: {list of hard constraints with evidence}
SOFT: {list of soft constraints — may override if justified}
POSITIVE: {list of patterns that worked before}

[PROFILE — from /profile-kernel output]
Current bottleneck: {roofline classification}
Limiting factor: {warp stalls, occupancy, BW, etc.}
Target metric: {what to improve}

[CHAMPION — current best]
{source code of current best kernel}
Performance: {TFLOPS, BW, occupancy}

[STATE VECTOR — current approach]
You are at State {N}: {approach description}
Previous states attempted: {what failed and why}

[TASK]
Generate an optimized kernel that improves on the champion.
Explain your optimization hypothesis in the header comment.
```

### Constraint Injection Rules

- **Hard constraints**: Always injected, cannot be overridden. Source: L0/L1 failures (compile errors) and architecture facts.
- **Soft constraints**: Injected with "prefer to avoid unless you have a specific reason." Source: L3 regressions.
- **Positive patterns**: Injected as "this approach worked before." Source: L4 promotions.
- Max constraint injection: 2000 tokens. If exceeded, prioritize by severity (hard > soft > positive) and recency.

### When NOT to Use

- Before `/profile-kernel` has identified the target region
- When the freeze manager has frozen this region
- When the constraint database is empty (run Phase 1 first to seed it)
