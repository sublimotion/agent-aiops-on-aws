---
name: diagnose-bottleneck
description: Analyze a failed or underperforming kernel candidate using ncu profile data and telemetry history. Produces a structured diagnosis (what's wrong, why, what to try next) and updates the constraint database. Use after /verify-kernel returns a failure or an L3/L4 regression, or when the agent needs guidance on which optimization direction to pursue.
---

# Diagnose Bottleneck

## Overview

Takes a kernel candidate's verification telemetry and profile data, compares against the champion and historical candidates, and produces an actionable diagnosis. This is the "feedback" step in the compound loop — it converts raw hardware signals into constraints and strategy recommendations.

## Usage

### Input

1. **Candidate telemetry**: Output from `/verify-kernel`
2. **Champion profile**: Current best kernel's ncu metrics
3. **Historical telemetry**: Last N candidates' telemetry (for trend analysis)
4. **Constraint database**: Current state

### Diagnosis Categories

| Category | Signal Pattern | Action |
|----------|---------------|--------|
| **Memory-bound plateau** | BW utilization >90%, TFLOPS low | Switch to compute-reducing approach (smaller tiles, less recomputation) |
| **Compute-bound plateau** | TFLOPS near peak, BW low | Unlikely to improve further on this kernel; freeze or try fusion |
| **Shared memory pressure** | Occupancy limited by smem | Reduce tile sizes, split into multiple kernels, use register-heavy approach |
| **Warp divergence** | High divergent_branch stalls | Restructure conditionals, pad to warp boundaries, sort by predicate |
| **Load imbalance** | Some SMs idle, others saturated | Expert routing imbalance (384-expert specific); rebalance tile assignment |
| **Launch overhead** | Kernel time < launch overhead | Fuse with adjacent kernel, use persistent kernel pattern |
| **Correctness failure** | L2 deviation pattern | Identify failing input shapes → race condition, sync issue, or precision loss |
| **Silent regression** | L3 passes but L4 regresses | Kernel-level improvement offset by integration overhead (CUDA graph invalidation, memory layout change) |

### Output

```json
{
  "diagnosis": "memory-bound plateau",
  "evidence": "BW utilization 91% (2.18 TB/s of 2.4 TB/s peak). Last 3 candidates improved TFLOPS by <1% each.",
  "root_cause": "Tile size 128x64 causes L2 cache thrashing on 384-expert dispatch. Each expert's data spans 2 L2 cache lines.",
  "recommendation": "Advance state vector to State 2 (TileLang rewrite). Try smaller tiles (64x32) with explicit L2 residency control.",
  "new_constraints": [
    {"type": "soft", "rule": "384-expert dispatch with tile_k=128 causes L2 thrashing", "evidence": "candidates 039-042 all plateau at 91% BW"},
    {"type": "positive", "rule": "tile_k=64 avoids L2 thrashing on 384-expert routing", "evidence": "candidate 038 achieved 93% BW before tile_k was increased"}
  ],
  "freeze_recommendation": false,
  "state_vector_advance": true
}
```

### Trend Analysis

Examines the last 10 candidates for:
- **Convergence**: Are improvements shrinking? (trigger freeze if <2% for 3 consecutive)
- **Oscillation**: Is the agent flip-flopping between two approaches? (inject "do not revisit" constraint)
- **Dead ends**: Has a specific approach been tried 3+ times without success? (advance state vector)
- **Regression clusters**: Are failures concentrated in one error class? (strengthen that constraint)

### Constraint Database Update

Every diagnosis adds constraints:
- **From L0/L1 failures**: Hard constraints (architecture limitations)
- **From L2 failures**: Correctness constraints (sync/ordering requirements)
- **From L3 regressions**: Performance constraints with measured deltas
- **From L4 null results**: Architectural constraints (interactions with serving pipeline)
- **From trends**: Meta-constraints (approach X is exhausted for region Y)

### When NOT to Use

- Before any candidates have been generated (nothing to diagnose)
- When the candidate was promoted successfully (record positive pattern instead)
- For initial profiling (use `/profile-kernel` — diagnosis requires comparison)
