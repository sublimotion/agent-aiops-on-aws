# Autoresearch Spec: Agent Swarm

## Status: DRAFT

## Overview

Two-phase experiment measuring how model capability (scaling and finetuning) affects agent swarm performance on a single GPU node. Phase 1 validates the bitter lesson empirically: same benchmark, same harnesses, four models spanning 17B-397B active parameters and base vs SERA-finetuned weights. Phase 2 builds a concurrent swarm demo on B200 using the best configuration from Phase 1.

**Core thesis**: At the SWE-bench horizon (< 30 min tasks), stronger models compress harness dependence, reduce turn budgets, and shrink the ensemble ceiling — the bitter lesson converging in real time. This experiment measures the convergence rate.

**Builds on**: Agent Harness experiment (Phase 1 turn degradation + Phase 2 seven-harness comparison on Devstral Small 2 24B).

## Phases

### Phase 1: Measure the Axes (g7e)

**Goal**: Quantify how model capability affects harness spread, turn budget, precision, and ensemble ceiling across two axes — scale and finetuning.

**Models**:

| Model | Total Params | Active Params | Source | Role |
|-------|-------------|--------------|--------|------|
| Devstral Small 2 FP8 | 24B | 24B (dense) | On NVMe | Baseline (existing data) |
| Qwen 2.5 Coder 32B | 32B | 32B (dense) | Download from HF | SERA finetuning control |
| SWE-smith Qwen 2.5 Coder 32B | 32B | 32B (dense) | `SWE-smith/Qwen2.5-Coder-32B-SWE-smith` | SERA-finetuned (the "after") |
| Qwen3.5-397B-A17B FP8 | 397B | 17B (MoE) | On NVMe | Scale axis — frontier MoE |

**Axes isolated**:

```
Finetuning axis (same arch, different weights):
  Base Qwen 2.5 Coder 32B  →  SWE-smith Qwen 2.5 Coder 32B
  Isolates SERA effect. Published: 20% → 40.2% (+20pp)

Scale axis (different models, all base):
  Devstral 24B  →  Qwen 2.5 Coder 32B  →  Qwen3.5-397B-A17B
  Isolates model capability: 24B dense → 32B dense → 17B active MoE

Cross-axis comparison:
  SWE-smith 32B (finetuned small) vs Qwen3.5-397B (base large)
  Does SERA finetuning match 10x+ scale increase?
```

**Bitter lesson predictions** (to validate or refute):

| Metric | Devstral 24B (measured) | SWE-smith 32B (predicted) | Qwen3.5 397B (predicted) |
|--------|------------------------|--------------------------|-------------------------|
| Best single harness | 22% | 35-45% | 35-45% |
| Harness spread | 22pp (0-22%) | ~10-12pp | ~10-12pp |
| Turns needed | 30 | 15-20 | 15-20 |
| Parkinson's % (explore before edit) | 58-65% | 30-40% | 30-40% |
| Ensemble ceiling (3 harnesses) | ~30% | ~45-50% | ~45-50% |
| Precision (best harness) | 53% (Claude Code) | 50-65% | 50-65% |

If both finetuned-32B and base-397B land at ~40%, that validates: (1) the bitter lesson at this horizon, (2) SERA finetuning as a scale-equivalent investment, (3) harness spread compression with model quality.

**Harnesses** (3 per model, subset of the 7 already tested):

| Harness | Why Included |
|---------|-------------|
| OpenCode | Best pass rate (22%) on Devstral, good baseline |
| Claude Code | Best precision (53%), tests conservative behavior at higher capability |
| SERA baseline | Continuity with existing Phase 1/2 data |

Three harnesses per model × four models = 12 configurations. Enough to measure harness spread without the diminishing returns of testing all 7.

**Benchmark**: Same SWE-bench Lite 50-issue subset (seed 42, stratified by 11 repos). Direct comparison to existing data.

### Phase 2a: The Swarm Demo — Naive Parallel (B200)

**Goal**: Run a concurrent agent swarm on one GPU node using the best model + harness from Phase 1. Establish the baseline swarm performance without scheduling optimizations.

**Architecture**:

```
┌──────────────────────────────────────────────────┐
│  Single B200 x8 Node (or g7e for smaller models) │
│                                                   │
│  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Orchestrator  │  │ Workers (best from Ph1)   │  │
│  │ Qwen3.5 TP4  │  │ × 4 replicas TP1          │  │
│  │ GPU 0-3      │  │ GPU 4-7                    │  │
│  └──────┬───────┘  └──────────┬────────────────┘  │
│         │                     │                    │
│  ┌──────▼─────────────────────▼────────────────┐   │
│  │  Local KV Cache (prefix caching + NVMe)     │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────┬────────────────────────────┘
                        │ localhost
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Agent 1 │   │ Agent 2 │...│Agent N  │  ← CPU processes
    │ OpenCode│   │ OpenCode│   │ OpenCode│
    └─────────┘   └─────────┘   └─────────┘
```

**Key design decisions**:
- **Single node**: All inference is localhost. No llm-d, no cross-node routing, no distributed KV.
- **Agents on CPU**: Agent processes (OpenCode/harness) are HTTP clients. ~200 MB RAM each. Hundreds fit on one node's system memory.
- **Orchestrator dispatches, workers execute**: Qwen3.5 classifies issues, groups by repo (prefix cache sharing), routes to worker queue. Workers run the OpenCode harness with the best Phase 1 model.
- **Before/after**: Run the swarm with base workers, then SERA-finetuned workers. Measure pass rate, wall time, cost.
- **No scheduling optimization**: vLLM handles request queuing natively. Agents submit inference requests directly to worker replicas. GPU bubbles during tool execution are wasted — this is the baseline to beat in Phase 2b.

**Configurations**:
- (A) Sequential baseline: 1 agent at a time, best model + harness
- (B) Naive parallel: N concurrent agents, round-robin across 4 replicas
- (C) Before/after: base workers vs SERA-finetuned workers (if different models win on each axis)

**Phase 2a is contingent on Phase 1 results**. If Phase 1 shows the finetuned 32B model matches the 397B base, the swarm uses finetuned 32B workers (4 replicas on 4 GPUs) + Qwen3.5 orchestrator (on the remaining 4 GPUs). If the 397B base wins, the swarm uses Qwen3.5 for everything (TP4 workers + orchestrator time-sharing).

### Phase 2b: ThunderAgent Scheduling (B200)

**Goal**: Add program-aware scheduling to the Phase 2a swarm and measure the GPU utilization improvement. ThunderAgent fills GPU bubbles during tool execution — the 40-60% of wall time where agents are reading files, running tests, or applying patches while GPUs hold idle KV cache.

**Architecture** (adds scheduling layer to Phase 2a):

```
┌──────────────────────────────────────────────────┐
│  Single B200 x8 Node                              │
│                                                   │
│  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Orchestrator  │  │ Workers × 4 replicas TP1  │  │
│  │ Qwen3.5 TP4  │  │ GPU 4-7                    │  │
│  │ GPU 0-3      │  │                             │  │
│  └──────┬───────┘  └──────────┬────────────────┘  │
│         │    ┌────────────────┤                    │
│         │    │  ThunderAgent  │                    │
│         │    │  Scheduler     │                    │
│         │    └───────┬────────┘                    │
│         │            │                             │
│  ┌──────▼────────────▼─────────────────────────┐   │
│  │  Local KV Cache (HiCache + NVMe)            │   │
│  │  Prefix cache shared across replicas        │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────┬────────────────────────────┘
                        │ localhost:9000 (scheduler)
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Agent 1 │   │ Agent 2 │...│Agent N  │  ← CPU processes
    │ OpenCode│   │ OpenCode│   │ OpenCode│
    └─────────┘   └─────────┘   └─────────┘
```

**ThunderAgent scheduler** (`thunder_proxy.py`):
- Sits on port 9000, proxies to vLLM workers on ports 8001-8004
- Each agent session gets a `program_id` (maps to ThunderAgent's "LLM Program" abstraction)
- Tracks program state: REASONING (on GPU) vs ACTING (executing tools, off GPU)
- When a program enters ACTING: marks its GPU capacity as reclaimable
- Other programs' inference requests backfill the freed slot ("optimistic scheduling")
- When the original program returns from tool execution: preempts the backfill, resumes with warm KV cache
- `tool_coefficient` parameter controls oversubscription aggressiveness

**Configurations**:
- (D) ThunderAgent parallel: same N agents as Phase 2a config B, but with scheduling
- (E) ThunderAgent oversubscribed: 2N agents (double the naive parallel count), testing whether bubble filling supports higher concurrency at similar latency

**Metrics to compare Phase 2a vs 2b**:

| Metric | Phase 2a (naive) | Phase 2b (ThunderAgent) |
|--------|-----------------|------------------------|
| GPU utilization (% time active) | ~40% (expected) | Target: 60-70% |
| Wall time for 50 issues | Baseline | Target: 30-50% reduction |
| Pass rate | Baseline | Should match (scheduling doesn't affect quality) |
| Max concurrent agents at p99 TTFT < 5s | Baseline | Target: 1.5-2x |
| Preemption overhead (wasted tokens) | 0 | Measure |
| Cost per issue | Baseline | Target: 40-60% reduction |

**ThunderAgent is novel engineering**. The minimal viable implementation needs:
1. Program registry (track `program_id` → state, backend, KV metadata)
2. State transition detection (inference response complete → ACTING; new request from same program → back to REASONING)
3. Optimistic scheduling queue (backfill freed GPU slots with waiting programs)
4. Preemption signal (returning program reclaims its slot, backfill yields)

This can start as a ~300-line Python proxy. Full ThunderAgent features (tool resource manager, adaptive `tool_coefficient`, multi-backend routing) are Phase 2b+ stretch goals.

## Components

### 1. Compute

**Phase 1**:
- **Instance**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each)
- **Already provisioned**: SSH access, vLLM installed, harness scripts deployed
- **GPU layout** (sequential for TP1 models, then TP4 for Qwen3.5):

| Model | TP | GPUs | VRAM/GPU | Context |
|-------|----|------|----------|---------|
| Devstral 24B FP8 | 1 | 1 | ~15 GB | 64K |
| Qwen 2.5 Coder 32B | 1 | 1 | ~18 GB | 64K |
| SWE-smith Qwen 2.5 Coder 32B | 1 | 1 | ~18 GB | 64K |
| Qwen3.5-397B-A17B FP8 | 4 | 4 | ~54 GB/GPU | 128K |

TP1 models can run 3 simultaneously (one per GPU). Qwen3.5 needs all 4 GPUs.

**Phase 2a + 2b**:
- **Instance**: p6-b200.48xlarge (8x B200, 183GB HBM3e each) — capacity block
- **Or**: g7e.24xlarge if Phase 1 best model fits TP1 (32B workers)
- **GPU layout** (B200):

| GPUs | Model | Config |
|------|-------|--------|
| 0-3 | Qwen3.5-397B-A17B (orchestrator) | TP4 |
| 4-7 | Best Phase 1 worker × 4 replicas | TP1 |

Phase 2a and 2b share the same GPU layout. The only difference is whether `thunder_proxy.py` sits between agents and workers (2b) or agents hit workers directly (2a).

### 2. Codebase

- **Source**: Existing harness scripts from `agent-harness` blueprint
- **Fixed files**:
  - SWE-bench Lite 50-issue subset (seed 42)
  - Gold test patches for verification
  - Evaluation pipeline (`harness_eval.py`)
- **Scripts to create**:
  - `scripts/swarm_eval.py` — Phase 1 runner: iterate models × harnesses, collect per-model metrics
  - `scripts/swarm_launcher.py` — Phase 2: concurrent agent process manager with orchestrator dispatch
  - `scripts/thunder_proxy.py` — Minimal ThunderAgent-style scheduler (Phase 2): program-aware queue, warm-KV priority, bubble backfill
  - `scripts/adapters/run_opencode.sh` — Already exists
  - `scripts/adapters/run_claude_code.sh` — Already exists
- **Reuse from agent-harness**:
  - `multi_harness_eval.py` — adapter infrastructure
  - `harness_eval.py` — workspace setup, gold test evaluation
  - All adapter scripts

### 3. Experiment Protocol

**Phase 1**:
- **Metric**: SWE-bench Lite pass rate (gold test patches), harness spread, turns used, precision, Parkinson's ratio (fraction of budget spent before first edit)
- **Matrix**: 4 models × 3 harnesses × 50 issues = 600 runs
- **Time budget**: ~5 hours for TP1 models (3 parallel) + ~15 hours for Qwen3.5 (sequential). Total: ~20 hours.
- **Turn budget**: 30 for all models (same as Phase 1 baseline). If a model converges early, record the natural turn count.
- **Logging**: One JSONL per model-harness pair: `results/swarm_phase1_{model}_{harness}.jsonl`
- **Termination**: All 12 configurations complete.

**Phase 2a**:
- **Metric**: Wall time for 50 issues, pass rate, cost, GPU utilization
- **Configurations**: (A) sequential baseline, (B) naive parallel, (C) before/after workers
- **Time budget**: ~2 hours per configuration (50 issues concurrent ≈ 15 min wall time + setup)
- **Logging**: `results/swarm_phase2a_{config}.jsonl` — per-issue metrics + GPU utilization trace

**Phase 2b**:
- **Metric**: Same as 2a + bubble reclamation rate, preemption overhead, max sustainable concurrency
- **Configurations**: (D) ThunderAgent parallel, (E) ThunderAgent oversubscribed (2N agents)
- **Time budget**: ~3 hours (including scheduler tuning of `tool_coefficient`)
- **Logging**: `results/swarm_phase2b_{config}.jsonl` — per-issue metrics + scheduler event trace

### 4. Model Weights

| Model | Location | Size | Action |
|-------|----------|------|--------|
| Devstral Small 2 FP8 | `/mnt/nvme/models/devstral-small-2-fp8` | 49 GB | None (exists) |
| Qwen3.5-397B-A17B FP8 | `/mnt/nvme/models/` | ~214 GB | On NVMe (already downloaded) |
| Qwen 2.5 Coder 32B | HuggingFace `Qwen/Qwen2.5-Coder-32B-Instruct` | ~18 GB FP8 | Download |
| SWE-smith Qwen 2.5 Coder 32B | HuggingFace `SWE-smith/Qwen2.5-Coder-32B-SWE-smith` | ~18 GB | Download (verify exact HF path) |

### 5. Networking

- **Phase 1**: localhost only. vLLM on port 8000 (swap models between runs).
- **Phase 2**: localhost only. Orchestrator on port 8000, workers on ports 8001-8004. ThunderAgent proxy on port 9000.
- **No external dependencies**: No Bedrock, no llm-d, no cross-node communication.

### 6. Storage

- **NVMe**: `/mnt/nvme/` — model weights (~300 GB total), workspaces, results
- **Results**: Blueprint results directory, JSONL per configuration

## Success Criteria

### Phase 1: Measure the Axes

1. All 12 configurations complete (4 models × 3 harnesses × 50 issues)
2. **Finetuning axis measured**: Base Qwen 2.5 Coder 32B vs SWE-smith, with pass rate, harness spread, turn compression, Parkinson's ratio
3. **Scale axis measured**: Devstral 24B → 32B → 397B MoE, with the same metrics
4. **Bitter lesson validated or refuted**: Does harness spread compress with model quality? Does the finetuned 32B match the base 397B?
5. **Best swarm config identified**: Which model + harness combination maximizes pass rate per GPU-second?
6. **Blog data ready**: Numbers to fill into the bitter lesson draft (harness spread at 3 model scales)

### Phase 2a: Naive Parallel Swarm

1. 50 SWE-bench issues processed concurrently on one node
2. Wall time < 20 min (vs ~8 hours sequential)
3. Pass rate matches or exceeds Phase 1 single-agent results (concurrent execution should not degrade quality)
4. Before/after comparison: base vs finetuned workers on the same swarm infrastructure
5. GPU utilization measured as baseline for Phase 2b comparison
6. Cost per issue < $0.05 (one node-hour amortized across 50 issues)

### Phase 2b: ThunderAgent Scheduling

1. ThunderAgent scheduler deployed as proxy between agents and workers
2. GPU utilization improves by >= 30% vs Phase 2a naive parallel
3. Wall time reduces by >= 30% at same concurrency, or sustains 1.5-2x concurrency at similar latency
4. Pass rate unchanged (scheduling must not degrade quality)
5. Preemption overhead quantified (wasted tokens from interrupted backfill inference)
6. `tool_coefficient` tuning documented for the model + workload combination

## Non-Requirements

- Multi-node deployment (single node is the architecture)
- Full SWE-bench Verified (500 issues) — 50-issue subset for direct comparison
- Training or finetuning any model (we use existing weights)
- llm-d, Gateway API, or distributed routing (localhost only)
- Ensemble of multiple harnesses in production (pick the best one)
- All 7 harnesses from Phase 2b (3 is sufficient to measure spread)

## Known Limitations

- **50-issue subset**: ~5% sampling variance. Results may not generalize to full 300-issue set.
- **SWE-smith model availability**: Exact HuggingFace path for the SWE-smith finetuned Qwen 2.5 Coder 32B needs verification. May be under a different org name.
- **Qwen3.5 tool calling**: Not tested with vLLM on g7e. May need `--tool-call-parser` flag discovery (check mdc).
- **Qwen 2.5 Coder chat template**: May differ from Devstral's Mistral template. Adapter scripts need per-model endpoint config.
- **ThunderAgent is novel engineering**: No existing implementation to deploy. Phase 2 needs a minimal scheduler built from the paper's design. Scope risk.
- **g7e PCIe bandwidth**: Qwen3.5 TP4 over PCIe Gen5 will be slower than NVSwitch. Phase 1 is measuring accuracy not throughput, so this is acceptable. Phase 2 throughput numbers should note the hardware constraint.
- **Docker-limited eval**: Only Django/pytest/sympy issues can be gold-test verified without Docker. Same limitation as the harness experiment.
- **NCCL bug irrelevant**: vLLM inference uses custom allreduce, not NCCL. Training is out of scope.

## Deliverables

### Phase 1

- `RESULTS.md` update with 4-model comparison table
- Visual explainer update (new section: "Bitter Lesson Validation")
- Blog draft update: fill in the empirical numbers for harness spread at 3 model scales
- Per-model JSONL results in blueprint results directory

### Phase 2

- Swarm demo: 50 issues, one node, < 20 min, with GPU utilization trace
- ThunderAgent minimal implementation (program-aware scheduling proxy)
- Before/after comparison (base vs finetuned workers)
- Cost analysis: $/issue for the swarm vs sequential vs Claude API

## Relationship to Blog Draft

The blog ["The Bitter Lesson Has a Time Horizon Problem"](obsidian://open?vault=obsidian-notes&file=01_Projects%2FBlog%20-%20PredictingTheNextToken%2Farticles%2Fbitter-lesson-time-horizon%2Fdraft-v1) argues that the bitter lesson holds at short horizons but breaks at long ones. This experiment provides the empirical backbone:

- **Phase 1** fills in the harness spread compression data at 3 model scales
- **Phase 1** tests whether SERA finetuning (compute) beats harness engineering (human knowledge) — Sutton's exact framing
- **Phase 2** demonstrates the practical architecture for short-horizon agent swarms where the bitter lesson has converged: minimal harness, strong model, parallel execution
- **Phase 2 before/after** shows one iteration of the leapfrog cycle: model improves → harness tricks become overhead

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
