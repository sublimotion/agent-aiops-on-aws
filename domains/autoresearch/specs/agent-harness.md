# Autoresearch Spec: Agent Harness

## Status: COMPLETE (Phase 1 + Phase 2 measured)

## Overview
Systematic investigation of how agent scaffolding affects coding task performance. Three phases: (1) turn degradation analysis, (2) multi-harness comparison, (3) model finetuning (future, out of scope for execution).

Based on the "harness problem" insight: a single tool design change can produce 10x gains (hashline example: Grok 6.7% → 68.3%), often exceeding the impact of switching frontier models. The harness is a universal multiplier.

## Phases

### Phase 1: Turn Degradation Analysis
**Goal**: Determine how turn count affects fix quality — are late turns helping or actively hurting?

**Observation**: Baseline averages 29.6 turns out of 30 budget. 82% of issues generate fixes but only 17.7% pass tests. Longer runs show sharper quality degradation. This phase isolates whether the model spirals after a threshold.

**Experiment matrix**:

| Config | Turn budget | Hypothesis |
|--------|------------|------------|
| A | 10 turns | Force early commitment, avoid spiraling |
| B | 15 turns | Tighter budget, still room to iterate |
| C | 20 turns | Potential sweet spot before degradation |
| D | 30 turns (baseline) | Current config for comparison |
| E | 15 turns + fresh restart at 15 | Reset context to fight degradation |
| F | 30 turns + context compaction at turn 15 | Keep budget, fight context pollution |

**Per-turn metrics to track**:
- Fix quality over time (does edit correctness decline in later turns?)
- Context pollution (what fraction of context window is wasted on failed attempts?)
- Repetition rate (is the model re-trying the same failing approach?)
- Turn of first correct fix (if the model found the right fix, when?)

**Expected outcome**: Identify the turn threshold where quality degrades, and whether restart/compaction strategies recover performance.

### Phase 2: Multi-Harness Comparison
**Goal**: Determine which scaffolding architecture extracts the most capability from a fixed model.

**Design**: Fix the model (Devstral Small 2 FP8), fix the benchmark (SWE-bench Lite 50-issue subset), vary only the harness. Produces a harness leaderboard for a specific model.

**Candidate harnesses**:

| Harness | Architecture | Key differentiator |
|---------|-------------|-------------------|
| SERA (baseline) | Custom Python agent loop | Current scaffolding |
| OpenHands | CodeAct — Jupyter + bash sandbox | Rich execution environment |
| SWE-agent | ACI (Agent-Computer Interface) | Purpose-built for SWE-bench |
| Aider | Edit-focused, repo-map + architect mode | Structural code understanding |
| Claude Code | CLI agent, auto-compaction, rich tool set | Production-grade context management |
| OpenCode | Minimal CLI agent | Lightweight, similar paradigm to Claude Code |
| LangGraph ReAct | ReAct agent with tool use | Standard agent framework |

**Per-harness metrics**:
- SWE-bench Lite pass rate (same 50-issue subset across all harnesses)
- Avg turns to resolution
- Turn degradation curve (from Phase 1 methodology)
- Context efficiency (tokens consumed per successful fix)

**Expected outcome**: Rank harness architectures by effectiveness, identify which scaffolding patterns matter most (tool design, context management, turn strategy, execution environment).

### Phase 3: Model Finetuning (Future — NOT executing)
**Goal**: After identifying the optimal harness (Phases 1-2), finetune the model specifically for that harness's tool format, turn structure, and failure patterns.

**Rationale for deferring**: The 82% fix generation rate with 17.7% pass rate indicates a scaffolding problem, not a weights problem. Finetuning is expensive and slow to iterate. The right time is after harness optimization plateaus.

**When to revisit**:
- Phases 1-2 have converged (harness improvements plateau)
- A consistent class of model errors is identified (e.g., wrong import patterns, bad test commands)
- A small LoRA targeting those patterns could compound with harness improvements

**Potential approach**:
- Collect successful turn traces from Phase 2 as training data
- LoRA finetune on (harness-specific tool format, successful fix trajectories)
- Evaluate whether finetuned model + optimal harness exceeds both individually

## Components

### 1. Compute
- **Platform**: Bare metal GPU instance (SSH)
- **Instance Type**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each)
- **Model serving**: 4x vLLM replicas (Devstral Small 2 24B FP8), round-robin load balancer on port 9000
- **Eval runner**: Python agent loop on same instance

### 2. Codebase
- **Source**: SERA scripts from `devstral-sera` blueprint (`/mnt/nvme/sera-scripts/`)
- **Fixed files** (agent must NOT edit):
  - SWE-bench Lite issue definitions (300 issues)
  - Test harness runner (pytest execution, patch validation)
  - Evaluation metric (`tests_pass` + `recall` threshold)
  - vLLM serving config and model weights
- **Agent-editable files** (Phase 1):
  - `scripts/harness_eval.py` — instrumented agent loop with per-turn metrics, configurable turn budgets, restart/compaction strategies
  - `scripts/setup_vllm.sh` — vLLM serving startup (4 replicas + load balancer)
- **External harnesses** (Phase 2):
  - Claude Code, OpenHands, SWE-agent, Aider, OpenCode, LangGraph — each with adapter scripts in `scripts/adapters/`
  - `scripts/multi_harness_eval.py` — orchestrator that runs each harness against the same 50-issue subset with consistent logging
- **Agent instructions**:
  - `program.md` — autoresearch loop protocol for harness optimization

### 3. Experiment Protocol
- **Metric**: SWE-bench Lite pass rate (tests pass + recall >= 0.8)
- **Eval subset**: 50 issues per experiment (sampled for diversity across repos, same subset across all harnesses)
- **Phase 1 time budget**: ~30 min per config (50 issues × variable turn budget × ~5s/turn), 6 configs = ~3 hours
- **Phase 2 time budget**: ~1 hour per harness (setup + 50-issue eval), 7 harnesses = ~7 hours
- **Loop structure**:
  - Phase 1: Run each turn-budget config on the 50-issue subset, record per-turn metrics
  - Phase 2: Run each harness on the same 50-issue subset, record pass rate + turn curves
- **Termination**: Phase 1 completes when all 6 configs run. Phase 2 completes when all harnesses run.
- **Logging**: Each experiment logs to `experiments.jsonl`: phase, config/harness name, turn budget, issues attempted, tests passed, SVG accepted, pass rate, per-turn breakdown

### 4. Networking
- **Access**: SSH to g7e instance
- **Model serving**:
  - Phase 1: localhost:9000 (round-robin across 4 vLLM replicas) — local serving for fine-grained per-turn control
  - Phase 2: Amazon Bedrock OpenAI-compatible endpoint — simplifies harness setup, no local GPU dependency. All harnesses point at the same Bedrock base URL via `OPENAI_BASE_URL` / `OPENAI_API_KEY` (SigV4 auth or Bedrock access key). Falls back to local vLLM if Bedrock model unavailable or rate-limited.
- **Bedrock considerations**: Check model availability (Devstral Small 2 or substitute), request quota increase before Phase 2 to avoid throttling across 7 harnesses

### 5. Storage
- **Model weights**: `/mnt/nvme/models/devstral-small-2-fp8` (49 GB)
- **SWE-bench repos**: `/mnt/nvme/sera-workspaces/`
- **Results**: `experiments.jsonl` in blueprint results directory

## Baseline

From SERA Phase 1 (`devstral-sera/lessons.md`):
- **17.7% pass rate** on SWE-bench Lite (300 issues, 53 tests pass, 28 SVG accepted)
- **82% fix generation rate** (246/300 issues got a fix)
- **29.6 avg turns** (nearly all issues exhaust 30-turn budget)
- **Django dominance**: 26/28 accepted examples are Django (dep install issues on other repos)

## Optimization Categories

Within each phase, the agent should explore these harness dimensions:

1. **Turn strategy** (Phase 1 focus) — turn budget, early termination, restart-with-summary, context compaction triggers
2. **System prompt** — instruction clarity, step-by-step debugging guidance, repo-specific hints
3. **Tool design** — edit granularity (line-level vs block-level), output truncation, error formatting
4. **Context management** — what to keep/drop across turns, file content summarization, compaction strategies
5. **Temperature/sampling** — per-turn temperature scheduling (creative exploration early, precise edits late)
6. **Repo adaptation** — detecting repo type (Django, pytest, etc.) and loading repo-specific instructions
7. **Scaffolding architecture** (Phase 2 focus) — execution environment (sandbox vs bare), agent paradigm (ReAct vs CodeAct vs edit-focused), structural code understanding (repo-map, AST)

## Success Criteria

### Phase 1: Turn Degradation
1. All 6 turn-budget configs complete on the 50-issue subset
2. Clear identification of the turn threshold where quality degrades
3. At least one config (restart or compaction) recovers performance vs naive long runs
4. Per-turn metrics logged for degradation curve analysis

### Phase 2: Multi-Harness Comparison
1. At least 4 harnesses successfully run against the same 50-issue subset
2. Harness leaderboard with pass rates, turn curves, and context efficiency
3. Identification of which scaffolding patterns matter most (tool design, context mgmt, execution env)
4. At least one harness exceeds the 17.7% SERA baseline

### Phase 3: Finetuning (future, not executing)
- Documented plan for when to revisit, based on Phase 1-2 outcomes
- Identified error classes that could benefit from weight updates

## Non-Requirements
- Changing the model (Devstral Small 2 FP8 is fixed for Phases 1-2)
- Multi-node distributed evaluation
- Full SWE-bench Verified (500 issues) — use 50-issue subset for iteration speed
- Executing Phase 3 (finetuning) — documented for future reference only
- Cost optimization — serving is already running

## Known Limitations
- 50-issue eval subset introduces sampling variance (~5% noise)
- Some improvements may be repo-specific (Django vs pytest) and not generalize
- Turn budget is the primary bottleneck — 82% of issues generate fixes but only 17.7% pass tests
- Dependency installation failures for non-Django repos limit eval diversity
- vLLM Mistral parser bug (#23180) breaks multi-turn tool-call IDs — affects harness evaluation
- Phase 2 harnesses may have different OpenAI-compatible API expectations — adapter scripts needed
- Some harnesses (OpenHands, SWE-agent) have their own sandbox requirements that may conflict with bare-metal setup
- Bedrock rate limits may throttle Phase 2 — request quota increase in advance. Local vLLM fallback available.
- Bedrock per-token cost for Phase 2 (~350 issues × ~30 turns × ~4K tokens/turn = ~42M tokens) — estimate before running

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
