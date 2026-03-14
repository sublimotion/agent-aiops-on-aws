# Autoresearch Spec: Agent Harness

## Status: DRAFT

## Overview
Apply the autoresearch loop to optimize a coding agent's harness — system prompts, tool definitions, turn strategy, and context management — targeting higher SWE-bench pass rates. The model weights are fixed; only the scaffolding around the model changes.

Based on the "harness problem" insight: a single tool design change can produce 10x gains (hashline example: Grok 6.7% → 68.3%), often exceeding the impact of switching frontier models. The harness is a universal multiplier.

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
- **Agent-editable files**:
  - `system_prompt.txt` — agent system instructions
  - `tool_definitions.py` — tool schemas (read_file, write_file, edit_file, run_command, etc.)
  - `agent_loop.py` — turn management, context truncation, retry logic
  - `config.yaml` — turn budget, temperature, tool formatting parameters
- **Agent instructions**:
  - `program.md` — autoresearch loop protocol for harness optimization

### 3. Experiment Protocol
- **Metric**: SWE-bench Lite pass rate (tests pass + recall >= 0.8)
- **Eval subset**: 50 issues per experiment (sampled for diversity across repos)
- **Time budget**: 30-60 minutes per experiment (50 issues × 30 turns × ~5s/turn)
- **Loop structure**: Measure baseline → hypothesize harness improvement → edit scaffolding → run 50-issue eval → compare pass rate → keep or revert
- **Termination**: Manual stop, or 20 experiments completed
- **Logging**: Each experiment logs to `experiments.jsonl`: experiment number, hypothesis, issues attempted, tests passed, SVG accepted, pass rate, delta from baseline

### 4. Networking
- **Access**: SSH to g7e instance
- **Model serving**: localhost:9000 (round-robin across 4 vLLM replicas)

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

The agent should explore these harness dimensions:

1. **System prompt** — instruction clarity, step-by-step debugging guidance, repo-specific hints
2. **Tool design** — edit granularity (line-level vs block-level), output truncation, error formatting
3. **Turn strategy** — when to pivot approach, backtracking signals, early termination on repeated failures
4. **Context management** — what to keep/drop across turns, file content summarization
5. **Temperature/sampling** — per-turn temperature scheduling (creative exploration early, precise edits late)
6. **Repo adaptation** — detecting repo type (Django, pytest, etc.) and loading repo-specific instructions

## Success Criteria

1. Autoresearch loop completes 10+ experiments without human intervention
2. At least one experiment exceeds the 17.7% baseline pass rate
3. Structured experiment log captures all hypotheses, configurations, and results
4. Lessons identify which harness dimensions have the highest leverage

## Non-Requirements
- Changing the model (Devstral Small 2 FP8 is fixed)
- Multi-node distributed evaluation
- Full SWE-bench Verified (500 issues) — use 50-issue subset for iteration speed
- Training/fine-tuning — this optimizes inference-time scaffolding only
- Cost optimization — serving is already running

## Known Limitations
- 50-issue eval subset introduces sampling variance (~5% noise)
- Some improvements may be repo-specific (Django vs pytest) and not generalize
- Turn budget is the primary bottleneck — 82% of issues generate fixes but only 17.7% pass tests
- Dependency installation failures for non-Django repos limit eval diversity
- vLLM Mistral parser bug (#23180) breaks multi-turn tool-call IDs — affects harness evaluation

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
