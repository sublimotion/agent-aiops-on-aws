# Autoresearch Spec: Pivot Point Analysis

## Status: DRAFT

## Overview

Identify high-variance decision points ("pivots") in agent trajectories from the verification primitives experiment. Quantify which moments — when to stop exploring, when to verify, when to submit — have the highest impact on final outcomes. This converts the Parkinson's Law observation from descriptive to prescriptive: instead of "agents waste 55% of budget exploring," we get "intervening at turn X has Y% impact on pass rate."

**Core hypothesis**: The verification decision point (invoke tools or skip) and the exploration-to-implementation transition are the two highest-variance pivots. The two-stage checkpoint (40% edit nudge, 55% verify nudge) already targets these, but pivot analysis will confirm or challenge the placement.

**Motivation**: PivotRL (arXiv:2603.21383) demonstrated that focused RL on just the high-variance intermediate turns achieves competitive accuracy with end-to-end RL at 4x fewer rollout turns. Even without RL infrastructure, identifying pivots tells us: (a) are our checkpoint positions optimal, (b) which decisions should a future RL policy focus on, (c) can we build an early-stopping rule from behavioral signals at pivot points.

**Depends on**: verification-primitives (300-issue trajectory data with telemetry), verification-primitives-swebench (Docker eval gold labels)

## Components

### 1. Compute
- **Platform**: Local laptop or EC2 — Python analysis
- **Instance Type**: Any (CPU-only)
- **GPUs**: None required

### 2. Codebase
- **Source**: New analysis scripts in blueprint directory
- **Fixed files**:
  - Trajectory telemetry: `blueprints/verification-primitives/results/telemetry/` (251 JSONL tool telemetry files + 300 Claude Code logs)
  - Gold labels: `blueprints/verification-primitives-swebench/results/eval_report.json` + `eval_report_errors_v2.json`
  - Predictions: `blueprints/verification-primitives-swebench/results/predictions_lite.jsonl`
- **Agent-editable files**:
  - `blueprints/pivot-analysis/scripts/extract_pivots.py`
  - `blueprints/pivot-analysis/scripts/analyze_variance.py`
- **Agent instructions**: N/A

### 3. Experiment Protocol
- **Metric**: Per-pivot outcome variance (higher = more impactful pivot)
- **Secondary**: Mutual information between pivot choice and gold outcome, pivot timing distribution
- **Time budget**: 2-3 hours (data parsing + analysis + visualization)
- **Loop structure**: Single-pass analysis
- **Termination**: Pivot ranking computed and validated
- **Logging**: `blueprints/pivot-analysis/results/pivot_report.md`

### 4. Networking
- **Access**: Local or SSH to EC2 for data access

### 5. Storage
- **Data**: Existing telemetry from verification-primitives experiment
- **Results**: `blueprints/pivot-analysis/results/`

## Analysis Design

### Step 1: Parse Trajectories into Decision Sequences

For each of the 300 issues, extract a decision sequence from the Claude Code logs:

```
[EXPLORE, EXPLORE, READ, EXPLORE, EDIT, EDIT, TOOL:generate_tests, TOOL:run_tests, EDIT, TOOL:adversarial_review, SUBMIT]
```

Key decision types to classify:
- **Explore→Implement**: First file edit (the Parkinson's transition)
- **Implement→Verify**: First verification tool invocation
- **Verify→Submit**: Decision to stop verifying and submit
- **Verify→Revise**: Decision to fix based on verification feedback
- **Skip-Verify→Submit**: Submit without any verification tool use

### Step 2: Compute Per-Pivot Outcome Variance

For each decision type, compute:

1. **Conditional pass rates**: P(gold_pass | chose_action_A) vs P(gold_pass | chose_action_B)
   - E.g., P(pass | used_tools) = 0.695 vs P(pass | skipped_tools) = 0.188 (already known)
   - But also: P(pass | early_first_edit) vs P(pass | late_first_edit)
   - P(pass | ran_adversarial_review) vs P(pass | only_ran_tests)
   - P(pass | revised_after_test_failure) vs P(pass | submitted_despite_failure)

2. **Outcome variance at each turn**: For turn T, what fraction of trajectories that reached turn T in the same state diverged to different outcomes? Higher divergence = higher-variance pivot.

3. **Information gain**: Mutual information I(pivot_choice; gold_outcome) for each pivot type.

### Step 3: Timing Analysis

For each pivot type:
- Distribution of when it occurs (turn number as % of budget)
- Correlation between pivot timing and outcome
- Does our 40%/55% checkpoint align with the natural pivot timing?

### Step 4: Composition Pattern Analysis

Cross-reference with the 5 emergent composition patterns from the primitives experiment:
- `ignore` (no tools)
- `generate_run` (generate + run tests)
- `gen_run_iterate` (generate + run + fix + re-run)
- `full_pipeline` (generate + run + adversarial review)
- `full_pipeline_iterate` (full pipeline + iteration)

Which pattern transitions (e.g., starting with `generate_run` then escalating to `full_pipeline`) correlate with success?

### Step 5: Early-Stopping Rule Derivation

If pivot analysis shows strong signal:
1. Define early-stopping criteria: e.g., "if by turn 18 (60% of budget) no edit has been made AND no tool has been invoked, abort"
2. Backtest on 300 trajectories: how many doomed trajectories would this catch?
3. Estimate compute savings: (aborted turns saved) / (total turns consumed)
4. Estimate false-abort rate: how many successful trajectories would we incorrectly terminate?

## Success Criteria

1. Pivot types ranked by outcome variance with statistical significance (Fisher exact or chi-squared)
2. Top 2-3 pivots identified with effect sizes
3. Checkpoint placement validated or adjusted: does 40%/55% align with empirical pivot timing?
4. Early-stopping rule proposed with backtested precision/recall on the 300-issue dataset
5. Visualization: pivot timing heatmap showing where high-variance decisions cluster in the turn budget

## Non-Requirements
- RL training or policy optimization
- New data collection or API calls
- Causal inference (observational analysis only — confounders acknowledged)
- Real-time pivot detection (offline analysis)

## Known Limitations
- **Observational, not causal**: Agents that chose to verify may differ from non-verifiers in ways beyond the verification decision (e.g., problem difficulty, agent "confidence"). The within-run comparison (tool users vs non-users) has this confounder.
- **Single checkpoint configuration**: All 300 trajectories used the same two-stage checkpoint. We can't compare different checkpoint positions from this data alone.
- **Claude Code log parsing**: Logs may not have perfectly structured turn-by-turn data. May need heuristic parsing.
- **n=300 limits subgroup analysis**: Splitting by composition pattern × pivot type × outcome may produce cells with < 10 instances.

## Relationship to Other Specs

- **verification-primitives**: Source of trajectory telemetry and composition pattern analysis
- **verification-primitives-swebench**: Source of gold labels for outcome measurement
- **verifier-reward**: v009 adversarial rubric provides the adversarial_review tool behavior data
- Future: Results feed into E7 (PivotRL) if RL infrastructure becomes available

## Key References

- PivotRL (arXiv:2603.21383) — focused RL on high-variance decision points, 4x compute savings
- AgentPRM (arXiv:2511.08325) — process rewards for agents based on "promise" and "progress"
- HISR (arXiv:2603.18683) — segment-level process rewards modulated by hindsight information
- CoderForge (arXiv:2503.01207) — early test fraction as strongest behavioral predictor of success

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
