# Autoresearch Spec: Pre-Loop Micro-Experiments

## Status: COMPLETE (2026-05-09)

All three experiments ran in a single ralph-loop session. See `blueprints/self-coding-agent-loop/pre-loop-micros-results.md` for consolidated results and `blueprints/self-coding-agent-loop/pre-loop-micros-progress.md` for the execution log.

**Outcomes**:
- E_env: `proceed_single_rf` (Δ AUC = -0.144) — no per-pipeline ensemble needed.
- E_attr: bake `v009_fail AND rf_pass` cell signal into Loop 1 drift monitoring (34.9pp gold-rate gap).
- E_constraint_agent: `negative_result` (P@R≥0.30 = 0.286) — v009 remains the only V1b candidate.

## Context

Three pre-loop micro-experiments that run before the full self-coding agent loop (`self-coding-agent-loop.md`) launches. Motivated by Eigen AI's "Reliable Post-Training for Interactive Tool-Using Agents" (arXiv 2601.22607) and EigenData findings, which provide empirical evidence that **environment variance reduction is the dominant single lever in agentic RL training** (their Telecom +20pp ablation from user-model fine-tuning alone).

Two relevant pieces of prior evidence sit in our own data:

- **Phase 3 (single-pipeline)**: On `claude_opencode_300` (300 Claude Sonnet 4.6 × OpenCode trajectories) the behavioral RF achieves AUC 0.730 and the 4-feature selected RF achieves AUC 0.756 (`learned-verifier/docs/phase3_report.md`). This is a within-pipeline ceiling — no pipeline heterogeneity is present.
- **E6 (cross-pipeline)**: When the same RF architecture is trained on pooled Claude/OpenHands/Nebius trajectories a per-model (family-routed) ensemble reaches AUC 0.801, while a single universal RF transfers poorly (`learned-verifier/docs/e6_cross_model_report.md`). This is the cross-pipeline signal.

Neither result directly tells us what happens to a single RF trained on the *pooled* mixed-pipeline set without per-model routing — which is the baseline the self-coding loop would actually deploy. These micros isolate that signal before committing to a loop design that assumes a single mixed-pipeline verifier is optimal.

## Overview

Three experiments, two pure-analysis and one pilot, total budget ~$50, duration 1.5 weeks. Results feed directly into the `self-coding-agent-loop.md` V1/V1b gate design and Loop 1 architecture.

| ID | Name | Cost | Duration | Type |
|---|---|---|---|---|
| E_env | Environment variance reduction test | $0 | 2 days | Analysis on pooled Claude/OpenHands/Nebius features |
| E_attr | Verifier-disagreement attribution | $0 | 3 days | Analysis on Phase 3 Claude×OpenCode data |
| E_constraint_agent | Per-instance constraint verification on agent patches | ~$25 | 4 days | Pilot on n=50 Qwen3.5×SERA agent patches (agent-swarm results) |

**Total:** ~$25-$50, ~1.5 weeks wall-clock.

## Why Pre-Loop, Not During

The `self-coding-agent-loop.md` spec gates Arms C/D/E on V1b (verifier precision ≥ 0.70 on Qwen3.5 traces). If V1b fails, the spec says: *"Loop 1 must recalibrate first. Block C/D/E."* What does "Loop 1 recalibration" look like? The spec doesn't say — you'd be designing the recovery path mid-loop with partial data.

These micros give Loop 1 recalibration a pre-built toolkit:
- **E_env** — tells you whether the verifier's real problem is trace heterogeneity vs. feature deficiency
- **E_attr** — gives drift monitoring an attribution layer (which component failed)
- **E_constraint_agent** — provides a second verifier candidate if v009 fails V1b

Cost asymmetry: ~$50 for the micros vs. ~$330-$570 per failed loop arm. Even 10% probability of gate-relevance makes the micros high-ROI.

## Components

### 1. Compute

- **E_env**: Local analysis on the pooled cross-pipeline feature CSVs (`features/combined_features.csv` Claude×OpenCode + `features/e6_openhands_features.csv` + `features/e6_nebius_features.csv`). No GPU.
- **E_attr**: Local analysis on the Phase 3 single-pipeline CSV + E9 disagreement data.
- **E_constraint_agent**: Bedrock Haiku API for constraint extraction + agent-patch evaluation. No GPU.

### 2. Codebase

- **Repo**: `/Users/phi/Documents/workbench/learned-verifier/`
- **Scripts** (new):
  - `experiments/e_env_environment_variance.py`
  - `experiments/e_attr_verifier_disagreement.py`
  - `experiments/e_constraint_agent_pilot.py`
- **Shared data loader**: `experiments/_shared_loader.py` (refactor if duplication emerges)
- **Backlog entry**: append to `experiments/backlog.md`

### 3. Experiment Protocol

Each experiment has its own protocol section below. Shared principles:

- **Fail-fast on analysis.** If E_env shows no single-pipeline AUC improvement, skip to E_attr immediately. Don't iterate on E_env.
- **Each experiment has one decision rule.** No multi-factor analysis paralysis.
- **Results feed `self-coding-agent-loop.md`** as updates to V1/V1b gate design.

### 4. Networking

- Bedrock API (for E_constraint_agent Haiku calls) — existing AWS account
- No external compute

### 5. Storage

- **Inputs**:
  - `learned-verifier/data/features/combined_features.csv` (Phase 3 Claude×OpenCode, n=300)
  - `learned-verifier/data/features/e6_openhands_features.csv` (OpenHands multi-model, n≈2,099)
  - `learned-verifier/data/features/e6_nebius_features.csv` (SWE-agent, n≈67,074)
  - `learned-verifier/data/features/phase3_results.json` (Phase 3 RF baselines)
  - `learned-verifier/data/features/e9/` (E9 disagreement data for E_attr)
  - `agent-swarm/results/swarm_phase1_qwen35-397b_sera.jsonl` (Qwen3.5×SERA traces, n=50, for E_constraint_agent)
  - `agent-swarm/results/eval_qwen25-coder-32b_sera.jsonl` (fallback if Qwen3.5×SERA patches are incomplete)
- **Outputs**:
  - `learned-verifier/docs/e_env_report.md`
  - `learned-verifier/docs/e_attr_report.md`
  - `learned-verifier/docs/e_constraint_agent_report.md`
  - Updated `experiments/backlog.md`
  - **Cross-reference**: `agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop/pre-loop-micros-results.md` (operational artifact — created after runs complete)

---

## E_env: Environment Variance Reduction Test

### Hypothesis

A single RF trained on the pooled cross-pipeline set (Claude×OpenCode + OpenHands-multi-model + SWE-agent/Nebius) is bounded by trajectory-source heterogeneity rather than sample size or feature count. Holding (agent_model, scaffold) constant — i.e. training a per-cell RF — will raise the sample-weighted mean AUC meaningfully above the pooled single-RF baseline, reproducing the per-family routing benefit E6 observed (AUC 0.801 routed vs. ~0.72 pooled) with the exact features available to Loop 1.

### Motivation (external)

Eigen AI's ablation: naive RL on Telecom = 75.6%. With user-model fine-tuning (reducing simulator variance) = 95.6%. **A +20pp gain from environment-variance reduction alone.**

### Design

1. Build a pooled feature frame by stacking `combined_features.csv` (Claude×OpenCode, n=300), `e6_openhands_features.csv` (OpenHands multi-model), and `e6_nebius_features.csv` (SWE-agent). Inner-join to the feature subset present in all three (behavioral cost/turns/loop-count/tokens + gold_pass). Stamp each row with a `(agent_model, scaffold)` identifier derived from its source dataset plus OpenHands per-row model metadata.
2. Fit a **baseline pooled RF** on the union using the same hyperparameters as Phase 3's 4-feature `selected_4` model. Record pooled AUC, P@R≥30%, ECE — this replaces the previously cited "0.727 mixed-pipeline" figure, which was actually a within-pipeline number.
3. For each `(agent_model, scaffold)` cell with n ≥ 50, fit a per-cell RF independently with the same hyperparameters.
4. Compute per-cell AUC, P@R≥30%, ECE, and **sample-size-weighted mean per-cell AUC**.
5. Compare sample-weighted per-cell AUC vs. baseline pooled AUC from step 2.

### Metrics

- Primary: Δ(sample-weighted per-cell AUC) vs pooled baseline AUC (measured fresh in step 2, not a hardcoded number)
- Secondary: Δ(sample-weighted per-cell ECE) vs pooled baseline ECE
- Diagnostic: feature-importance stability across cells (are the top 4 features stable or do they vary per cell?)
- Sanity check: pooled AUC in step 2 should fall between 0.66 (Nebius-only, reported in registry) and 0.80 (E6 routed ensemble), giving the true baseline Loop 1 would deploy.

### Decision Rule

- **If Δ AUC > 0.05**: Redesign Loop 1 as a per-trace-source verifier ensemble rather than a single RF. Update `self-coding-agent-loop.md` V1 gate to specify which pipeline(s) the verifier must transfer to.
- **If Δ AUC ≤ 0.05**: Current single-RF design is fine. Proceed with V1b as written.

### Cost Estimate

- $0 — pure analysis on existing data.

### Duration

- 2 days (1 day implementation, 1 day analysis + writeup).

### Output

- `docs/e_env_report.md` with per-cell AUC table, feature-importance stability matrix, and decision rule verdict.
- Backlog entry updated to `DONE`.

### Connection to Loop Spec

Updates `self-coding-agent-loop.md` §V1 (Verifier transfers to SWE-ReBench) and §Loop 1 (Reward Model Calibration) architecture section.

---

## E_attr: Verifier-Disagreement Attribution

### Hypothesis

When v009 (rubric) and behavioral RF disagree on trajectory outcome, the *pattern* of disagreement is informative about failure mode. Specifically:

- **v009-fail + RF-pass** → likely adversarial patch (BenchJack-style exploit, wrong-but-looks-right)
- **v009-pass + RF-fail** → likely Simpson's Paradox (hard task, flailing agent)
- **v009-fail + RF-fail** → clear failure (catch both, high confidence reject)
- **v009-pass + RF-pass** → clear success (high confidence accept)

### Motivation (external)

EigenData's three-agent attribution architecture (DatabaseAgent vs CodingAgent vs DataAgent) is a verifier cascade applied to training-data quality. When a trajectory fails, blame is attributed to specific sources. This is load-bearing for EigenData's self-evolving property.

Our Phase 3 has no attribution layer. When behavioral RF misclassifies a trajectory, we don't know whether the label is wrong (outcome metric failure), the trajectory is ambiguous (scaffold noise), or the features are model-specific (E6 cross-model transfer).

### Design

1. For n=300 Phase 3 trajectories, tabulate the 2x2 confusion matrix of (v009_verdict, rf_verdict) against gold labels.
2. Within each disagreement cell, measure:
   - Distribution of gold-label outcomes
   - Average behavioral-feature profile (cost, loop_count, edit_fraction, etc.)
   - Whether Phase 3 already categorized these as exploits or over-edits (if existing labels available)
3. Cross-reference with known failure modes:
   - BenchJack-style exploits (conftest.py manipulation)
   - Simpson's Paradox cases (high cost + high read:edit + fail)
   - Over-editing patterns (high edit distance + fail)
4. Test: does the disagreement pattern predict failure mode better than either verifier alone?

### Metrics

- Primary: Per-cell gold-label accuracy breakdown (what fraction of v009-fail + RF-pass disagreements are adversarial exploits?)
- Secondary: Feature-profile separation between disagreement cells (does the behavioral feature distribution differ by disagreement type?)
- Diagnostic: Sample trajectories from each cell for manual inspection (does the attribution make sense qualitatively?)

### Decision Rule

- **If disagreement cells show distinct failure-mode distributions** (e.g., v009-fail+RF-pass cell is ≥60% adversarial patches): Bake attribution into Loop 1's drift-monitoring design. Add `disagreement_pattern` as a signal in Phase 2 drift detection.
- **If disagreement cells are uniform noise** (no distinct distributions): Attribution doesn't help. Document the finding; skip attribution layer.

### Cost Estimate

- $0 — pure analysis.

### Duration

- 3 days (1 day data integration, 1 day analysis, 1 day writeup + sample inspection).

### Output

- `docs/e_attr_report.md` with 2x2 disagreement matrix, per-cell feature profiles, sample trajectories, and decision rule verdict.
- Backlog entry updated to `DONE`.
- If decision rule fires: updated `self-coding-agent-loop.md` §Loop 1 Phase 2 drift monitoring section.

### Connection to Loop Spec

Updates `self-coding-agent-loop.md` §Loop 1 drift monitoring design and Phase 2 recalibration trigger logic.

---

## E_constraint_agent: Per-Instance Constraint Verification on Agent Patches

### Hypothesis

Per-instance extracted behavioral constraints checked against **agent-generated patches** (not gold patches — distinct from completed E5) achieve recall ≥ 0.30 at precision ≥ 0.85, providing a viable v009 replacement for agent-patch evaluation.

### Motivation (external)

Eigen AI's EigenData uses per-instance executable verification functions. On τ²-bench Telecom, the combined system achieves 98.3% post-RL (up from 53.7%). Per-instance verifiers are substantially stronger than generic rubrics for agentic tasks.

Our completed E5 ran constraint verification on **gold patches** — a ceiling analysis showing potential, but not tested as a practical verifier on agent output. V1b gate in `self-coding-agent-loop.md` requires v009 precision ≥ 0.70 on Qwen3.5 agent traces. If v009 fails, we need a pre-tested alternative.

### Distinction from E5

- **E5 (DONE)**: Gold patches from SWE-bench, ceiling analysis. Result: AUC 0.624 standalone, 50% RF false-negative recovery. Ceiling only.
- **E_constraint_agent (this experiment)**: Agent-generated patches from Qwen3.5 + SERA pipeline. Tests real-world transfer.

### Design

1. Use n=50 from the agent-swarm Qwen3.5-397B × SERA run at `agent-aiops-on-aws/domains/autoresearch/blueprints/agent-swarm/results/swarm_phase1_qwen35-397b_sera.jsonl` (Phase 3 itself does not contain Qwen3.5 or SERA — it is Claude×OpenCode only). Each record carries the agent patch and the gold label. If more than 8-10 records have empty patches, top up from `eval_qwen25-coder-32b_sera.jsonl` to keep n=50.
2. For each instance, extract 3-5 behavioral constraints from the issue description + failing test (Haiku via Bedrock, ~$0.05 per extraction).
3. For each agent patch, evaluate constraint satisfaction:
   - Deterministic constraints (e.g., "function returns None on empty input"): regex / AST checks
   - Semantic constraints (e.g., "error message must reference file path"): Haiku judge
4. Score = weighted fraction of constraints satisfied.
5. Compare against gold labels:
   - Precision at recall = 0.30
   - AUC vs v009 alone on same n=50
   - ECE vs v009

### Metrics

- Primary: Precision at recall ≥ 0.30 (target > 0.85)
- Secondary: AUC on n=50 (target > v009's in-sample AUC)
- Diagnostic: Constraint satisfaction correlation with gold outcome (target r > 0.3)
- Operational: Per-instance extraction cost (must stay < $0.10 average)

### Decision Rule

- **If precision ≥ 0.85 at recall ≥ 0.30**: Add as backup verifier for V1b. If v009 fails V1b, use this as replacement in Arms C/D (or complement v009 via ensemble).
- **If precision ≥ 0.80 but recall < 0.30**: Use as v009 ensemble partner (agreement = accept), not standalone replacement.
- **If precision < 0.80**: Document negative result. v009 is the only candidate for V1b; if v009 fails, Loop 1 recalibration is the only path.

### Cost Estimate

- Haiku extraction: 50 × $0.05 = $2.50
- Haiku semantic evaluation: 50 patches × 5 constraints × $0.03 = $7.50
- Buffer for rework: $15
- **Total: ~$25**

### Duration

- 4 days (1 day constraint-extraction prompt design, 1 day extraction run, 1 day evaluation run, 1 day analysis + writeup).

### Output

- `docs/e_constraint_agent_report.md` with n=50 confusion matrix, precision-recall curve, per-instance cost breakdown, decision rule verdict.
- Backlog entry updated to `DONE`.
- If decision rule fires positive: v009 ensemble design added to `self-coding-agent-loop.md` V1b gate logic.

### Connection to Loop Spec

Updates `self-coding-agent-loop.md` §V1b gate and Arms C/D verifier specification. If this experiment shows positive transfer, V1b passes automatically with the ensemble rather than requiring Loop 1 recalibration.

---

## Execution Order

Strict sequence — later experiments depend on earlier ones being green:

```
Day 1-2:   E_env (analysis)
             │
             ├─ Δ AUC > 0.05 → update Loop 1 to per-pipeline ensemble
             └─ Δ AUC ≤ 0.05 → proceed

Day 3-5:   E_attr (analysis, in parallel with E_constraint_agent design)
             │
             ├─ Disagreement patterns distinct → bake into drift monitoring
             └─ Uniform noise → document, skip attribution

Day 5-7:   E_constraint_agent extraction + evaluation (in parallel with E_attr writeup)

Day 8:     Consolidated writeup in blueprints/self-coding-agent-loop/pre-loop-micros-results.md
            Update self-coding-agent-loop.md V1/V1b gate logic based on all three results

Day 9-10:  Begin V1b validation (from original spec) with updated gate design
```

Total: ~1.5 weeks wall-clock.

## Success Criteria

- **E_env** completes analysis and produces a decision on Loop 1 architecture (single-RF vs per-pipeline ensemble).
- **E_attr** completes analysis and produces a decision on whether to include disagreement attribution in drift monitoring.
- **E_constraint_agent** completes n=50 pilot and produces a decision on whether constraint verification is a viable v009 replacement/ensemble partner.
- **Consolidated results** flow back into `self-coding-agent-loop.md` V1/V1b gates, Loop 1 architecture section, and Phase 2 drift monitoring design.
- **Total cost stays under $50.**
- **Total duration stays under 2 weeks.** If E_constraint_agent extraction is slow, defer the pilot and proceed with E_env/E_attr only.

## Non-Requirements

- **Statistical power at n=50.** E_constraint_agent is explicitly a pilot. If it shows promise, a full n=300 run is a follow-up experiment.
- **Proposing the routed verifier architecture.** E_env asks whether pooled-vs-routed matters in the features Loop 1 would actually use; it does not specify the routing implementation. That's a follow-up if E_env fires.
- **New architectural claims.** These micros test whether existing verifier architecture is the bottleneck; they don't propose new architectures.
- **Updating other loop specs.** Only `self-coding-agent-loop.md` is updated. Other specs (verifier-reward, pivot-analysis, etc.) are untouched unless micros surface something unexpected.

## Known Limitations

- **Phase 3 data is SWE-bench Verified, not SWE-ReBench V2.** Insights may not transfer directly. The transfer we care about is *methodological* (environment variance matters, attribution helps drift monitoring) — not specific AUC numbers. Verified data suffices for this.
- **n=50 is small for E_constraint_agent.** A single run may not be decisive. If precision hovers around 0.80 ± 0.05, the decision is ambiguous. Plan for a follow-up n=150 run if the pilot is borderline.
- **Feature-column intersection is lossy.** The pooled feature frame for E_env is the intersection of columns present in `combined_features.csv`, `e6_openhands_features.csv`, and `e6_nebius_features.csv`. The OpenHands and Nebius CSVs carry a narrower behavioral feature set than Phase 3, so the pooled RF will likely run on ~5-8 features rather than the full 30+. That's still informative for the pooled-vs-routed comparison, but the absolute AUC is not comparable to Phase 3's `selected_4` AUC of 0.756.
- **Cell count is small.** Expect 3-4 (agent_model, scaffold) cells meeting n ≥ 50: Claude×OpenCode, SWE-agent (Nebius, treated as one cell), and 1-2 OpenHands model cells. If fewer than 3 cells meet the threshold, fall back to (scaffold-only) partitioning.
- **v009 agreement with gold labels on Qwen3.5 traces may be too low to define "failure mode" reliably.** E_attr assumes gold labels are the ground truth; BenchJack work suggests they're not always.

## References

- Eigen AI "Reliable Post-Training for Interactive Tool-Using Agents" (arXiv 2601.22607)
- EigenData blog (self-evolving function-calling data): https://www.eigenai.com/blog/self-evolving-llm-function-calling-data-eigendata
- Phase 3 report: `/Users/phi/Documents/workbench/learned-verifier/docs/phase3_report.md`
- E6 cross-model transfer report: `/Users/phi/Documents/workbench/learned-verifier/docs/e6_cross_model_report.md`
- E5 constraint verification report (gold patches): `/Users/phi/Documents/workbench/learned-verifier/docs/e5_constraint_report.md`
- Parent loop spec: `/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/autoresearch/specs/self-coding-agent-loop.md`
- Vault essay seed: `/Users/phi/Documents/obsidian-notes/01_Projects/Blog - PredictingTheNextToken/articles/trajectories-verifier-moat/SEED.md`

---

> **Note**: Operational artifacts (lessons learned, actual experiment results, analysis writeups)
> belong in the blueprint directory at `domains/autoresearch/blueprints/self-coding-agent-loop/pre-loop-micros-results.md`.
> This spec describes the *plan*; the blueprint holds the *outcomes*.
