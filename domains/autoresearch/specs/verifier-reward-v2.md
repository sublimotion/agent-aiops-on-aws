# Learned Verifier: Cross-Model Transfer & Continuous Learning

**Status**: E6/E6b COMPLETE — results inform architecture decision
**Created**: 2026-03-19
**Updated**: 2026-04-08 (E6b OpenHands 7-model results, continuous learning framing)
**Depends on**: Phase 3 4-feature RF (COMPLETE, AUC=0.756)

## Context (Completed Phases)

The learned verifier pipeline completed Phases 0-3 plus E_new1/E_new2/E_new3 experiments. Key results:

- **4-feature RF**: `total_cost_usd`, `tokens_per_edit`, `svg_accepted`, `loop_count` — AUC=0.756, P@R>=30%=0.966
- **Forward selection on 99 features** picks the same 4 features every time
- **v009 adversarial rubric**: precision=0.92, $0.030/patch (model-agnostic by design)
- **Simpson's Paradox confirmed**: behavioral features reverse cross-sectionally vs longitudinally

Results: `domains/autoresearch/blueprints/learned-verifier/results/`

## E6 Results: Transfer Fails, Ensemble Wins

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Zero-shot transfer AUC (Claude RF → Qwen3.5) | **0.363** | > 0.65 | FAIL |
| Reverse transfer AUC (Qwen3.5 RF → Claude) | **0.410** | > 0.65 | FAIL |
| Per-family ensemble AUC | **0.801** | > 0.72 | PASS |
| Model-agnostic 3-feat AUC (Claude CV) | **0.738** | > 0.70 | PASS |
| svg_accepted ablation drop | -0.019 | — | Minimal |

**The features carry real signal, but decision boundaries are model-specific.** Claude's "this cost level means struggling" threshold doesn't map to Qwen3.5's cost distribution. Loop count means different things — Qwen uses 1.5x more turns (mean 22 vs 15) with 4x higher variance.

Transfer fails in both directions (bidirectional mean AUC=0.387), confirming this isn't a data size issue — the models genuinely occupy different regions of feature space.

**Per-model ensemble (AUC=0.801) exceeds the single-model baseline (0.756).** When each model family gets its own RF, the signal is cleaner than mixing families.

Script: `learned-verifier/scripts/e6_cross_model_transfer.py`
Full results: `learned-verifier/results/e6_cross_model_results.json`

## Implication: Continuous Learning Architecture

E6 rules out a universal static verifier. The production architecture must:

1. **Bootstrap per-model**: When a new model is onboarded, collect ~50-100 labeled traces and train a model-specific RF. The 3 behavioral features (`total_cost_usd`, `tokens_per_edit`, `loop_count`) are the right features — they just need model-specific thresholds.

2. **Fall back to v009 rubric**: Until enough traces exist for a new model family, the v009 adversarial rubric (precision=0.92, model-agnostic) handles cold-start verification. No RF needed.

3. **Route at inference**: `model_family` field determines which RF to use. The per-model ensemble (AUC=0.801) already demonstrates this works.

4. **Refresh on drift**: Model updates (new checkpoint, changed system prompt, harness changes) shift feature distributions. The RF needs periodic retraining on recent traces — not a one-time fit.

### Two-tier verification cascade

```
Tier 0: Per-model RF (fast, ~0ms, AUC 0.75-0.80)
  ↓ uncertain
Tier 1: v009 rubric (4 Haiku calls, $0.03, precision 0.92)
```

- New model with no RF → skip Tier 0, go straight to Tier 1
- Model with enough traces → Tier 0 filters obvious cases, Tier 1 handles edge cases
- RF confidence < threshold → escalate to Tier 1

### Data flywheel

Each verification creates a training example for the next RF iteration:
1. Agent generates patch → behavioral features captured automatically
2. v009 rubric provides label (or gold tests if available)
3. Labeled trace feeds back into model-specific RF training set
4. RF improves → handles more cases without rubric → lower cost per verification

This is continuous learning, not batch training. The verifier improves as it's used.

## RL Implication: RF as Per-Model Reward Accelerator

The RF is model-specific — but so is RL. Fine-tuning is inherently per-model (a Claude RL checkpoint can't be applied to Qwen). This means the RF's model-specificity is a feature, not a bug: a Claude RF provides reward signal for Claude RL, a Qwen RF for Qwen RL.

The RF's advantage over v009 as RL reward is **reward density**:

| Reward signal | Recall | Precision | Rollouts with signal | Cost |
|--------------|--------|-----------|---------------------|------|
| v009 rubric (4/4) | 0.14 | 0.92 | 14% — 86% wasted | $0.03/patch |
| **Per-model RF** | **0.30** | **0.966** | **30% — 2x denser** | **~0ms** |
| RF + v009 cascade | ~0.40+ | ~0.93 | ~40% | $0.03 for escalated |
| Gold tests | 1.00 | 1.00 | 100% | ~$0.50 (Docker) |

v009 alone leaves 86% of rollouts without confident signal. The RF doubles that coverage at higher precision and near-zero cost. The cascade (RF labels easy cases, v009 handles uncertain ones, gold tests for periodic calibration) gives the best density-to-cost ratio.

### RL bootstrap sequence

1. **Cold start** (new model, no RF): v009 rubric as reward. Sparse (14% coverage) but model-agnostic. Sufficient for initial RL exploration.
2. **After ~50-100 traces**: Train model-specific RF. Switch to RF as primary reward for easy cases, v009 for uncertain cases. Reward density jumps to ~40%.
3. **At scale**: RF handles majority of reward labeling. v009 spot-checks for calibration drift. Gold tests for periodic ground truth.

### Constraints

- **Cross-model RL is not viable.** A Claude RF cannot reward Qwen RL — the thresholds are wrong (AUC=0.363 cross-model). Each model family needs its own RF before RF-as-reward is available.
- **ECE must be monitored.** SWE-RM showed 7x ECE gap between verifiers caused RL collapse. The RF's ECE (0.083 on Claude) must be re-measured after each retraining cycle. If ECE drifts above 0.15, fall back to v009.
- **Reward hacking risk.** The RF rewards behavioral patterns (low cost, few loops), not semantic correctness. An RL policy could learn to produce cheap, short traces that look good to the RF but are wrong. Mitigation: v009 semantic checks on a random sample of RF-accepted rollouts.

### The cloud model gap

Our richest traces (n=300, Claude) come from a model we can't fine-tune. RL requires an open-weight base model — which means we need traces from that model to train its RF.

This is a **data acquisition problem, not a methodology problem.** The RF methodology is validated (features, architecture, AUC expectations). Each open-weight model needs its own labeled traces.

### Dataset survey: telemetry matters

We evaluated all public SWE-bench trajectory datasets for RF feature extraction:

| Dataset | Model | n | Real API telemetry | RF AUC |
|---------|-------|---|-------------------|--------|
| **OpenHands eval outputs** | 7 model families | 2,098 | **Yes** (4/7 models have per-call tokens) | **0.55-0.78** |
| **Nebius OpenHands RFT** | Qwen3-30B | 67K | **No** — HF export strips usage fields | 0.660 (char-count estimates) |
| **CoderForge** (Together AI) | Qwen3-Coder-480B | 258K | Unknown — trajectories not released | — |
| **SWE-Gym / SWE-smith** | Various | 400K+ | **No** — no API telemetry | — |
| **SERA datagen** | Any (self-hosted) | On demand | Yes (we control logging) | — |

**Only OpenHands/openhands-evaluation-outputs has real API telemetry** (`metrics.accumulated_cost`, `usage.prompt_tokens`, `usage.completion_tokens`, `tool_call_metadata.function_name`). Nebius was tested and gave AUC=0.660 with character-count token estimates — too coarse to be useful.

### E6b: OpenHands multi-model RF (7 models, 2,098 instances)

Script: `learned-verifier/scripts/e6_openhands_rf.py`
Results: `learned-verifier/results/e6_openhands_rf_results.json`

**Transfer gap at scale**: Self-model AUC = **0.984** (near-perfect), cross-model AUC = **0.532** (barely above random). Transfer gap = **0.452** — even larger than our Claude→Qwen experiment.

| Model | Standalone AUC | n | Pass rate | Real telemetry |
|-------|---------------|---|-----------|----------------|
| llama-70b | **0.785** | 300 | 10.7% | Yes (per-call tokens) |
| deepseek | **0.707** | 300 | 7.7% | Yes (per-call tokens) |
| gpt-4o-mini | 0.628 | 300 | 7.7% | No (cost-estimated) |
| claude-haiku | 0.628 | 299 | 28.8% | Yes (per-call tokens) |
| qwen-72b | 0.619 | 300 | 7.7% | No (cost-estimated) |
| claude-sonnet | 0.546 | 299 | 43.5% | Yes (per-call tokens) |
| gpt-4o | 0.434 | 300 | 21.3% | No (cost-estimated) |
| **claude-ours (Phase 3)** | **0.738** | 300 | 58.3% | Yes |

Notable: Our Phase 3 Claude RF (0.738) outperforms the OpenHands Claude-Sonnet RF (0.546) despite using the same model. The harness matters — SERA/OpenCode produce cleaner behavioral signals than CodeActAgent.

**Transfer matrix highlights**: Some model pairs share behavioral signatures (llama-70b→deepseek: 0.802, qwen-72b→deepseek: 0.712). GPT-4o-mini transfers worst (0.24-0.51 to others).

**RL reward density**: Most models get 0% coverage at P>=90% precision. Only DeepSeek achieves 8.7%. Low pass rates (7-10%) limit the RF's ability to learn discriminative boundaries.

**Ensemble**: Per-model ensemble AUC=0.735 vs single RF 0.687 (+0.048).

### Implications for RL roadmap

The OpenHands results refine the RL roadmap:

1. **Acquire trajectories** — Generate with target open-weight model using SERA/OpenCode harness (not CodeActAgent — harness design affects RF quality)
2. **Extract 3 features** — adapter exists (`e6_openhands_rf.py`)
3. **Need >10% pass rate** — Models with 7% pass rate give AUC=0.62-0.71. Higher base rates (like our 58% Claude) give better RF signal.
4. **Train model-specific RF** — minutes on CPU
5. **RL with RF + v009 cascade as reward** — RF handles easy cases, v009 handles uncertain
6. **Periodic gold-test calibration** — prevent reward hacking and ECE drift

## What's Next

E6 closes the learned verifier experiment series. The findings inform production design:

| Decision | Recommendation | Evidence |
|----------|---------------|----------|
| Universal vs per-model RF | **Per-model** | Transfer gap=0.452 at 7-model scale, ensemble beats single RF |
| Feature set | **3 features** (no SVG) | svg_accepted drop is only -0.019; SVG is model-specific |
| Cold-start strategy | **v009 rubric fallback** | Model-agnostic, 0.92 precision, $0.03/patch |
| Retraining trigger | **Distribution shift detection** | Feature means shift 0.84-1.50x across models |
| Minimum pass rate for RF | **>10%** | 7% pass models cap at AUC~0.70; 58% pass achieves 0.74 |
| Harness choice for traces | **SERA/OpenCode over CodeActAgent** | Same model (Claude Sonnet): 0.738 vs 0.546 AUC |
| RF as RL reward | **Yes — per-model only** | 2x reward density vs v009 (30% vs 14% recall at high precision) |
| RL cold-start reward | **v009 rubric** | Model-agnostic, 14% coverage, sufficient for initial exploration |

### Open questions

1. **Can quantile normalization recover transfer?** Normalizing features to percentile ranks within each model family might enable a shared RF. Untested.
2. **Does the data flywheel converge?** If v009 labels have systematic bias (e.g., always reject small diffs), the RF inherits that bias. Need gold-test spot checks.
3. **Does harness quality dominate model quality for RF?** Our Claude+SERA (AUC=0.738) vs OpenHands Claude+CodeActAgent (0.546) suggests the harness contributes more to RF signal than the model. Needs controlled experiment.
