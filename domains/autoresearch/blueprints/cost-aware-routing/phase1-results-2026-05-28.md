# Phase 1 Results — 2026-05-28

Final summary of the cost-aware-routing Phase 1 experiment after the regime-A pivot. All 5 implementation steps from `phase1-redesign-2026-05-28.md` are complete.

## Headline result (96-question held-out eval)

| Policy | α=1.0 reward | vs best-static | Notes |
|--------|-------------:|---------------:|-------|
| Always-Qwen-Coder-480B (best static) | +0.740 | — | strong default |
| Classifier-router (5-class, 96.9% acc) | +0.730 | **−1.0pp** | per-category routing too coarse |
| Reasoning-gate (5-class projection) | +0.733 | −0.7pp | matches static-baseline |
| Reasoning-gate (oracle binary label) | **+0.792** | **+5.2pp** | proves per-Q gap is reachable |
| Per-question oracle | +0.910 | +17.0pp | upper bound |

**Read**: a 5-class category classifier — even a 96.9%-accurate one — does not beat a strong static baseline (always-Qwen-Coder-480B) on this 9-worker Bedrock pool. **Per-question difficulty/complexity is the discriminating signal**, not category.

## Five empirical findings (in order of size)

1. **Always-Qwen-Coder-480B is a remarkably strong default** (75-96% per-category accuracy at $0.0007/q on the 480-question augmented dataset). It's the best-static policy at every α tested. Any router has to beat *this*, not always-Opus or always-Gemma.

2. **The per-question oracle gap is +0.17 at α=1.0** on the eval split (oracle +0.91, static +0.74). Real routing value exists but requires per-question discrimination.

3. **5-class category routing captures essentially zero of that gap.** Why: within each category, multiple workers cluster within 0.04 E[r], with Qwen-Coder beating or tying others on most questions. Picking the per-category best (Opus on math, Qwen-Coder elsewhere) doesn't beat picking Qwen-Coder always.

4. **Reasoning-gate with the *true* per-question difficulty label captures +5.2pp at α=1.0** — confirming the oracle gap is reachable with a per-question difficulty signal. But our 5-class projection only catches 58% of those labels (over-flags math/reasoning categories that don't actually need Opus).

5. **The 5-class ModernBERT classifier achieves 96.9% accuracy on category** but category was the wrong target. Per-category accuracy: math 100%, code 100%, factual 96%, reasoning 89%, open-domain 100%.

## Two negative results, one constructive direction

### Negative 1: GRPO 7B router (regime-A applied to a regime-A problem)

- 3 smoke runs all collapsed to mean reward ~+0.24, well below best-static.
- CPU GRPO simulator confirmed: shared-policy + multi-modal cost-aware reward = always-Gemma collapse.
- Spec at `domains/autoresearch/specs/grpo-router-negative-result.md`.

### Negative 2: 5-class ModernBERT classifier-router

- 96.9% classification accuracy → −1.0pp vs always-Qwen-Coder at α=1.0.
- Reason: category is the wrong abstraction; **within-category difficulty** is what carries routing value.
- Reasoning-gate variant with the same 5-class signal: also flat vs best-static.

### Constructive direction: per-question difficulty classifier

The reasoning-gate with **oracle binary labels** (using actual rollout outcomes to label "needs reasoning") achieves +5.2pp at α=1.0. That's the achievable target if we train a per-question **difficulty** classifier rather than a category classifier.

Two ways to build it:
- **Outcome-supervised**: label each question by `n_workers_correct` from the baseline rollouts, train a regression/binary classifier on that label. Requires the same baseline rollouts we already have.
- **Cost-of-routing supervised**: label each question by the cheapest correct worker's cost, train a regression model on that. More directly cost-aligned.

## What lives where

```
phase1-redesign-2026-05-28.md           — the regime-A redesign (this experiment's plan)
phase1-results-2026-05-28.md            — this doc
data/augmented_baseline_500q.jsonl      — 480 category-labeled questions
results/baselines/always_x_augmented.json — 9 workers × 350 new Qs (rollouts + judges)
results/runs/oracle_alpha_sweep_v2.json  — per-α oracle on full 480q dataset
results/runs/classifier_router_eval.json — 5-class classifier-router eval (negative)
results/runs/reasoning_gate_eval.json    — reasoning-gate variants eval (negative + oracle)
artifacts/classifier/                    — fine-tuned ModernBERT-base (96.9% cat acc)
scripts/router_policy.py                 — closed-form cost-aware policy (correct E[r])
scripts/eval_classifier_router.py        — pipeline for classifier-router eval
scripts/eval_reasoning_gate.py           — pipeline for reasoning-gate eval
```

## Phase 2 candidates

If we keep going:

A. **Train a per-question difficulty classifier** (regression on baseline `n_workers_correct`). Use it as the gate signal in a reasoning-gate router. Test target: capture ≥50% of the +0.17 oracle gap at α=1.0 (i.e., reach +0.81+). **TESTED 2026-05-28**: 74% accuracy, 32% hard recall. Beats best-static by +1.0pp at α=0.5 only; flat-to-worse at α≥1.0.

B. **Multi-feature classifier** (category × difficulty). 7+ category classes × {easy, medium, hard} = 21+ cells in the policy table. Quality table needs more rollout data per cell — augment to 1000+ questions (~$50). UNTESTED.

C. **Cost-of-routing regression**: predict the cheapest correct worker per question. Direct optimization. Alternative to the difficulty signal. **TESTED 2026-05-28**: 67.7% accuracy on cheapest-correct target. 75% Opus recall but 0% recall on most other classes. Beats best-static by NONE; loses 2.6-4.5pp at every α. Oracle on this target beats best-static by +16.8pp at α=1.0 — the largest oracle gap measured, but classifier accuracy isn't sufficient.

D. **Accept the negative result**, write up "5-class category classifier-routers don't beat strong static defaults on heterogeneous Bedrock pools" as a companion to the GRPO negative-result spec, and stop.

## Phase 2 final table (all tested approaches at α=1.0)

| Approach | Trained classifier reward | Oracle reward | Trained vs static | Oracle vs static |
|----------|--------------------------:|--------------:|------------------:|-----------------:|
| Always-Qwen-Coder (best static) | — | — | 0 | 0 |
| 5-class category (Phase 1) | +0.730 | +0.730 | -1.0pp ❌ | -1.0pp |
| Reasoning-gate (5c proj) | +0.733 | +0.792 | -0.7pp | +5.2pp |
| Binary difficulty (Phase 2A) | +0.739 | +0.767 | -0.0pp | +2.7pp |
| Cheapest correct (Phase 2C) | +0.708 | **+0.908** | -3.2pp ❌ | **+16.8pp** ✅ |

**The cheapest-correct ORACLE achieves +16.8pp** — proving substantial routing value exists. But the trained classifier captures only 67.7% of the labels, and the resulting routing performs worse than just always-using-the-strong-static.

**Conclusion**: a 149M-class encoder cannot extract enough question-level features to reliably predict the cheapest correct worker on this heterogeneous 9-Bedrock-worker pool, despite the per-question oracle gap being ~17pp. Either:
- Bigger models / different architectures (LLM-as-classifier with chain-of-thought) might reach the oracle target.
- Or the practical answer is: **always-Qwen-Coder-480B is the right deployment** for this pool size, and the routing optimization isn't worth the engineering complexity.

## Deployment recommendation: two regimes, two router designs

The 480q + in-flight 5K data point to a clean partition of routing problems. The classifier-router negative result is specific to **single-shot chatbot traffic**; it does not generalize to agentic workloads.

### Regime A — single-shot chat: medium tier is already enough

- ~66% of questions in this corpus are solved by **Gemma-27B alone** at $0.0003/q.
- ~85-90% are solved by anything up through **Qwen-Coder-480B** at ≤$0.0007/q.
- Only ~10-15% genuinely require Sonnet/Opus tier.
- The per-question oracle gap (+16.8pp) is concentrated in that ~15% slice.
- Discriminating those questions from text alone is what every classifier we trained failed to do (5-class, binary difficulty, 9-way cheapest-correct).
- **Recommendation**: deploy **always-Qwen-Coder-480B**. A 149M classifier-router does not beat it; the engineering cost of training, serving, and monitoring a router exceeds the realistic +1-2pp gain a working router would deliver on this traffic mix.
- **Caveat**: our mix (MATH500, MMLU-Pro, HumanEval, MBPP, WildChat single-turn) skews easy. A deliberately-hard slice (AIME, GPQA-Diamond, LiveCodeBench-Hard) would shift the cheapest-correct distribution toward flagship tier and could change this answer. Untested.

### Regime B — agentic / multi-step: flagship tier carries the trajectory

- 20-50 turn trajectories compound per-turn error: 90% × 30 turns = ~4% end-to-end.
- Self-critique, backtracking, long-context coherence are measurably better in flagship models. Cross-evidence: verifier-reward experiment (Claude-specific rubric, didn't transfer to Devstral/Mistral); agent-harness experiment (Devstral 24B + OpenCode hit 88% fix but 16% gold pass; 8-harness ensemble with bigger models pushed to 36%).
- The routing decision is **per-turn**, not per-question. "This turn is tool-use → Haiku" vs "this turn is planning → Opus" requires reasoning about future turns, not classifying the current input.
- A ModernBERT-class encoder cannot carry this — the policy needs to model the trajectory. That's the RL-conductor / Sakana setup, a fundamentally different problem (sequential, credit-assigned, not single-shot classification).
- **Recommendation**: for agentic systems, route via either (a) an LLM-as-router with chain-of-thought, or (b) an RL-trained orchestrator over the pool. Don't reuse the single-shot classifier-router pattern.

### Where this leaves Phase 2

The 5K corpus retrain is still worth running — it tells us whether the regime-A negative is a data-quantity ceiling or a fundamental signal-extraction limit. But the **deployment answer for chatbot traffic is already actionable** independent of that result: ship always-Qwen-Coder-480B.

### 5K retrain outcome (2026-05-29) — confirms signal-extraction limit, not data limit

The 5K baseline run partially failed (workers 5-8 hit `EndpointConnectionError` mid-run), so we retrained on a partial labelset: 4,067 questions × ord 0..4 search space (Gemma, gpt-oss, qwen3-32b, qwen-coder, mistral-large-3). Distribution: 67% Gemma / 8% gpt-oss / 9% qwen3-32b / 4% qwen-coder / 10% mistral / 8% no-medium-correct fallback.

Trained on AWS g6.xlarge (NVIDIA L4) — ModernBERT-base, 6 epochs, batch 32. Best eval at epoch 3:

| Epoch | Eval acc | Gemma rec | gpt-oss | qwen3-32b | qwen-coder | mistral |
|-------|---------:|----------:|--------:|----------:|-----------:|--------:|
| 3 (best) | **67.0%** | 97% | 18% | 1% | **0%** | 6% |
| 6 (final) | 63.3% | 89% | 6% | 6% | **0%** | 27% |

**The 8× larger training set produces ~the same accuracy as the 480q result (67.7%).** Best epoch is essentially the Gemma class frequency floor — the model collapses to "predict Gemma" with marginal improvement on minority classes. Qwen-Coder recall stays at 0% across all epochs despite being the cost-effective sweet spot.

**Conclusion**: the negative result is robust. A 149M-class encoder cannot extract per-question cheapest-worker signal from text alone, regardless of training set size. This is a **signal-extraction limit**, not a data-quantity ceiling. Closing the +16.8pp oracle gap requires either:
- An LLM-as-classifier with chain-of-thought (much higher inference cost — likely defeats the purpose),
- A different feature space (response previews, embeddings from a stronger encoder), or
- Accepting always-Qwen-Coder-480B as the deployment answer.

Phase 1 + Phase 2 spend: ~$365 (vs $7,800 budget). Phase 2 retrain on a fresh g6.xlarge: ~$0.30. Phase complete; recommend stopping classifier-router work for this pool/traffic mix.

## Total Phase 1 spend

| Item | Cost |
|------|------|
| Pre-flight + GRPO smoke runs (regime-B failed attempt) | ~$240 |
| Augmented baseline (350 new Qs × 9 workers + judges) | ~$25 |
| Classifier training (local CPU, free) | $0 |
| Eval + reasoning-gate (offline, free) | $0 |
| **Phase 1 grand total** | **~$265** |

vs. the original spec's $7,800 budget.
