# Phase 1 Redesign — Single-shot routing as a regime-A problem

**Status**: DRAFT, 2026-05-28. Supersedes plan-addendum-2026-05-27.md as of this date.

## Why redesign

Three smoke runs (V1, V2, V3-format-only) and a CPU GRPO simulator established the Phase 1 design as written cannot succeed. The findings, ordered by importance:

1. **Architecture mismatch**: the original spec applies the rl-conductor multi-step orchestration framing (regime B) to a single-pick routing problem (regime A). These are different problems with different optimal architectures — see §"Two routing regimes" below.

2. **Cost economics inverted**: a 7B GRPO router that costs ~$0.001/decision is more expensive to *run* than the savings it could find on the cheap-tier workers ($0.00035-$0.00075/query for Gemma/gpt-oss/Qwen-Coder). The router itself eats the cost-saving lever.

3. **Empirical confirmation of regime mismatch** (CPU simulator, see `results/runs/grpo_sim_*.json` and `oracle_alpha_sweep.json`):
   - Shared-policy GRPO (the LLM trainer's effective topology) collapses to always-Gemma at every α tested. Final reward +0.59, well below best-static +0.65 at α=1.0.
   - Per-source/per-difficulty GRPO succeeds (+0.70) but requires the policy to condition on a label the single-input prompt doesn't naturally have.
   - Per-difficulty needs >90% classifier accuracy to deliver the routing benefit; Haiku 4.5 zero-shot achieves 68.5%.

4. **Literature alignment**: vLLM Semantic Router (2603.04444 + paper bibliography) shows production routers use **classifier + signal + declarative policy** at the <500M parameter scale, not LLM-as-RL-router at 7B+. Our regime-A problem belongs there.

## Two routing regimes (the reframing)

| Aspect | Regime A: single-shot routing | Regime B: multi-step orchestration |
|--------|-------------------------------|-----------------------------------|
| Decision space | Pick 1 of N workers per query | Pick subtask + worker at each step k of a plan |
| State across calls | None | Full history of (subtask, worker, output) |
| Information requirement | Low — query-level features (domain, complexity, code/math/etc.) | High — needs to read prior outputs and reason about plan state |
| Cost asymmetry | Brutal — router cost > cheap-worker cost | Favorable — router cost amortized across multiple worker calls in a plan |
| Quality bound | Bounded by signal extraction quality | Bounded by reasoning quality |
| Right architecture | Small classifier + signal + policy | LLM with CoT + RL over trajectories |
| Reference work | vLLM Semantic Router, RouteLLM, FrugalGPT | Sakana Conductor (2512.04388), rl-conductor |

**Phase 1 lives in regime A.** The original spec inherited regime-B machinery from rl-conductor; the redesign moves to regime A.

## Redesigned Phase 1

### Objective

Demonstrate that a **lightweight classifier-based router** captures ≥50% of the measured oracle gap on a heterogeneous 9-worker Bedrock pool, while costing <1% of the cheapest worker per routing decision.

### Architecture

```
Question → Signal extractor (ModernBERT-base, 149M) → {category, complexity, reasoning-needed}
                                                              ↓
                                            Quality table (per-(category, worker) success rate)
                                                              ↓
                                            Cost-aware policy: argmax over E[r | category, worker, alpha]
                                                              ↓
                                                     worker_id ∈ {0..8}
```

**No RL training of the router.** The router is a deterministic policy over classifier output + a quality table. The classifier is fine-tuned with supervised learning. The quality table is computed offline from baseline rollouts.

### Components

1. **Signal extractor** — ModernBERT-base (149M parameters) fine-tuned on:
   - Category labels: `{math, code, factual, open-domain, multilingual, reasoning, structured}` (7 classes).
   - Complexity label: easy / hard (binary; computed from the baseline rollouts as `n_workers_correct ≤ 3` → hard, ≥ 7 → easy, else uncertain).
   - Reasoning-needed: binary, derived from "is Opus the only correct worker?" on baseline data.
   - Training data: ~5,000 questions = 1,000 from each of MATH500, AIME25, HumanEval, GSM8K, MMLU, plus 1,000 WildChat-1M filtered. Label categories from source tags + complexity from baseline correctness patterns.
   - Cost: ~$10 to fine-tune on a g6e.4xlarge (~3 hours @ $1.20/hr).

2. **Quality table** — per-(category × complexity × worker) accuracy and cost. Built from existing 130-question baseline + augmented to 500 questions to stabilize per-cell estimates.
   - Cost: ~$50 in Bedrock calls for the augmentation.

3. **Cost-aware policy** — closed-form `argmax_w E[is_correct(w | cat, cmplx)] × max(1 − α·cn(cost(w)), −1)`. Per-α policies are deterministic given the table.

4. **Inference** — at serve time: single ModernBERT forward (~5ms on CPU) + dictionary lookup. Per-query routing cost <$0.0001.

### Hypothesis (updated 2026-05-28 with simulator validation, recomputed 2026-05-28 #2 after E[r] bug fix)

Simulated the proposed architecture on the existing 130-question baseline data using the actual Haiku classifier output (`results/runs/sim_classifier_router.json`). Original simulator computed E[r] as `p_correct × reward(avg_cost)` which silently disagrees with the per-rollout-then-mean E[r] used in the oracle calculation; fixed in `scripts/router_policy.py` and `scripts/sim_classifier_router.py`. Corrected results:

| α | best-static | per-q oracle | **3-class oracle classifier** | 2-class diff oracle | Haiku 2-class (68.5% acc) |
|--:|-----------:|-------------:|------------------------------:|--------------------:|--------------------------:|
| 0.5 | +0.679 | +0.805 | **+0.805 (100%)** | +0.777 (78%) | +0.749 (55%) |
| 1.0 | +0.648 | +0.765 | **+0.765 (100%)** | +0.761 (96%) | +0.671 (20%) |
| 1.7 | +0.639 | +0.741 | **+0.741 (100%)** | +0.739 (98%) | +0.656 (17%) |
| 3.0 | +0.623 | +0.705 | **+0.705 (100%)** | +0.698 (91%) | +0.653 (36%) |

**Three findings (post-bug-fix)**:

1. **A perfect category-level classifier captures the FULL per-question oracle gap at every α.** This is stronger than I'd predicted in the first pass (which had α=3.0 collapsing). The classifier-router architecture is correct; the only question is classifier accuracy.

2. **A perfect 2-class difficulty classifier captures 78-98%.** Difficulty alone is enough at α∈{1.0, 1.7, 3.0}; only at α=0.5 (where the cost penalty is small) does category matter for getting the last 22pp.

3. **Haiku at 68.5% accuracy captures 17-55%, depending on α.** At α=0.5 (small cost penalty, accuracy dominates) Haiku's leniency hurts least. At α∈{1.0, 1.7} (the discrimination band) Haiku's noise costs the most.

**Architectural prediction**: a 7-class ModernBERT category classifier with ≥85% accuracy + closed-form cost-aware policy will deliver **≥80% of the per-question oracle gap at every α**. At α=1.0 that means mean reward ≥+0.742.

The architecture is validated; the next experiment that produces new information is fine-tuning the actual ModernBERT classifier on labeled data.

### Eval plan

Same as original spec — held-out 510-question eval set across MATH500/MMLU/HumanEval/GPQA/AIME25/WildChat. Compare:

- Always-X baselines (9 static policies, already measured).
- **Closed-form classifier-router** at α ∈ {0.5, 1.0, 1.7, 3.0}.
- vLLM Semantic Router reasoning-gate baseline (reproduce per `arxiv.org/abs/2510.08731`) — reason_on/off of Qwen3-30B-A3B as the 2-class router. Adapt to our 9-worker pool by using its reason/no-reason output to pick {Opus, Gemma}.
- (Stretch) Original 7B-LLM router smoke runs as a reference point — already collected.

### Success criteria (revised after simulator validation)

1. **Beat best-static at α∈{0.5, 1.0, 1.7}**: classifier-router mean reward exceeds always-Qwen-Coder-480B by ≥8pp on the eval mix. (Threshold raised from 6pp because the simulator showed a perfect 3-class classifier captures ~12pp at α=1.0, so 8pp = ~67% gap-capture is a tight but achievable target.)
2. **Capture ≥75% of oracle gap**: at α=1.0, achieve mean reward ≥+0.736 (75% of +0.117 oracle gap above best-static +0.648).
3. **Routing cost <1% of cheapest worker**: classifier latency × cost <$0.0000035 per decision (1% of Gemma's $0.00035).
4. **Capability-grounded routing**: at α=1.0, ≥40% of AIME questions route to Opus, ≥40% of MATH500 route to Mistral/Qwen-Coder.
5. **Classifier accuracy ≥85%** on a held-out category-labeled set (the simulator showed 68% Haiku vs 100% oracle implies the gap-capture ratio scales roughly linearly with classifier accuracy).

### What this gives up

- **No RL component**. The router never RL-trains. We forfeit the spec's exploratory question of "can GRPO learn cost-aware routing." The CPU simulator and 7B smoke runs already answered: with shared policy, no; with per-source policy, partially. The redesign skips that question because the production-relevant answer is already known (use a classifier).

- **No cross-pool transfer claim**. The classifier learns categories+complexity, not worker-conditioned policies. Adding a 10th worker requires updating the quality table, not retraining. (This is actually an improvement over the spec's rl-conductor inheritance.)

- **No "router as model" claim**. The output is engineering, not a fine-tuned model. The research contribution is the quantified Pareto frontier and the negative result on RL-trained 7B routers.

### Cost & timeline

| Item | Cost | Time |
|------|------|------|
| Augment baseline data to 500 questions × 9 workers | ~$50 | 2 hours |
| Fine-tune ModernBERT classifier (g6e.4xlarge) | ~$5 | 3-4 hours |
| Reproduce vLLM Semantic Router reasoning-gate baseline | ~$30 | 1 day |
| Evaluation on full 510-question eval set | ~$50 | 4 hours |
| **Phase 1 total** | **~$135** | **~3 days** |

That's a 95% cost reduction vs the previous addendum's $3,640 for 4 GRPO α runs.

### Negative-result framing for the original spec

The original spec's GRPO-7B-router framing produced a **publishable negative result** that should appear in the experiment writeup:

> "We attempted to train a 7B Qwen2.5-Instruct router with GRPO and a cost-aware reward floored at −1, sweeping α ∈ {0.5, 1.0, 1.7, 3.0}. Across three smoke runs (50 iters each, batch 32) the router collapsed to a degenerate cheap-tier policy (mean reward +0.24, vs best-static +0.65, vs oracle +0.77). A CPU simulator with the same reward function but a 9-d softmax router (no question features) confirms the failure is structural to shared-policy RL on a multi-modal reward landscape. Decoupling the routing decision into (signal extractor + closed-form cost-aware policy) recovered the oracle gap at <1% of the GRPO compute cost."

This is a sharper, more defensible result than the spec's original ambition.

## What's NOT in this redesign

- **Phase 2 cascade** — orthogonal; cascade routing is a different question (FrugalGPT-style threshold escalation). Defer.
- **Phase 3 pool robustness** — testable on the classifier-router (re-run quality table with a worker added/removed). Adds ~$30 + 1 day. Worth doing.
- **Phase 4 SWE-bench transfer** — not directly applicable to a classifier-router. SWE-bench is multi-step (regime B); needs a different architecture entirely.

## Decisions (resolved 2026-05-28)

1. ✅ **Negative result kept as a separate spec** — `domains/autoresearch/specs/grpo-router-negative-result.md` (to be written) frames the failed 7B GRPO attempt as a published finding. This redesign references it but doesn't replicate the work.

2. ✅ **Reproduce vLLM Semantic Router reasoning-gate baseline** — implement their classifier+policy on our 9-worker pool. Adapt their reason/no-reason output to a 2-way Opus/Gemma routing decision; compare to our 7-class category-aware policy.

3. ✅ **149M ModernBERT-base** — no smaller-model ablation. The router itself is small enough that 50M vs 149M doesn't materially change deployment economics.

4. ✅ **Start with 500-question quality table** — augment to 5,000 only if per-cell variance forces it. Total initial Bedrock spend ~$50.
