# Autoresearch Spec: GRPO 7B Router — Negative Result Writeup

## Status: DRAFT (pulled from cost-aware-routing experiment, 2026-05-28)

## Summary

A 7B Qwen2.5-Instruct router trained with cost-aware GRPO over a 9-worker Bedrock pool **fails to learn per-question routing**. Across three controlled smoke runs (50 iters, batch 32, α=1.0) the router collapses to a degenerate cheap-tier policy with mean reward +0.24, well below best-static (+0.65) and the per-question oracle (+0.77). A CPU GRPO simulator with the same reward function but a 9-d softmax router (no question features) reproduces the failure exactly, showing the issue is structural to shared-policy RL on a multi-modal cost-aware reward landscape — not specific to Qwen2.5-7B, ModernBERT, or any architectural detail of the 7B router.

The negative result motivates the regime-A redesign documented in `domains/autoresearch/blueprints/cost-aware-routing/phase1-redesign-2026-05-28.md`: cost-aware single-shot routing belongs in classifier-and-policy architectures (vLLM Semantic Router-class, sub-200M parameters), not in 7B+ RL-trained LLMs.

## Why this is publishable

1. **Naive GRPO + cost-aware reward + heterogeneous pool** is the obvious thing to try given recent fashion (Sakana Conductor 2512.04388, RouteLLM, GRPO papers). Showing it doesn't work — and specifically why — is non-trivial and saves future researchers months.

2. **The failure mode is reproducible from first principles** in a 50-line CPU simulator. The simulator gives a clean theoretical statement: a single-policy GRPO router cannot resolve cost-quality tradeoffs across heterogeneous question types because the within-batch advantage signal is multi-modal and pulls in opposite directions on easy vs. hard questions.

3. **Production economics don't survive scrutiny**: a 7B router that costs ~$0.001/decision is more expensive to *run* than the savings it could find on cheap-tier workers (Gemma at $0.0004/query). Even a successful 7B router would have negative ROI in production.

## What was tested (briefly)

- **Pool**: 9 Bedrock workers from $0.0004 (Gemma-3-27B) to $0.021 (Opus 4.7), 60× cost spread, measured per-source accuracy on MATH500 / AIME25 / WildChat (130 questions × 9 workers = 1,170 rollouts).
- **Reward**: `is_correct ? max(1 − α·cost_normalized, −1) : 0`, α ∈ {0.5, 1.0, 1.7, 3.0}.
- **Trainer**: GRPO on Qwen2.5-7B-Instruct, ported from rl-conductor's Phase 1 trainer; 8 GPUs (p5.48xlarge), batch 32 rollouts/iter, 50 iters, cosine LR 1e-6.
- **Three prompt variants**:
  - V1: 9-shot per-worker examples (one per ord; "balanced few-shot"). Failed.
  - V2: V1 + per-token GRPO loss (gradient on the digit token only, not full response). Failed.
  - V3: Format-only few-shot (single generic question, picks rotated 0..8). Failed.
- **All three converged** to mean reward ~+0.24, with worker picks heavily concentrated on Gemma (32%) + gpt-oss-120b (44%) + scattered Opus (~12%) regardless of question difficulty.

## What the simulator showed

CPU GRPO simulator (`results/runs/grpo_sim_*.json`):

| Topology | α=1.0 mean reward | Notes |
|----------|------------------:|-------|
| Shared policy | +0.59 | Collapses to always-Gemma everywhere |
| Stratified-batch shared | +0.38 | Worse (alternating gradients) |
| Per-difficulty (2-class) | +0.70 | Beats best-static by ~5pp |
| Per-source (3-class) | +0.70 | Same as per-difficulty on this data |
| Per-question oracle | +0.77 | Theoretical upper bound |
| Best static (Always-Qwen-Coder) | +0.65 | Single-worker baseline |

The shared-policy collapse to +0.59 < +0.65 (best-static) is the critical finding: **GRPO with a single-policy router is strictly worse than picking one good worker statically**, regardless of compute budget.

## Why naive GRPO fails (the formal argument)

1. **Reward landscape is multi-modal**: at α=1.0, cheap-worker rewards on easy questions (~+1.0) are similar in magnitude to expensive-worker rewards on hard questions (~+0.6, since cost penalty cancels accuracy gain). There's no monotone direction in policy space that improves all questions.

2. **Within-batch advantage normalization mixes signals**: GRPO computes `(r − mean(group)) / std(group)` for advantage. If the group contains both easy and hard questions, the advantage estimates are noisy because the per-question optimal worker differs.

3. **No question-conditioning in shared policy**: a 9-d softmax with no input features cannot represent "Gemma for math500, Opus for aime25, Qwen-Coder for wildchat" — those require 3 different distributions. The LLM trainer has question features in principle but doesn't extract the routing-relevant ones at the scale of training compute we tested (50 iters × 32 rollouts).

4. **Cost asymmetry inverts the gradient on hard questions**: at α=1.0, Opus correct on AIME = `1 − 1.0·1.0 = 0`. Gemma wrong on AIME = `0`. The within-group variance is dominated by Gemma being correct/wrong on easy questions, not by Opus's hard-question performance. So the gradient pulls toward Gemma even when Opus is the right answer — the "correct" Opus rollouts give zero reward, indistinguishable from "wrong" Gemma rollouts.

## What changed when we tested fixes

- **Per-token GRPO loss** (compute CE only on the worker_id digit token): no behavioral improvement. The reward landscape is still multi-modal.
- **Format-only few-shot** (rotate worker picks across the same question): improved iter-0 entropy but didn't unlock learning. The model still converged to the cheap-tier collapse.
- **Increased batch size** (CPU sim): per-source policy converges faster but shared policy never beats best-static.
- **Increased iters** (CPU sim 2000 iters): same collapse, just longer to confirm.

## Recommendation

For single-shot cost-aware routing problems on heterogeneous worker pools, **prefer classifier+policy architectures** (vLLM Semantic Router, our regime-A redesign). Reserve LLM-RL routers for **multi-step orchestration** (Sakana Conductor regime), where the routing decision depends on plan state and the router's compute cost is amortized across multiple worker calls.

## Where the artifacts live

- Smoke run training logs: `domains/autoresearch/blueprints/cost-aware-routing/results/runs/alpha1.0-smoke50-{v1,v2,v3-fo}-training.jsonl`
- Iter-0 histogram diagnostics: `results/preflight/iter0_*.json`
- Introspection (base vs trained): `results/runs/introspect_{base,v1_iter49}.json`
- Prompt-variant experiment: `results/runs/prompt_variants.json`
- CPU GRPO simulator: `scripts/grpo_sim.py`, results in `results/runs/grpo_sim_alpha*.json`
- Oracle table: `results/runs/oracle_alpha_sweep.json`
- This spec: `domains/autoresearch/specs/grpo-router-negative-result.md`

## Phases (this is a writeup spec, not a run-it spec)

### Phase A: Final writeup

- Convert the simulator findings into a 4-page workshop paper.
- Target venue: NeurIPS ML for Systems workshop (where vLLM Semantic Router published) or arxiv preprint with Sakana Conductor + RouteLLM as related work.
- Negative result framing matched against the formal argument above.

### Phase B (optional): Larger-scale replication

- If reviewers request, re-run with the spec's original 256-rollout × 200-iter budget on multiple α values to show the collapse persists at scale.
- Estimated cost: ~$3,000 (the original spec's budget). Skip unless reviewer-requested.

## Non-requirements

- This spec does NOT advocate replacing GRPO. GRPO works when its assumptions hold (single-objective reward, monotone advantage signal, sufficient compute).
- This spec does NOT claim cost-aware routing is impossible — see `phase1-redesign-2026-05-28.md` for what works.
- This spec does NOT critique rl-conductor or Sakana Conductor — those are regime-B architectures and the negative result here doesn't transfer.
