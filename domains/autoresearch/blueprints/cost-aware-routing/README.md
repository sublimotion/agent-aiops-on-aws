# cost-aware-routing — blueprint navigation

## Active plan

**`phase1-redesign-2026-05-28.md`** — current Phase 1 design. ModernBERT classifier + closed-form cost-aware policy (regime-A architecture). Replaces the superseded GRPO 7B framing.

Read alongside:
- Spec entry point: `domains/autoresearch/specs/cost-aware-routing.md` (with SUPERSEDED banner pointing here).
- Negative-result writeup: `domains/autoresearch/specs/grpo-router-negative-result.md` (the abandoned GRPO 7B framing).

## Historical context

- `plan-addendum-2026-05-27.md` — pre-launch decisions for the GRPO 7B framing. Has SUPERSEDED banner at top. Useful for understanding why each parameter was set, what we tried, and why the framing failed.
- `llm_routing_literature_review.md` — prior art survey (FrugalGPT / RouteLLM / UCCI / etc.) collected while planning the original spec.

## Artifacts

### Pre-flight (validated)

- `results/preflight/judge_agreement_n50.json` — Haiku-Sonnet agreement on math (98%, n=50).
- `results/preflight/worker_probe.json` — 9-worker Bedrock probe (ping + token + 32-conc burst).
- `results/preflight/wildchat_judge_calibration.json` — Haiku-Sonnet agreement on open-domain (85%, n=20).
- `results/preflight/iter0_gate_qwen25_7b.json` — Iter-0 router pick distribution on real Qwen2.5-7B.
- `results/preflight/difficulty_classifier_probe.json` — Haiku as difficulty classifier (68.5% accuracy).

### Baselines (already collected)

- `results/baselines/always_x_math500.json` — 9 workers × 50 MATH500 questions.
- `results/baselines/always_x_aime25_n30.json` — 9 workers × 30 AIME25 questions.
- `results/baselines/always_x_wildchat_n50.json` — 9 workers × 50 WildChat questions.
- `results/baselines/oracle_router.json` — perfect-routing simulator output.

### Failed GRPO smoke runs (kept for the negative-result writeup)

- `results/runs/alpha1.0-smoke50-training.jsonl` — V1 (full-response loss, balanced few-shot).
- `results/runs/alpha1.0-smoke50-v2-training.jsonl` — V2 (per-token loss, balanced few-shot).
- `results/runs/alpha1.0-smoke50-v3-fo-training.jsonl` — V3 (per-token loss, format-only few-shot).
- `results/runs/introspect_base.json`, `introspect_v1_iter49.json` — base vs v1-trained introspection.
- `results/runs/prompt_variants.json` — V0/V1/V2/V3 prompt variant comparison (the meta-prompt finding).

### CPU GRPO simulator + analysis

- `scripts/grpo_sim.py` — 50-line NumPy GRPO simulator.
- `results/runs/grpo_sim_alpha*_per_source_*.json` — sweep over α × topology.
- `scripts/oracle_alpha_sweep.py` — closed-form per-α oracle calculator.
- `results/runs/oracle_alpha_sweep.json` — oracle / best-static / gap per α.
- `scripts/sim_classifier_router.py` — closed-form classifier-router simulator (validates the redesign).
- `results/runs/sim_classifier_router.json` — 3-class vs 2-class vs Haiku classifier comparison.

### Tooling

- `scripts/worker_pool.py` — single source of truth for the 9-worker Bedrock pool.
- `scripts/cost_reward.py` — cost-aware reward + Haiku-as-judge.
- `scripts/few_shot.py`, `scripts/few_shot_format_only.py` — original and revised few-shot prompts.
- `scripts/wildchat_judge.py` — open-domain quality judge.
- `scripts/run_baselines.py`, `scripts/run_baselines_wildchat.py` — Always-X baseline runners.
- `scripts/probe_workers.py` — Bedrock TPM/cost probe.
- `scripts/probe_difficulty_classifier.py` — Haiku classifier probe.
- `scripts/build_train_data.py`, `scripts/lmsys_loader.py` — training data assembly (for the abandoned GRPO framing; some still useful for the redesign).
- `scripts/train_cost_aware_router.py` — the abandoned GRPO trainer. Kept for the negative-result writeup.
- `vendor/rl-conductor-phase1/` — vendored rl-conductor source (ancestor of the abandoned trainer).

### Data

- `data/train.jsonl` — 620-question training mix (300 MATH500 + 20 AIME25 + 300 WildChat).
- `data/lmsys_train_300.jsonl`, `data/lmsys_eval_100.jsonl` — WildChat-1M filtered splits.
