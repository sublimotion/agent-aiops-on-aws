# trinity-coordinator — blueprint navigation

Reproduce **Trinity** (Xu et al., *"TRINITY: An Evolved LLM Coordinator"*, ICLR 2026, arXiv:2512.04695) — the evolved multi-agent coordinator behind Sakana's Fugu — on an all-Bedrock worker pool, run via the repo's agent runtime.

## Spec

- `domains/autoresearch/specs/trinity-coordinator.md` — the reproduction spec (Bedrock pool, CMA-ES + SVD, agent-runtime execution, Phase 0 → 0.5 smoke → 1 full → 2 head-to-head).

## Why this exists

Direct successor to the **GRPO router negative result** (`domains/autoresearch/specs/grpo-router-negative-result.md`). That writeup proved a single-policy GRPO LLM router collapses below best-static on a multi-modal cost-aware reward. Trinity is the architecture the writeup named as the alternative: **gradient-free CMA-ES over <20K params (SVD scales + linear head), not RL over a 7B policy.** The Trinity paper's own bake-off corroborates: `sep-CMA-ES 0.615 > SFT 0.592 > RS 0.374 > REINFORCE 0.253` — REINFORCE (≈ our GRPO) last, matching our +0.24 collapse.

## What Trinity is

- Coordinator: **Qwen3-0.6B** reading its own penultimate-token **hidden state** (does not generate routing text).
- Head: linear `hidden → (L workers + 3 roles)`, ~10K params.
- SVD fine-tuning: learn only singular-value *scales* of one layer (layer 26), U/V fixed.
- Optimizer: **CMA-ES** (gradient-free), <20K total params.
- Roles: **Thinker / Worker / Verifier**, multi-turn loop (≤5 turns), verifier ACCEPT halts.

## Layout

- `vendor/trinity-upstream/` — the authors' OpenReview code submission, verbatim except large regenerable weights (`svd_weights.pt`, model `.npy` kept only where small). **Reference; do not edit in place.**
  - `fugu/algorithms/es.py` — CMA-ES trainer.
  - `fugu/head_modules.py` — router head architectures.
  - `fugu/core.py` — multi-turn routing loop + role prompts.
  - `fugu/llm_clients.py` — provider clients (`query_anthropic` already uses Bedrock Converse).
  - `decompose_model.py` — SVD of Qwen3-0.6B.
  - `evaluate_trinity_livecodebench.py` — eval harness.
  - `logs/ckpt/models/model_iter_60.npy` — a trained coordinator checkpoint (Phase 0 eval target).
  - `.data_splits/livecodebenchv6_42_v0.2_t0.2.json` — eval split.
- `scripts/` (to be written during Phase 0) — the Bedrock + agent-runtime adaptation layer:
  - `bedrock_clients.py`, `worker_pool_bedrock.py`, `cost_bedrock.py`, `run_trinity_agent.py`.

## Adaptations from upstream

1. **Worker pool → all-Bedrock** (7 workers, closed + open) via Converse API. No GPU serving fleet for workers (the big infra simplification vs upstream's 4×vLLM ports).
2. **Coordinator stays local** on a small GPU (g6e-class) — it's 0.6B and its hidden states must be read directly.
3. **Execution → agent runtime** (`agent-runner` on EKS, IRSA-scoped Bedrock + S3, resumable).

## Carryover from cost-aware-routing (hard-won; enforced in the spec's gates)

- Gate 0.0 model-ID drift re-verification + Qwen3-32B dual-mode (reasoning/direct) probe.
- Gate 0.2b per-(worker × role) output-parser audit — 21 cells, BLOCKING.
- Phase 0.5 tiny-scale CMA-ES smoke before the $1K-5K full run.
- Checkpoint at iter 0, sync every iter, rollouts to S3 every iter, artifact-durability exfil before teardown.

## Status

DRAFT spec, code vendored. Not yet run. Phase 0 (eval-only with bundled checkpoint) is the cheapest first validation (~$50-150 Bedrock, no training).
