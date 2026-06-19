# Autoresearch Spec: RL Conductor Reproduction

## Status: DRAFT

## Overview

Reproduce the core results of Sakana AI's *Conductor* (arxiv:2512.04388, ICLR 2026): a 7B model trained with GRPO and a binary correctness reward to orchestrate a pool of larger worker agents. The Conductor outputs three synchronized Python lists per workflow step — `subtasks`, `model_id`, `access_list` — defining a flexible multi-agent topology up to depth 5.

Beyond pure reproduction, the experiment tests two questions the paper does not answer:

1. **Does emergent verification appear in the trained workflow logs?** The paper claims verification rounds emerge from binary reward alone, with no verification-specific training signal. We want to detect and quantify this independently.
2. **How does a mixed open + closed worker pool change orchestration behavior?** The paper used closed frontier models only. We use a hybrid pool spanning self-hosted (GLM-5, Kimi K2.6, Qwen3.5, Devstral) and Anthropic-hosted models (Opus 4.7, Sonnet 4.6, Haiku 4.5).

This connects directly to our Verifier Reward and Verification Primitives findings: if RL produces emergent verification, the recall ceiling we hit at 38 iterations of rubric tuning may be addressable by training the orchestrator instead of the rubric.

**Depends on**: existing serving infra (Qwen3.5/GLM-5/Kimi K2.6 endpoints), agent-harness eval rig, SWE-bench Lite gold-test pipeline.

## Research Questions

1. **Can the Conductor recipe (Qwen2.5-7B + GRPO, batch 256, lr 1e-6, 200 iters, no KL, ternary reward) be reproduced on a single 8× A100 node?** Paper used 2× H100 80GB; p4de.24xlarge spot has 4× the parallelism budget.

2. **Does a Conductor trained with a mixed open/closed worker pool route differently than one trained on open-only or closed-only pools?** Specifically: does it learn to gate Opus 4.7 calls behind cheaper checks, the way our verifier-reward cascade does manually?

3. **Do verification rounds emerge from binary reward alone?** Measured by parsing workflow logs across training: % of workflows containing a "review" or "check" step on a worker's output, by training iteration.

4. **Does the trained Conductor transfer to SWE-bench Lite tasks?** Paper reports MATH/MMLU/LiveCodeBench. SWE-bench is harder, longer-horizon, and our home turf. We have a 175/300 baseline (verification-primitives-swebench).

5. **Cost-adjusted performance vs fixed pipelines.** Paper claims 2.4× advantage over MoA at equal accuracy. Does this hold when the worker pool includes ~free self-hosted models alongside paid APIs?

## Phases

### Phase 1: Reproduction (single-node training)

**Goal**: Reproduce the paper's headline numbers on MATH500, LiveCodeBench V6, and GPQA-Diamond using the published worker pool composition (frontier closed-source) so we have a clean reference point before changing variables.

**Steps**:

1. **Training data**: 960 problems — MATH500 train (300), MMLU sample, LiveCodeBench V1, RLPR. All public.
2. **Worker pool (closed, paper-faithful)**: Opus 4.7, Sonnet 4.6, Haiku 4.5, GPT-5, Gemini 2.5 Pro. Substitute Opus 4.7 for the paper's Claude Sonnet 4 (model has rolled forward).
3. **Base model**: Qwen2.5-7B (paper's exact base).
4. **GRPO config (paper-exact)**:
   - 200 iterations
   - Batch 256 = 4 questions × 64 rollouts
   - lr 1e-6, cosine schedule, AdamW
   - **No KL regularization**
   - Reward: 0 (malformed) / 0.5 (parseable, wrong) / 1 (correct)
   - Max workflow depth: 5 steps
5. **Hardware**: p4de.24xlarge (8× A100 80GB, spot, us-east-1 az6) — see infra memory.
6. **Eval**: MATH500 test (100), LiveCodeBench V6 (sample), GPQA-Diamond, AIME25 OOD.

**Success target**: within ±3pp of paper's headline (LiveCodeBench 83.9%, GPQA-D 87.5%, MATH 95%+). Anything inside that band counts as reproduction.

### Phase 2: Mixed open/closed worker pool

**Goal**: Train a second Conductor with a hybrid pool and compare routing behavior + cost-adjusted performance.

**Worker pool (hybrid)**:

| ID | Model | Source | Approx cost/1M out |
|----|-------|--------|---------------------|
| 0 | Qwen3.5-122B-A10B-FP8 | Self-hosted (g7e.24xl, vLLM) | ~$0 |
| 1 | GLM-5-FP8 | Self-hosted (B200, SGLang HiCache) | ~$0 |
| 2 | Kimi K2.6 | Self-hosted (B300, vLLM) | ~$0 |
| 3 | Devstral Small 2 24B | Self-hosted (g7e.12xl, vLLM) | ~$0 |
| 4 | Haiku 4.5 | Anthropic API | $1.25 |
| 5 | Sonnet 4.6 | Anthropic API | $15 |
| 6 | Opus 4.7 | Anthropic API | $75 |

**Same training config as Phase 1** so the pool is the only variable.

**Comparisons**:
- Headline accuracy on identical eval sets vs Phase 1 closed-only Conductor.
- Cost-per-correct-answer (closed-only vs hybrid).
- Routing histogram by task type: does the hybrid Conductor learn to send code-heavy tasks to GLM-5/Kimi and reasoning-heavy tasks to Opus 4.7?
- Workflow depth distribution: does access to cheap self-hosted models encourage longer multi-step workflows?

### Phase 3: Emergent verification detection

**Goal**: Empirically test the paper's emergent-verification claim on our trained Conductors.

**Method**:
1. Sample workflow logs at iterations {0, 25, 50, 100, 150, 200}.
2. For each sampled workflow, classify each step into one of: `decompose`, `solve`, `aggregate`, `verify`, `revise`. Use an LLM judge (Sonnet 4.6) with an explicit rubric — a "verify" step is one where the subtask instruction asks the worker to evaluate, check, find errors in, or critique a prior step's output.
3. Plot fraction of workflows containing ≥1 verify step vs training iteration.
4. Test: is the verify-step fraction at iter 200 > 2× the rate at iter 0? (Paper's qualitative claim, our quantitative bar.)
5. Correlate: do workflows with verify steps have higher reward than those without, controlling for task difficulty?

### Phase 4: SWE-bench Lite transfer

**Goal**: Apply the Phase 2 hybrid Conductor to SWE-bench Lite (n=300, our existing eval) without retraining, and compare against:
- Best single-model baseline: Claude Code + Opus 4.7 (estimate via small subset)
- Verification-primitives 175/300 (58.3%) baseline
- Cost-matched fixed pipeline: SERA + Devstral, OpenCode + Qwen3.5, etc.

This is the novel contribution beyond the paper.

### Phase 5: Recursive self-correction (optional)

**Goal**: Reproduce the paper's recursive Conductor extension (20 fine-tuning iters, 350 filtered samples, 0.25 discount on non-recursive rounds) and measure BigCodeBench delta.

## Components

### 1. Compute

- **Phase 1-2 training**: p4de.24xlarge (8× A100 80GB) — spot. Backup checkpoints to S3 every 25 iterations.
- **Inference for self-hosted workers**: existing g7e/B200/B300 endpoints (already deployed).
- **Inference for closed workers**: Anthropic API, OpenAI API, Google API.
- **Eval**: m7i.4xlarge (existing, 54.210.193.49) for SWE-bench Docker eval.

### 2. Codebase

- **Source**: New blueprint `domains/autoresearch/blueprints/rl-conductor/`.
- **Training framework**: `verl` or `trl`'s GRPO trainer. Prefer `verl` — it's the published Conductor's reference framework class and handles 64-rollout batches cleanly.
- **Worker proxy**: thin OpenAI-compatible router that maps `model_id` integers to backends (vLLM/SGLang/Anthropic/OpenAI/Google). Reuse `thunder_proxy.py` patterns from agent-swarm memory.
- **Workflow runtime**: parses Conductor output → executes subtask graph → returns aggregated answer for reward computation.
- **Agent-editable**: training script, reward function, worker proxy.
- **Fixed**: eval harness, gold-answer datasets.

### 3. Experiment Protocol

- **Metric**: Eval accuracy on held-out set (MATH500-test, LCB-V6, GPQA-D, AIME25, SWE-bench Lite). Cost-per-correct-answer as secondary.
- **Time budget**: 24h wall clock per training run (paper reports < 12h on 2× H100; 8× A100 should fit comfortably).
- **Loop structure**: GRPO outer loop (200 iters) with 64-rollout inner sampling. Log every iteration to S3.
- **Termination**: 200 iterations OR reward plateau (no improvement over 25-iter window).
- **Logging**: per-iteration: mean reward, reward std, format-failure rate, mean workflow depth, worker_id histogram, verify-step fraction. Per-rollout: full subtasks/model_id/access_list/answer/reward.

### 4. Networking

- Training node SSH: `ssh -i ~/.ssh/g7e-bench.pem ec2-user@<p4de-ip>` (TBD on launch).
- Worker endpoints reachable from training node via VPC peering or public endpoint with token auth.
- Anthropic/OpenAI/Google API keys via env vars on training node only — never logged.

### 5. Storage

- **Training data**: S3 `s3://agent-aiops-research/rl-conductor/data/` — 960 train problems + eval sets.
- **Checkpoints**: S3 `s3://agent-aiops-research/rl-conductor/checkpoints/{phase}/iter-{n}/`. Local NVMe mirror; spot-resilient.
- **Logs**: S3 `s3://agent-aiops-research/rl-conductor/logs/{phase}/` — JSONL per iteration.
- **Eval results**: blueprint-local `domains/autoresearch/blueprints/rl-conductor/results/`.

## Success Criteria

1. **Phase 1 reproduction**: ≥1 of {MATH500, LCB-V6, GPQA-D} within ±3pp of paper's reported number. Format-failure rate < 5% by iter 100.
2. **Phase 2 hybrid**: trained Conductor produces a non-degenerate worker_id histogram (no single worker > 70% of calls) AND beats the strongest single worker in the pool by ≥2pp on at least one eval.
3. **Phase 3 verification emergence**: verify-step fraction at iter 200 is statistically higher than at iter 0 (paired bootstrap, p < 0.05). Negative result is also publishable.
4. **Phase 4 SWE-bench transfer**: hybrid Conductor exceeds 58.3% baseline OR delivers same accuracy at < 50% the cost of best-single-harness Opus 4.7.

## Non-Requirements

- Trinity reproduction (arxiv:2512.04695) — separate spec if Phase 1 succeeds.
- Fugu commercial features (recursion-on-recursion, latency optimization).
- Distributed multi-node training — single p4de is sufficient at this scale.
- Custom GRPO implementation — use `verl` or `trl` off the shelf.

## Known Limitations

- **Spot reclaim**: p4de spot can be reclaimed mid-training. Checkpoint every 25 iters; design resume path.
- **Closed API rate limits**: 64 rollouts × 5 max steps × per-step API call = bursty load on Anthropic/OpenAI tiers. May need request batching or tier upgrade.
- **Worker variance**: closed model behavior drifts over time (silent model updates). Snapshot worker model identifiers per training run.
- **Reward hacking**: ternary reward (0/0.5/1) with no KL is aggressive. Watch for format-degenerate solutions in early iterations and add format check tightening if needed.
- **Verification detection is judge-dependent**: Phase 3's LLM-judge classification of workflow steps introduces measurement noise. Run inter-judge agreement on a sample (Sonnet vs Haiku vs human).

---

> **Note**: Operational artifacts (training logs, checkpoint analysis, lessons) belong in `domains/autoresearch/blueprints/rl-conductor/` once Phase 1 begins.
