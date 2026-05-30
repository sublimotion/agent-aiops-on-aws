# Autoresearch Spec: Cost-Aware LLM Routing (OptiRoute)

## Status: SUPERSEDED (2026-05-28) — see redesign

This spec's "train a 7B Qwen2.5-Instruct router with GRPO" framing was empirically tested and **abandoned** after a structured 14-iteration investigation:

- **Active spec**: `domains/autoresearch/blueprints/cost-aware-routing/phase1-redesign-2026-05-28.md` — pivots Phase 1 to a regime-A architecture (ModernBERT-base classifier + closed-form cost-aware policy). 95% cheaper than this spec; ~3-day timeline; predicted to capture ≥75% of the per-question oracle gap.
- **Negative-result writeup**: `domains/autoresearch/specs/grpo-router-negative-result.md` — formalizes why the GRPO 7B framing failed (multi-modal reward landscape, shared-policy collapse, production-cost inversion). Targeted at NeurIPS ML-for-Sys workshop.
- **Plan addendum** (decisions made en route): `domains/autoresearch/blueprints/cost-aware-routing/plan-addendum-2026-05-27.md`.
- **Empirical evidence**: 3 GRPO smoke runs in `results/runs/alpha1.0-smoke50-{v1,v2,v3-fo}-training.jsonl`; CPU GRPO simulator in `scripts/grpo_sim.py`; oracle table in `results/runs/oracle_alpha_sweep.json`.

The rest of this document is preserved for historical context. Do **not** treat sections below as the active plan.

## Overview (original — superseded)

Train a cost-aware LLM router using GRPO that optimizes the Pareto frontier of quality vs cost. This is a strategic pivot from the rl-conductor reproduction effort (Phase 1.6), reframing the problem from "multi-step workflow orchestration" to "cost-constrained routing with quality maximization."

### Why Pivot from rl-conductor-repro

The rl-conductor reproduction (Phase 1.6) achieved ~64% MATH500 accuracy by LLM-judge after extensive debugging, validating the core GRPO training recipe. However, three structural issues emerged:

1. **Non-transferable routing**: The trained conductor is pinned to its 8-model pool via ordinal IDs (`Model id 0`, `Model id 1`...). Adding/removing a worker requires full retraining. Cross-seed pre-flight diagnostics (recommended in rl-conductor spec) would reveal whether this is fixable via profile-aware training (Phase 1.7-A) or an architectural limitation.

2. **Quality-only optimization**: The paper's binary reward `r = is_correct(response, gold) ? 1 : 0` optimizes for accuracy alone. Production systems need `f(quality, cost)` — the cost axis is missing. At $X budget per query, what's the maximum achievable quality? The conductor doesn't answer this.

3. **Reward-parsing conflation**: Phase 1.6 showed parser-based reward (14%) diverged from LLM-judged reward (64%). The regex extractor was a compounding confound. LLM-as-judge for reward (recommended in `feedback_grpo_resume_state.md`) is now the baseline, not an optional improvement.

### The Cost-Aware Routing Problem

The production question for any LLM deployment is: **at budget $X per query, what's the maximum quality I can achieve?** An ideal router delivers Sonnet-class quality at near-Haiku cost by routing simple questions to cheap models and escalating intelligently when needed.

This is not hypothetical. FrugalGPT (Chen et al. 2023, arxiv:2305.05176) and RouteLLM (Anyscale/LMSYS 2024) demonstrated measurable ROI from learned routing vs static policies. Our worker pool spans 60× in cost (~$0.00035/query for Gemma-3-27B to ~$0.021/query for Opus 4.7, at 200 in + 800 out tokens) — the optimization space is real.

### Core Hypothesis

A 7B router trained with cost-aware GRPO will learn to **optimize the cost-quality Pareto frontier**, producing a family of routers (one per cost tier) where each router beats any single-model policy at its target cost point. The router generalizes to OOD tasks (GPQA, AIME) without retraining and remains operational when workers are added/removed (measured via cross-pool transfer eval).

### Key Changes from rl-conductor

| Dimension | rl-conductor | cost-aware-routing |
|-----------|--------------|-------------------|
| **Reward function** | Binary: `is_correct ? 1 : 0` | Cost-aware: `is_correct ? f(cost_normalized) : 0` where f is monotonically decreasing |
| **Worker prompt** | `mask_style='names_and_params'` → bare ordinals (`Model id 0`, `Model id 1`...) | `mask_style='full'` → include per-worker metadata: provider, parameter count, $/1M tokens, p50 latency, qualitative strengths |
| **Eval metric** | Single number (% correct) | Pareto curve (quality vs $/query) + dominated-area metric |
| **Judge for reward** | Regex parser (14% false negative) | LLM-as-judge (Haiku, 96% agreement with Sonnet per verifier-reward T4) |
| **Routing depth** | Multi-step workflows (up to 5 steps) | Single-pick routing (simpler, faster, interpretable) + optional cascade extension |
| **Training objective** | Maximize correct rate | Sweep α ∈ {0.5, 1.0, 1.7, 3.0} (4 runs) — narrow band where per-question routing dominates static policies |

### Relationship to Existing Work

- **rl-conductor Phase 1.6**: Proves GRPO recipe works, provides training infrastructure, worker proxy, rollout capture. This spec inherits those components.
- **verifier-reward**: Provides Haiku-as-judge calibration (precision 0.92, recall 0.14 on hard tasks), LLM-judge cost model ($0.001/question for Haiku).
- **agent-swarm**: Provides multi-model fix-rate matrix (Devstral 88%, Qwen3.5 88%, SERA 64%) and harness transfer findings (precision varies 0.33-1.00 by model-harness pairing). Establishes that model choice + routing matters.
- **FrugalGPT / RouteLLM / USC**: Prior art on learned routing (FrugalGPT: cascade with confidence thresholds; RouteLLM: preference-trained router; USC: robust aggregation). This spec adds cost-awareness and Pareto optimization to the GRPO conductor recipe.

### Prior Art Comparison (per `blueprints/cost-aware-routing/llm_routing_literature_review.md`)

| Prior work | Approach | Cost-aware? | Method | What's novel here |
|---|---|---|---|---|
| FrugalGPT (2305.05176) | Cascade w/ confidence threshold | Post-hoc (cost as outcome, not loss) | Supervised + threshold tuning | We optimize cost in the reward directly |
| RouteLLM (LMSYS, 2024) | Binary router (strong vs weak) | Post-hoc | Preference data classifier | Multi-worker pool, RL not supervised |
| UCCI (2026) | Calibrated uncertainty router | Post-hoc | Conformal prediction over base classifier | Strongest production validation (31% cost cut on 75k queries) — **add as a baseline** |
| GraphRAG-Router (2025-26) | RL router for RAG | **Yes** (curriculum) | RL with cost-shaped reward (formulation undisclosed) | Closest prior. Theirs is RAG-specific; ours is general-purpose |
| AWS Bedrock Intelligent Prompt Routing | Closed-source SaaS | Yes | Unknown | Public benchmark target: ~30% cost reduction |
| Token-level routing (TIDE, RelayLLM) | Per-token model switching | Yes | Different paradigm | Out of scope here (no per-token API on Bedrock); flag as future work |
| Sakana Conductor (2512.04388) | RL-trained meta-LLM, multi-step | No (correctness only) | GRPO | Inherited recipe; add cost dimension |

**Headline gap**: no prior work explicitly optimizes `quality - λ·cost` (or `quality / cost`) as primary RL objective on a heterogeneous Bedrock-class pool. GraphRAG-Router is the closest, but its formulation is undisclosed and the domain is narrow (RAG only).

## Research Questions

1. **Does cost-aware GRPO produce a non-dominated Pareto frontier?** At each budget ∈ [$0.01, $0.05, $0.10, $0.50], does the trained router beat the best single-model policy (always-Haiku, always-Sonnet, always-Opus)?

2. **What is the optimal α sweep granularity?** Train routers at α ∈ {0.5, 1.0, 1.7, 3.0} (4 points). Always-X baselines (see Phase 1 §Pre-flight findings below) showed that α≤0.3 collapses to always-Opus and α≥5.0 collapses to always-Gemma; α ∈ {0.5, 1.0, 1.7, 3.0} is the band where the oracle simulator finds distinct per-question routing patterns. Does each α produce a distinct operating point on the Pareto curve?

3. **Does worker metadata improve routing over bare ordinals?** Compare `mask_style='full'` (rich metadata) vs `mask_style='names_and_params'` (bare ordinals). Measure via cross-seed eval: same pool, different ordinal→worker mapping. If metadata helps, accuracy should transfer; if ordinal-locked, accuracy collapses.

4. **Does the router generalize to OOD tasks?** Train on MATH500 + MMLU + HumanEval + LiveCodeBench + AIME25 (20-question train split) + WildChat-1M filtered (300 real user prompts). Eval zero-shot on GPQA-Diamond + AIME25 (10-question eval split) + WildChat holdout (100). Measure quality degradation and whether cost-routing pattern transfers (cheap-first, escalate on uncertainty). WildChat prompts are open-domain real user queries (LMSYS Chatbot Arena was gated; WildChat-1M is the public substitute); AIME25 is the hardest math available — together they exercise the router's ability to discriminate easy vs hard at inference.

5. **Can a cost-aware router match cascade baselines?** FrugalGPT-style cascade: try Haiku → if confidence < 0.7, escalate to Sonnet → if < 0.9, escalate to Opus. Does learned routing beat or lose to this hand-engineered policy?

6. **How robust is the router to pool changes?** Train on 9-worker pool (see Phase 1). Post-training: (a) remove ord_2 (Qwen3-32B), re-eval; (b) swap ord_8 (Opus 4.7) → ord_8' (Opus 4.6), re-eval. Does accuracy degrade gracefully or collapse?

## Phases

### Phase 1: Single-Pick Routing with Cost-Aware Reward (Weeks 1-3)

**Pre-flight findings (run 2026-05-27)** — informed the Phase 1 design:

1. **Haiku-as-judge agreement on math** (n=50 from rl-conductor v4 iter-074): Haiku 4.5 vs Sonnet 4.6 agree on 49/50 (98%). Haiku is reliable for the cost-aware reward signal. (`results/preflight/judge_agreement_n50.json`)

2. **All 9 Bedrock workers verified** (`results/preflight/worker_probe.json`):
   - Ping + 5-call token probe + 32-way burst all clean.
   - Top-3 cost workers TPM ceiling: 75-101K (15-21× over GRPO requirement).
   - Opus 4.7 deprecates the `temperature` parameter; trainer must omit it.

3. **Always-X baselines on MATH500** (n=50): cost spread is 26× ($0.00022 → $0.00566), tighter than the 60× reference because Opus's actual responses (140 out tokens) are much shorter than the 800-token reference. **At α≤0.3, always-Opus dominates; at α≥5.0, always-Gemma dominates.** Useful α band is {0.5, 1.0, 1.7, 3.0}. (`results/baselines/always_x_math500.json`)

4. **Always-X baselines on AIME25** (n=30): wide spread, 70% Opus vs 0-27% everyone else. The MATH500/AIME25 mix gives the trained router meaningful per-question routing signal: easy questions → Gemma at 78%, hard questions → Opus at 70%. (`results/baselines/always_x_aime25_n30.json`)

5. **Pool re-ranked by published prices**: Gemma-3-27B is now ord_0 (cheapest); Haiku 4.5 is ord_6 (mid-tier). Opus 4.7 priced at $5/$25 per 1M tok (3× cheaper than Opus 4.1).

**Goal**: Train a family of routers (one per α ∈ {0.5, 1.0, 1.7, 3.0}) that optimize `reward = is_correct ? max(1 - α·cost_normalized, -1) : 0`. **Demonstrate per-question routing beats the best static policy** at each α; produce the (cost, accuracy) curve and compare to all 9 Always-X baselines.

**Reward floor (-1)**: Allowing negative rewards on correct-but-expensive rollouts preserves the cost-ordering signal inside GRPO advantage normalization (clipping at 0 saturates the gradient on prompts where every rollout is correct). The floor at -1 bounds catastrophic rollouts so a single Opus pick can't blow up the within-group standard deviation. See plan addendum (`blueprints/cost-aware-routing/plan-addendum-2026-05-27.md`) for the full literature analysis (PPO/Engstrom 2020, Safe-RLHF/Dai 2024, FrugalGPT/Chen 2023, DeepSeekMath/Tülu 3 GRPO ablations).

**Worker pool (9 models, all Bedrock — no self-hosted)**:

Costs are computed per-rollout at 200 input tokens (system prompt + question) + 800 output tokens (CoT). Prices sourced from Anthropic published rates (Opus/Sonnet/Haiku) and AWS Bedrock published rates for the rest (us-west-2, with extrapolations noted in the plan addendum).

| Ord | Model | Bedrock model_id | $/1M in | $/1M out | $/query | Qualitative strengths |
|-----|-------|------------------|--------:|---------:|--------:|----------------------|
| 0 | **Gemma-3-27B-IT** | `google.gemma-3-27b-it` | 0.23 | 0.38 | **$0.00035** | Mid-tier generalist, fast |
| 1 | **gpt-oss-120b** | `openai.gpt-oss-120b-1:0` | 0.15 | 0.60 | **$0.00051** | Cheap, fast, weak reasoning |
| 2 | **Qwen3-32B** | `qwen.qwen3-32b-v1:0` | 0.15 | 0.62 | **$0.00053** | Mid-tier, strong tool calling |
| 3 | **Qwen3-Coder-480B** | `qwen.qwen3-coder-480b-a35b-v1:0` | 0.50 | 1.20 | **$0.00106** | Code specialist, frontier-class |
| 4 | **Mistral Large 3** | `mistral.mistral-large-3-675b-instruct` | 0.50 | 1.50 | **$0.00130** | Strong generalist, multilingual |
| 5 | **DeepSeek V3.2** | `deepseek.v3.2` | 0.62 | 1.85 | **$0.00160** | Strong reasoning, good value |
| 6 | **Haiku 4.5** | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 1.00 | 5.00 | **$0.00420** | Fast Anthropic frontier, weak at hard math |
| 7 | **Sonnet 4.6** | `us.anthropic.claude-sonnet-4-6` | 3.00 | 15.00 | **$0.01260** | Strong frontier, balanced |
| 8 | **Opus 4.7** | `us.anthropic.claude-opus-4-7` | 5.00 | 25.00 | **$0.02100** | Top-tier reasoning, expensive |

All workers are Bedrock-served. Self-hosted endpoints were removed from the pool to eliminate cluster-availability dependencies (g7e/B300 reservations) and simplify the rollout path. **Cost spread: 60×** ($0.00035 → $0.02100). The earlier "300×" target assumed retired Opus 4.1 prices ($75/1M out); Opus 4.7 dropped to $25/1M out, compressing the cost spread substantially.

**Cost normalization**: `cost_normalized = (cost_this_call - min_cost) / (max_cost - min_cost)` where min = $0.00035 (Gemma-3-27B), max = $0.02100 (Opus 4.7). All costs in range [0, 1].

**Note on Opus 4.7 API**: Opus 4.7 deprecates the `temperature` inference parameter; training/eval scripts must omit it from `inferenceConfig` for this worker only. Other workers accept temperature as usual.

**Qualitative strengths** (in worker prompt): "Gemma-3-27B: mid-tier generalist, fastest cheap option" | "gpt-oss-120b: cheap, fast, weak reasoning" | "Haiku 4.5: fast Anthropic frontier, weak at hard math" | "Qwen3-Coder-480B: code specialist, frontier-class" | "Opus 4.7: top-tier reasoning, expensive". This metadata helps the router learn capability-grounded routing rather than position-grounded.

**Training setup**:

| Parameter | rl-conductor Phase 1.6 | cost-aware-routing Phase 1 |
|-----------|------------------------|---------------------------|
| Base model | Qwen2.5-7B-Instruct | **Qwen2.5-7B-Instruct** (proven) |
| Trainer | verl GRPO | **verl GRPO** (proven) |
| Training data | 960 questions (MATH500 + MMLU + LCB + RLPR) | **1,520 questions** (MATH500 train 300 + MMLU 300 + HumanEval 300 + LiveCodeBench 300 + AIME25 train 20 + WildChat-1M filtered 300) |
| Worker pool | 8 models (closed frontier) | **9 models** (all Bedrock; mixed open- and closed-weight) |
| Worker prompt style | `names_and_params` (bare ordinals) | **`full`** (metadata-rich) |
| Reward | Binary: `is_correct ? 1 : 0` | **Cost-aware (floored at -1)**: `is_correct ? max(1 - α·cost_normalized, -1) : 0` |
| Judge | Regex parser (14% FN) | **Haiku-as-judge** (96% agreement) |
| α sweep | N/A (single objective) | **α ∈ {0.5, 1.0, 1.7, 3.0}** (4 training runs in the per-question-routing band; outer points {0.1, 0.3, 5.0} dropped after pre-flight + oracle simulator showed they reduce to always-X) |
| Batch size | 256 = 4 questions × 64 rollouts | **256** (proven) |
| Iterations | 200 | **200** (proven) |
| Learning rate | 1e-6, cosine | **1e-6, cosine** (proven) |
| KL penalty | 0 | **0** (paper-faithful) |
| Max routing depth | 5 steps (multi-step workflows) | **1 step** (single-pick routing) |
| Hardware | p4de.24xlarge (8× A100, spot) | **p5.48xlarge** (8× H100 80GB, spot, us-west-2b) — ~2× tokens/s for +8% price |

**α interpretation** (re-targeted after pre-flight + oracle-router simulation):

| α | Oracle picks (math / aime / wildchat) | Oracle E[reward] | Oracle accuracy | Oracle $/q | Best-static E[reward] |
|---|---|---:|---:|---:|---:|
| **0.5** | Opus / Opus / Qwen-Coder | +0.804 | 84.6% | $0.00404 | always-Opus +0.79 |
| **1.0** | Mistral / Opus / Qwen-Coder | +0.765 | 80.0% | $0.00284 | 3-way tie at +0.78 |
| **1.7** | Gemma / Opus / Qwen-Coder | +0.739 | 78.5% | $0.00254 | always-Gemma +0.74 |
| **3.0** | Gemma / Gemma / Qwen-Coder | +0.632 | 65.4% | $0.00046 | always-Gemma +0.66 |

The outer points {0.1, 0.3, 5.0} were dropped after Always-X baselines + oracle simulator showed:
- α≤0.3: oracle reduces to always-Opus (no per-question routing payoff).
- α≥5.0: oracle reduces to always-Gemma (cost penalty so brutal that even 70% Opus on AIME loses to 13% Gemma).
- α=5.0 oracle is identical to α=3.0 — no new information.

The remaining 4 alphas are where the trained router's per-question conditioning has measurable room to outperform any static policy. **This is no longer a "produce a 5-point Pareto curve" experiment — it's a "demonstrate per-question routing beats the best static policy at 4 narrow operating points" experiment.**

The oracle gap (oracle E[r] − best-static E[r]) ranges from +0.014 (α=0.5) to +0.025 (α=1.0). That's the maximum the trained router could possibly close; the **3pp success threshold corresponds to capturing >75% of the oracle gap** at α=1.0, which is the most discriminating regime.

**Eval sets** (held-out, zero-shot):

| Set | Size | Purpose |
|-----|------|---------|
| MATH500 test | 100 | In-distribution math |
| MMLU test | 100 | In-distribution factual |
| HumanEval test | 100 | In-distribution code |
| **AIME25 eval split** | 10 | In-distribution hard math (20 train / 10 eval split) |
| **GPQA-Diamond** | 100 | **OOD reasoning** |
| **WildChat holdout** | 100 | **OOD real user prompts** from WildChat-1M (LLM-as-judge scored, no gold; LMSYS Chatbot Arena was gated, this is the public substitute) |

**Comparison baselines** (on same eval sets):

| Baseline | Cost | Implementation |
|----------|------|---------------|
| Always-Gemma | $0.00035/q | Static routing to ord_0 (Gemma-3-27B) |
| Always-gpt-oss-120b | $0.00051/q | Static routing to ord_1 |
| Always-Haiku | $0.00420/q | Static routing to ord_6 (Haiku 4.5) |
| Always-Sonnet | $0.01260/q | Static routing to ord_7 (Sonnet 4.6) |
| Always-Opus | $0.02100/q | Static routing to ord_8 (Opus 4.7) |
| Random routing | ~$0.035/q | Uniform random over 9 workers |
| **FrugalGPT cascade** | Variable | Haiku → (if conf < 0.7) → Sonnet → (if conf < 0.9) → Opus |
| **RouteLLM (simulated)** | Variable | Preference-trained router from LMSYS Chatbot Arena data (use published routing matrix) |
| **UCCI** (2026) | Variable | Conformal-prediction uncertainty router. Reproduce per UCCI paper as supervised baseline. |
| **AWS Bedrock IPR** | Variable | Closed-source. Run the same eval through their service if budget allows; report black-box result for context. |

**Metrics**:

1. **Pareto curve**: For each α, plot (cost, quality) on same axes. A non-dominated frontier means no baseline strictly dominates any router.
2. **Dominated area**: Area under the Pareto curve. Lower = better (less cost for same quality). Compare router family vs baselines.
3. **Per-α accuracy**: Correct rate on eval sets (MATH, MMLU, HumanEval, GPQA, AIME).
4. **Routing histogram**: Distribution of worker calls by task type (math vs code vs reasoning). Does α=3.0 learn "cheap-first, escalate on hard"?
5. **Cross-seed transfer**: Re-eval α=1.7 router with `pool_seed=42` (different ordinal→worker mapping). If metadata helps, accuracy should hold; if ordinal-locked, accuracy collapses.

**Exit criteria** (re-targeted around per-question routing, not Pareto-spread):

1. **Beat the best static policy at all 4 α values**: at each α ∈ {0.5, 1.0, 1.7, 3.0}, the trained router's mean reward exceeds the best Always-X baseline at that α by ≥3pp. This is the experiment's core scientific claim — that learned per-question routing produces value beyond any pool-level static choice.
2. **Pareto-frontier non-dominance**: at least 3 of 4 α routers occupy non-dominated points on the (cost, accuracy) curve relative to the 9 Always-X baselines.
3. **Capability-grounded routing on the cost/quality split** (thresholds match the oracle simulator picks; clearing them demonstrates the router conditions on question type, not just the α prior):
   - α=0.5: Opus ≥40% on AIME25 (where Opus's reasoning advantage justifies its cost), Qwen-Coder ≥40% on WildChat.
   - α=1.0: Mistral or Qwen-Coder ≥40% on MATH500, Opus ≥40% on AIME25, Qwen-Coder ≥40% on WildChat.
   - α=1.7: Gemma ≥40% on MATH500, Opus ≥40% on AIME25, Qwen-Coder ≥40% on WildChat.
   - α=3.0: Gemma ≥60% across all eval sets (cost dominates).
4. **Cross-seed eval**: α=1.7 router retains ≥80% of seed=17 accuracy when re-evaled with seed=42 (proves metadata matters, ordinal-lock is broken).
5. **OOD transfer**: GPQA + LMSYS-holdout mean accuracy ≥ 70% of in-distribution MATH accuracy at the same α (validates task generalization).

**Estimated cost** (revised after pre-flight findings + p5 switch):

| Item | Cost |
|------|------|
| p5.48xlarge spot training (4 α × ~28 hr × $17.25/hr) | ~$1,932 |
| Worker calls during training (4 α × 200 iter × 256 rollouts × ~$0.0036/q) | ~$737 |
| Haiku-as-judge reward calls (same volume × $0.0042/q) | ~$860 |
| Eval (4 routers × 510 eval questions × ~$0.005 avg/q) | ~$10 |
| Baseline policies (already measured for MATH500 + AIME25; remaining: 4 sets × 9 workers × ~$0.005/q) | ~$80 |
| WildChat data prep + 100-question holdout judging | ~$5 |
| Mid-training calibration + iter-0 gates | ~$15 |
| **Total Phase 1** | **~$3,652** |

Phase 1 budget revised from ~$7,800 (original spec, 5 α × inflated worker prices on p4de) to **~$3,650** (4 α × measured Bedrock prices on p5).

### Phase 2: Cascade Routing Extension (Week 4, conditional)

**Goal**: Train a cascade router that emits ordered worker sequence: try cheap model → if confidence < threshold, escalate to expensive. Compare to Phase 1 single-pick routing.

**Prerequisites**: Phase 1 shows α=3.0 router produces measurable cost savings over the best Always-X policy at matched accuracy (validates that single-pick routing has Pareto room to improve).

**Architecture change**:

| Dimension | Phase 1 (single-pick) | Phase 2 (cascade) |
|-----------|----------------------|------------------|
| Router output | `worker_id` (single int) | `[worker_id_1, worker_id_2, ..., worker_id_N]` (ordered sequence) |
| Execution | Call `worker_id`, return response | Call `worker_id_1` → if `confidence(response_1) < threshold`, call `worker_id_2` → ... |
| Termination | After first call | When confidence ≥ threshold OR sequence exhausted |
| Reward | `is_correct ? max(1 - α·cost_1, -1) : 0` | `is_correct ? max(1 - α·Σcost_i, -1) : 0` (sum over cascade) |

**Confidence estimation**: Use worker's output probability or response length as proxy. Alternatively, train a small confidence predictor (BERT-style classifier on `(question, response)` → confidence) using Phase 1 rollout data.

**Training setup**: Same as Phase 1 (1,520 questions including AIME25 + WildChat, 9-worker pool, GRPO, 200 iters) but with cascade-aware reward. Train 3 cascade routers: α ∈ {0.5, 1.0, 1.7, 3.0} (matches Phase 1).

**Comparison**: Phase 1 single-pick router vs Phase 2 cascade router at matched α. Does cascade improve quality at same cost, or just add latency?

**Exit criteria**:

1. Cascade router (α=1.7) beats single-pick router (α=1.7) by ≥3pp quality at matched cost.
2. Cascade length distribution is non-degenerate (not always 1-step or always 3-step).
3. Per-question cost variance is higher than single-pick (validates adaptive escalation).

**Estimated cost**: ~$4,500 (3 α values × $1,500/run) + ~$200 eval.

### Phase 3: Pool Robustness (Week 5, conditional)

**Goal**: Measure router robustness to pool changes (worker removal, worker substitution).

**Prerequisites**: Phase 1 completes with cross-seed transfer ≥80% (proves metadata matters).

**Tests**:

| Test | Change | Expected behavior |
|------|--------|------------------|
| **Worker removal** | Remove ord_2 (Qwen3-32B) from pool, re-eval α=1.7 router | Accuracy degrades gracefully (within 5pp) as router redistributes load to ord_1 and ord_3 |
| **Worker substitution** | Swap ord_8 (Opus 4.7) → ord_8' (Opus 4.6, similar cost/quality) | Accuracy holds within 3pp (metadata match → routing transfers) |
| **Worker upgrade** | Swap ord_6 (Haiku 4.5) → ord_6' (Haiku 4.6 if available, same cost, better quality) | Accuracy improves ≥2pp (router benefits from better worker without retraining) |

**Exit criteria**: At least 2 of 3 tests pass (accuracy within tolerance). If all fail, metadata-rich prompts are insufficient for pool robustness — architectural changes (learned worker embeddings, dynamic pool discovery) required.

**Estimated cost**: ~$25 (3 tests × 510 questions × ~$0.005 avg/q × 3 pool variants).

### Phase 4: Comparison to Agent-Swarm Baselines (Week 6, optional)

**Goal**: Apply the best cost-aware router (α=1.0 or α=3.0) to SWE-bench Lite (50-issue subset) and compare to agent-swarm 8-harness ensemble (36% pass rate).

**Setup**: Treat SWE-bench issues as router inputs. Router picks a worker (harness + model pairing) to generate the patch. Gold eval via existing Docker pipeline.

**Comparison**:

| System | Pass rate | $/issue | $/resolved |
|--------|-----------|---------|-----------|
| 8-harness ensemble (agent-swarm Phase 2b) | 36% (18/50) | ~$0.50 | ~$1.39 |
| Cost-aware router (α=1.0) + worker pool | TBD | TBD | TBD |

**Exit criteria**: Router pass rate ≥ 30% (within 6pp of ensemble) while $/resolved < $0.70 (50% cost reduction).

**Estimated cost**: ~$50 (50 issues × ~$0.02 avg routing + worker call × 5 attempts for tuning).

## Components

### 1. Compute

- **Training (Phase 1-2)**: p5.48xlarge (8× H100 80GB, spot, us-west-2b / usw2-az2). H100s give ~2× rollout throughput vs p4de A100s for ~+8% price; mature NVLink 4 / NVSwitch stack avoids Blackwell-era NCCL pitfalls. Checkpoint to S3 every 25 iters for spot resilience.
- **Inference for all 9 workers**: Bedrock API in us-west-2 (no self-hosted endpoints).
- **Judge inference**: Haiku 4.5 via Bedrock. Pre-flight on 50 MATH500 (rl-conductor v4 iter-074 rollouts) measured **98% agreement with Sonnet 4.6** on math, 70% with regex parser — confirming Haiku-as-judge is reliable for the cost-aware reward signal.

### 2. Codebase

- **Source**: `domains/autoresearch/blueprints/cost-aware-routing/` (new blueprint, inherits from rl-conductor).
- **Reuse from rl-conductor**:
  - `trainer_v3.py` (verl GRPO trainer)
  - `worker_proxy_v2.py` (maps ordinals to backends, already tracks $/1M tok in `WorkerConfig`)
  - `rollout_capture.py` (JSONL logging per iteration)
- **New files**:
  - `cost_reward.py` — Cost-aware reward function: `is_correct(response, gold, judge_model) ? max(1 - alpha * cost_normalized, -1) : 0`
  - `metadata_prompt.py` — Worker prompt generator with `mask_style='full'` (includes qualitative strengths, $/1M tok, latency)
  - `pareto_eval.py` — Eval script that plots Pareto curve (cost vs quality) for all α routers + baselines
  - `cascade_router.py` — Phase 2 cascade extension (ordered worker sequence, confidence-based escalation)
- **Fixed files** (agent must NOT edit):
  - Eval datasets (MATH500 test, MMLU test, HumanEval test, GPQA, AIME25)
  - Worker pool config (9 models, costs, metadata)
  - Gold answers for eval

### 3. Experiment Protocol

- **Primary metric**: Dominated area under Pareto curve (cost vs quality). Lower = better.
- **Secondary metrics**: Per-α accuracy (MATH, MMLU, HumanEval, GPQA, AIME), routing histogram (worker call distribution), cross-seed transfer (accuracy retention with seed=42).
- **Training budget**: 200 iterations per α (proven from rl-conductor Phase 1.6).
- **Eval budget**: 510 questions per router (MATH 100 + MMLU 100 + HumanEval 100 + GPQA 100 + AIME25 eval 10 + WildChat holdout 100).
- **Logging**: Per-iteration JSONL with mean reward, cost distribution, worker histogram, format-failure rate. Per-rollout: `(question, worker_id, response, cost, is_correct, reward)`.
- **Checkpointing**: S3 every 25 iters to `s3://agent-aiops-research/cost-aware-routing/checkpoints/{alpha}/iter-{n}/`.

### 4. Networking

- Training node SSH: `ssh -i ~/.ssh/g7e-bench.pem ec2-user@<p5-ip>` (TBD on launch).
- Worker endpoints: VPC peering or public with token auth.
- Bedrock API: HTTPS, token auth via env vars.

### 5. Storage

- **Training data**: `s3://agent-aiops-research/cost-aware-routing/data/` — 1,520 train questions (includes AIME25 train split + LMSYS Chatbot Arena 300) + eval sets (510 questions).
- **Checkpoints**: `s3://agent-aiops-research/cost-aware-routing/checkpoints/{alpha}/iter-{n}/`.
- **Logs**: `s3://agent-aiops-research/cost-aware-routing/logs/{alpha}/` — JSONL per iteration.
- **Results**: Blueprint-local `domains/autoresearch/blueprints/cost-aware-routing/results/`.

## Success Criteria

### Phase 1: Single-Pick Routing

1. **Beats best static policy at α ≤ 1.7**: at each α ∈ {0.5, 1.0, 1.7}, the trained router's mean reward (on the held-out eval mix) exceeds the best Always-X baseline (Always-Qwen-Coder-480B / ord_3) by ≥6pp. The oracle gaps measured on rollout data are +15pp (α=0.5), +12pp (α=1.0), +11pp (α=1.7); the 6pp threshold corresponds to capturing ≥50% of the oracle gap. (See `results/baselines/oracle_router.json`.)
   - At α=3.0 the oracle gap is only +0.2pp (cost domination collapses oracle ≈ always-Gemma); this α is included in the sweep for completeness but is not expected to materially beat best-static.
2. **Pareto non-dominance**: at least 3 of 4 α routers occupy non-dominated points on the (cost, accuracy) curve relative to the 9 Always-X baselines.
3. **Capability-grounded routing on the cost/quality split** (matches oracle picks):
   - α=0.5 router picks Opus ≥40% on AIME25, ≥40% Qwen-Coder on WildChat;
   - α=1.0 router picks Mistral or Qwen-Coder ≥40% on MATH500, Opus ≥40% on AIME25;
   - α=3.0 router picks Gemma ≥60% across all eval sets;
   - on WildChat at every α, ≥40% of picks go to Qwen3-Coder-480B.
4. **Metadata matters**: cross-seed eval (α=1.0, seed=42) retains ≥80% of seed=17 accuracy.
5. **OOD generalization**: GPQA + WildChat-holdout accuracy ≥ 70% of in-distribution MATH accuracy at the same α.
6. **Judge-based reward works**: LLM-judge reward converges faster than rl-conductor's regex parser (measured via iters-to-first-correct).

### Phase 2: Cascade Routing (conditional)

1. Cascade router (α=1.7) beats single-pick router (α=1.7) by ≥3pp quality at matched cost.
2. Cascade length distribution is non-degenerate (mean length 1.3-2.5 steps, not always 1 or 3).
3. Per-question cost variance > single-pick (validates adaptive escalation).

### Phase 3: Pool Robustness (conditional)

1. Worker removal: accuracy degrades ≤5pp when ord_2 removed.
2. Worker substitution: accuracy holds within 3pp when ord_8 swapped.
3. Worker upgrade: accuracy improves ≥2pp when ord_6 upgraded.

### Phase 4: SWE-bench Transfer (optional)

1. Router pass rate ≥ 30% on SWE-bench Lite (50 issues).
2. $/resolved < $0.70 (50% cost reduction vs 8-harness ensemble).

### Negative Results (Still Valuable)

- **Pareto collapse**: All α routers converge to same operating point → cost-aware reward doesn't produce diverse strategies. This would indicate the α sweep is too narrow or the pool lacks cost diversity. Publishable as "cost-aware GRPO requires X orders of magnitude cost spread."
- **Ordinal lock persists**: Cross-seed eval collapses even with metadata-rich prompts → architectural change required (learned worker embeddings, not prompt-based metadata). Publishable as "prompt-based transfer is insufficient for pool robustness."
- **Cascade doesn't beat single-pick**: Adds latency without quality gain → FrugalGPT-style cascades are overfit to specific cost structures. Publishable as "learned routing beats hand-engineered cascades in heterogeneous pools."

## Non-Requirements

- **Trinity reproduction** (arxiv:2512.04695) — separate spec.
- **Multi-step workflows** — Phase 1 is single-pick routing only. Cascade (Phase 2) is ordered sequence, not DAG.
- **Fugu commercial features** — not relevant.
- **Custom GRPO implementation** — use verl or trl off the shelf.
- **Distributed multi-node training** — single p5 (8× H100) is sufficient for 7B GRPO.
- **Profile-evolution loop** (Phase 1.7 from rl-conductor spec) — deferred. Phase 1 uses static profiles; online updates are future work.

## Known Limitations

### Training & Infrastructure

- **Spot reclaim**: p5 spot can be reclaimed mid-training. Checkpoint every 25 iters; design resume path. See `feedback_grpo_resume_state.md` for RNG persistence + checkpoint-reload lessons. Note rl-conductor's earlier finding: cross-region spot resume can collapse format quality (ISDD/trust-region overshoot) — keep all 5 α runs in us-west-2b.
- **Closed API rate limits**: 9 workers × 64 rollouts = bursty load on Bedrock. May need request batching or tier upgrade.
- **Worker variance**: Closed models drift over time (silent updates). Snapshot worker model identifiers + costs per training run.
- **Bedrock-only pool**: All 9 workers are Bedrock-served. Removes self-hosted cluster dependencies but introduces Bedrock TPM/RPM as the sole capacity constraint. 9 workers × 64 rollouts/iter = bursty. Pre-budget per-model TPM headroom (especially Opus 4.7 and Sonnet 4.6) before committing to 200-iter run; tier upgrade may be required.

### Reward & Evaluation

- **LLM-judge calibration**: Haiku-as-judge agrees 96% with Sonnet on code patches (verifier-reward T4), but math/reasoning calibration is less proven. Pre-flight: measure Haiku-Sonnet agreement on 100 MATH500 questions before committing to judge-based reward.
- **Cost model assumptions**: Costs are current Bedrock rates (2026-05). Relative ordering matters more than absolute values, but large shifts (e.g., Opus $75 → $30) would require retraining to re-optimize Pareto frontier.
- **Reward hacking**: Cost-aware reward `max(1 - α·cost, -1)` with no KL is aggressive. Watch for format-degenerate solutions (router always picks ord_0 at α=5.0, regardless of question difficulty). Add format check tightening if needed.

### Generalization & Transfer

- **Ordinal lock risk**: If cross-seed eval collapses, metadata-rich prompts are insufficient. Profile-aware training (rl-conductor Phase 1.7-A) or learned worker embeddings required. Current spec assumes metadata helps but doesn't guarantee it.
- **Task distribution shift**: Training on MATH + MMLU + HumanEval may not transfer to SWE-bench (Phase 4). Code patches are 30+ turn trajectories; routing is 1-shot decision. If Phase 4 fails, it's a domain mismatch, not a routing failure.
- **Pool composition matters**: 9-worker pool spans 300× in cost ($0.001 → $0.300). If Bedrock model availability changes mid-experiment (deprecation, throttling), Pareto optimization may degenerate. Diversity is a precondition for the experiment.

### Lessons from rl-conductor Phase 1.6

- **Log raw artifacts**: Capture full rollouts (question, worker_id, response, cost, is_correct, reward) per iteration, not just aggregated metrics. Post-hoc re-grading is impossible without raw responses. See `feedback_log_raw_artifacts.md`.
- **RNG persistence**: GRPO trainer must save/restore RNG state on checkpoint resume, or reward distributions shift across spot interruptions. See `feedback_grpo_resume_state.md`.
- **Brand bias in base model**: Qwen2.5-7B has ~52% Opus skew when picking workers from pool (rl-conductor Phase 1.6 finding). If this persists, it biases the GRPO reward signal toward expensive models. **Adopted mitigation (Phase 1)**: stack (a)+(b) — (a) prepend balanced 9-shot examples in the system prompt, one per worker, showing the worker being correctly selected for a question that suits its strengths; (b) iter-0 histogram diagnostic gate — run 256 rollouts at iter 0 before consuming training compute and verify each worker is picked between 5-20% (target 11% ± 6pp). If Opus > 25%, regenerate few-shot examples or shuffle ordinal mapping. Costs ~$5, prevents wasting $1,500 on a contaminated training run. Hold (c) histogram-deviation regularizer as a Phase 1.5 fallback if (a)+(b) is insufficient.

## Open Questions

These are exploratory directions, not committed phases. Promotion to phases requires re-planning with fresh evidence from Phase 1-3 results.

### OQ1: Multi-Objective Pareto Optimization

**Question**: Instead of sweeping α manually (5 training runs), can we train a single router with multi-objective GRPO that learns the entire Pareto frontier in one run?

**Approach**: Replace scalar reward `(1 - α·cost)` with vector reward `[quality, -cost]`. Use multi-objective RL (e.g., Pareto-DQN, MO-TRPO) to learn a set of non-dominated policies. At inference, provide α as input to the router → router outputs worker_id conditioned on α.

**Why exploratory**: Multi-objective RL is less mature than scalar GRPO. Implementation complexity is higher. Single α-sweep (Phase 1) is proven; multi-objective is speculative.

### OQ2: Learned Confidence Predictor for Cascade

**Question**: Phase 2 cascade uses heuristic confidence (output probability or response length). Can we train a small confidence predictor (BERT-style classifier on `(question, response)` → confidence) that improves cascade escalation decisions?

**Approach**: Use Phase 1 rollout data (1,200 questions × 9 workers × 64 rollouts = ~691K labeled examples of `(question, worker_id, response, is_correct)`). Train a binary classifier to predict `is_correct`. At inference, cascade escalates when `P(correct) < threshold`.

**Why exploratory**: Requires Phase 1 to produce high-quality rollout data. Confidence calibration (ECE) is critical; miscalibrated predictor could harm cascade. Only promote if Phase 2 shows cascade beats single-pick — otherwise confidence predictor is solving the wrong problem.

### OQ3: Profile-Aware Training (Cross-Pool Generalization)

**Question**: If cross-seed eval (Phase 1) shows ordinal lock persists even with metadata-rich prompts, can profile-aware training (rl-conductor Phase 1.7-A) fix it?

**Approach**: Bake worker profiles into system prompt at training time, but **shuffle ordinal→worker mapping every N iterations** (e.g., N=25). Forces the router to learn "read profile, route by capability" rather than "memorize ord_3 is strong at code."

**Why exploratory**: Depends on Phase 1 cross-seed eval result. If metadata already enables transfer (≥80% accuracy retention), profile-aware training is unnecessary. Only promote if cross-seed eval fails (<70% retention).

### OQ4: Routing for Long-Context Tasks (SWE-bench Full)

**Question**: Does cost-aware routing transfer to long-horizon tasks (SWE-bench Lite 300 issues)?

**Approach**: Use Phase 1 best router (α=1.0 or α=3.0) on SWE-bench Lite. Instead of routing to a single worker, route to a (harness, model) pairing. Compare to agent-swarm 8-harness ensemble (36% pass rate, $0.50/issue).

**Why exploratory**: SWE-bench is 30+ turn trajectories; routing is 1-shot decision. The router chooses a harness but doesn't participate in the trajectory. This is architecturally different from MATH/MMLU (1-shot Q&A). Only promote if Phase 4 quick test (50 issues) shows promise.

## Relationship to Other Specs

- **rl-conductor-repro**: Provides training infrastructure (verl GRPO, worker proxy, rollout capture), proves recipe works. This spec pivots the objective (quality-only → cost-aware) while reusing components.
- **verifier-reward**: Provides Haiku-as-judge calibration (96% agreement with Sonnet on code, precision 0.92), LLM-judge cost model.
- **agent-swarm**: Provides multi-model baselines (fix rates, pass rates, precision by model-harness pairing). Establishes that model choice + routing matters.
- **coderforge-eval**: Parallel experiment on training cost. Pool is intentionally Bedrock-only here; CoderForge-derived weights are out of scope unless deployed via Bedrock Custom Model Import.

## Future: Learned Worker Embeddings

If Phase 3 pool robustness tests fail (accuracy collapses when workers change), metadata-rich prompts are insufficient. Next step: **learned worker embeddings**.

**Approach**: Replace ordinal IDs with learned embeddings. Each worker has a vector `e_w ∈ R^d` (d=64 or 128). Router's policy is conditioned on `[question_embed, e_w]` for each candidate worker. At training time, worker embeddings are learned jointly with policy weights. At inference, new workers can be hot-added by embedding their profiles (via few-shot or zero-shot embedding prediction from metadata).

**Why deferred**: Requires architectural change (router must consume embeddings, not ordinals). Only needed if prompt-based transfer fails. If Phase 1 cross-seed eval succeeds (≥80% retention), learned embeddings are unnecessary.

---

> **Note**: Operational artifacts (lessons learned, training logs, Pareto curves, routing histograms) belong in `domains/autoresearch/blueprints/cost-aware-routing/`, not in this spec.
