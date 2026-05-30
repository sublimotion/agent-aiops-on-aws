# Plan Addendum — Cost-Aware Routing (2026-05-27)

> **STATUS NOTICE (2026-05-28)**: This addendum captured pre-launch decisions for the GRPO 7B router framing. That framing was tested in 3 smoke runs and abandoned after the per-question oracle gap could not be closed. **Active plan**: see `phase1-redesign-2026-05-28.md` in this same directory (regime-A pivot to ModernBERT classifier + closed-form policy). Negative result writeup: `domains/autoresearch/specs/grpo-router-negative-result.md`.

Pre-launch decisions made before kicking off Phase 1 RALPH loop. Records the deltas from the original spec.

## 1. Worker pool — all Bedrock (no self-hosted), ranked by published prices

**Original spec**: 6 Bedrock + 3 self-hosted (Qwen3.5-32B on g7e, Qwen3.5-Coder-480B on B300, Kimi K2.6 on B300), with $/query estimates from outdated rate cards.

**Adopted**: 9 Bedrock workers, ordered by ascending published $/query at 200 input + 800 output tokens (realistic GRPO rollout shape).

| Ord | Model | model_id | $/1M in | $/1M out | $/query | Source |
|-----|-------|----------|--------:|---------:|--------:|--------|
| 0 | Gemma-3-27B-IT | `google.gemma-3-27b-it` | 0.23 | 0.38 | $0.00035 | Bedrock us-west-2 |
| 1 | gpt-oss-120b | `openai.gpt-oss-120b-1:0` | 0.15 | 0.60 | $0.00051 | Bedrock (extrapolated; Sydney + safeguard variant) |
| 2 | Qwen3-32B | `qwen.qwen3-32b-v1:0` | 0.15 | 0.62 | $0.00053 | Bedrock (extrapolated from Sydney) |
| 3 | Qwen3-Coder-480B | `qwen.qwen3-coder-480b-a35b-v1:0` | 0.50 | 1.20 | $0.00106 | Bedrock proxy (Qwen3 Coder Next) |
| 4 | Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` | 0.50 | 1.50 | $0.00130 | Bedrock us-west-2 |
| 5 | DeepSeek V3.2 | `deepseek.v3.2` | 0.62 | 1.85 | $0.00160 | Bedrock us-west-2 |
| 6 | Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 1.00 | 5.00 | $0.00420 | Anthropic published |
| 7 | Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 3.00 | 15.00 | $0.01260 | Anthropic published |
| 8 | Opus 4.7 | `us.anthropic.claude-opus-4-7` | 5.00 | 25.00 | $0.02100 | Anthropic published |

**Why**: removes cluster-availability dependencies (g7e/B300 reservations), simplifies the rollout path to a single API.

**Surprise from probe**: published prices are dramatically different from the original spec assumptions:
- **Opus 4.7 is $5/$25, not $15/$75** (Opus 4.1 prices). The latest Anthropic generations cut pricing ~3× from the 4.1 series.
- **Cost spread is 60×, not 300×** as the spec claimed. The "300× spread" was based on retired Opus 4.1 prices.
- **Cost ranking changed**: Gemma-3-27B is now ord_0 (cheapest), not gpt-oss-120b. Haiku 4.5 dropped from ord_2 → ord_6 (cost-wise it's mid-tier, not cheap-tier).
- Implication: Opus's quality-cost tradeoff is now far more favorable than the spec assumed. The router may converge to "Opus for everything" more readily; we should monitor that during training.

**Cost normalization**: min = $0.00035, max = $0.02100. `cost_normalized ∈ [0, 1]`.

**Opus 4.7 API quirk**: deprecates the `temperature` param. Probe captured this; training scripts must omit `temperature` from `inferenceConfig` for Opus 4.7 only (other workers accept it normally).

## 2. Region + instance — us-west-2b on p5.48xlarge

Spot capacity in us-west-2b (us-west-2-az2):
- p4de.24xlarge: ~$15.97/hr
- **p5.48xlarge: ~$17.25/hr** ← adopted default for Phase 1
- p5en.48xlarge: ~$26.60/hr (held in reserve)

**Why p5 over p4de**: only +8% on price, but the H100s deliver ~2× tokens/sec on 7B models vs A100 (3 TB/s HBM3 bandwidth vs 1.5 TB/s on p4de) and run on a mature NVLink 4 / NVSwitch stack with no Blackwell NCCL pitfalls (H100 sm_90 is a known-good target). Expected to cut per-α wall-time from ~50hr → ~25-30hr, dropping Phase 1 compute from ~$4K to ~$2.5K.

p5en held in reserve for Phase 2 (cascade routing) or if rollout batch grows to 128 and exceeds H100 80GB headroom.

S3 buckets and worker calls all in us-west-2 to avoid inter-region egress.

## 3. Pre-flight: Haiku-as-judge calibration on math (DONE)

The spec (line 344) flagged that the cited 96% Haiku-Sonnet agreement is from code patches, not math. Phase 1 reward signal depends on Haiku-as-judge being reliable on MATH500-style problems.

**Pre-flight result (n=50, rl-conductor v4 iter-074 rollouts as test set)**:
- Haiku 4.5 vs Sonnet 4.6: **49/50 = 98% agreement**
- Haiku vs regex parser: 36/50 = 72% (parser severely undercounts — ~28pp false-negative rate)
- Sonnet vs regex parser: 37/50 = 74%
- Haiku correct rate: 98%; Sonnet correct rate: 96%
- Confusion (Haiku, Sonnet): TT=48, TF=1, FT=0, FF=1 — Haiku is one notch more lenient

Cost: $0.23. Output: `results/preflight/judge_agreement_n50.json`.

**Verdict**: Haiku-as-judge is reliable for math reward. The 28pp parser FN gap confirms why we cannot use the regex extractor for reward.

**Mid-training calibration check**: re-run 50 training rollouts through Sonnet at Phase 1 iter 50 of each α run. Expect ≥95% agreement; if drift > 5pp, swap reward judge to Sonnet (4× cost).

## 4. Brand-bias mitigation — stack (a)+(b)

**Background**: Qwen2.5-7B picks Opus ~52% of the time at iter 0 in an 8-worker pool labeled by name (memory: `project_qwen25_brand_bias.md`). With cost-aware reward and Haiku-as-judge (no parser FN to mask it), this would contaminate early GRPO advantages.

**Adopted**:
- **(a) Balanced 9-shot examples**: prepend 9 system-prompt examples, one per worker, showing the worker being correctly selected for a question that suits its strengths. E.g., gpt-oss-120b → trivial arithmetic, Qwen3-Coder-480B → code generation, Opus 4.7 → multi-step proof.
- **(b) Iter-0 histogram gate**: before consuming any training compute, run 256 rollouts at iter 0 and verify each worker is picked between 5–20% (target 11% ± 6pp). If Opus > 25%, regenerate few-shot examples or shuffle ordinal mapping. Cost ~$5; prevents wasting $1,500 on a contaminated run.

**Held in reserve**:
- **(c) Histogram-deviation regularizer** `−λ·KL(p_router || uniform)`. Adds a hyperparameter that interacts with α and would confound Pareto curve interpretation. Promote to Phase 1.5 only if (a)+(b) fails the gate.

## 5. Reward sign — floor at −1, drop α=10

**Original spec**: `reward = is_correct ? (1 - α·cost_normalized) : 0`, α sweep ∈ {0.1, 0.3, 1.0, 3.0, 10.0}.

**Adopted (initial pivot)**: `reward = is_correct ? max(1 - α·cost_normalized, -1) : 0`, α sweep ∈ {0.1, 0.3, 1.0, 3.0, 5.0}.

**Adopted (after AIME25 + WildChat baselines + oracle simulator on rollout data)**: same reward function, **α sweep ∈ {0.5, 1.0, 1.7, 3.0}** (4 runs).

Iterations:
1. After MATH500-only baselines suggested {1.0, 1.7, 3.0}: 3 runs in the per-question-routing band.
2. After AIME25 baselines added (huge spread, only Opus reaches 70%): same 3-run sweep, much wider underlying Pareto.
3. After WildChat baselines (Haiku 90%, Opus 88%; gpt-oss 8% catastrophic) + oracle simulator on actual 130 rollouts at each α: re-added α=0.5, the most accuracy-friendly point with a meaningful per-question pattern (Opus on hard, Qwen-Coder-480B on open-domain). Final sweep is **{0.5, 1.0, 1.7, 3.0}** (4 runs).

Oracle gaps measured on rollout data (oracle E[r] − best-static E[r], 130 questions):

| α | oracle | best-static | gap |
|--:|------:|-----------:|----:|
| 0.5 | +0.804 | +0.653 (always-Qwen-Coder) | **+0.151** |
| 1.0 | +0.765 | +0.645 (always-Qwen-Coder) | **+0.120** |
| 1.7 | +0.740 | +0.634 (always-Qwen-Coder) | **+0.106** |
| 3.0 | +0.632 | +0.630 (always-Gemma) | +0.002 (collapses) |
| 5.0 | +0.617 | +0.630 (always-Gemma) | -0.013 (oracle worse) |

**Surprises from the oracle simulator**:
- Always-Qwen-Coder-480B (ord_3) is the cost-aware best static policy at α ≤ 1.7. Not the highest-accuracy worker (66% vs Opus's 86%) but its low cost compensates.
- Oracle picks Qwen-Coder-480B (not Haiku) on WildChat at every α. Haiku wins WildChat on raw accuracy (90% vs 84%) but loses on cost-weighted reward (Haiku is 5× more expensive per call).
- At α=3.0, even 70%-accurate Opus on AIME25 *loses* to 13%-accurate Gemma (cost penalty so brutal that 0.13 × +1.0 > 0.70 × −1.0).
- α=5.0 oracle is *worse* than always-Gemma. Confirmed correctly dropped.

This is not a "produce a 5-point Pareto curve" experiment; it's a **"demonstrate per-question routing captures ≥50% of the oracle gap at α ∈ {0.5, 1.0, 1.7}"** experiment. Phase 1 cost reduced from 5 runs → 4 runs (~$480 compute saved vs original spec).

**Why floor at −1, not 0** (clipping at 0 was the alternative):

1. **Clipping at 0 saturates the GRPO gradient**. GRPO normalizes advantage within a 64-rollout group: `A = (r - mean(r)) / std(r)`. On easy prompts where every rollout is correct, clipping at 0 collapses rewards into a narrow band in `[0, 1]` and α loses effect at high values. Engstrom et al. ("Implementation Matters in Deep Policy Gradients", ICLR 2020) make this argument generally for PPO.
2. **Allowing negatives preserves cost ordering**. With `r = max(1 − α·cost, −1)`, even at α=5 the cost ranking of correct rollouts is preserved (gpt-oss-120b correct → 0.998, Opus correct → −0.5).
3. **Constrained-RL prior art prefers vector reward + Lagrangian** (Safe-RLHF / Dai et al. 2024) over scalar clipping. We use scalar for engineering simplicity but keep the spirit (no information loss in ordering).
4. **Floor at −1 (not unbounded)** prevents a single Opus pick from blowing up within-group standard deviation. With α=5 and Opus (cost_norm=1), `1 − 5·1 = −4` would have huge std impact in a group of 64; floor at −1 caps it.

**Why drop α=10**: at α=10 with floor=−1, the only sensible learnable policy is "gpt-oss-120b for everything and accept wrong answers" — a degenerate Pareto point, not a useful one. **α=5** still produces a strongly cost-biased operating point (`1 − 5·cost_norm < 0` for any worker above ord_4 / Qwen3-32B) without the degenerate regime, and keeps 5 distinct training runs.

## 6. Estimated cost (revised down with measured Bedrock prices)

| Item | Cost |
|------|------|
| Pre-flight: Haiku-Sonnet judge calibration (n=50) | ~$0.25 (DONE) |
| Pre-flight: 9-worker probe (ping + token + burst) | ~$0.10 (DONE) |
| Pre-flight: Always-X baselines on MATH500 + AIME25 (50+30 questions × 9 workers × $0.003) | ~$5 (DONE) |
| Iter-0 histogram gates (4 runs × ~$1.50 / 256 rollouts) | ~$6 |
| Phase 1 worker calls (4 α × 200 iter × 256 rollouts × ~$0.0036 avg/q) | **~$737** |
| Haiku-as-judge reward calls (same volume × $0.0042/q) | **~$860** |
| p5.48xlarge spot training (4 runs × ~28 hr × $17.25/hr) | **~$1,932** |
| Phase 1 eval (4 routers × 510 questions × ~$0.005 avg/q) | ~$10 |
| Remaining baselines (4 eval sets × 9 workers × ~$0.005/q × ~100 q) | ~$80 |
| Mid-training calibration checks (50 rollouts × Sonnet × 4 runs) | ~$6 |
| **Phase 1 total** | **~$3,640** |

The original spec's $7,800 estimate was dominated by inflated worker-call costs (assumed $0.30/Opus-call vs measured $0.021). The compute is now the dominant line item, as expected for GRPO training.

## 7. Bedrock TPM headroom (probe results)

Burst probe at 32-way concurrency, top 3 cost workers:

| Worker | wall_s | tokens | effective TPM | required for GRPO* | headroom |
|--------|-------:|-------:|--------------:|-------------------:|---------:|
| Qwen3-Coder-480B | 5.6s | 9,382 | 101,029 | 4,850 | 21× |
| Sonnet 4.6 | 5.5s | 7,167 | 77,761 | 4,850 | 16× |
| Opus 4.7 | 3.6s | 4,498 | 74,746 | 4,850 | 15× |

\*Required = uniform-routing baseline: 256 rollouts/iter ÷ 9 workers × ~340 tokens/call × 60/120s. Real load is bursty (rollouts complete in seconds within an iter, then iter waits on training step), so peak TPM is higher than the average. 15× headroom on Opus is comfortable.

All 9 workers passed Stage A (ping). All 9 passed Stage B (5 calls × MATH question, no errors). Stage C burst probe was clean (32/32 ok on each of the top 3 workers, no throttling).

## 8. Open items deferred to Phase 1.5+

- KL anchor (β > 0) — spec uses 0 for paper faithfulness; reconsider if entropy collapses by iter 50.
- Multi-objective GRPO (vector reward) — OQ1 in spec; only revisit if α sweep collapses.
- Profile-aware training — OQ3 in spec; only revisit if cross-seed eval (Phase 1 SC #3) fails.

## 9. Launch criteria

Before running training compute:

- [x] Spec reviewed end-to-end
- [x] Pool finalized (all-Bedrock, 9 workers, ranked by published prices)
- [x] Region locked (us-west-2b)
- [x] Pre-flight: Haiku-Sonnet judge agreement ≥95% on math (measured 98%)
- [x] Brand-bias mitigation chosen (stack a+b: balanced 9-shot + iter-0 histogram gate)
- [x] Reward sign decided (floor at −1)
- [x] α sweep ∈ {0.5, 1.0, 1.7, 3.0} (was {0.1, 0.3, 1.0, 3.0, 5.0}) — narrowed via Always-X baselines + oracle simulator
- [x] Bedrock TPM headroom probed (75-101K TPM, 15-21× headroom)
- [x] Per-worker $/query confirmed against published prices
- [x] Opus 4.7 API quirk captured (no `temperature` param)
- [ ] Spec + addendum edits committed

## 10. Training pre-implementation tasks (loop-internal)

These belong inside the RALPH loop, not as pre-flight, but listing here for clarity:

1. ~~Implement `cost_reward.py` with floored reward and Haiku-as-judge.~~ DONE — `scripts/cost_reward.py`, self-test passed end-to-end.
2. ~~Implement `metadata_prompt.py` with the balanced 9-shot examples.~~ DONE — `scripts/few_shot.py` + `scripts/worker_pool.py` `build_metadata_prompt()`.
3. ~~Build the worker proxy mapping the 9 ords to Bedrock model_ids (Opus 4.7 special-case).~~ DONE — `scripts/worker_pool.py` `invoke_worker()`.
4. ~~WildChat open-domain judge.~~ DONE — `scripts/wildchat_judge.py`, calibration 85% Haiku-Sonnet agreement.
5. ~~Always-X baselines on MATH500, AIME25, WildChat.~~ DONE — see §11.
6. ~~Oracle router simulator.~~ DONE — `scripts/oracle_router.py`, gives per-α targets.
7. Run iter-0 histogram diagnostic gate on actual Qwen2.5-7B (currently a Qwen3-32B proxy gate exists; needs the real base model). **Blocked on p5 spot provisioning.**
8. Sync rl-conductor `trainer_v3.py`, `worker_proxy_v2.py`, `rollout_capture.py` from the existing spot box (`ubuntu@98.87.153.245:/opt/dlami/nvme/rl-conductor/`); adapt to use our `cost_reward.py`, `worker_pool.py`, `few_shot.py`. **Blocked on p5 spot provisioning.**
9. Launch α=1.0 smoke run (50 iters) on p5 before full sweep. **Blocked on p5 spot provisioning.**

## 11. Cross-dataset Pareto findings (oracle simulator results)

Always-X baselines run on three datasets (sources: rl-conductor v4 iter-074 cherry-picks for math, WildChat-1M filtered for open-domain):

| Dataset | n | best worker (raw) | best acc | best $/q | gpt-oss-120b acc |
|---------|--:|-------------------|---------:|---------:|-----------------:|
| MATH500 | 50 | Opus 4.7 | 94% | $0.00410 | 50% |
| AIME25 | 30 | Opus 4.7 | 70% | $0.00941 | **0%** |
| WildChat | 50 | Haiku 4.5 | 90% | $0.00387 | **8%** |

`gpt-oss-120b` is consistently catastrophic on hard / open-domain (0-8%) — kept in pool as a known-bad option for the router to *avoid*.

**Cost-aware best static policy** (highest E[reward] aggregated over all 130 rollouts at α ≤ 1.7) is **always-Qwen-Coder-480B**, not always-Opus. Qwen-Coder's 80%/13%/84% accuracy is mid-tier but its $0.00075/q on WildChat (vs Opus's $0.02913) wins the cost-weighted comparison.

**Oracle router gap** measured on real rollouts:

| α | oracle E[r] | best-static E[r] | gap | implication |
|--:|------------:|------------------:|----:|-------------|
| 0.5 | +0.804 | +0.653 | +0.151 | Big payoff possible |
| 1.0 | +0.765 | +0.645 | +0.120 | Big payoff possible |
| 1.7 | +0.740 | +0.634 | +0.106 | Meaningful payoff |
| 3.0 | +0.632 | +0.630 | +0.002 | Cost dominates; oracle ≈ always-Gemma |
| 5.0 | +0.617 | +0.630 | -0.013 | Oracle worse than always-Gemma → drop |

**Headline target**: trained router captures ≥50% of the oracle gap at α ∈ {0.5, 1.0, 1.7}. Equivalently, ≥6pp expected reward over best-static.

**Oracle's per-source picks** (used as Phase 1 capability-grounding test):
- α=0.5: math→Opus, aime→Opus, wildchat→Qwen-Coder
- α=1.0: math→Mistral, aime→Opus, wildchat→Qwen-Coder
- α=1.7: math→Gemma, aime→Opus, wildchat→Qwen-Coder
- α=3.0: math→Gemma, aime→Gemma, wildchat→Qwen-Coder

**WildChat → Qwen-Coder-480B at every α** is the most surprising finding. Qwen-Coder is fundamentally a code model but its open-domain quality matches Mistral/DeepSeek (84%) at much lower cost ($0.00075 vs $0.00186-0.00194). The router's main job is to learn this non-obvious mapping.
