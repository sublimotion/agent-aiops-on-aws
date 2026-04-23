# Autoresearch Spec: Verifier Reward

## Status: DRAFT

## Overview

Measure Claude models (Haiku, Sonnet, Opus) as coding agents on the same SWE-bench 50-issue subset used in agent-harness, then build verification and coaching skills that improve trajectory efficiency and patch selection. API-driven — no GPU infrastructure required for Phase 1-3.

The existing agent-harness data covers only self-hosted models (Devstral 24B, Qwen variants). Without Claude baselines on the same benchmark, we cannot measure whether skills, coaches, or verifiers add value versus simply using a stronger model. Baselines first, interventions second.

**Core hypothesis**: Claude models combined with lightweight verification and coaching skills can match or exceed self-hosted model performance at competitive cost, and the skills produce structured evaluation data that feeds future learned verifiers and RL reward signals.

**Depends on**: agent-harness (Phase 1+2 complete, provides eval infrastructure and comparison data)

## Research Questions

1. **Where do Claude models sit on the fix rate / pass rate / precision frontier?** Devstral 24B: 88% fix, 22% pass (OpenCode). Claude Code's system prompt gave Devstral best precision (52.6%). Do Claude models inherently produce higher-precision patches?

2. **Does Parkinson's Law hold for Claude models?** Devstral delays first edit to ~60% of turn budget. If Claude models edit earlier, they may not need trajectory coaching at all.

3. **What's the cost-performance frontier?** If Haiku at $0.05/issue matches Devstral at $0/issue (self-hosted), the self-hosted path loses its advantage. If Sonnet at $0.50/issue reaches 40%+ pass rate, it may already exceed what skills can add to weaker models.

4. **Can a patch verification skill (Claude-as-judge) match SVG consensus?** SVG needs 3 inference calls on a self-hosted model. A single Claude Haiku call at $0.001/patch would be 100x cheaper if it achieves comparable precision.

5. **Can a trajectory coach skill break Parkinson's Law and improve pass rate?** Mid-trajectory intervention that summarizes progress and nudges toward editing — does it reduce first_edit_turn and improve fix-to-pass conversion?

## Phases

### Phase 1: Claude Model Baselines (Week 1-2)

**Goal**: Establish fix rate, pass rate, precision, and behavioral metrics for Claude Haiku, Sonnet, and Opus on the same 50-issue subset and harnesses used in agent-harness.

**Steps**:

1. **Harness setup** (OpenCode + Claude Code only):
   - **Claude Code**: Native — already uses Anthropic SDK. Point at real Claude models instead of Devstral-via-vLLM. Set `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, etc.
   - **OpenCode**: Use `@ai-sdk/anthropic` provider (native Anthropic support in Vercel AI SDK) or `@ai-sdk/openai-compatible` with Anthropic's OpenAI-compatible endpoint.

2. **Run baseline matrix** (start with cheapest, scale up):

   | Priority | Model | Harness | Issues | Est. Cost | Rationale |
   |----------|-------|---------|--------|-----------|-----------|
   | P0 | Haiku 4.5 | OpenCode | 50 | ~$3 | Cheapest baseline, best single harness |
   | P0 | Sonnet 4.6 | OpenCode | 50 | ~$25 | Mid-tier, likely production sweet spot |
   | P1 | Haiku 4.5 | Claude Code | 50 | ~$5 | Native harness, cheapest model |
   | P1 | Sonnet 4.6 | Claude Code | 50 | ~$40 | Native harness, strong model |
   | P2 | Opus 4.6 | OpenCode | 50 | ~$125 | Frontier ceiling |
   | P2 | Opus 4.6 | Claude Code | 50 | ~$200 | Frontier ceiling, native harness |

3. **Capture full telemetry** for each run:
   - `patch_diff` (git diff HEAD after agent completes)
   - `tests_pass` (gold test evaluation)
   - `turns_used`, `first_edit_turn`, `action_distribution`, `repeat_rate`, `context_growth`
   - `tokens_consumed` (input + output), `total_cost`
   - `total_latency_ms`

4. **Analyze and compare** to existing self-hosted baselines:

   | Metric | Compare Against |
   |--------|----------------|
   | Fix rate | Devstral × OpenCode: 88%, Qwen3.5 × OpenCode: 88% |
   | Pass rate | OpenCode: 22%, Claude Code: 20% (both Devstral) |
   | Precision | Claude Code × Devstral: 52.6%, OpenCode × Devstral: 25% |
   | Harness spread | Devstral: 50pp, Qwen3.5: 16pp |
   | Parkinson's ratio | Devstral: 46%, Qwen3.5: 46%, Qwen 2.5: 24% |
   | Cost/issue | Self-hosted: ~$0 (amortized), Haiku: ~$0.05, Sonnet: ~$0.50 |

**Exit criteria**:
- At least 4 model × harness cells completed (Haiku + Sonnet × OpenCode + Claude Code)
- Results table with pass rate, fix rate, precision, behavioral metrics, cost/issue
- Parkinson's ratio measured for Claude models
- Cost-performance frontier plotted (pass rate vs $/issue)
- Decision: which Claude tier is the baseline for Phase 2-3 interventions?

**Estimated cost**: P0 runs ~$30, P0+P1 ~$75, full matrix ~$400

### Phase 2: Patch Verification Skill (Week 3-4)

**Goal**: Build a Claude-based verification skill that scores patches, measure whether it can select the best patch from N candidates, and compare to SVG consensus.

**Prerequisites**: Phase 1 baselines completed. Need N>1 candidate patches per issue — use existing multi-harness data (8 harnesses × Devstral) plus Phase 1 Claude patches.

**Steps**:

1. **Design the verification rubric** — structured criteria, not a single "is this good?" judgment:
   - **Problem alignment**: Does the patch address the specific bug described in the problem statement?
   - **Minimality**: Does the patch make only necessary changes? (penalize reformatting, unrelated modifications)
   - **Test safety**: Does the patch modify test files? (suspicious — agent may be gaming tests)
   - **Logic correctness**: Does the fix logic match the error described? (targets the 42% "wrong fix" failure mode)
   - **Scope**: Does the patch touch the right files based on the traceback/error?
   - **Completeness**: Does the patch handle edge cases mentioned in the issue?

2. **Implement as a skill folder**:
   ```
   skills/patch-verifier/
   ├── SKILL.md              # Rubric + scoring instructions
   ├── scripts/
   │   ├── verify_patch.py   # Calls Claude API with rubric, returns structured score
   │   └── rank_patches.py   # Given N patches, rank by verification score
   └── references/
       └── rubric.md         # Detailed scoring criteria with examples
   ```

3. **Evaluate across Claude tiers**:
   - Input: 50 issues × N candidate patches (from multi-harness + Phase 1 data)
   - Each tier (Haiku/Sonnet/Opus) scores all patches
   - Pick top-1 by score, measure pass rate
   - Compare to: random selection, SVG consensus (AUC 0.981), shortest-patch heuristic

4. **Measure verifier properties**:
   - Precision: when skill says "good", how often does it pass tests?
   - Recall: of all passing patches, how many does the skill catch?
   - ECE: does the confidence score match actual pass probability?
   - Cost per verification: tokens consumed × $/token

5. **Cross-tier analysis**:
   - Is Haiku-as-verifier + Sonnet-as-generator better than Sonnet-as-generator alone?
   - What's the cost-optimal pairing? (cheap verifier × expensive generator, or vice versa)

**Exit criteria**:
- Verification skill deployed and tested on >= 200 patches
- Top-1 pass rate measured for each tier and compared to baselines
- Precision/recall/ECE measured
- If any tier beats random by >5pp: verification skill adds value
- If Haiku matches SVG consensus: cheap verification path validated
- Structured evaluation data saved (becomes training data for future learned verifier)

**Estimated cost**: ~$20-100 (depends on N candidates and tiers tested)

### Phase 2b: Adversarial Self-Critique in Generation (Week 3-4, parallel with Phase 2)

**Goal**: Test whether injecting adversarial self-critique into the generation prompt improves pass rate — zero extra API cost. If it works, this is directly comparable to SERA/RL approaches because it changes the generator's effective quality, not post-hoc filtering.

**Motivation**: Phase 2's adversarial verification skill (v009) eliminates all false positives as a post-hoc filter at $0.038/patch. But this is apples-to-oranges with SERA (which bakes quality into weights via SVG-filtered SFT) or RL (which bakes quality into weights via reward optimization). If the same adversarial reasoning works *during* generation — "before submitting, assume your patch is wrong and find the bug" — then we get a free capability upgrade that IS comparable to weight-based approaches.

**Prerequisites**: Phase 1 baselines completed (Haiku + Sonnet pass rates established as control).

**Research questions**:
1. Does adversarial self-critique during generation improve pass rate vs Phase 1 baseline? (The core question)
2. Does it reduce the fix-to-pass gap? (82% fix rate but only 12% pass rate — does self-critique catch "plausible but wrong" patches before submission?)
3. Does it transfer across model tiers? (If Haiku self-critique improves, does Sonnet also improve, or is Sonnet already implicitly self-critical?)
4. What's the cost? (Self-critique may consume more turns/tokens from the same budget — is the pass rate gain worth the throughput loss?)

**Steps**:

1. **Design prompt variants** (two treatments + control):

   | Variant | Prompt Addition | Hypothesis |
   |---------|----------------|------------|
   | `control` | (Phase 1 baseline prompt — no change) | Baseline pass rate |
   | `self-critique` | "After writing your fix, review it critically: assume the patch is wrong and try to find a bug. If you find one, fix it before finishing." | Adversarial self-review improves patch quality |
   | `self-critique-strong` | "IMPORTANT: Before you finish, you MUST do a self-review. Assume your patch is incorrect. Try to construct an input that would make the patched code fail. If you find a plausible failure, fix the patch. Only finish when you cannot break your own fix." | Stronger adversarial framing, mirrors v009 rubric language |

2. **Run experiment matrix** (cheapest models first):

   | Priority | Model | Variant | Issues | Est. Cost | Rationale |
   |----------|-------|---------|--------|-----------|-----------|
   | P0 | Haiku | self-critique | 50 | ~$3 | Cheapest test of the hypothesis |
   | P0 | Haiku | self-critique-strong | 50 | ~$3 | Stronger variant |
   | P1 | Sonnet | self-critique | 50 | ~$25 | Does it help stronger models too? |
   | P1 | Sonnet | self-critique-strong | 50 | ~$25 | Stronger variant on stronger model |

3. **Compare to Phase 1 baselines**:
   - Same 50-issue subset, same harness (OpenCode), same gold evaluation
   - Primary metric: pass rate (gold tests)
   - Secondary metrics: fix rate, precision (pass/fix), tokens consumed, turns used, cost/issue
   - Does self-critique increase tokens/turns? (Budget trade-off)

4. **Compare to SERA baselines** (from agent-swarm data):
   - SERA-32B pass rate: ~16% (SERA harness), ~11% (OpenCode)
   - If Claude Haiku + self-critique exceeds SERA-32B's pass rate, adversarial reasoning in prompts competes with SVG-filtered SFT

5. **Analyze failure modes**:
   - Did self-critique catch any FM-003 (plausible but wrong) errors during generation?
   - Did self-critique cause any NEW failures (e.g., correct patch revised to incorrect)?
   - What fraction of runs actually perform a self-review step? (Model may ignore the instruction)

**Exit criteria**:
- At least 2 model × variant cells completed (Haiku × self-critique + self-critique-strong)
- Pass rate compared to Phase 1 control baseline
- If pass rate improves by >3pp: adversarial self-critique adds value at zero extra cost
- If no improvement: self-critique doesn't transfer from post-hoc to in-generation (model can't find bugs in its own code)
- Token/turn overhead measured — is the cost neutral?
- Comparison to SERA-32B baseline documented

**Estimated cost**: ~$6 (P0 only), ~$56 (P0 + P1)

**Connection to transferability experiments**: This is T5 from the transferability matrix. The remaining tests (T1-T4) require g7e GPU capacity to re-run Devstral/Qwen patch generation with fixed diff capture, then run the post-hoc ensemble verifier on those patches.

### Phase 3: Trajectory Coach Skill (Week 5-7)

**Goal**: Build a mid-trajectory coaching skill that provides intelligent process feedback, targeting Parkinson's Law and the "wrong fix" failure mode. Test whether coaching improves pass rate.

**Prerequisites**: Phase 1 baselines show Parkinson's Law holds for Claude models (if it doesn't, skip this phase — the model doesn't need coaching).

**Steps**:

1. **Design the coaching skill**:
   ```
   skills/trajectory-coach/
   ├── SKILL.md               # Coaching protocol + intervention rules
   ├── scripts/
   │   ├── coach.py           # Analyzes trajectory state, produces intervention
   │   └── inject.py          # Wires coach into OpenCode agent loop (or API proxy for Claude Code)
   └── references/
       └── intervention_types.md  # Catalog of coaching interventions
   ```

2. **Define intervention types** (grounded in agent-harness findings):

   | Trigger | Intervention | Targets |
   |---------|-------------|---------|
   | `turns_used > 0.5 * budget AND first_edit_turn == None` | "You've explored N files. The traceback points to X. Try editing now." | Parkinson's Law |
   | `repeat_count > 2 for same file` | "You already read this file at turn K. Here's what you found: [summary]. Move forward." | Looping (37% repeat rate) |
   | `context_tokens > 0.7 * max_context` | "Context is 70% full. Summarize what you know before proceeding." | Context overflow |
   | `edit_count > 3 AND no test_run` | "You've made 3 edits without testing. Verify your changes." | Thrashing |
   | `files_touched > 3` | "You're modifying 4+ files. The issue likely needs a single-file fix." | Scope creep |

3. **Offline evaluation first** (cheapest):
   - Replay 50 existing Devstral trajectories (Phase 1 turn metrics data, 6,599 rows)
   - At each turn, ask the coach: "Given this trajectory state, what intervention would you suggest?"
   - Grade: does the intervention align with what would have helped? (using known outcome)
   - Measure: intervention precision (when coach says "edit now", would editing have helped?)

4. **Live experiment**:
   - Wire coach into the OpenCode agent loop via `inject.py` (OpenCode is open-source, loop is modifiable)
   - For Claude Code: use an API proxy that intercepts responses and injects coaching context
   - Every K turns (K=3 or K=5), the coach reviews the trajectory and injects a guidance message
   - Coach model: Haiku (cheapest, <$0.01/intervention)
   - Agent model: best Claude tier from Phase 1
   - Run on same 50-issue subset
   - Compare: coached vs uncoached (Phase 1 baseline)

5. **Measure coaching impact**:
   - Does `first_edit_turn` decrease? (Parkinson's Law broken?)
   - Does `repeat_rate` decrease? (Less looping?)
   - Does pass rate increase? (The actual goal)
   - Does cost increase justify the pass rate gain?

**Exit criteria**:
- Offline evaluation on 50 trajectories completed
- At least one live coached run (50 issues) completed
- First_edit_turn and pass rate compared to Phase 1 uncoached baseline
- If pass rate improves by >3pp: coaching adds value
- If Parkinson's ratio decreases by >10pp: trajectory efficiency improved
- Cost-adjusted comparison: coached Haiku vs uncoached Sonnet

**Estimated cost**: Offline ~$5, live ~$30-100 (depends on agent model tier)

### Phase 4: Combined System + SVG Bridge (Week 8-10, contingent)

**Goal**: Combine verification and coaching skills into an end-to-end system. Generate labeled data that bridges to the GPU-heavy SVG-as-RL-reward experiments.

**Prerequisites**: Phase 2 or Phase 3 shows positive result.

**Steps**:

1. **Compose skills**: Coach guides trajectory → agent generates patch → verifier scores patch → select or retry
2. **Best-of-N with coaching**: Generate N=4 coached patches, verify with skill, pick top-1
3. **Compare to the full existing baseline matrix**:
   - Coached Claude Haiku + verification vs uncoached Devstral × 8-harness ensemble (36%)
   - Coached Claude Sonnet + verification vs Qwen3.5 × OpenCode (88% fix)
4. **Produce labeled dataset**: Every (patch, rubric_score, tests_pass) triple becomes training data
5. **Measure SVG agreement**: Run SVG consensus on Claude-generated patches. Does SVG agree with the verification skill? This calibrates the skill against SVG and produces the bridge data for the GPU-heavy verifier-reward Phase 1 (ECE measurement).

**Exit criteria**:
- Combined coached + verified system evaluated on 50 issues
- Pass rate compared to all existing baselines
- Labeled dataset of (patch, rubric_scores, tests_pass) produced
- SVG agreement rate measured
- Decision: proceed to GPU-heavy phases (SVG-as-RL-reward spec) or iterate on skills

**Estimated cost**: ~$50-200

## Components

### 1. Compute
- **Platform**: Local machine + Anthropic API (Phase 1-3). No GPU required.
- **Phase 4 SVG runs**: g7e.24xlarge if SVG comparison needed (self-hosted Devstral for SVG pipeline)
- **Concurrency**: Anthropic API supports parallel requests. 50 issues × 3 tiers can run concurrently.

### 2. Codebase
- **Source**: Existing harness infrastructure
  - OpenCode — use `@ai-sdk/anthropic` provider or `@ai-sdk/openai-compatible`
  - Claude Code — native Anthropic SDK, no adaptation needed
  - `multi_harness_eval.py` — orchestrator (adapt for API-based runs)
- **New code** (agent-editable):
  - `scripts/run_claude_baseline.py` — baseline runner for Claude models via API
  - `scripts/eval_baselines.py` — analysis and comparison to existing results
  - `skills/patch-verifier/` — verification skill (rubric + scoring)
  - `skills/trajectory-coach/` — coaching skill (intervention logic)
  - `scripts/compose_skills.py` — Phase 4 combined system
- **Fixed files** (agent must NOT edit):
  - SWE-bench Lite issue definitions and gold test patches
  - Existing agent-harness results (comparison baselines)
- **Agent instructions**: `program.md` — autoresearch loop protocol

### 3. Experiment Protocol
- **Primary metric**: Pass@1 on SWE-bench Lite 50-issue subset (verified by gold tests)
- **Secondary metrics**: fix rate, precision (pass/fix), cost/issue, first_edit_turn, Parkinson's ratio, repeat_rate, tokens consumed
- **Eval subset**: Same 50 issues as agent-harness (seed 42, stratified by 11 repos)
- **Gold evaluation**: Apply agent patch + gold test_patch, run FAIL_TO_PASS tests. Same method as agent-harness Phase 2.
- **Gold eval constraint**: Only Django/pytest/sympy evaluable without Docker (same as agent-harness). Pass rate is a lower bound.
- **Logging**: JSONL per run with full telemetry. Manifest file pinning dataset version and instance_ids.

### 4. Networking
- **Anthropic API**: Direct HTTPS calls. No SSH, no GPU instance needed for Phase 1-3.
- **Rate limits**: Haiku/Sonnet should be fine for 50 sequential issues. Opus may need rate limit awareness.
- **g7e** (Phase 4 only): SSH for SVG pipeline runs on self-hosted Devstral.

### 5. Storage
- **Results**: `domains/autoresearch/blueprints/verifier-reward/results/`
- **Skills**: `domains/autoresearch/blueprints/verifier-reward/skills/`
- **Comparison data**: agent-harness results (read-only reference)

## Data Assets (Existing)

| Asset | Use in This Spec |
|-------|-----------------|
| Agent-harness Phase 2b results (8 harnesses × Devstral) | Comparison baselines, candidate patches for Phase 2 verification |
| Agent-harness Phase 1 turn metrics (6,599 rows) | Offline coaching evaluation (Phase 3), behavioral feature reference |
| Agent-swarm multi-model matrix | Comparison baselines across model scales |
| SVG production-run1 (300 rows, precision=1.0) | Phase 4 SVG agreement calibration |
| Learned-verifier VERIFICATION_FRAMEWORK.md | Skill design reference (Phase 0.5 methodology) |

## Success Criteria

### Phase 1: Baselines Established
- Claude Haiku, Sonnet measured on >= 2 harnesses each
- Cost-performance frontier plotted
- Parkinson's ratio measured — determines whether Phase 3 is needed
- Clear ranking: Claude tiers vs self-hosted models on pass rate, precision, cost

### Phase 2: Verification Skill Works
- Top-1 selection from N candidates beats random by >5pp
- Precision/recall/ECE measured for at least 2 Claude tiers
- If Haiku-as-verifier works: $0.001/patch verification path validated

### Phase 2b: Adversarial Self-Critique Works
- Self-critique variant pass rate exceeds Phase 1 control by >3pp
- Token/turn overhead measured (is the budget trade-off worth it?)
- Comparison to SERA-32B baseline documented
- If no improvement: self-critique during generation doesn't transfer from post-hoc verification (still valuable as negative result — the model can find bugs in others' code but not its own)

### Phase 3: Coaching Improves Trajectories
- Coached agent shows lower Parkinson's ratio than uncoached (>10pp decrease)
- Coached agent pass rate exceeds uncoached baseline (>3pp)
- Cost-adjusted value: coaching cost < pass rate gain value

### Phase 4: System Composition
- Combined system (coach + verifier) pass rate exceeds best single-method result
- Labeled dataset of >= 200 (patch, rubric_score, tests_pass) triples produced
- SVG agreement measured — bridges to GPU-heavy experiments

### Negative Results (Still Valuable)
- Phase 1: Claude Opus already exceeds 36% (8-harness ensemble ceiling) with no intervention → skills unnecessary, just use a better model. Valid and important finding.
- Phase 2: Verification skill doesn't beat random → LLM-as-judge insufficient for patch selection at this scale. Validates need for SVG or execution-based verification.
- Phase 3: Parkinson's Law doesn't hold for Claude → the pattern is model-specific (Devstral/Qwen), not universal. Coaching is unnecessary for capable models.

## Non-Requirements
- GPU infrastructure (Phase 1-3 are API-only)
- RL training, GRPO, weight updates (that's the separate GPU-heavy spec if Phase 4 succeeds)
- Fine-tuning any model
- Full SWE-bench Verified (500 issues) — use 50-issue subset
- Docker-based test execution
- Multi-node anything

## Known Limitations
- 50-issue eval subset introduces ~5% sampling variance
- Gold evaluation only covers Django/pytest/sympy without Docker — pass rate is a lower bound
- Anthropic API rate limits may slow Opus runs
- Coaching injection requires modifying the agent loop — Claude Code's agent logic is in the CLI binary, so coaching must use a proxy approach (intercept API calls) rather than direct loop injection. OpenCode's open-source codebase allows direct modification.
- Cost estimates assume current Anthropic pricing — may change
- Claude Code's system prompt is ~22K tokens — high fixed cost per issue that disproportionately affects Haiku (where 22K tokens is a significant fraction of the budget)

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Claude Opus solves everything, no skill needed | MEDIUM | Phases 2-3 unnecessary | This is a good outcome — document the cost-performance frontier |
| Anthropic API rate-limited during eval | LOW | Runs slower | Sequential issues, exponential backoff, or request limit increase |
| Claude Code incompatible with coaching injection | HIGH | Phase 3 limited to OpenCode | Design coach as a proxy that wraps API calls, not as agent-internal injection |
| Verification rubric doesn't correlate with test outcomes | MEDIUM | Phase 2 negative | Iterate rubric based on failure analysis; compare multiple rubric designs |
| Haiku too weak for OpenCode tool calling | LOW | Phase 1 incomplete | Fall back to Claude Code (native Anthropic SDK, simpler integration) |
| Cost exceeds budget on Opus runs | LOW | Opus cells skipped | Opus is P2 priority — skip if Sonnet results are sufficient |

## Relationship to Other Specs

- **agent-harness**: Provides the eval framework, 50-issue subset, comparison baselines, and harness adapters
- **agent-swarm**: Provides multi-model comparison data; Phase 1 extends this to Claude models
- **learned-verifier**: Provides verification framework methodology; Phase 2 implements Phase 0.5 (skill verifier) from that framework
- **verifier-reward (GPU-heavy future)**: Phase 4 of this spec produces the bridge data (SVG agreement, labeled patches) that feeds Phase 1 of the GPU-heavy SVG-as-GRPO-reward experiment. The GPU-heavy work becomes a follow-on spec if this experiment validates the approach.

## Future: GPU-Heavy Follow-On

If Phase 4 succeeds (skills produce labeled data, SVG agreement measured), the next spec covers:
1. ECE measurement of SVG consensus (RL gate)
2. Best-of-N with SVG + behavioral ensemble
3. Multi-harness rejection sampling SFT
4. SVG as GRPO reward signal

This follows the phased approach from VERIFICATION_FRAMEWORK.md: skills (Phase 0.5) → consensus (Phase 1) → behavioral (Phase 2) → learned (Phase 3) → RL (Phase 4). The current spec covers Phase 0.5. The GPU-heavy spec covers Phases 1-4.

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
