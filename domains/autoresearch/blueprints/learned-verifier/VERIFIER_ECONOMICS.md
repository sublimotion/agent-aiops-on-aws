# The Economics of Learned Verification for Coding Agents

**Date**: 2026-03-20
**Context**: Analysis of where verification fits in the coding agent stack and what it unlocks — with empirical data from our SVG consensus experiments (n=300, AUC 0.981, precision 1.0).

## The Problem

A coding agent generates patches. Some pass tests. Most don't.

From our 7-harness experiment across 50 SWE-bench issues:
- **Fix rate** (generates a plausible patch): 38-96% depending on harness
- **Pass rate** (patch actually passes tests): 14-25%
- **Gap**: The agent produces plausible code but cannot judge quality

This gap is the verification problem. On open-source benchmarks with gold test suites, you can run tests. On private codebases — where enterprises actually deploy coding agents — you can't.

## Where Verification Fits in the Agent Stack

A coding agent has three phases where verification creates value:

### Phase 1: During Generation — Steering Signal

```
Agent starts task
  → Turn 1: search files
  → Turn 2: read file
  → ...
  → Turn 15: VERIFIER CHECK → trajectory looks bad → abort early
  → (saved 15 turns of wasted compute)
```

Failed runs consume the same compute as successful ones. From our SVG data (n=300):
- Average trajectory: 29.6 turns, ~40K tokens
- 193 of 300 runs failed (64%)
- Failed runs consumed ~7.7M tokens producing nothing useful

A mid-trajectory verifier that aborts at turn 15 instead of 30 saves ~50% of compute on failed runs. That's 3.8M tokens per 300 issues — modest in dollar terms on self-hosted inference, but the wall-clock savings matter. Each failed run takes ~4 minutes. Cutting failures in half saves hours of latency in a best-of-N pipeline.

### Phase 2: After Generation — Selection (Primary Use Case)

```
Generate N=16 candidate patches
  → Verifier scores each: P(pass)
  → Submit top-1
```

This is best-of-N selection with a learned reward model. The verifier replaces test execution as the selection criterion.

#### Verification Methods Compared

| Method | Cost per Patch | Precision | Recall | AUC | Needs Tests? |
|--------|:---:|:---:|:---:|:---:|:---:|
| Test execution (Docker) | $0.05-0.10, 5 min | ~100% | ~100% | 1.0 | **Yes** |
| SVG consensus (our data) | $0.042, 2 calls | **1.000** | 0.528 | **0.981** | **No** |
| Learned verifier (R4P) | $0.004, 1 call | ~72% | ~72% | — | **No** |
| LLM-as-judge (zero-shot) | $0.01-0.05, 1 call | ~55-65% | ~55-65% | — | **No** |
| Blind submission | $0 | ~20% | 100% | 0.5 | No |

SVG consensus is remarkably effective: 100% precision means zero false positives. When it says a patch is good, it's always right. The weakness is recall (0.528) — it misses about half of correct patches by being conservative.

#### Selection Pipeline Economics

For a team processing 100 issues per week, generating N=16 candidates per issue:

| Strategy | Gen Cost | Verify Cost | Pass Rate | Human Review | Weekly Total |
|----------|:---:|:---:|:---:|:---:|:---:|
| Blind submit (N=1) | $4 | $0 | ~20% | $10,000 | **$10,004** |
| Test execution (N=16) | $64 | $128 | ~60% | $4,000 | **$4,192** |
| SVG consensus (N=16) | $64 | $67 | ~53% | $4,700 | **$4,831** |
| Learned verifier (N=16) | $64 | $6 | ~72% | $2,800 | **$2,870** |
| SVG + early stop (N≤4) | $16 | $17 | ~45% | $5,500 | **$5,533** |

Assumptions:
- Agent: 32B model, ~40K tokens/trajectory, $0.001/1K tokens (self-hosted)
- Human review: $100/patch (30-60 min senior engineer time)
- SVG: 2 extra inference calls (describe + reproduce)
- Learned verifier: 1 inference call (~4K tokens)
- Test execution: $0.08/patch (Docker + CPU + 5 min wall clock)

**The dominant cost is human review, not GPU.** A verifier that moves pass rate from 20% → 72% saves $7,130/week = **$370K/year** in engineer time on a 100-issue/week workload.

### Phase 3: For Training — Reward Model

This is the highest-leverage application. To train a coding agent via RL, you need a reward signal for every generated patch:

```
RL training loop (100K+ iterations):
  Agent generates patch → Reward signal → Policy gradient update
```

| Reward Source | Cost per Iteration | Infrastructure | Wall Clock (100K iters) |
|---------------|:---:|:---:|:---:|
| Test execution | $0.08 (Docker) | 512 containers (DeepSWE) | **6 days** |
| Learned verifier | $0.004 (1 call) | 1 GPU | **~6 hours** |
| SVG consensus | $0.042 (2 calls) | 1 GPU | **~3 days** |

DeepSWE's training required 64 H100 GPUs for 6 days with 512 parallel Docker containers orchestrating test execution. Total compute: ~$100K+.

Replace test execution with a learned verifier: **250x cost reduction, 24x wall-clock speedup.**

R4P proved this works. Their Mini-SE agent was trained entirely with learned rewards (no test execution) and hit 32.8% Pass@1 on SWE-bench.

## The Private Codebase Problem

Everything above assumes you have a choice between test execution and a verifier. For enterprise private codebases, you often don't:

| Verification Tier | Open Source (SWE-bench) | Enterprise Private Code |
|---|:---:|:---:|
| **Hard** (compiler, formal proof) | Available for some | Available for some |
| **Strong** (test suite execution) | Gold tests exist | **Often missing or incomplete** |
| **Soft** (SVG consensus, behavioral) | Works | **Works — no tests needed** |
| **Learned** (trained verifier) | Train on test results | **Train on SVG signal** |
| **None** (blind submit) | 19% pass rate | ~20% pass rate |

For a typical enterprise codebase:
- Test coverage averages 40-60% (not 100%)
- Many modules have zero test coverage
- Integration tests are slow and flaky
- Setting up test environments is a DevOps burden

**SVG consensus bypasses all of this.** It works by having the model describe a patch as a PR, then independently regenerate a fix from only the description. If two independent generations converge (line-recall >= 0.8), the patch is likely correct. No tests, no Docker, no CI pipeline.

From our empirical data: **100% precision on 300 instances.** Every patch SVG accepted actually passed tests.

## Training a Verifier on Private Code

The pipeline for bringing learned verification to a private codebase:

```
Step 1: Deploy coding agent on private repo (generates patches)
Step 2: Run SVG consensus on each patch (no tests needed)
         → SVG-accepted patches = positive examples (precision 1.0)
         → SVG-rejected patches = negative examples
Step 3: Train lightweight verifier on (patch, SVG label) pairs
         → XGBoost on behavioral features (works at n=200+)
         → Or LLM fine-tuning (works at n=10K+)
Step 4: Deploy verifier in agent loop
         → 1 inference call per patch instead of 2 (SVG)
         → Better recall than SVG (recovers false negatives)
Step 5: Use verifier as reward model for RL fine-tuning
         → Agent improves on private codebase without tests
```

**Step 5 is the unlock nobody has demonstrated yet.** RL fine-tuning on private code has been impossible because there's no scalable reward signal. A learned verifier trained on SVG consensus provides that signal.

## Scaling Economics

### Small Team (10 engineers, 50 issues/week)

| Metric | Without Verifier | With Verifier | Delta |
|--------|:---:|:---:|:---:|
| Pass rate | 20% | 53-72% | +33-52pp |
| Human reviews/week | 40 | 14-24 | -40-65% |
| Engineer hours on review | 20 hrs | 7-12 hrs | -8-13 hrs/week |
| Annual savings | — | $42K-68K | |
| Verifier cost | — | ~$2K/year (inference) | |

### Medium Team (50 engineers, 250 issues/week)

| Metric | Without Verifier | With Verifier | Delta |
|--------|:---:|:---:|:---:|
| Pass rate | 20% | 53-72% | +33-52pp |
| Human reviews/week | 200 | 70-118 | -41-65% |
| Engineer hours on review | 100 hrs | 35-59 hrs | -41-65 hrs/week |
| Annual savings | — | **$213K-338K** | |
| Verifier cost | — | ~$8K/year | |

### Large Team (200 engineers, 1000 issues/week)

| Metric | Without Verifier | With Verifier | Delta |
|--------|:---:|:---:|:---:|
| Pass rate | 20% | 53-72% | +33-52pp |
| Human reviews/week | 800 | 280-470 | -41-65% |
| Engineer hours on review | 400 hrs | 140-235 hrs | -165-260 hrs/week |
| Annual savings | — | **$858K-$1.35M** | |
| Verifier cost | — | ~$30K/year | |

The ROI is **30-45x** at every scale. The verifier pays for itself in the first week.

## Beyond Cost: What Verification Enables

The economic analysis understates the impact because some benefits aren't reducible to dollars:

### 1. Autonomous Horizon Extension

Without verification, a coding agent operates for minutes — generate one patch, submit, wait for human. With verification, the agent can iterate autonomously:

```
Without verifier:  Generate → Submit → Wait for human (minutes)
With verifier:     Generate → Verify → Improve → Verify → Submit (hours)
```

This is the difference between a "coding assistant" (human in the loop every step) and a "coding agent" (human reviews the final output). The bitter lesson time horizon equation:

```
Model capability × Harness sophistication × Verifier strength = Autonomous horizon
```

Our SVG verifier with precision=1.0 enables the agent to self-iterate until it produces something it's confident about. At 53% recall, that means roughly every other iteration produces a verified-good patch.

### 2. Confidence Scores for Triage

A verifier doesn't just accept/reject — it provides a confidence score. This enables:

- **Auto-merge**: P(pass) > 0.95 → merge without review (SVG precision=1.0 justifies this)
- **Prioritized review**: P(pass) 0.5-0.95 → human reviews, but with confidence context
- **Auto-reject**: P(pass) < 0.2 → regenerate automatically

This transforms code review from "look at everything" to "look at what matters."

### 3. RL on Private Code (The Strategic Unlock)

Every published RL success for coding agents (DeepSWE, SWE-RL, DeepCoder, Kimi K2) relies on test execution as the reward signal. This restricts RL to codebases with comprehensive test suites — primarily open-source benchmarks.

A learned verifier trained on SVG consensus makes RL feasible on any codebase:
- No test suite required
- No Docker infrastructure for test execution
- Reward signal from inference alone
- Enables domain-specific agent fine-tuning

**Nobody has published RL training with SVG-derived rewards yet.** This is an open research direction.

## Generalization: Which Features Transfer to New Codebases?

A verifier trained on SWE-bench won't automatically work on your private Java monolith. But some features are about the *agent's process*, not the *code's content* — and those transfer.

### Features That Generalize (Agent Process)

These measure how the agent behaves, not what the code looks like:

| Feature | Why It's Generic | Published Evidence |
|---------|-----------------|-------------------|
| `loop_behavior` / `repeat_rate` | Agent stuck in a loop = bad, regardless of repo | Critic Rubrics: AUC 0.94, strongest learnable feature. Works on both SWE-bench and real-world data. |
| `action_pct_search` vs `action_pct_edit` | Ratio of exploration to exploitation is universal. Agent searching endlessly without editing is lost. | Our data: r=+0.412, consistent across configs A-F |
| `context_growth_rate` | Context bloat = lost focus. About the agent's cognitive process, not the codebase. | Novel signal not in Critic Rubrics — mechanism is codebase-independent |
| `first_edit_turn` / Parkinson's ratio | Late edits = wasted budget exploring. Same dynamic in Python or Java. | Our Phase 1 data: consistent across 6 configs |
| `misunderstood_intention` | Agent misread the task. Detectable from trace regardless of language. | Critic Rubrics: -0.25 delta, strongest single predictor |
| `insufficient_testing` | Agent skipping validation = lower quality. Universal. | Critic Rubrics: -0.21 delta on SWE-bench |

**These are about the agent, not the code.** A model that learns "looping agents produce bad patches" transfers to any codebase.

### Features That Partially Generalize (SVG Consensus)

SVG consensus is structural, not learned — two independent generations converge means the patch is likely correct. This principle is language-agnostic and repo-agnostic. Same insight as Agentless's majority voting (+18.5% on SWE-bench).

**What transfers:** The mechanism (generation agreement = quality signal).

**What doesn't transfer:** The threshold. Our 0.8 recall threshold gave precision=1.0 on SWE-bench repos. On a private codebase:
- Harder problems → lower base recall even for correct patches → threshold should drop
- Different coding styles → line-level matching may be noisier
- Monorepos with many files → reproductions may diverge more

Calibrate on a small labeled set (50-100 issues) from the target codebase.

### Features That Do NOT Generalize (Patch Content)

| Feature | Why It's Repo-Specific |
|---------|----------------------|
| `diff_size` (lines changed) | A typical Django fix is 5-15 lines. A data pipeline fix might be 200. "Small patch = good" is repo-dependent. |
| `files_touched` | Some architectures require touching many files for any change (microservices, generated code). |
| `test_file_modified` | Only meaningful if tests exist and follow naming conventions. |
| AST-level features | Language-specific. Python AST ≠ Java AST ≠ TypeScript AST. |
| `import changes` | Framework-specific. |
| Patch embedding similarity | Trained on one distribution of code, won't match another. |

These features encode "what correct code looks like" for a specific codebase. They overfit by design.

### Published Evidence on Generalization

Critic Rubrics has the clearest finding:

> "Benchmark-only critics fail on real-world data: AUC 0.48 for PR merge, 0.45 for code survival. **Below random.**"

Critics trained only on SWE-bench data actively *hurt* on real-world code. But:

> "Rubric-supervised critics generalize across LLM backbones (Sonnet 4.5 vs Opus 4.5). Success-Only critics overfit to one backbone."

Behavioral rubrics (process features) transfer. Outcome predictions (success-only) don't.

### Practical Deployment: Tiered Verifier

For a private codebase, deploy in tiers by generalization strength:

```
Tier 1: Process features (transfer directly, no labeled data needed)
  → loop detection, action distribution, context growth, Parkinson's ratio
  → Train once on any codebase, deploy everywhere
  → Expected: modest signal (AUC 0.6-0.7)
  → Cost: $0 (uses existing agent telemetry)

Tier 2: SVG consensus (mechanism transfers, calibrate threshold)
  → Run on 50-100 issues from target repo
  → Human reviews ~50 patches to calibrate precision/recall threshold
  → Expected: high precision, moderate recall
  → Cost: ~$50 inference + 1 day of engineer time for calibration

Tier 3: Patch features (retrain per codebase)
  → Use SVG-accepted patches as positive examples (no test suite needed)
  → Train XGBoost or LLM verifier on repo-specific patterns
  → Expected: additive to Tier 1+2, marginal gains
  → Cost: ~200+ labeled examples needed, ongoing retraining
```

**Tier 1 is free** — just observe the agent's behavior. An agent that loops, bloats context, and edits late is producing bad patches regardless of the codebase.

**Tier 2 is cheap** — 50 issues with human review is feasible for any team. Our SVG data shows precision=1.0 at this tier alone.

**Tier 3 is where the investment lives** — and also where the marginal gain is smallest. The question is whether Tier 1 + Tier 2 are "good enough." For most teams, SVG precision=1.0 is sufficient; the Tier 3 investment only makes sense if you need higher recall (catching more correct patches that SVG misses).

## The Bitter Lesson Applied to Verification

The bitter lesson (Sutton, 2019): methods that leverage computation scale better than methods that leverage human knowledge. The verification spectrum is this lesson playing out in real time.

### The Progression

```
Human code review                       ← Pure human knowledge. Doesn't scale.
  → Patch features per repo             ← Human knowledge encoded as features.
                                           Scales poorly (retrain per codebase).
    → SVG consensus                     ← Compute-based. Two generations agree.
                                           Scales with inference budget.
      → Learned verifier on SVG signal  ← Compute + learning. Scales with data.
        → Agent absorbs verifier via RL ← Pure compute. Scales indefinitely.
```

Each step replaces human knowledge with computation:

| Stage | Human Knowledge Required | Compute Required | Scales? |
|-------|:---:|:---:|:---:|
| Human code review | Expert reviewer per patch | None | No — bottlenecked on people |
| Patch features (Tier 3) | Feature engineering per repo, labeled data | Training | Poorly — new repo = new features |
| SVG consensus (Tier 2) | Threshold calibration (one-time) | 2 inference calls/patch | Yes — more compute = more patches |
| Learned verifier (Tier 2→1) | Initial training set | 1 inference call/patch | Yes — more data = better verifier |
| RL-absorbed verification | None | Training + inference | Yes — indefinitely |

The final stage is what Cursor Composer 2 demonstrated: compaction-in-the-loop RL where the model learns its own context management through training. No explicit verifier — the agent absorbs verification into its weights. Their 50% reduction in compaction error wasn't engineered; it was learned. The harness trick becomes a weight update.

### Connection to the Time Horizon Equation

The bitter lesson time horizon equation for coding agents:

```
Model capability × Harness sophistication × Verifier strength = Autonomous horizon
```

Each variable in this equation has a bitter lesson trajectory:

**Model capability**: Scaling laws. More compute → more capable model. This is the original bitter lesson.

**Harness sophistication**: Currently hand-engineered (tool definitions, prompts, scaffolds). Cursor Composer 2 shows the model can absorb harness features via RL. The harness simplifies as the model improves.

**Verifier strength**: Currently the bottleneck. Moving from human review → SVG consensus → learned verifier → RL-absorbed verification follows the same curve. Each step trades human knowledge for computation.

The verification spectrum IS the bitter lesson applied to the third axis. And our Phase 0 finding — SVG consensus achieving AUC 0.981 with zero human knowledge beyond a threshold — is empirical evidence that compute-based verification already works.

### What This Predicts

If the bitter lesson holds for verification (and the evidence says it does):

1. **Patch feature engineering will be obsoleted.** Just as hand-crafted features lost to neural nets in vision, hand-crafted code features will lose to learned representations. Investing in Tier 3 is a short-term play.

2. **SVG consensus (or equivalent compute-based verification) will be the standard.** It's already at precision=1.0 with zero domain knowledge. The only question is efficiency — can we get the same signal with fewer inference calls?

3. **Verification will be absorbed into the model.** The Leanstral pattern (external verifier in loop) is a stepping stone. The endpoint is an agent that knows when its own patches are wrong — no external verification needed. Cursor's RL results suggest this is already happening.

4. **The enterprise moat is compute infrastructure, not domain expertise.** The team that can run N=16 candidates with SVG verification fastest wins — not the team with the best hand-crafted features for their specific codebase. This is Together AI's thesis (90% cache hit rate on 452B tokens for CoderForge) and it applies to verification too.

## Key Assumptions and Risks

| Assumption | Risk | Mitigation |
|-----------|------|------------|
| SVG precision generalizes beyond Devstral/SWE-bench | May be lower on private code | Validate on internal repos before deploying |
| 32B model sufficient for SVG consensus | Larger codebases may need more context | Test with 128K context models |
| $100/review for human review cost | Varies by seniority and complexity | Use team-specific rates; even at $50/review, ROI > 15x |
| N=16 candidates is feasible at scale | Inference cost scales linearly | Use early-stopping (N=4 avg with SVG) |
| Learned verifier recall > SVG recall | Training may not improve over SVG | SVG alone is already valuable; learned verifier is upside |

## Summary

| Application | GPU Savings | Time Savings | Strategic Value |
|-------------|:---:|:---:|:---:|
| **Early stopping** (bad trajectories) | ~40% of wasted gen compute | Hours of latency | Low |
| **Best-of-N selection** | 20x (vs test execution) | **$370K-$1.35M/yr** (engineer time) | **High** |
| **Confidence-based triage** | — | 40-65% fewer reviews | High |
| **RL reward model** | **250x** (vs test execution) | 6 days → 6 hours | **Strategic** |
| **Private code unlock** | ∞ (enables what was impossible) | — | **Transformative** |

The verifier is not primarily about saving GPU compute. It's about **saving human time** at the selection layer and **enabling RL training** at the reward layer. For private codebases where test suites don't exist, it transforms coding agents from "generate and hope" to "generate, verify, and iterate."

---

*Empirical basis: 300 SVG consensus results (Devstral Small 2 on SWE-bench Lite, production-run1). Published references: R4P (72.2% accuracy, 50x faster), SWE-RM (100K trajectories, +10.4pp), DeepSWE (512 Docker containers, 64 H100s, 6 days).*
