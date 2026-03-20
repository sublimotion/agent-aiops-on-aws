# The Verification Framework: Primitives and Implementation Methodology

**Date**: 2026-03-20
**Audience**: Forward deploy teams implementing agent systems for knowledge workflows
**Status**: Draft — grounded in empirical coding agent data, extending to general knowledge work

## What This Document Is

A methodology for analyzing any knowledge workflow, identifying verification opportunities, and implementing a tiered verifier that extends an agent's autonomous horizon. The pattern applies to any domain where an AI agent produces outputs that need quality assessment — coding, legal, financial, research, operations.

The framework decomposes into **five primitives** and a **four-phase implementation playbook**.

---

## Part 1: The Primitives

Every agent system that produces quality-controlled output is composed of five primitives. Understanding these lets you decompose any workflow and identify where verification creates value.

### Primitive 1: Generator

**What it does**: Produces candidate outputs from a task specification.

```
Input:  Task specification (bug report, legal question, analysis request)
Output: One or more candidate outputs (patch, document, report)
```

**Properties that matter**:
- **Throughput**: How many candidates can it produce per unit time/cost?
- **Diversity**: Do N candidates explore different approaches, or converge?
- **Fix rate**: What fraction of candidates are plausible (not obviously broken)?
- **Pass rate**: What fraction actually meet quality standards?

The gap between fix rate and pass rate is the **verification opportunity**. From our coding agent data: fix rate 82%, pass rate 17.7%. The agent produces plausible output most of the time — it just can't tell good from bad.

**Key insight**: Generator quality matters less than you think. A cheap generator with a strong verifier often beats an expensive generator with no verifier. Our Devstral 24B (free, self-hosted) + SVG consensus outperforms GPT-4 class models submitting blind.

### Primitive 2: Verifier

**What it does**: Scores a candidate output on quality.

```
Input:  (Task specification, candidate output, [optional context])
Output: Score ∈ [0, 1] or binary accept/reject
```

Every verifier has three properties in tension:

| Property | Definition | Trade-off |
|----------|-----------|-----------|
| **Precision** | When it says "good," how often is it right? | Higher precision → fewer false positives → safer to auto-accept |
| **Recall** | Of all good outputs, how many does it catch? | Higher recall → fewer missed opportunities → less human fallback |
| **Cost** | Time, compute, or human effort per verification | Lower cost → more candidates verified → better best-of-N |

The **verification spectrum** orders verifiers by these properties:

```
                    Precision    Recall    Cost         Needs Domain Setup?
                    ─────────    ──────    ────         ───────────────────
Hard verifier       ~100%        ~100%     Free         Yes (formal spec)
Strong verifier     80-100%      80-100%   High         Yes (test suite/expert)
Consensus verifier  ~100%*       40-60%    Medium       No
Behavioral verifier 60-75%       60-75%    Low          No
Learned verifier    70-85%       70-85%    Low          Yes (training data)
No verifier         N/A          N/A       Free         No

* Our empirical data: SVG consensus precision = 1.000 on n=300
```

### Primitive 3: Selector

**What it does**: Given N scored candidates, picks the one(s) to submit.

```
Input:  [(candidate_1, score_1), ..., (candidate_N, score_N)]
Output: Top-k candidates for submission or human review
```

**Selection strategies**:

| Strategy | When to Use | Behavior |
|----------|------------|----------|
| **Top-1** | High-precision verifier, auto-submit | Submit highest-scored candidate |
| **Threshold** | Mixed confidence, triage | Auto-submit if score > τ_high, reject if < τ_low, human reviews middle band |
| **Top-k for review** | Low-precision verifier | Human reviews k candidates, verifier reduces from N to k |
| **Majority vote** | Multiple generators | Pick the candidate most generators agree on |
| **Early stop** | Cost-sensitive | Stop generating once a candidate exceeds threshold |

The selector is where the verifier's precision/recall trade-off becomes an operational decision. High-precision verifiers enable auto-submit. Low-precision verifiers enable triage (reducing human workload from N to k).

### Primitive 4: Telemetry

**What it does**: Captures signals from the generator's process, not just its output.

```
Input:  Agent execution trace (tool calls, reasoning steps, timing)
Output: Behavioral feature vector
```

Telemetry is the raw material for building verifiers. Every agent system already produces it — the question is whether anyone collects and uses it.

**Universal behavioral signals** (transfer across domains):

| Signal | What It Measures | Verification Intuition |
|--------|-----------------|----------------------|
| **Action diversity** | Ratio of exploration to exploitation | Agent that only searches (or only edits) is unbalanced |
| **Repetition / looping** | Same action repeated N times | Agent is stuck — output quality is low |
| **Time-to-first-output** | When the agent first produces substantive work | Late production = low confidence (Parkinson's pattern) |
| **Context consumption** | How much input the agent processes | Runaway consumption = lost focus |
| **Source diversity** | Number of distinct sources consulted | More sources = more thorough (for research/analysis tasks) |
| **Self-correction count** | Times agent revised its own output | Some revision is good; excessive revision is thrashing |
| **Completion ratio** | Budget consumed / budget available | Using 100% of budget = likely didn't converge |

These signals work because they measure **process competence**, not domain knowledge. An agent looping on a legal research task is the same signal as an agent looping on a coding task.

### Primitive 5: Feedback Loop

**What it does**: Routes verifier signals back to improve the generator.

```
Input:  (candidate, verifier_score, [human_decision])
Output: Updated generator (weights, prompts, or context)
```

**Feedback mechanisms by cost and impact**:

| Mechanism | What Changes | Cost | Latency | Impact |
|-----------|-------------|------|---------|--------|
| **Prompt refinement** | System prompt / examples | Free | Immediate | Low-medium |
| **Context curation** | RAG retrieval, few-shot selection | Low | Minutes | Medium |
| **SFT on accepted outputs** | Model weights | Medium | Hours-days | Medium-high |
| **RL with verifier as reward** | Model weights + policy | High | Days-weeks | High |
| **RL absorbed verification** | Model internalizes quality judgment | Very high | Weeks | Transformative |

The feedback loop is where the bitter lesson plays out. Each mechanism replaces more human knowledge with more computation, and each scales better than the last.

---

## Part 2: The Verification Spectrum by Domain

The five primitives are universal. What changes by domain is which verification tiers are available.

### Domain Mapping

| Domain | Hard Verifier | Strong Verifier | Consensus | Behavioral | Notes |
|--------|:---:|:---:|:---:|:---:|---|
| **Coding** | Compiler, type checker | Test suite execution | SVG / majority vote | Tool call traces | Best case — all tiers available |
| **Legal** | Regulatory clause check, citation exists | Attorney review ($500/hr) | Two independent analyses | Search depth, source diversity | Strong verifier is expensive |
| **Financial** | Formulas balance, data validated | Market outcomes (delayed) | Two independent valuations | Scenario coverage, assumption checks | Strong verifier has months of latency |
| **Medical** | Drug interaction check, dosage bounds | Patient outcome (delayed) | Two independent diagnoses | Differential diagnosis breadth | Hard verifier exists for safety-critical subset |
| **Research** | Citation verification, statistical checks | Peer review (months) | Two independent literature reviews | Source count, methodology coverage | Strong verifier extremely slow |
| **Operations/DevOps** | `terraform validate`, `helm lint` | Deploy + monitor (hours) | Two independent configs | Runbook coverage, rollback plan | Hard verifier exists for IaC |
| **Sales/Marketing** | Brand guideline checker, fact verification | A/B test (weeks) | Two independent drafts | Audience analysis depth | Weak hard verifier |
| **Customer Support** | Policy compliance check | Customer satisfaction (delayed) | Two independent responses | Ticket analysis depth, escalation patterns | Fast feedback via CSAT |

**Key observation**: Coding and Operations have hard verifiers. Most knowledge work tops out at "strong" (human expert) with significant cost and latency. This makes the soft/consensus/learned tiers proportionally more valuable in non-coding domains.

### The Cost Inversion

In coding, verification is cheap (run tests: $0.08/patch). The expensive part is generation.

In knowledge work, verification is expensive (expert review: $100-500/instance). The expensive part is verification.

This means the ROI of a learned verifier is **higher for knowledge work than for coding**:

| Domain | Expert Verification Cost | Learned Verifier Cost | Savings per Instance |
|--------|:---:|:---:|:---:|
| Coding | $0.08 (test execution) | $0.004 | $0.076 (20x) |
| Legal review | $500 (attorney hour) | $0.04 | **$499.96 (12,500x)** |
| Financial analysis | $200 (analyst hour) | $0.04 | **$199.96 (5,000x)** |
| Medical second opinion | $300 (specialist) | $0.04 | **$299.96 (7,500x)** |
| Research peer review | $100 (reviewer time) | $0.04 | **$99.96 (2,500x)** |

The learned verifier doesn't replace the expert entirely — but it triages. If the verifier pre-filters from 100 outputs to the 20 that need expert review, you've cut expert costs by 80%.

---

## Part 3: Implementation Playbook

A four-phase methodology for deploying the verification framework on any knowledge workflow. Each phase builds on the previous and can be stopped at any point — every phase delivers standalone value.

### Phase 0: Workflow Decomposition (1-2 days)

**Goal**: Map the customer's workflow to the five primitives.

**Steps**:

1. **Identify the generator**: What AI system produces the output? (Coding agent, document drafter, analysis pipeline, etc.)

2. **Identify existing verification**: How does quality get checked today? Map to the spectrum:
   - Is there any automated check? (Hard verifier)
   - Is there expert review? How much does it cost? (Strong verifier)
   - Is there any telemetry from the agent? (Behavioral signals)

3. **Measure the gap**: What's the fix rate vs. pass rate?
   - Fix rate: % of outputs that look plausible
   - Pass rate: % that actually meet quality standards
   - Gap = verification opportunity

4. **Estimate verification cost**: What does the customer spend on quality control today?
   - Expert review hours × hourly rate
   - Rework cycles × cost per cycle
   - Escaped defects × cost per defect

**Deliverable**: Workflow decomposition document mapping primitives, current verification tiers, gap analysis, and cost baseline.

**Template**:

```
Workflow: [name]
Generator: [system]
  Fix rate: [X%]
  Pass rate: [Y%]
  Gap: [X-Y]pp

Current verification:
  Hard: [what exists, if any]
  Strong: [expert review process, cost]
  Soft: [any telemetry collected?]
  Learned: [none / existing model]

Verification cost baseline:
  Expert reviews/month: [N]
  Cost/review: [$X]
  Annual verification spend: [$Y]
  Rework cost: [$Z]
```

### Phase 1: Consensus Verifier (1-2 weeks)

**Goal**: Deploy SVG-pattern consensus verification. No training needed.

**Steps**:

1. **Implement consensus generation**: For each output, generate a second independent version:
   - Summarize the first output as a specification
   - Generate a new output from only the specification
   - Compare the two outputs

2. **Define comparison metric**: Domain-specific similarity measure:
   - Coding: line-level diff recall (our SVG approach)
   - Legal: key clause overlap + conclusion agreement
   - Financial: numerical range overlap + directional agreement
   - Research: citation overlap + claim agreement
   - Operations: config diff + resource overlap

3. **Calibrate threshold**: Run on 50-100 historical examples with known outcomes:
   - Sweep threshold from 0.5 to 1.0
   - Find precision/recall trade-off point
   - Target: precision > 0.9 at maximum recall

4. **Deploy as triage layer**:
   - Above threshold → auto-accept (or light review)
   - Below threshold → full expert review
   - Track precision/recall in production

**Deliverable**: Deployed consensus verifier with calibrated threshold. Expected result: 40-60% of outputs auto-accepted, reducing expert review volume by 40-60%.

**Cost**: ~2 inference calls per output ($0.04-0.08). Engineering time for comparison metric and threshold calibration.

**Our empirical baseline**: SVG consensus on coding achieved precision=1.000 and recall=0.528 at threshold=0.8, reducing test execution by 52.8%.

### Phase 2: Behavioral Verifier (2-4 weeks)

**Goal**: Add process telemetry signals to the consensus verifier.

**Steps**:

1. **Instrument the generator**: Capture behavioral telemetry on every run:
   - Action sequence (what tools/sources used, in what order)
   - Timing (how long per step, time-to-first-output)
   - Repetition (loops, retries, dead-end explorations)
   - Context consumption (how much input processed)
   - Source diversity (how many distinct sources consulted)

2. **Correlate with outcomes**: Join telemetry with consensus scores and/or expert decisions:
   - Run feature-level analysis (point-biserial correlation, univariate AUC)
   - Identify which behavioral features predict quality
   - Expect: 2-4 features with significant signal, rest are noise

3. **Build lightweight classifier**: XGBoost or logistic regression on behavioral features:
   - No LLM fine-tuning — classical ML is correct at n=200-1000
   - Leave-one-out or k-fold cross-validation
   - Measure: AUC, precision at target recall, ECE (calibration)

4. **Combine with consensus**: Ensemble the behavioral score with the consensus score:
   - Behavioral verifier identifies outputs consensus misses (false negatives)
   - Combined verifier should have higher recall at same precision

**Deliverable**: Behavioral feature set, trained classifier, ensemble verifier with improved recall over consensus alone.

**Our empirical baseline**: At n=23, behavioral features alone are underpowered (AUC 0.542). At n=300, consensus dominates (AUC 0.981). The hypothesis: behavioral features add to consensus by recovering false negatives. Untested at adequate sample size.

### Phase 3: Learned Verifier (1-3 months)

**Goal**: Train a model that approximates expert verification at inference cost.

**Prerequisites**: Phase 1 consensus verifier has been running in production, accumulating labeled data (consensus-accepted + expert-reviewed outcomes).

**Steps**:

1. **Accumulate training data**: From Phase 1-2 production deployment:
   - Consensus-accepted outputs with downstream outcomes (auto-labeled)
   - Expert-reviewed outputs with accept/reject decisions (human-labeled)
   - Target: 1,000+ labeled examples for classical ML, 10,000+ for LLM fine-tuning

2. **Train verifier model**:
   - **Option A (n=1K-10K)**: XGBoost/gradient boosting on feature vector (behavioral + output features)
   - **Option B (n=10K+)**: LLM fine-tuning with generative classification (YES/NO token, following SWE-RM methodology)
   - **Option C (n=100K+)**: Full reward model with pairwise ranking objective (R4P methodology)

3. **Measure calibration (critical)**:
   - Compute ECE (Expected Calibration Error)
   - SWE-RM finding: two verifiers with identical ranking diverged catastrophically under RL due to calibration gap
   - If ECE > 0.3, do not use for RL integration — recalibrate first

4. **Deploy as replacement for consensus**:
   - 1 inference call instead of 2 (consensus)
   - Higher recall (recovers consensus false negatives)
   - Same or better precision

**Deliverable**: Deployed learned verifier, reducing per-output verification cost to a single inference call while maintaining precision.

### Phase 4: Feedback Loop (Ongoing)

**Goal**: Route verifier signal back to improve the generator.

**Steps (escalating investment)**:

1. **Prompt optimization**: Use verifier scores to identify failure modes → update system prompt with guardrails. Free, immediate.

2. **Few-shot curation**: Use verifier to select best historical outputs as few-shot examples. Low cost, high impact.

3. **SFT on verified outputs**: Fine-tune the generator on verifier-accepted outputs. Medium cost, improves base pass rate.

4. **RL with verifier reward**: Use the learned verifier as the reward model for RL training of the generator. High cost, highest impact. The verifier becomes the training signal.

**Phase 4 is where the bitter lesson completes.** The generator absorbs the verifier's judgment into its weights. Over time, the generator needs less verification because it's internalized quality standards. Cursor Composer 2's RL results (50% less compaction error) demonstrate this is already happening for coding.

---

## Part 4: Decision Framework

### Should You Implement This?

```
Q1: Does the agent produce outputs that need quality review?
    ├─ NO  → Framework doesn't apply
    └─ YES → Continue

Q2: What's the current verification cost?
    ├─ < $1K/month  → Phase 1 only (consensus). ROI is modest.
    └─ > $1K/month  → Continue

Q3: Is expert review the bottleneck?
    ├─ NO (verification is fast/cheap)  → Invest in generator instead
    └─ YES (review is slow/expensive)   → Continue — high ROI

Q4: Can you generate two independent outputs?
    ├─ NO  → Skip Phase 1, go to Phase 2 (behavioral only)
    └─ YES → Start Phase 1 (consensus). Expected: 40-60% review reduction.

Q5: Do you have 200+ labeled examples?
    ├─ NO  → Stay at Phase 1-2, accumulate data
    └─ YES → Phase 3 (learned verifier). Expected: 70-85% review reduction.

Q6: Do you have 10K+ examples and RL infrastructure?
    ├─ NO  → Stay at Phase 3
    └─ YES → Phase 4 (feedback loop). Generator improves autonomously.
```

### Expected ROI by Phase

| Phase | Investment | Timeline | Review Reduction | Typical Annual Savings |
|-------|:---:|:---:|:---:|:---:|
| **0: Decompose** | 2 days consulting | 1-2 days | — (baseline) | — |
| **1: Consensus** | 1-2 weeks eng | 2 weeks | 40-60% | Verification cost × 0.5 |
| **2: Behavioral** | 2-4 weeks eng | 1 month | 50-70% | Verification cost × 0.6 |
| **3: Learned** | 1-3 months eng | 3 months | 70-85% | Verification cost × 0.8 |
| **4: Feedback** | Ongoing | 6+ months | 80-95% | Verification cost × 0.9 + generator improvement |

### Sizing Guide

| Team Size | Issues/Month | Expert Cost/Review | Annual Verification Spend | Framework ROI (Phase 1-3) |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 200 | $100 | $240K | $120K-200K saved |
| 50 | 1,000 | $100 | $1.2M | $600K-1M saved |
| 50 | 1,000 | $500 (legal/medical) | $6M | $3M-5M saved |
| 200 | 4,000 | $100 | $4.8M | $2.4M-4M saved |

---

## Part 5: The Bitter Lesson Trajectory

The framework is not static. It follows the bitter lesson — each phase replaces human knowledge with computation:

```
Phase 0  Human expert reviews everything
           100% human knowledge, 0% compute
           ↓
Phase 1  Consensus verifier auto-accepts high-confidence outputs
           Compute replaces ~50% of human judgment
           ↓
Phase 2  Behavioral signals triage the rest
           Compute replaces ~70% of human judgment
           ↓
Phase 3  Learned verifier approximates expert
           Compute replaces ~85% of human judgment
           ↓
Phase 4  Generator absorbs verification via RL
           Compute replaces ~95% of human judgment
           ↓
Endstate Agent knows when its own output is wrong
           100% compute, human reviews exceptions only
```

The implication for implementation: **don't over-invest in any single phase.** Each phase is a stepping stone. The domain-specific features you engineer in Phase 2 will be obsoleted by the learned verifier in Phase 3, which will be absorbed by the generator in Phase 4. Build each phase to be good enough, extract value, and move forward.

The enterprise moat is not domain expertise encoded as features. It's the **infrastructure to run the loop** — generate candidates fast, verify cheaply, accumulate data, and retrain. The team that iterates this loop fastest wins.

---

## Appendix: Empirical Foundation

This framework is grounded in empirical data from coding agent experiments. Key results that inform the methodology:

| Finding | Source | Implication |
|---------|--------|-------------|
| SVG consensus: precision=1.000, AUC=0.981 (n=300) | Our Phase 0 experiment | Consensus verification works with zero domain engineering |
| Behavioral features: below baseline at n=23 | Our Phase 0 experiment | Need n=200+ for behavioral signals; don't skip to Phase 2 early |
| Critic Rubrics: +15.9 Best@8 from 24 behavioral features | arXiv:2603.03800 | Process features predict quality across codebases |
| Benchmark-only critics: AUC 0.48 on real-world data | Critic Rubrics paper | Domain-specific training doesn't transfer; process features do |
| SWE-RM: ECE matters more than ranking accuracy | arXiv:2512.21919 | Calibration is critical before using verifier for RL |
| R4P: 72.2% accuracy, 50x faster than tests | arXiv:2510.22775 | Learned verifiers can replace execution-based verification |
| DeepSWE: 250x cost reduction with verifier reward | Together AI | RL with learned rewards is feasible and economic |
| Cursor Composer 2: model absorbs verification via RL | Cursor blog | The endstate — verification internalized in weights |
| Fix rate 82% vs pass rate 17.7% (our n=300) | Our SVG data | The verification gap is the primary bottleneck |

---

*This methodology is designed to be adapted. The primitives are universal; the domain-specific details (comparison metrics, behavioral features, threshold calibration) change per deployment. Start with Phase 0 decomposition and let the data guide which phases deliver value.*
