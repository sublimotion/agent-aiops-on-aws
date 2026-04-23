# The Verification Framework: Primitives and Implementation Methodology

**Date**: 2026-03-20
**Audience**: Forward deploy teams implementing agent systems for knowledge workflows
**Status**: Draft — grounded in empirical coding agent data, extending to general knowledge work

## What This Document Is

A methodology for analyzing any knowledge workflow, identifying verification opportunities, and implementing a tiered verifier that extends an agent's autonomous horizon. The pattern applies to any domain where an AI agent produces outputs that need quality assessment — coding, legal, financial, research, operations.

The framework decomposes into **five primitives** and a **four-phase implementation playbook**.

---

## Context: Why Not Just Generate Tests?

The obvious objection to this entire framework: if the problem is missing test suites, why not have LLMs generate comprehensive tests from project artifacts (PRDs, design docs, business requirements) and get a hard verifier? Then it's just throwing compute at the problem — bitter lesson applies directly.

**This objection is largely correct for well-documented codebases, and it represents the likely 12-18 month endgame.** The verification spectrum exists because of the practical gap between that endgame and today's reality.

### Where LLM-generated tests work

Given sufficient context — PRDs, design documents, acceptance criteria, existing test patterns — an LLM has enough signal to generate meaningful tests. This is exactly what QA engineers do: read the spec, understand intended behavior, write tests that assert it. The capability is real and improving fast.

For a specific repo, the path is:
1. Feed PRD + design docs + existing test patterns to an LLM
2. Generate a comprehensive test suite once
3. Now you have a hard verifier for that repo going forward

The amortized cost of generating the test suite is trivial compared to the ongoing value of having a hard verifier. This is the most bitter-lesson-aligned approach to verification.

### Why this isn't the dominant approach today

**Artifact availability**: Most enterprise codebases don't have clean, current, colocated project artifacts. The PRD exists in Confluence, the design doc is stale, the requirements are spread across Jira tickets. The context *exists* but isn't structured for LLM consumption. This is a context engineering problem that's closing fast (context windows went from 4K → 1M in two years, RAG is improving), but it's a real barrier today.

**Test execution infrastructure**: Even perfect generated tests require environment setup — databases, message queues, external services, correct dependency versions. This is the DeepSWE problem: 512 Docker containers, 64 H100s, 6 days. Test *generation* isn't expensive; test *execution* is. For the RL inner loop (100K+ iterations), execution infrastructure is the dominant cost. SVG consensus and learned verifiers produce signal from inference calls alone — zero infrastructure overhead.

**Generated test reliability**: A generated test that passes on a wrong patch is worse than no test — it gives false confidence. SWE-bench Verified exists because even gold human-written tests had ~15% noise. LLM-generated tests have their own precision/recall problem, creating a recursive verification need.

### The practical landscape

| Codebase Type | Best Verification Path |
|---------------|----------------------|
| Well-documented + CI/CD mature | LLM-generated tests → hard verifier (your endgame) |
| Legacy + sparse docs | SVG consensus → learned verifier (this framework) |
| No test infra + compliance-heavy | Skill verifier → consensus → learned (Phases 0.5-3) |
| RL training loop (any codebase) | Learned verifier as reward (execution too slow at 100K+ iters) |

### Where this framework fits

The verification spectrum is the bridge for the gap between "generate tests and run them" and today's reality:

- **Today**: Most repos lack clean artifacts, test infra is friction, context engineering isn't solved. The verification spectrum provides tiered approaches that work now.
- **Near-term**: Context engineering matures, test generation from specs becomes reliable, execution infrastructure commoditizes. LLM-generated test suites become standard for well-documented codebases.
- **Endgame**: LLM-generated tests as the default hard verifier. The learned verifier persists for the RL reward loop (where execution latency matters) and for codebases where test execution is impractical.

The framework below is designed for the full spectrum — including organizations that will eventually generate their way to a hard verifier but need verification *today*.

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

Every verifier has four properties in tension:

| Property | Definition | Trade-off |
|----------|-----------|-----------|
| **Precision** | When it says "good," how often is it right? | Higher precision → fewer false positives → safer to auto-accept |
| **Recall** | Of all good outputs, how many does it catch? | Higher recall → fewer missed opportunities → less human fallback |
| **Calibration (ECE)** | Does the confidence score match the actual probability? | Low ECE → safe as RL reward signal. High ECE → use for ranking only, not RL (SWE-RM: identical ranking, 7x ECE difference → RL collapse) |
| **Cost** | Time, compute, or human effort per verification | Lower cost → more candidates verified → better best-of-N |

**Why calibration matters separately from precision/recall**: A verifier can rank outputs correctly (high AUC) while being poorly calibrated (high ECE). For best-of-N selection, ranking is sufficient. For RL training, miscalibration "couples multiplicatively with the policy gradient, injecting additional variance" (SWE-RM, arXiv:2512.21919), causing training collapse. This distinction determines which use cases a verifier supports. See RLVR_AND_VERIFICATION.md for the full analysis.

**Noise tolerance**: "Noisy Data is Destructive to RLVR" (arXiv:2603.16140) proved that algorithmic improvements cannot compensate for noisy rewards — reward signal quality is more important than algorithm choice. This validates precision-first verification design: SVG consensus at precision=1.0 means zero noise in positive labels.

The **verification spectrum** orders verifiers by these properties:

```
                         Precision    Recall    ECE     Cost           Needs Domain Setup?
                         ─────────    ──────    ───     ────           ───────────────────
Hard verifier            ~100%        ~100%     ~0      Free           Yes (formal spec)
Strong verifier          80-100%      80-100%   Low     High           Yes (test suite/expert)
Adversarial verifier     Improving    Improving Low     Medium-High    Yes (co-evolving tests)
Skill verifier (v009)    96.3%*       14.9%*    0.092*  $0.030         Yes (rubric design)
Consensus (SVG)          96.3%*       14.9%*    0.031†  $0             No
Multi-agent debate       72.5%*       59.2%*    0.234*  $0.024         No
Behavioral/process       71.3%*       ~78%*     0.055*  ~$0            No
Learned verifier (RF)    93.3%‡       30%‡      0.072*  ~$0            Yes (training data)
Learned verifier (RL)    AUC 0.707*   —         0.031*  ~$0            Yes (all signals)
Implicit PRM (PRIME)     70-85%       70-85%    Online  Near-zero      Yes (outcome labels)
No verifier              N/A          N/A       N/A     Free           No

*  Our empirical data from 6 experiments (SWE-bench Lite, n=300)
†  After Platt scaling (raw ECE = 0.512)
‡  P@R≥30% operating point
```

**New tiers from RLVR research**:
- **Adversarial verifier** (EvolveCoder, arXiv:2603.12698): Test cases co-evolve with the policy, getting harder as the model improves. Verification quality increases over training rather than staying fixed. Expensive but addresses the Goodhart plateau.
- **Implicit PRM** (PRIME, arXiv:2502.01456): Derives process rewards from token-level log-probability ratios between policy and reference model. No separate reward model, no step-level annotations. 2.5x sample efficiency vs outcome-only RL. Online updates prevent reward hacking.
- **Auto-generated environments** (ReSyn, arXiv:2602.20117): Autonomously generates reasoning environments with built-in verifiers. Most bitter-lesson-aligned approach — the verification environment itself is learned, not engineered.

The **Skill verifier** is the critical bootstrap tier. It encodes domain expertise as a structured verification procedure (rubric, checklist, or evaluation prompt) that an agent executes against an output. It works at N=1, requires zero training data, and generates the labeled evaluations needed to eventually train a learned verifier. In domains without test suites (legal, financial, research), the skill verifier is often the highest available tier. Anthropic's [Complete Guide to Building Skills](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) provides the implementation patterns; our framework adds verification as the missing Pattern 0.

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

**Feedback mechanisms by cost, impact, and overoptimization risk**:

| Mechanism | What Changes | Cost | Latency | Impact | Overoptimization Risk |
|-----------|-------------|------|---------|--------|-----------------------|
| **Prompt refinement** | System prompt / examples | Free | Immediate | Low-medium | None |
| **Context curation** | RAG retrieval, few-shot selection | Low | Minutes | Medium | None |
| **Rejection sampling → SFT** | Model weights (curated data) | Medium | Hours-days | Medium-high | Low (data selection only) |
| **Best-of-N selection** | No weights changed | Low | Per-inference | Medium | Low (log form, Gao et al.) |
| **RL with verifier as reward** | Model weights + policy | High | Days-weeks | High | **Medium-High (√KL form)** |
| **RL absorbed verification** | Model internalizes quality judgment | Very high | Weeks | Transformative | Requires careful KL control |

The feedback loop is where the bitter lesson plays out. Each mechanism replaces more human knowledge with more computation, and each scales better than the last.

**The overoptimization boundary** (Gao et al., arXiv:2210.10760): For RL, gold reward follows `R_gold = α√(d_KL) − β·d_KL` — initial improvement then decline as KL divergence increases. For best-of-N, the functional form is logarithmic — less susceptible to overoptimization. This means **best-of-N with a learned verifier is inherently safer than RL with the same verifier**. The phased approach (Phase 1-2: best-of-N, Phase 3: learned verifier, Phase 4: RL) is not just incremental — it follows the overoptimization safety gradient.

**RL infrastructure requirements**: Phase 4 requires the RL environment stack — training loop framework (veRL, TRL, OpenRLHF), rollout engine (vLLM with sleep mode or disaggregated), and reward model serving. OpenEnv (Meta + HuggingFace) provides a standardized `step()/reset()/close()` API for environment instantiation. See RLVR_AND_VERIFICATION.md for the full stack architecture and RL-Post-Training-Infrastructure-Patterns for framework comparison.

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

### Phase 0.5: Skill Verifier (2-5 days)

**Goal**: Build and deploy a verification skill — a structured evaluation procedure that encodes domain expertise. This is the minimum viable verifier and the bootstrap mechanism for all subsequent phases.

**Why this phase exists**: In domains without test suites (most knowledge work), there's no free source of labels. A skill verifier generates labeled evaluations as a side effect of running, solving the cold start problem for Phases 2-3.

**Steps**:

1. **Identify what "good" looks like**: Work with domain experts to enumerate quality criteria. Not a single "is this good?" judgment, but decomposed checks:
   - Coding: Does the patch compile? Does it modify the right files? Is the diff minimal?
   - Legal: Are all citations valid? Does the analysis address all clauses? Is the conclusion supported?
   - Financial: Do the numbers balance? Are assumptions stated? Is the methodology appropriate?

2. **Build the verification skill**: Following Anthropic's skill patterns, create a skill folder:
   ```
   verification-skill/
   ├── SKILL.md                    # Core verification rubric
   ├── scripts/
   │   ├── validate.py             # Deterministic checks (hard verifier subset)
   │   └── score_rubric.py         # Structured rubric scoring
   └── references/
       └── quality-criteria.md     # Domain-specific standards
   ```

3. **Instrument for telemetry**: The skill should output structured evaluations, not just pass/fail:
   ```json
   {
     "output_id": "...",
     "rubric_scores": {"criterion_1": 0.8, "criterion_2": 1.0, ...},
     "deterministic_checks": {"compiles": true, "lint_pass": true},
     "overall_score": 0.85,
     "confidence": "high",
     "failure_reasons": []
   }
   ```
   These structured evaluations ARE the training data for Phase 3's learned verifier.

4. **Test and calibrate**: Run the skill on 20-50 historical examples with known outcomes:
   - Does the skill's scoring correlate with expert judgment?
   - Which rubric criteria are most predictive?
   - Iterate the rubric based on disagreements (the guide's "iterate with Claude" pattern)

5. **Deploy as initial gate**: Use the skill as a first-pass filter:
   - High-score outputs → light review
   - Low-score outputs → full expert review
   - Track: how often does the expert agree with the skill?

**Deliverable**: Deployed verification skill generating structured evaluations on every output. Expert agreement rate as the initial quality metric. Structured evaluation data accumulating for Phase 3.

**Cost**: ~1 LLM inference call per verification ($0.02-0.04). Engineering time for rubric design and skill creation (2-5 days, following Anthropic's skill-creator pattern).

**Key design principles** (from Anthropic's guide + our framework):
- **Decomposability**: Check independent aspects separately (Pattern 3's `scripts/check_report.py` approach)
- **Deterministic where possible**: Bundle validation scripts for checks that don't need LLM judgment (p.26: "code is deterministic; language interpretation isn't")
- **Progressive disclosure**: Keep the core rubric lean, link detailed criteria from `references/`
- **Calibration**: Consistent scoring across runs — test by running the same output 5 times

**Connection to subsequent phases**: The skill verifier's structured evaluations become:
- Phase 1: Comparison metric calibration data
- Phase 2: Behavioral feature correlation labels
- Phase 3: Training data for the learned verifier
- Phase 4: Initial reward signal before the learned verifier is ready

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

3. **Measure calibration (critical — this is the RL gate)**:
   - Compute ECE (Expected Calibration Error) — not just AUC
   - SWE-RM finding: two verifiers with identical TTS ranking (+4.7% vs +4.5%) had 7x ECE difference (0.481 vs 0.069). RL training with the poorly-calibrated one **collapsed**.
   - **ECE < 0.1**: Safe for RL reward signal (Phase 4)
   - **ECE 0.1-0.3**: Safe for best-of-N and rejection sampling, needs calibration for RL
   - **ECE > 0.3**: Use for ranking/triage only — recalibrate before any training use
   - SWE-RM achieved ECE 0.047-0.051 with 400K+ trajectories and 2:1 positive-to-negative ratio
   - R4P mitigates calibration issues via group-wise verification (comparing N=4 patches against each other, denser signal)

4. **Deploy as replacement for consensus**:
   - 1 inference call instead of 2 (consensus)
   - Higher recall (recovers consensus false negatives)
   - Same or better precision

5. **Determine use-case boundary based on ECE**:
   - Low ECE → Phase 4 RL use is viable
   - High ECE → Stay at best-of-N selection (inherently safer per Gao et al. overoptimization scaling laws: log form vs √KL form)
   - Consider hybrid reward (SWE-RM pattern): combine learned verifier with test execution where available for +3pp over either alone

**Deliverable**: Deployed learned verifier with measured ECE, reducing per-output verification cost to a single inference call while maintaining precision. ECE determines whether the verifier is Phase 4-ready.

### Phase 4: Feedback Loop (Ongoing)

**Goal**: Route verifier signal back to improve the generator.

**Steps (escalating investment, following the overoptimization safety gradient)**:

1. **Prompt optimization**: Use verifier scores to identify failure modes → update system prompt with guardrails. Free, immediate. No overoptimization risk.

2. **Few-shot curation**: Use verifier to select best historical outputs as few-shot examples. Low cost, high impact. No overoptimization risk.

3. **Rejection sampling → SFT**: Use verifier to filter the best outputs, fine-tune generator on accepted outputs only. This is what Allen AI's SERA did with SVG — SFT on SVG-accepted trajectories, $2K total cost, 24.4% → 49.5% on SWE-bench. SERA's finding that "verification threshold doesn't matter for SFT" simplifies this step. Low overoptimization risk.

4. **Best-of-N selection at inference**: Generate N candidates, verifier selects top-1. No weight changes, but requires N× inference budget. Overoptimization follows log form (Gao et al.) — inherently safe. This is the recommended default for most deployments.

5. **RL with verifier reward (GRPO)**: Use the learned verifier as reward model in a GRPO training loop. **Requires ECE < 0.1** (Phase 3 gate). Overoptimization follows √KL form — monitor proxy vs gold reward divergence. Infrastructure: veRL/TRL/OpenRLHF + vLLM rollout engine. R4P demonstrated this works: Mini-SE trained entirely with learned rewards achieved 32.8% Pass@1.

6. **Implicit PRM (advanced)**: PRIME-style token-level rewards derived from policy log-probabilities. No separate reward model. Online updates prevent reward hacking. 2.5x sample efficiency vs outcome-only RL. Highest leverage but most complex to implement.

**Failure modes to monitor at Steps 5-6**:
- **Reward hacking**: Agent learns surface features that fool the verifier. Mitigate with group-wise verification (R4P) or mixed-policy training data (SWE-RM).
- **Calibration drift**: Policy improves, verifier becomes less calibrated on new distribution. PRIME's online updates solve this; static verifiers need periodic recalibration.
- **Goodhart plateau**: Gold reward peaks at `d_KL = α²/(4β²)` then declines. Monitor KL divergence throughout training.
- **Context truncation**: SWE-RM found truncated trajectories receive no valid reward. Need 256K context support for long agent traces.

**Phase 4 is where the bitter lesson completes.** The generator absorbs the verifier's judgment into its weights. Over time, the generator needs less verification because it's internalized quality standards. Cursor Composer 2's RL results (50% less compaction error) demonstrate this is already happening for coding. The emerging RL environment ecosystem (35+ companies per SemiAnalysis, OpenEnv standardizing APIs) is making this phase increasingly accessible.

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

## Appendix A: Metrics Glossary

This glossary defines every metric used in the verification framework, explains what each one measures in plain language, and — critically — specifies *when each metric matters and when it doesn't*. The verification research program spans six experiments producing dozens of numbers. Not all numbers are equally relevant in every context. A metric that is essential for RL reward selection (ECE) is irrelevant for auto-accept decisions (where precision is what matters). Choosing the wrong metric for the task leads to failures that look correct on paper — SWE-RM demonstrated this when two verifiers with nearly identical AUC produced opposite RL training outcomes because their ECE differed 7x.

Each entry follows the same structure: **what it measures** (formula), **intuition** (plain-language analogy), **when it matters** (which decisions depend on it), and **when it doesn't** (where it misleads). Where relevant, entries include our empirical values from SWE-bench Lite (n=300) to ground the definitions in real data.

### Precision

**What it measures**: Of all outputs the verifier labels as "correct," what fraction actually are correct?

```
Precision = True Positives / (True Positives + False Positives)
```

**Intuition**: If precision = 0.85, then 15% of auto-accepted patches are wrong. A precision of 0.96 (v009) means you can trust its "likely_correct" verdict almost unconditionally.

**When it matters**: Auto-accept decisions. Every false positive is a bad patch shipped to production. In high-stakes domains (medical, financial), precision requirements are 0.95+.

**When it doesn't matter**: RL training. RL can tolerate some label noise — what matters is calibration (see ECE below).

### Recall

**What it measures**: Of all outputs that actually are correct, what fraction does the verifier catch?

```
Recall = True Positives / (True Positives + False Negatives)
```

**Intuition**: If recall = 0.15 (v009), the verifier misses 85% of correct patches — they go to expensive human review unnecessarily. If recall = 0.59 (debate), the verifier catches most correct patches.

**When it matters**: Cost reduction. Low recall means most good work still needs human review. High recall is what reduces the review burden.

**The precision-recall trade-off**: You can always increase recall by lowering the acceptance threshold, but precision drops. The operating point depends on the cost of false positives vs. false negatives in your domain.

### P@R≥30% (Precision at Recall ≥ 30%)

**What it measures**: What's the highest precision achievable while still catching at least 30% of correct outputs?

**Why this specific metric**: It captures the sweet spot for triage — you want to auto-accept the top tier (high precision) without being so selective that you only speak on 5% of patches (low recall). The "30%" threshold means the verifier is useful on a meaningful fraction of outputs.

**Our best**: P@R≥30% = 0.933 (Phase 3 selected RF). This means: at a threshold where the model accepts at least 30% of truly correct patches, 93.3% of its acceptances are actually correct.

### AUC (Area Under the ROC Curve)

**What it measures**: The probability that the verifier ranks a random correct output higher than a random incorrect output.

```
AUC = P(score(correct) > score(incorrect))
```

**Intuition**: AUC = 0.5 means random guessing. AUC = 1.0 means perfect ranking. AUC = 0.727 (our best) means if you pick one correct and one incorrect patch, the model gives the correct one a higher score 72.7% of the time.

**When it matters**: Best-of-N selection. If you generate N patches and want to pick the best one, ranking ability (AUC) is what you need. The actual probability values don't matter — only the ordering.

**When it doesn't matter**: RL training. Two verifiers with identical AUC can have 7x different ECE, and only the well-calibrated one works for RL (SWE-RM finding). AUC measures ranking; RL needs magnitudes.

### ECE (Expected Calibration Error)

**What it measures**: How well the model's predicted probabilities match actual outcomes. When the model says "80% confident," do 80% of those patches actually pass?

```
ECE = Σ (n_bin / n_total) × |avg_confidence_in_bin − actual_pass_rate_in_bin|
```

**How it's computed**:
1. Bin predictions by confidence level (e.g., 0-10%, 10-20%, ..., 90-100%)
2. In each bin, compare the **average predicted probability** to the **actual pass rate**
3. ECE = weighted average of these gaps across all bins

**Intuition**: ECE is like reliability in weather forecasting. When the weather app says "80% chance of rain," it should rain ~80% of those days. If it rains only 50% of the time when the forecast says 80%, the forecast is poorly calibrated — high ECE. The forecast might still *rank* rainy days above dry ones correctly (good AUC), but the *numbers themselves are wrong*.

**Why it matters for RL**: RL policy gradients multiply by the reward value. If the verifier says "0.80" but the true probability is 0.50, the gradient is systematically biased by 0.30. Over thousands of training steps, this bias compounds and the policy diverges. SWE-RM demonstrated this empirically: two verifiers with nearly identical ranking (AUC within 0.2pp) had 7x different ECE (0.481 vs 0.069). The poorly-calibrated one collapsed RL training. The well-calibrated one worked.

**Thresholds**:
- ECE < 0.05: Excellent — safe for RL reward signal
- ECE < 0.10: Good — usable for RL with monitoring
- ECE 0.10-0.30: Marginal — use for ranking/triage only, or recalibrate first
- ECE > 0.30: Poor — not usable as a continuous signal

**ECE vs confidence intervals**: These measure different things. A confidence interval says "the true metric lies in this range given our sample size" — it measures **uncertainty about a point estimate**. ECE measures **systematic bias in probability predictions** — whether the model's stated confidences are honest. A model can have narrow confidence intervals (lots of data, low uncertainty) but terrible ECE (systematically overconfident).

**How to fix bad ECE**: Post-hoc calibration methods remap predictions without changing ranking:
- **Platt scaling** (logistic regression): fits a sigmoid to map predictions → calibrated probabilities. We used this on SVG scores: ECE dropped from 0.512 → 0.031.
- **Isotonic regression**: non-parametric monotonic mapping. More flexible than Platt, but can overfit at small n.
- **Temperature scaling**: divides logits by a learned temperature T. Simple, preserves ranking, but only corrects overall confidence level (not per-bin).

**Our ECE landscape (from 6 experiments)**:

| Verifier | ECE | Notes |
|----------|-----|-------|
| SVG raw | 0.512 | Degenerate — 82% of scores are zero |
| Debate (2-round) | 0.234 | Overconfident on CORRECT verdicts |
| v009 RF (10 features) | 0.092 | Close to threshold |
| Phase 3 selected RF (4 features) | 0.072 | RL-ready |
| Behavioral-only RF | 0.055 | Good |
| Phase 3 all-signals RF | 0.031 | Best at full coverage (94%) |
| SVG + Platt | 0.031 | Best but only 9.6% coverage |
| Phase 3 all-signals RF + Platt | 0.026 | Best overall |

### Brier Score

**What it measures**: Mean squared error between predicted probabilities and actual outcomes.

```
Brier = (1/n) Σ (predicted_probability − actual_outcome)²
```

**Intuition**: Combines both calibration and discrimination into a single number. A lower Brier score is better. Unlike ECE (which only measures calibration), Brier penalizes both miscalibration and poor ranking.

**When it matters**: When you need a single number to compare models on both ranking and calibration simultaneously. Useful as a tiebreaker when AUC and ECE point in different directions.

**Our range**: 0.202 (Phase 3 best) to 0.266 (XGBoost worst). For reference, always predicting the base rate (0.583) gives Brier = 0.243.

### F1 Score

**What it measures**: Harmonic mean of precision and recall at a specific threshold (usually 0.5).

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Intuition**: Balances precision and recall into a single number. F1 = 1.0 means perfect precision and recall. F1 = 0.652 (debate) means a reasonable balance but neither precision nor recall is stellar.

**When it matters**: Comparing verifiers at their default operating points. Useful for quick comparisons but hides the precision-recall trade-off.

**When it's misleading**: When your application cares much more about precision than recall (or vice versa). In verification, we often want very high precision with acceptable recall — F1 doesn't capture this preference. Use P@R≥30% instead.

### Mutual Information (MI)

**What it measures**: How much knowing one variable reduces uncertainty about another. Used in pivot analysis to rank behavioral signals by their predictive power.

```
MI(X; Y) = Σ P(x,y) × log(P(x,y) / (P(x) × P(y)))
```

**Intuition**: MI = 0 means the variables are independent (knowing X tells you nothing about Y). Higher MI means stronger association. MI is symmetric and captures non-linear relationships that correlation misses.

**Our range**: Tool usage (MI=0.0668) is the strongest single predictor. For context, MI is measured in bits — 0.0668 bits means knowing whether VP tools were used reduces uncertainty about the gold outcome by ~6.7% of the maximum possible.

**When it matters**: Feature selection and understanding which behavioral signals carry the most predictive value. MI is threshold-free, unlike precision/recall which depend on a cutoff.

---

## Appendix B: Empirical Foundation

This framework is grounded in empirical data from coding agent experiments. Key results that inform the methodology:

| Finding | Source | Implication |
|---------|--------|-------------|
| SVG consensus: precision=1.000, AUC=0.981 (n=300) | Our Phase 0 experiment | Consensus verification works with zero domain engineering |
| Behavioral features: below baseline at n=23 | Our Phase 0 experiment | Need n=200+ for behavioral signals; don't skip to Phase 2 early |
| Critic Rubrics: +15.9 Best@8 from 24 behavioral features | arXiv:2603.03800 | Process features predict quality across codebases |
| Benchmark-only critics: AUC 0.48 on real-world data | Critic Rubrics paper | Domain-specific training doesn't transfer; process features do |
| SWE-RM: ECE matters more than ranking accuracy | arXiv:2512.21919 | Calibration is critical before using verifier for RL |
| R4P: 72.2% accuracy, 50x faster than tests | arXiv:2510.22775 | Learned verifiers can replace execution-based verification |
| R4P Mini-SE: 32.8% trained entirely with learned rewards | arXiv:2510.22775 | RL with learned verifier reward works end-to-end |
| DeepSWE: 512 containers, 64 H100s, 6 days, ~$100K+ | Together AI / SemiAnalysis | Test-execution RL is expensive; learned verifier = 250x cheaper |
| SWE-RM: ECE 7x difference → RL collapse | arXiv:2512.21919 | Calibration (not ranking) is the gate for RL use |
| SWE-RM: hybrid reward +3pp over execution-only | arXiv:2512.21919 | Combining learned + execution rewards is better than either alone |
| Noisy rewards: 8-10pp underperformance, algorithm can't fix | arXiv:2603.16140 | Reward quality > algorithm quality; validates precision-first design |
| Gao et al.: RL overoptimizes (√KL), best-of-N doesn't (log) | arXiv:2210.10760 | Best-of-N with learned verifier is inherently safer than RL |
| PRIME: 2.5x sample efficiency, implicit PRM | arXiv:2502.01456 | Process rewards from policy itself — no separate model needed |
| GRPO = implicit PRM (mathematical proof) | arXiv:2509.21154 | GRPO already does token-level credit assignment from outcome rewards |
| SERA SVG: $2K training, 49.5% SWE-bench | arXiv:2601.20789 | SVG-filtered SFT is cheap and effective (Phase 4 Step 3) |
| Agentic Rubrics: 54.2%, designed as RL reward | arXiv:2601.04171 | Context-grounded checklists bridge verification → RL |
| EvolveCoder: co-evolving tests, +4.2pp | arXiv:2603.12698 | Adversarial verification quality improves over training |
| Cursor Composer 2: model absorbs verification via RL | Cursor blog | The endstate — verification internalized in weights |
| Fix rate 82% vs pass rate 17.7% (our n=300) | Our SVG data | The verification gap is the primary bottleneck |
| 35+ RL environment companies | SemiAnalysis (2026-03-22) | Verification/reward infrastructure is a distinct market |
| OpenEnv: standardized step/reset/close API | Meta + HuggingFace | Environment layer is being standardized for RL |
| SVG raw ECE=0.512, Platt-scaled ECE=0.031 | Our SVG ECE experiment | Post-hoc calibration can rescue degenerate scores |
| Debate 2-round: prec=0.725, rec=0.592, F1=0.652 | Our debate experiment | Multi-agent argumentation breaks the recall barrier (4x v009) |
| v009+debate combined: prec=0.882, rec=0.662 | Our debate experiment | Debate recovers 58% of v009 FNs; v009 catches 100% of debate FPs |
| Pivot: tool usage MI=0.0668, risk diff +46.3% | Our pivot analysis | Tool usage is the #1 behavioral predictor of patch success |
| Tiny judge: 5-feat RF, P@R30=0.949 | Our tiny-judge experiment | 4 novel behavioral features beat v009 (AUC 0.670 vs 0.550) |
| Phase 3: 4-feat RF, AUC=0.727, ECE=0.072 | Our Phase 3 experiment | Behavioral features dominate; LLM signals help calibration not discrimination |
| Phase 3 all-signals RF: ECE=0.031 at 94% coverage | Our Phase 3 experiment | Best RL reward: v009+debate+SVG improve calibration even when they hurt AUC |
| Behavioral features dominate forward selection | Our Phase 3 experiment | At n=300, adding LLM features (v009/debate/SVG) to behavioral hurts AUC (curse of dimensionality) |

---

## Companion Documents

- **VERIFIER_ECONOMICS.md** — Cost analysis: $370K-$1.35M/yr savings, ROI by team size
- **RLVR_AND_VERIFICATION.md** — Deep research on RLVR, RL environments, and the verification-to-reward bridge
- **RL-Post-Training-Infrastructure-Patterns** (vault) — Framework comparison: veRL, OpenRLHF, TRL, NeMo-RL
- **PHASE0_REPORT.md** — Empirical results: SVG AUC 0.981, behavioral feature analysis

---

*This methodology is designed to be adapted. The primitives are universal; the domain-specific details (comparison metrics, behavioral features, threshold calibration) change per deployment. Start with Phase 0 decomposition and let the data guide which phases deliver value.*
