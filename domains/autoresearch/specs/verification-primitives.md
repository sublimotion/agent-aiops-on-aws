# Autoresearch Spec: Verification Primitives

## Status: COMPLETE

## Overview

Arm coding agents with verification primitives — test generation, test execution, and adversarial review — as composable tools, then measure whether agents learn to compose them into effective workflows. This is explicitly NOT an engineered multi-agent pipeline (InfCode, ChatDev). The agent itself decides when and how to use verification tools.

**Core hypothesis**: Agents given verification primitives as callable tools will discover effective verification patterns (front-loaded testing, adversarial review before submission) that improve pass rate, without human-engineered orchestration. The composition is learned, not designed.

**Motivation**: CoderForge's 413K-trajectory analysis found that early test fraction is the single strongest predictor of agent success (56.3% concordance, 12,286 within-issue pairs). But those agents only had access to existing repo tests. What happens when you give agents the ability to *generate* their own verification? This extends the verification spectrum from "run existing tests" (hard verifier) to "create then run tests" (skill → hard composition).

**Bitter lesson connection**: InfCode's approach (79.4% SWE-bench Verified) hard-wires Test Agent ↔ Code Agent ↔ Selector as a fixed pipeline. The topology is human knowledge encoded in architecture. This spec tests whether giving the agent durable primitives and letting it compose them produces comparable results — and whether the composition generalizes better than the engineered pipeline.

**Counter-evidence to address**: arXiv:2602.07900 ("Rethinking the Value of Agent-Generated Tests") found that ad-hoc test writing during issue resolution has marginal utility. Agents prefer print statements over formal assertions. The hypothesis: this fails because casual test writing is confirmatory. Adversarial test generation ("write tests designed to break your patch") should produce qualitatively different results — same as our v009 finding that adversarial framing beats confirmatory by 2.3x.

**Depends on**: verifier-reward (Phase 2 verification skill data, v009 adversarial rubric baseline), agent-harness (eval infrastructure, 50-issue subset, behavioral telemetry)

**Distinction from post-training on tool calling**: This spec targets a specific level of verification capability. There are four levels on the progression, and they are not interchangeable:

1. **Tool competence** — Model can call pytest, format tool calls correctly. This is baseline capability. Post-training on tool-calling data (e.g., fine-tuning on OpenHands tool-call format) targets this level.
2. **Behavioral cloning via RFT** — Model replicates successful trajectory patterns end-to-end, including testing behavior. Nebius (25.2% → 50.3%) and SERA (24.4% → 49.5%) demonstrate this level. The model learns *that* testing leads to success but doesn't develop judgment about *when* and *which kind* of verification to apply. It's pattern replication, not composition.
3. **Primitive composition** (this spec) — Model learns to compose verification primitives with judgment: when verification is needed, what kind, how much. The agent develops the meta-skill of deploying verification strategically — not just "run tests" but "generate adversarial tests for this specific edge case, then run them, then review if results are ambiguous." This is qualitatively different from level 2 because the agent must reason about verification, not just replicate patterns.
4. **Absorbed verification** — Model generates verified-quality output without needing external tools. Cursor Composer 2 (RL post-training on Kimi K2.5) demonstrates early-stage level 4. Requires RL with calibrated verifier reward (ECE < 0.1). This is the endstate on the verification framework.

RFT (level 2) is a necessary precursor — the model needs tool competence and behavioral patterns. But this spec asks: can an agent go beyond replicating successful patterns to developing verification *judgment*? The measurement is whether the agent's tool invocation patterns are contextually adaptive (different verification strategies for different problem types) rather than uniform (always do the same thing).

## Research Questions

1. **Does giving agents a test-generation tool improve pass rate?** Compare: agent with standard tools vs agent with standard tools + test-generation primitive. Same model, same issues, same budget.

2. **Does adversarial framing matter for test generation?** Compare: "write tests for your patch" (confirmatory) vs "write tests designed to break your patch" (adversarial). The counter-evidence paper tested only confirmatory framing.

3. **Do agents discover TDD patterns when given the primitives?** Measure early test fraction, test-generation timing, and test-execution timing. Does the agent front-load testing (CoderForge's success pattern) when it has the tools to do so?

4. **Does skill→hard composition beat skill-only or hard-only?** Compare: v009 adversarial rubric alone (0.92 precision, 0.14 recall) vs generated-test execution alone vs the composition (generate adversarial tests, run them, then v009 on survivors). Does the composition close the recall gap?

5. **What's the cost-performance frontier?** Test generation adds an LLM call ($0.01-0.05) + test execution time (10-60s). Is the pass rate gain worth the latency and cost?

6. **Does the composition transfer across model tiers?** If Haiku discovers TDD patterns with primitives, does the same emerge for Sonnet/Opus? Or do stronger models not need the scaffolding?

## Phases

### Phase 1: Primitive Design & Baseline (Week 1-2)

**Goal**: Design the verification primitives as agent-callable tools and establish baselines without them.

**Steps**:

1. **Design three verification primitives as tools**:

   **Tool 1: `generate_tests`**
   - Input: problem_statement + current_patch_diff + optional: source_files
   - Output: test file content (pytest format)
   - Two variants:
     - `generate_tests_confirmatory`: "Write tests that verify this patch correctly fixes the issue"
     - `generate_tests_adversarial`: "Write tests designed to break this patch. Target edge cases, boundary conditions, and assumptions the patch might have missed. Your goal is to find a failing test."
   - Implementation: Claude API call (Haiku for cost, Sonnet for quality comparison)
   - Cost: ~$0.01-0.03/call

   **Tool 2: `run_tests`**
   - Input: test file content + target repo state (base + patch applied)
   - Output: test results (pass/fail per test, stderr, stdout)
   - Implementation: subprocess in sandboxed workspace (same as gold eval infrastructure)
   - Timeout: 60s per test suite
   - Cost: compute only, ~$0.01/run

   **Tool 3: `adversarial_review`**
   - Input: problem_statement + patch_diff + optional: test_results
   - Output: structured verdict (reuse v009 schema: problem_alignment, logic_correctness, completeness, scope, test_safety)
   - Implementation: v009 adversarial rubric, single call (not 4-call ensemble — agent can call multiple times if it wants)
   - Cost: ~$0.008/call

2. **Implement tools as OpenCode extensions**:
   ```
   skills/verification-primitives/
   ├── SKILL.md                    # Tool descriptions for agent context
   ├── tools/
   │   ├── generate_tests.py       # LLM call → test file
   │   ├── run_tests.py            # Sandbox execution → results
   │   └── adversarial_review.py   # v009 single-call → verdict
   └── prompts/
       ├── confirmatory.md         # "Write tests that verify..."
       └── adversarial.md          # "Write tests that break..."
   ```

3. **Run baselines (no verification primitives)**:
   - Reuse verifier-reward Phase 1 data if available
   - Otherwise: 50 issues × Haiku + Sonnet × OpenCode, standard tool set
   - Capture: pass rate, fix rate, behavioral telemetry (turns, actions, timing)

**Exit criteria**:
- Three tools implemented and tested on 5 known-outcome issues
- Generated tests actually run (no syntax errors, import errors, etc.)
- Baseline pass rates established for comparison
- Cost per tool call measured

**Estimated cost**: ~$30-50 (baseline runs + tool testing)

### Phase 2: Single-Primitive Experiments (Week 3-4)

**Goal**: Test each verification primitive in isolation. Does having the tool available improve outcomes?

**Steps**:

1. **Experiment matrix** (each cell = 50 issues):

   - **A: generate_tests_confirmatory** — Agent has standard tools + confirmatory test generation. Tool description: "You can generate tests to verify your fix. Call this before submitting."
   - **B: generate_tests_adversarial** — Agent has standard tools + adversarial test generation. Tool description: "You can generate adversarial tests designed to break your fix. Call this before submitting."
   - **C: run_tests only** — Agent has standard tools + ability to run existing repo tests (no generation). Mirrors CoderForge's finding directly.
   - **D: adversarial_review only** — Agent has standard tools + v009 single-call review. Tool description: "You can request an adversarial code review of your patch."
   - **Control**: Standard tools only (Phase 1 baseline)

2. **Critical design choice — no forced usage**:
   - Tools are AVAILABLE, not REQUIRED
   - Agent system prompt mentions them but doesn't mandate when to use them
   - Measure: how often does the agent voluntarily invoke each tool?
   - Measure: does invocation timing match CoderForge's TDD pattern (early = better)?

3. **For cells A and B, chain with execution**:
   - If agent calls `generate_tests`, automatically offer `run_tests` on the generated tests
   - This tests the skill→hard composition naturally — the agent generates (skill), then validates (hard)

4. **Capture per-run**:
   - Standard telemetry (turns, actions, cost, latency)
   - Tool invocation log: which verification tools called, when (turn number), how many times
   - Tool results: test generation quality (do generated tests compile? run? catch real bugs?)
   - Final outcome: pass rate on gold tests

5. **Analyze**:
   - Pass rate per cell vs control
   - Tool usage patterns: when does the agent invoke verification? (Early, late, never?)
   - Does adversarial test generation outperform confirmatory? (Addresses counter-evidence)
   - Does voluntary usage correlate with success? (Validates CoderForge at primitive level)

**Exit criteria**:
- All 5 cells completed (250 total runs)
- Pass rate per cell with confidence intervals
- Tool usage frequency and timing analyzed
- Adversarial vs confirmatory test generation compared
- If any cell beats control by >3pp: that primitive adds value
- If agent never uses the tools: prompt design needs iteration (→ Phase 2b)

**Estimated cost**: ~$50-150 (depends on model tier, 250 runs × Haiku ~$0.10-0.30/run)

### Phase 2b: Guided Composition (Week 4-5, if Phase 2 shows low voluntary usage)

**Goal**: If agents don't voluntarily use verification primitives, test whether light guidance improves adoption without hard-wiring the pipeline.

**Steps**:

1. **Guidance variants** (not forced, but encouraged):

   - **Nudge**: Add to system prompt: "Best practice: before submitting your final patch, generate adversarial tests and run them. Patches that pass adversarial tests are more likely to be correct."
   - **Checkpoint**: At 70% of turn budget, inject: "You have N turns remaining. Before submitting, consider running adversarial tests on your patch."
   - **TDD-first**: Add to system prompt: "Start by running the existing test suite to reproduce the bug. After your fix, generate additional tests targeting edge cases, then run all tests before submitting."

2. **Compare**: nudge vs checkpoint vs TDD-first vs unguided (Phase 2) vs control (no tools)

3. **Measure**: Does guidance increase tool usage? Does increased usage translate to higher pass rate? Or does guidance just waste budget (the counter-evidence paper's concern)?

**Exit criteria**:
- Guidance variants tested on 50 issues each
- Tool adoption rate measured (% of runs that invoke verification tools)
- Pass rate compared to Phase 2 unguided and Phase 1 control
- If TDD-first guidance + primitives beats control by >5pp: the combination works

**Estimated cost**: ~$30-100

### Phase 3: Composition Analysis (Week 5-6)

**Goal**: Analyze how agents compose verification primitives and compare emergent patterns to engineered pipelines.

**Steps**:

1. **Classify emergent composition patterns** from Phase 2/2b trajectories:
   - **TDD pattern**: generate tests early → edit → run tests → iterate (CoderForge's success pattern)
   - **Review-before-submit**: edit → adversarial review → fix → submit
   - **Belt-and-suspenders**: edit → generate tests → run tests → adversarial review → submit
   - **Ignore**: agent never uses verification tools
   - **Late check**: agent uses tools only in final 2-3 turns (probably too late)

2. **Correlate patterns with outcomes**:
   - Which composition pattern has the highest pass rate?
   - Does the best emergent pattern match InfCode's engineered pipeline?
   - Or does a novel pattern emerge that no one designed?

3. **Compare to engineered baselines** (from literature):
   - InfCode: adversarial test-patch co-evolution (79.4% SWE-bench Verified)
   - Agentless: LLM-generated tests + AST-normalized voting
   - TDAD: dependency-graph-targeted test execution (70% regression reduction)
   - Our v009: adversarial rubric alone (0.92 precision, 0.14 recall)

4. **Measure the composition gap**:
   - Best emergent composition vs best engineered pipeline
   - If emergent ≈ engineered: primitives are sufficient, pipelines are unnecessary
   - If engineered >> emergent: agents can't yet compose verification effectively, need scaffolding
   - If emergent > engineered: agent discovers novel patterns human engineers missed

**Exit criteria**:
- Composition patterns classified and mapped to outcomes
- Comparison to InfCode / Agentless / v009 documented
- Clear answer: do agents compose verification primitives effectively?

**Estimated cost**: Analysis only (no new runs needed)

### Phase 4: Cross-Model Transfer & Scaling (Week 7-8, contingent)

**Goal**: Test whether verification primitive composition transfers across model tiers and task difficulty.

**Prerequisites**: Phase 2 or 2b shows positive results (>3pp improvement).

**Steps**:

1. **Model tier comparison** (best primitive/guidance from Phase 2/2b):
   - Haiku + primitives vs Haiku alone
   - Sonnet + primitives vs Sonnet alone
   - Does the lift decrease for stronger models? (Sonnet may already self-verify)

2. **Cost-performance analysis**:
   - Haiku + primitives vs Sonnet alone (cheaper model + tools vs expensive model solo?)
   - What's the cost-equivalent comparison? (same $/issue, different strategies)

3. **Scale test**:
   - Expand from 50 to 200 issues (full SWE-bench Lite subset)
   - Do results hold at larger N?
   - Do new composition patterns emerge with more diverse tasks?

4. **Produce training data for RFT pipeline**:
   - Every (patch, generated_tests, test_results, gold_outcome) tuple → labeled dataset
   - Generated tests that correctly predict gold outcome → supervision signal for learned verifier
   - This bridges to the learned verifier pipeline (Phase 3 of verification framework)
   - **RFT connection**: Nebius showed rejection fine-tuning on verified trajectories doubles pass rate (Qwen3-30B: 25.2% → 50.3%, no RL needed). SERA's SVG-filtered SFT achieved similar gains (24.4% → 49.5%). If verification primitives produce higher-quality trajectories, those become better SFT training data. The flywheel: primitives → better trajectories → filtered SFT → better base model → better primitive usage.
   - Format trajectories as OpenHands-compatible JSONL for direct RFT compatibility with Nebius/CoderForge pipelines

**Exit criteria**:
- Cross-model comparison completed (at least Haiku + Sonnet)
- Cost-performance frontier plotted (primitives vs model scaling)
- If Haiku + primitives ≈ Sonnet alone: verification primitives substitute for model capability
- Labeled dataset of 200+ (patch, tests, outcome) triples produced
- Trajectories formatted for RFT compatibility
- Decision: proceed to learned verifier training, RFT pipeline, or iterate on primitives

**Estimated cost**: ~$100-300

## Components

### 1. Compute
- **Platform**: Local machine + Anthropic API (Phase 1-3). No GPU required.
- **Sandbox**: Local workspace per issue (git clone at base_commit, apply patch, run tests). Reuse agent-harness eval infrastructure.
- **Concurrency**: Anthropic API parallel requests. 50 issues can run concurrently.

### 2. Codebase
- **Source**: Existing harness infrastructure from agent-harness + verifier-reward specs
  - OpenCode — primary agent harness (open-source, tool set extensible)
  - `patch_eval.py` — gold test evaluation (reuse)
  - `verify_patch.py` — v009 adversarial verifier (reuse as adversarial_review tool)
- **New code** (agent-editable):
  - `skills/verification-primitives/tools/generate_tests.py` — test generation via Claude API
  - `skills/verification-primitives/tools/run_tests.py` — sandboxed test execution
  - `skills/verification-primitives/tools/adversarial_review.py` — v009 single-call wrapper
  - `scripts/run_primitives_experiment.py` — experiment runner (matrix of cells)
  - `scripts/analyze_composition.py` — Phase 3 composition pattern classifier
- **Fixed files** (agent must NOT edit):
  - SWE-bench issue definitions and gold test patches
  - Existing baseline results (comparison data)
  - v009 adversarial rubric text (used as-is in adversarial_review tool)
- **Agent instructions**: `program.md` — autoresearch loop protocol

### 3. Experiment Protocol
- **Primary metric**: Pass@1 on SWE-bench Lite 50-issue subset (verified by gold tests)
- **Secondary metrics**: fix rate, precision (pass/fix), cost/issue, tool invocation rate, tool invocation timing (turn number), composition pattern classification
- **Behavioral metrics** (per CoderForge): early test fraction, file scatter count, repeated command count, turns to first edit
- **Eval subset**: Same 50 issues as agent-harness and verifier-reward (seed 42, stratified by 11 repos)
- **Gold evaluation**: Apply agent patch + gold test_patch, run FAIL_TO_PASS tests. Same method as agent-harness.
- **Logging**: JSONL per run with full telemetry + tool invocation log. Tool results stored alongside.

### 4. Networking
- **Anthropic API**: Direct HTTPS calls. No SSH, no GPU instance needed.
- **Rate limits**: Haiku/Sonnet should be fine for 50 sequential issues.

### 5. Storage
- **Results**: `domains/autoresearch/blueprints/verification-primitives/results/`
- **Skills**: `domains/autoresearch/blueprints/verification-primitives/skills/`
- **Comparison data**: verifier-reward + agent-harness results (read-only reference)
- **Generated tests**: stored per-issue for analysis and training data

## Data Assets (Existing)

- **Verifier-reward Phase 2 results**: v009 adversarial rubric baseline (0.92 precision, 0.14 recall on n=483). Comparison point for adversarial_review tool.
- **Agent-harness Phase 1 turn metrics** (6,599 rows): Behavioral baseline, Parkinson's Law data.
- **Agent-harness Phase 2 multi-harness results**: Comparison baselines across harnesses.
- **CoderForge-Preview dataset** (413K trajectories): External validation of behavioral signals, TDD finding.
- **Nebius OpenHands trajectories** (67K trajectories, 32K successful): RFT recipe — Qwen3-30B 25.2% → 50.3% with verified-trajectory SFT. Validates Feedback Loop primitive (mechanism #3).
- **Counter-evidence paper** (arXiv:2602.07900): Null result for ad-hoc test writing — our adversarial framing addresses this.

## Data Assets (New — produced by this spec)

- **Tool invocation logs**: When and how agents use verification primitives (Phase 2)
- **Generated test suites**: Per-issue adversarial/confirmatory tests (Phase 2)
- **Composition pattern classifications**: Emergent verification workflows (Phase 3)
- **Labeled training data**: (patch, generated_tests, test_results, gold_outcome) tuples (Phase 4) — feeds learned verifier training

## Success Criteria

### Phase 1: Primitives Work
- Three tools implemented, tested, and callable by agent
- Generated tests compile and execute (>80% of generated suites run without errors)
- Baselines established for comparison

### Phase 2: Primitives Improve Outcomes
- At least one primitive cell beats control by >3pp pass rate
- Adversarial test generation outperforms confirmatory (addresses counter-evidence)
- Agent voluntarily uses tools in >50% of runs (validates tool design)
- If agent never uses tools: proceed to Phase 2b (guidance)

### Phase 3: Composition Patterns Emerge
- Identifiable composition patterns classified
- Best emergent pattern documented with pass rate
- Comparison to InfCode/Agentless quantified
- Clear answer: agents compose verification primitives effectively (or not)

### Phase 4: Results Generalize
- Cross-model transfer confirmed (Haiku + Sonnet)
- Cost-performance frontier: primitives vs model scaling
- 200+ labeled (patch, tests, outcome) triples produced
- Decision point: learned verifier training viable?

### Negative Results (Still Valuable)

- **Agents don't use primitives voluntarily** → Composition doesn't emerge naturally; engineered pipelines (InfCode) are necessary. This validates the "human knowledge in architecture" approach and challenges the bitter lesson for agent composition.
- **Adversarial test generation = confirmatory** → The framing doesn't matter for test generation (unlike for rubric-based verification). Would narrow the adversarial advantage to reasoning-only verification.
- **Generated tests don't catch real bugs** → LLM-generated tests have low discriminative power. Validates SWE-ABS finding that coverage-driven test augmentation needs structural analysis, not just LLM generation.
- **Primitives help Haiku but not Sonnet** → Stronger models internalize verification. Verification primitives are scaffolding for weaker models, not a universal enhancement. This would be a bitter lesson datapoint: model capability absorbs the need for external verification tools.

## Non-Requirements
- GPU infrastructure (API-only experiment)
- Multi-agent orchestration (single agent with tools, not agent pipeline)
- RL training or weight updates
- Full SWE-bench Verified (500 issues) — use 50-issue subset, expand to 200 in Phase 4
- Docker-based test execution (use local sandbox, same as agent-harness eval)
- InfCode reproduction (we compare against their published numbers, don't reimplement)

## Known Limitations
- 50-issue eval subset introduces ~5% sampling variance (mitigated in Phase 4 at 200 issues)
- Gold evaluation limited to Django/pytest/sympy without Docker — pass rate is lower bound
- Generated test quality depends on model tier — Haiku may generate weaker tests than Sonnet
- Test execution in sandbox may miss environment-specific failures (missing dependencies, etc.)
- Agent may game the tools (generate trivially-passing tests to satisfy "run tests" without real verification)
- No existing test suite for all SWE-bench issues — some issues' repos may lack pytest infrastructure, making generated tests harder to run

## Risk Register

- **Agent games verification tools** (MEDIUM probability, HIGH impact): Agent generates trivial tests that always pass. Mitigation: measure generated test discriminative power (do they distinguish good patches from known-bad patches?). If tests pass on both good and bad patches, they're useless.
- **Generated tests don't compile** (MEDIUM, MEDIUM): LLM-generated tests have import errors, syntax issues. Mitigation: add a test-repair step (1 retry with error message). Measure compilation rate as a quality metric.
- **Phase 2 counter-evidence replicates** (MEDIUM, LOW): Adversarial framing doesn't help test generation either. This is a valid negative result — document and publish.
- **Tool invocation overhead exceeds budget** (LOW, MEDIUM): Agent spends all turns on verification, no time to edit. Mitigation: monitor turn allocation. If >70% of turns go to tool use, the budget is misallocated.
- **Anthropic API rate limits** (LOW, LOW): 250+ runs may hit limits. Mitigation: sequential issues, exponential backoff.

## Relationship to Other Specs

- **verifier-reward**: Provides v009 adversarial rubric (reused as adversarial_review tool), Claude baselines (Phase 1 data), and verification skill methodology. This spec extends verification from post-hoc scoring to in-loop primitives.
- **agent-harness**: Provides eval infrastructure, 50-issue subset, behavioral telemetry framework, and Parkinson's Law baselines. This spec adds new behavioral metrics (tool invocation patterns).
- **learned-verifier**: This spec produces training data (Phase 4) that feeds the learned verifier pipeline. Generated tests that correctly predict outcomes become supervision signal.
- **coderforge-eval**: CoderForge-Preview data provides external validation for behavioral signals. This spec tests whether CoderForge's TDD finding can be induced through tool design.

## Key References

- CoderForge 413K trajectory analysis (Li et al., 2026): TDD as strongest behavioral predictor
- InfCode (Li et al., arXiv:2511.16004): Adversarial test-patch co-evolution, 79.4% SWE-bench Verified
- "Rethinking Agent-Generated Tests" (Chen et al., arXiv:2602.07900): Counter-evidence for ad-hoc test writing
- Otter (Ahmed et al., arXiv:2502.05368): Test generation from issue descriptions
- TDAD (Alonso et al., arXiv:2603.17973): Dependency-graph targeted test execution, 70% regression reduction
- UTBoost/SWE-ABS (Yu et al., arXiv:2506.09289): 20% of "solved" patches actually wrong — weak test suites
- Nebius OpenHands trajectories (2026): 67K trajectories, RFT doubles Qwen3-30B pass rate without RL
- SERA/SVG (Allen AI, arXiv:2601.20789): SVG-filtered SFT 24.4% → 49.5%, consensus AUC 0.981
- Our v009 adversarial verifier: 0.92 precision, adversarial > confirmatory by 2.3x
- Verification Framework (this project): Five primitives, four phases, spectrum model

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
