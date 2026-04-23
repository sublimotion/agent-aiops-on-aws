# Verification Primitives — Progress

## Phase 1: Primitive Design & Baseline

### Iteration 1 — Blueprint scaffolding

**Status**: COMPLETE

**Completed**:
- Blueprint directory structure created
- Three verification primitives implemented:
  - `generate_tests.py` — LLM-generated test cases (confirmatory + adversarial modes) via Bedrock Haiku
  - `run_tests.py` — sandboxed pytest execution with per-test results
  - `adversarial_review.py` — v009 adversarial rubric single-call wrapper (reuses verifier-reward infrastructure)
- Prompt templates written (confirmatory.md, adversarial.md)
- SKILL.md tool descriptions for agent context
- Experiment runner (`run_primitives_experiment.py`) with 6 cells:
  - control: standard tools only
  - A: + confirmatory test generation
  - B: + adversarial test generation
  - C: + run_tests (repo tests only)
  - D: + adversarial_review
  - E: all verification primitives
- Smoke test script (`smoke_test.py`) with known-good and known-bad patches

### Iteration 2 — Smoke tests (2026-03-29)

**Status**: COMPLETE

All 3 verification primitives validated:
- `generate_tests` (adversarial): 5717 chars, compiles OK, $0.008, 10.4s latency
- `generate_tests` (confirmatory): 4565 chars, compiles OK, $0.006, 6.8s latency
- `run_tests`: 1 pass + 1 fail as expected, 550ms
- `adversarial_review` (good patch): verdict=likely_correct, score=0.99, $0.003
- `adversarial_review` (bad patch): verdict=likely_incorrect, score=0.10
- Discriminative power: good patch 0.99 vs bad patch 0.10 (correct ordering)

### Iteration 3 — Control baseline (2026-03-29)

**Status**: COMPLETE

Ran 2 control baselines:
- Prior run (30 turns): 2/3 fix rate (from smoke_control.jsonl)
- New run (15 turns): 0/5 fix rate, $0.50 total cost

**Finding**: 15 turns too short for Haiku to reliably edit. Agent spends all turns exploring (Parkinson's Law confirmed again — consistent with agent-harness Phase 1 finding).

## Phase 2: Single-Primitive Experiments

### Iteration 4 — All cells, 5-issue smoke (2026-03-29)

**Status**: COMPLETE

Ran all 6 cells on 5 issues (pylint-5859, requests-1963, matplotlib-18869, pytest-11143, sklearn-10297) with 15-turn budget, Haiku model.

| Cell | Description | Fix Rate | Tool Users | Cost |
|------|-------------|----------|------------|------|
| control | Standard tools only | 0/5 (0%) | 0/5 (0%) | $0.50 |
| A | + confirmatory test gen | 1/5 (20%) | 0/5 (0%) | $0.45 |
| B | + adversarial test gen | 1/5 (20%) | 0/5 (0%) | $0.57 |
| C | + run_tests | 0/5 (0%) | 1/5 (20%) | $0.52 |
| D | + adversarial review | 1/5 (20%) | 0/5 (0%) | $0.43 |
| E | All primitives | 1/5 (20%) | 0/5 (0%) | $0.59 |

**Critical finding: 0% voluntary tool usage** across cells A, B, D, E. The agent never invokes verification primitives (generate_tests, adversarial_review) when they are merely available. Only `run_tests` was used once (cell C, pylint-5859 at turn 8), and it did not lead to a fix.

**Fix rate analysis**: Fixes are distributed across different issues per cell (A: requests-1963, B: pytest-11143, D: requests-1963, E: pylint-5859). This is noise at n=5 — no cell reliably beats control. The 20% fix rate in treatment cells vs 0% in control is not statistically significant (Fisher exact p > 0.4).

**Root causes for zero tool adoption**:
1. **Parkinson's Law dominates**: Agent uses all 15 turns exploring code. No turns remain for verification.
2. **No edit-first habit**: Agent must FIRST generate a patch before verification tools are useful. With Haiku's late-editing behavior, the tools are never reachable in the workflow.
3. **Tool description insufficient**: The system prompt mentions tools but doesn't create urgency to use them. The agent treats them as optional enrichment, not workflow-critical.

**Decision**: This triggers Phase 2b (guided composition). The spec anticipated this: "If agent never uses the tools: prompt design needs iteration."

## Phase 2b: Guided Composition

### Iteration 5 — Guidance variants, 5-issue smoke, 30-turn budget (2026-03-29)

**Status**: COMPLETE

Increased turn budget to 30 (from 15 in iter 4) to give agent room for edit+verify cycle. Added 4 guidance variants.

| Cell | Description | Fix Rate | Tool Users | First Tool Turn | Cost |
|------|-------------|----------|------------|----------------|------|
| control | No tools | 4/5 (80%) | 0/5 | N/A | $1.69 |
| B | Adversarial (unguided) | 3/5 (60%) | 0/5 | N/A | $1.43 |
| D | Review (unguided) | 4/5 (80%) | 0/5 | N/A | $1.68 |
| B_mandatory | "You MUST use tools" | 3/5 (60%) | **2/5 (40%)** | 23.5 | $1.73 |
| **B_checkpoint** | **Injection at 70% budget** | **4/5 (80%)** | **3/5 (60%)** | **25.7** | **$1.85** |
| B_tdd | TDD workflow prompt | 2/5 (40%) | 0/5 (0%) | N/A | $1.49 |

**Key findings**:

1. **Checkpoint injection is the best guidance variant**: 60% tool adoption + 80% fix rate (matches control).
2. **Mandatory language works but hurts**: 40% adoption but lower fix rate (60%). The verbose system prompt and verification overhead consume turns.
3. **TDD-first guidance is worst**: Complex 7-step workflow overwhelms the agent. 0% tool adoption, 33% fix rate. Too much instruction = less action.
4. **Unguided primitives = zero adoption**: Confirmed at 30 turns too. Merely describing tools doesn't cause usage.
5. **Tool usage is always late**: First tool call at turn 23-29 (77-97% of budget). Consistent with Parkinson's Law.
6. **Generated test quality mixed**: 0/0 test collection (import errors) for requests-1963 in both mandatory and checkpoint. 9/9 pass for pytest-11143. 3/4 for pylint-5859. The generate_tests prompt needs work for repo-specific imports.

**Per-issue tool invocation detail (B_checkpoint)**:
- pylint-5859: FIX, generate_tests at T28 (8325 chars, 3/4 pass), run_tests at T29
- requests-1963: FIX, generate_tests at T21 (14K chars, 0/0 collected — import error), run_tests at T22
- matplotlib-18869: FIX, no tools used
- pytest-11143: FIX, generate_tests at T28 (5690 chars, 9/9 pass), run_tests at T29
- sklearn-10297: NOEDIT, no tools used

**Emerging pattern**: Agent reliably chains generate_tests → run_tests when prompted. This is the skill→hard composition from the spec (generate tests = skill, run them = hard verifier). However, adversarial_review is never used even when available alongside.

### Iteration 6 — Full 50-issue experiment (2026-03-29)

**Status**: COMPLETE

Ran control and B_checkpoint on all 50 issues, 30-turn budget, Haiku model. Then ran gold evaluation on all generated diffs.

#### Fix Rate Results

| Cell | Fix Rate | Tool Users | Cost/Issue | Total Cost |
|------|----------|-----------|-----------|-----------|
| Control | 22/50 (44%) | 0/50 (0%) | $0.340 | $17.01 |
| **B_checkpoint** | **30/50 (60%)** | **9/50 (18%)** | $0.356 | $17.82 |

**Fix rate improvement: +16pp** (60% vs 44%). Net +8 issues (gained 9, lost 1).

#### Gold Eval (Pass Rate on Gold Tests)

| Cell | Applied | Pass | Precision |
|------|---------|------|-----------|
| Control | 22/22 | 1/22 (5%) | 5% |
| B_checkpoint | 30/30 | 1/30 (3%) | 3% |

**Both cells pass the same single issue**: mwaskom__seaborn-3010. Pass rate is ~2-5% for Haiku regardless of verification primitives. This is consistent with verifier-reward findings — Haiku's pass rate ceiling is very low on SWE-bench.

**Critical insight**: The pass rate is a Haiku model limitation, NOT a verification primitive failure. The verification tools increase fix rate (+16pp) but can't overcome the model's inability to produce correct patches. The agent generates "plausible-looking" edits that don't actually fix the underlying bugs.

#### Tool Usage Analysis

**Within B_checkpoint**:
- Tool users: 9/50 (18%), all 9 generated fixes (100% fix rate among tool users)
- Non-tool-users: 21/41 fixes (51% fix rate)
- **Tool use correlates with fix**, but the tool users' single gold pass (seaborn-3010) shows tools don't lift pass rate

**Tool invocation patterns**:
- generate_tests: 11 calls
- run_tests: 11 calls
- adversarial_review: 2 calls
- **Dominant chain**: generate_tests → run_tests (every tool user follows this)
- **Iterate pattern**: 2/9 users called generate→run twice (retry on failure)
- Average first tool turn: 20.9 (70% of budget)

**Test generation quality**:
- 11 suites generated, 6 compiled (55%), 50 tests run, 47 passed, 3 failed

#### Key Findings

1. **Checkpoint guidance increases fix rate by 16pp** (44% → 60%) at negligible cost (+$0.016/issue)
2. **Verification primitives do NOT improve pass rate** at the Haiku tier — model capability is the bottleneck, not verification
3. **100% fix rate among tool users** — agents that engage with verification tools always produce edits (vs 51% for non-users)
4. **Tool adoption is 18% at scale** — lower than the 60% seen in smoke test, but consistent
5. **The only gold pass (seaborn-3010) came from a tool user** — anecdotally positive but n=1

#### Implications

- **Verification primitives are most valuable for stronger models**: Haiku can't produce correct patches, so verification adds overhead without improving outcomes. A model like Sonnet or Opus (with 20-40% pass rate) would benefit more — verification could catch incorrect patches before submission.
- **The fix rate improvement is real but misleading**: More fixes ≠ more correct fixes. The verification tools encourage the agent to commit to an edit rather than abandoning, but the edits aren't actually correct.
- **The spec's Phase 4 (cross-model transfer) is now the critical next step**: Does checkpoint + verification primitives improve pass rate for Sonnet?

## Phase 4: Cross-Model Transfer (Sonnet)

### Iteration 7 — Full 50-issue Sonnet experiment (2026-03-29)

**Status**: COMPLETE

Ran control and B_checkpoint on all 50 issues with Sonnet 4.6, 30-turn budget.

#### Fix Rate

| Cell | Haiku | Sonnet |
|------|-------|--------|
| Control | 22/50 (44%) | 34/50 (68%) |
| B_checkpoint | 30/50 (60%) | **44/50 (88%)** |
| Delta | +16pp | **+20pp (p=0.028, significant)** |

#### Gold Pass Rate

| Cell | Haiku | Sonnet |
|------|-------|--------|
| Control | 1/22 (5%) | 3/34 (9%) |
| B_checkpoint | 1/30 (3%) | 3/44 (7%) |

Same 3 issues pass in both Sonnet cells: seaborn-3010, flask-4992, sphinx-11445. All 3 were tool users in checkpoint cell. No net change in pass rate.

#### Tool Adoption (Sonnet vs Haiku)

| Metric | Haiku | Sonnet |
|--------|-------|--------|
| Tool adoption | 18% | **58%** |
| Avg first tool turn | 20.9 (70% budget) | **19.0 (63% budget)** |
| generate_tests | 11 calls | **37 calls** |
| run_tests | 11 calls | **52 calls** |
| adversarial_review | 2 calls | **18 calls** |
| Test compile rate | 55% | **86%** |
| Tool user fix rate | 100% | **100%** |

#### Composition Patterns (Sonnet)

Sonnet discovers richer verification patterns than Haiku:

1. **Full verification pipeline** (6/29 tool users): generate → run → review → iterate
   - Example: requests-3362: gen(t16) → run(t17) → run(t18) → review(t19) → gen(t25) → run(t26) → run(t27) → review(t28)
2. **Generate-run-review** (9/29): The most common pattern. Generate tests, run, review before finishing.
3. **Generate-run** (14/29): Simple chain, same as Haiku's dominant pattern.
4. **Iterate after failure** (8/29): Agent re-generates tests or re-runs after failures — a behavior Haiku almost never exhibited.

Earliest tool use: turn 3 (seaborn-3010 — same early adopter as Haiku). Latest: turn 27.

#### Key Findings

1. **Sonnet adopts verification tools 3x more than Haiku** (58% vs 18%). Stronger models are better tool users.
2. **Fix rate gain is LARGER for Sonnet** (+20pp vs +16pp) and **statistically significant** (Fisher p=0.028).
3. **Pass rate does NOT improve** for either model tier. Verification primitives catch fixable issues but don't discriminate correct from incorrect patches — the gold tests require specific behaviors that the verifier can't predict.
4. **Sonnet uses adversarial_review 9x more** (18 vs 2 calls). Haiku barely used it; Sonnet actively seeks code review as part of verification.
5. **Test quality scales with model**: Sonnet compiles 86% of generated tests vs Haiku's 55%. The generated tests are more correct.
6. **All 3 Sonnet gold passes came from tool users** — but 26/29 tool users still fail gold tests. The 10% tool-user pass rate matches the 9% control precision.
7. **Verification overhead is negligible**: +$0.076/issue (7%) for Sonnet, +$0.016 (5%) for Haiku.

#### Implications

- **The spec's RQ6 ("Does composition transfer across model tiers?") is answered**: YES — stronger models adopt tools more readily, use richer patterns, and produce higher-quality generated tests. The lift in fix rate scales with model capability.
- **The spec's RQ4 ("Does skill→hard composition beat skill-only or hard-only?") is partially answered**: The generate→run chain (skill→hard) is the dominant emergent pattern. But it doesn't beat the v009 rubric alone on precision — it's a different axis (fix rate vs correctness).
- **The recall gap from verifier-reward persists**: v009's 0.14 recall means most correct patches get rejected. Verification primitives address a different problem — they help agents _produce_ patches, not _evaluate_ them.
- **For production use**: Checkpoint-guided verification primitives are a cheap (+7%) way to boost fix rate by 20pp. Whether this matters depends on downstream filtering — if you have a strong verifier, more fixes = more opportunities for a correct one to pass.

## Phase 3: Composition Analysis (from existing data)

### Iteration 8 — Pattern classification and cross-model comparison (2026-03-23)

**Status**: COMPLETE

Script: `scripts/analyze_composition.py`

#### Emergent Composition Patterns

Five distinct patterns emerged from B_checkpoint trajectories:

| Pattern | Description | Haiku | Sonnet |
|---------|-------------|-------|--------|
| ignore | No tools used | 41/50 (82%) | 21/50 (42%) |
| generate_run | gen → run (skill→hard) | 4/50 (8%) | 6/50 (12%) |
| generate_run_iterate | gen → run with retries | 3/50 (6%) | 6/50 (12%) |
| full_pipeline | gen → run → review | 2/50 (4%) | 7/50 (14%) |
| full_pipeline_iterate | gen → run → review with retries | 0/50 (0%) | 10/50 (20%) |

#### Adversarial Review Is the Differentiator

**Sonnet tool users with adversarial_review**: 3/17 gold pass (18%)
**Sonnet tool users without adversarial_review**: 0/12 gold pass (0%)

All 3 Sonnet gold passes came from full_pipeline users (with review). The review step appears to improve patch quality, not just fix rate.

Best emergent pattern: `full_pipeline` — 29% gold pass rate among 7 Sonnet users (2/7). Small n but the only pattern with gold passes.

#### Early Tool Use Predicts Success

| Timing Bucket | N | Fix Rate | Gold Pass |
|---------------|---|----------|-----------|
| <50% of budget | 3 | 100% | 33% (1/3) |
| 50-70% | 7 | 100% | 14% (1/7) |
| >70% | 19 | 100% | 5% (1/19) |

Consistent with CoderForge's finding that early test fraction is the strongest success predictor. But agents don't discover this pattern naturally — even with checkpoint injection, average first tool call is at 69% of budget.

#### Composition Complexity Scales with Model

| Metric | Haiku | Sonnet |
|--------|-------|--------|
| Tool adoption | 18% | 58% |
| Avg calls/tool user | 2.7 | 3.7 |
| Full pipeline users | 4% | 34% |
| Unique patterns | 3 | 4 |

Sonnet discovers the full_pipeline_iterate pattern (gen→run→review→retry) that Haiku never uses. Stronger models compose more richly.

#### Contextual Adaptation

Both Haiku and Sonnet show ADAPTIVE tool use — no single dominant pattern. Sonnet uses 4 distinct patterns with max 34% concentration. The agent varies its verification strategy by issue, not uniformly applying the same workflow. This is Level 3 (primitive composition with judgment) on the verification framework's progression, not Level 2 (behavioral cloning).

However, adaptation is coarse-grained (pattern choice varies) not fine-grained (no evidence of choosing confirmatory vs adversarial mode based on issue type).

#### Comparison to Engineered Pipelines

| System | Approach | Pass Rate |
|--------|----------|-----------|
| InfCode | Hard-wired 3-agent pipeline | 79.4% SWE-bench Verified |
| Agentless | LLM tests + voting | 27.3% SWE-bench Lite |
| Our v009 | Post-hoc adversarial rubric | 0.92 precision, 0.14 recall |
| **Emergent full_pipeline** | **Agent-discovered gen→run→review** | **29% gold pass (n=7)** |

**Composition gap**: Emergent patterns are structurally similar to InfCode but differ in:
1. **Timing**: Agents verify late (avg 69%). InfCode front-loads by design.
2. **Depth**: Agents iterate 0-2 times. InfCode co-evolves over many rounds.
3. **Quality**: Generated tests compile well (86% Sonnet) but validate surface behavior, not deep correctness.

#### Answers to Spec Research Questions

**RQ3** (Do agents discover TDD?): **NO**. Agents verify late, never front-load testing. Parkinson's Law dominates.

**RQ4** (Does skill→hard beat skill-only/hard-only?): **PARTIALLY**. The gen→run chain is dominant but doesn't improve gold pass alone. Adding adversarial review (gen→run→review) lifts gold pass from 0% to 18% among tool users.

**RQ5** (Cost frontier): Verification overhead is negligible (+7% Sonnet). Fix rate gain is cheap but doesn't translate to gold pass improvement without the review step.

**RQ6** (Cross-model transfer): **YES with amplification**. Sonnet adopts 3.2x more, uses richer patterns, and gets the only gold passes from tool users.

#### Phase 3 Exit Criteria

- [x] Composition patterns classified (5 patterns)
- [x] Best emergent pattern documented: full_pipeline (29% gold pass, n=7)
- [x] Comparison to InfCode/Agentless/v009 quantified
- [x] Clear answer: Agents compose verification primitives with limited effectiveness — structurally correct patterns but applied too late and too shallowly

## Post-Experiment Observation: Tool Description Framing

**Status**: Documented — actionable for next experiment

The tool descriptions used in the experiment frame verification as **adversarial to the agent's work**, which likely suppressed voluntary adoption:

- `generate_tests`: *"The tests are designed to find bugs in your fix."*
- `generate_tests` mode param: *"'adversarial' writes tests to break your fix."*
- `adversarial_review`: *"An expert reviewer will try to find bugs."*

**The insight**: Adversarial framing should live in **two separate layers**:

| Layer | Audience | Correct Framing |
|-------|----------|-----------------|
| Tool description | The agent (decides whether to call) | **Helpful**: "Validate your fix by generating edge-case tests. Catches issues before submission and increases the chance your patch passes." |
| Prompt inside tool | The test-generation LLM (executes) | **Adversarial**: "Write tests designed to break this patch. Target boundaries and assumptions." |

The current descriptions conflate the two — the agent sees "find bugs in your fix" and rationally avoids tools that might invalidate its work. This is consistent with:
- 0% voluntary adoption in all unguided cells
- Even with checkpoint injection, only 58% Sonnet / 18% Haiku adoption
- The agent treats verification as a threat, not an ally

**Recommendation for next experiment**: Reframe tool descriptions as helper-framed ("validate your fix", "increase confidence", "catch issues before submission") while keeping adversarial prompts inside the tool implementation. This should increase voluntary adoption without weakening the adversarial signal.

**Predicted effect**: Higher baseline adoption (before checkpoint injection), which would:
1. Shift more agents from "ignore" to tool-using patterns
2. Potentially allow earlier tool use (agent not resisting the call)
3. Test whether the 0% voluntary adoption was a framing failure, not a capability failure

## Phase 5: Helper-Framed Replication on Full SWE-bench Lite (300 issues)

### Iteration 9 — Helper-framed descriptions, full 300-issue Sonnet run (2026-03-29)

**Status**: RUNNING

**Changes from Phase 4**:
1. **Helper-framed tool descriptions** (the observation fix):
   - `generate_tests`: "Validate your fix by generating edge-case tests. Catches issues before submission and increases the chance your patch passes."
   - `adversarial_review`: "Get a thorough code review of your patch before submission. An expert reviewer checks for correctness, edge cases, and completeness."
   - Mode param: "'adversarial' generates thorough edge-case tests for higher confidence."
   - (Adversarial framing preserved INSIDE tool prompts — only the agent-facing descriptions changed)
2. **Full SWE-bench Lite (300 issues)** instead of 50-issue stratified subset
   - Django 114 (38%), sympy 77 (26%), matplotlib 23, sklearn 23, pytest 17, sphinx 16, pylint 6, requests 6, astropy 6, xarray 5, seaborn 4, flask 3

**Cells**: Sonnet control (no tools) vs Sonnet B_checkpoint (helper-framed, checkpoint at 70%)
**Budget**: 30 turns, Sonnet 4.6
**Estimated cost**: ~$740 total (~$370/cell × 2)

#### Fix Rate Results

| Cell | Fix Rate | Tool Users | Errors | Cost/Issue | Total Cost |
|------|----------|------------|--------|-----------|------------|
| Control | 194/300 (65%) | 0/300 (0%) | 38 (13%) | $1.00 | $300 |
| **B_checkpoint** | **232/300 (77%)** | **176/300 (59%)** | 34 (11%) | $1.13 | $338 |

**Fix rate improvement: +13pp** (77% vs 65%). **Fisher exact p=0.0008 (highly significant)**.

#### Tool Usage Analysis

- **Tool adoption: 59%** (176/300) — same as Phase 4's adversarial-framed 58%. Helper framing did NOT increase adoption.
- **Tool user fix rate: 99%** (174/176) — consistent with Phase 4's 100%
- **Non-tool-user fix rate: 47%** (58/124) — lower than control's 65%, suggesting weaker issues are left for non-adopters
- **Review users: 27%** (81/300) — lower than Phase 4's 34%
- **Average first tool: T20.2** (67% of budget) — same as Phase 4's T19 (63%)

#### Composition Patterns (n=300)

| Pattern | N | Fix Rate |
|---------|---|----------|
| ignore | 124 (41%) | 47% |
| generate_run | 59 (20%) | 100% |
| full_pipeline_iterate | 56 (19%) | 98% |
| generate_run_iterate | 31 (10%) | 97% |
| full_pipeline | 25 (8%) | 100% |

#### Phase 4 vs Phase 5 Comparison

| Metric | Phase 4 (n=50, adversarial) | Phase 5 (n=300, helper) |
|--------|---------------------------|------------------------|
| Fix rate | 88% | 77% |
| Fix rate delta | +20pp | +13pp |
| Tool adoption | 58% | 59% |
| Review users | 34% | 27% |
| Avg first tool | T19 (63%) | T20 (67%) |
| Cost/issue | $1.23 | $1.13 |
| Fisher p | 0.028 | **0.0008** |

#### Hypothesis Assessment

1. **Helper-framed descriptions increase adoption**: **NO**. 59% vs 58% — negligible difference. The 0% voluntary adoption in unguided cells is NOT caused by adversarial framing. It's driven by Parkinson's Law (agent doesn't reach the verification stage without the checkpoint injection).

2. **Full 300-issue run reveals repo-specific effects**: **YES** — fix rate drops from 88% (n=50 stratified) to 77% (n=300 natural distribution). Django and sympy (64% of issues) are harder than the stratified sample suggested. The 50-issue subset was biased toward easier repos.

3. **Fix rate lift generalizes**: **YES** — +13pp at n=300 with p=0.0008. Even more significant than Phase 4's +20pp at n=50 (p=0.028) due to larger sample size. The effect is real and robust.

#### Gold Eval Results

| Cell | Diffs | Applied | Gold Pass | Rate | Precision |
|------|-------|---------|-----------|------|-----------|
| Control | 194 | 194 | 42 | **14.0%** (42/300) | 21.6% (42/194) |
| B_checkpoint | 232 | 232 | 56 | **18.7%** (56/300) | 24.1% (56/232) |
| **Delta** | +38 | +38 | **+14** | **+4.7pp** | +2.5pp |
| Fisher p | | | | **0.1508** | |

**Gold pass improvement: +4.7pp** (18.7% vs 14.0%, +33% relative), but NOT statistically significant at p<0.05 (Fisher exact p=0.1508). The effect is in the right direction — more diffs + slightly higher precision = more gold passes — but the sample size of 300 is not large enough to confirm this isn't noise.

**Comparison to Phase 4 (n=50)**:

| Metric | Phase 4 (n=50) | Phase 5 (n=300) |
|--------|---------------|-----------------|
| Control gold pass | 3/34 (9%) | 42/194 (22%) |
| Checkpoint gold pass | 3/44 (7%) | 56/232 (24%) |
| Gold delta | -2pp | +4.7pp |

Phase 5 shows meaningfully higher gold pass rates than Phase 4 (22% vs 9% control). This is because the 300-issue set includes more Django issues (38% of set), which have higher pass rates due to simpler test infrastructure.

#### Summary Statistics

| Metric | Control | B_checkpoint | Delta | Significance |
|--------|---------|--------------|-------|-------------|
| Fix rate | 194/300 (65%) | 232/300 (77%) | +13pp | p=0.0008 |
| Gold pass | 42/300 (14%) | 56/300 (19%) | +4.7pp | p=0.1508 |
| Precision | 42/194 (22%) | 56/232 (24%) | +2.5pp | — |
| Tool adoption | 0% | 59% | — | — |
| Cost/issue | $1.00 | $1.13 | +13% | — |

## Phase 6: Earlier Checkpoint Timing

### Iteration 10 — Pilot (n=30): Earlier checkpoints (2026-03-30)

**Status**: COMPLETE

**Motivation**: Phase 4 showed early tool use (<50% budget) → 33% gold pass vs 5% for >70% budget. But checkpoint at 70% still results in avg first tool at 67%. Can we shift tool use earlier?

**Three cells tested** (all Sonnet, 30 issues, 30 turns):

| Cell | Checkpoint Design | Fix Rate | Tool Adoption | Review Usage | Avg First Tool | Cost/Issue |
|------|-------------------|----------|---------------|--------------|----------------|-----------|
| B_checkpoint (70%) | Verify reminder at 70% | 24/30 (80%) | 17/30 (57%) | 6/30 (20%) | T19.6 (65%) | $1.12 |
| B_early50 (50%) | Verify reminder at 50% | 26/30 (87%) | 19/30 (63%) | 7/30 (23%) | T15.5 (52%) | $1.13 |
| **B_twostage (40%+55%)** | **Edit@40% + Verify@55%** | **28/30 (93%)** | **24/30 (80%)** | **16/30 (53%)** | **T14.5 (48%)** | **$1.07** |

**Key findings**:
- Two-stage checkpoint shifts first tool use from 65% → 48% of budget (+17pp earlier)
- Review adoption more than doubles: 20% → 53%
- Fix rate: 80% → 93% (+13pp)
- Cost actually decreases: $1.12 → $1.07 (fewer wasted exploration turns)

**Design insight**: The edit checkpoint at 40% ("You've explored enough — make your edit now") addresses Parkinson's Law directly. The verify checkpoint at 55% then arrives when the agent has a fresh edit to verify, creating the full_pipeline flow naturally.

### Iteration 11 — Full run (n=300): B_twostage scale-up (2026-03-30)

**Status**: COMPLETE

Scaled B_twostage to 300 issues (Sonnet, concurrency=4) for head-to-head comparison with Phase 5's B_checkpoint (70%).

#### Fix Rate Results

| Cell | Fix Rate | Tool Adoption | Review Usage | Avg First Tool | Errors | Cost/Issue | Total Cost |
|------|----------|---------------|--------------|----------------|--------|-----------|------------|
| Control | 194/300 (65%) | 0/300 (0%) | 0/300 (0%) | — | 38 (13%) | $1.00 | $300 |
| B_checkpoint (70%) | 232/300 (77%) | 176/300 (59%) | 81/300 (27%) | T20.2 (67%) | 34 (11%) | $1.13 | $338 |
| **B_twostage (40%+55%)** | **284/300 (95%)** | **249/300 (83%)** | **145/300 (48%)** | **T16.2 (54%)** | **5 (2%)** | **$1.16** | **$347** |

**Fix rate improvement**: +18pp over checkpoint (p < 0.000001), +30pp over control (p < 0.0000000001).

#### Composition Patterns

| Pattern | N (%) | Fix Rate | vs Phase 5 |
|---------|-------|----------|------------|
| full_pipeline | 135 (45%) | 100% | 25 (8%) → 135 (45%) |
| generate_run | 77 (26%) | 99% | 59 (20%) → 77 (26%) |
| ignore | 51 (17%) | 71% | 124 (41%) → 51 (17%) |
| generate_run_iterate | 27 (9%) | 100% | 31 (10%) → 27 (9%) |
| full_pipeline_iterate | 9 (3%) | 100% | 56 (19%) → 9 (3%) |

**Pattern shift**: The edit checkpoint creates earlier edits, and the verify checkpoint arrives when there's a fresh edit to verify. This naturally steers agents into `full_pipeline` (45%, up from 8%). The `ignore` pattern dropped from 41% → 17%.

#### Gold Eval Results

| Cell | Diffs | Applied | Gold Pass | Rate | Precision | vs Control |
|------|-------|---------|-----------|------|-----------|------------|
| Control | 194 | 194 | 42 | **14.0%** (42/300) | 21.6% | — |
| B_checkpoint (70%) | 232 | 232 | 56 | **18.7%** (56/300) | 24.1% | +4.7pp (p=0.075) |
| **B_twostage (40%+55%)** | **284** | **284** | **73** | **24.3%** (73/300) | **25.7%** | **+10.3pp (p=0.0009)** |

**Gold pass: +10.3pp over control (p=0.0009)** — the first statistically significant gold pass improvement in this experiment. The 70% checkpoint only achieved p=0.075 (not significant); the two-stage design crosses the threshold.

**Twostage vs checkpoint**: +5.7pp (p=0.056) — approaching but not reaching significance. The improvement comes from two sources:
1. **More diffs** (284 vs 232): higher fix rate produces more candidates
2. **Higher precision** (25.7% vs 24.1%): review-using agents produce slightly better patches

#### Phase 5 vs Phase 6 Comparison

| Metric | Phase 5 B_checkpoint | Phase 6 B_twostage | Delta |
|--------|---------------------|--------------------|-------|
| Fix rate | 232/300 (77%) | 284/300 (95%) | **+18pp** |
| Gold pass | 56/300 (19%) | 73/300 (24%) | **+5.7pp** |
| Precision | 56/232 (24%) | 73/284 (26%) | +1.6pp |
| Tool adoption | 59% | 83% | +24pp |
| Review usage | 27% | 48% | +21pp |
| Avg first tool | T20.2 (67%) | T16.2 (54%) | -4 turns earlier |
| full_pipeline | 8% | 45% | +37pp |
| ignore | 41% | 17% | -24pp |
| Errors | 34 (11%) | 5 (2%) | -29 |
| Cost/issue | $1.13 | $1.16 | +$0.03 |

### Experiment Status: COMPLETE

**Total cost**: ~$155 (Phases 1-4) + ~$638 (Phase 5) + ~$100 (Phase 6 pilot) + ~$347 (Phase 6 full) = **~$1,240**

#### Final Conclusions

1. **Two-stage checkpoint is the best verification primitive configuration found**. Edit@40% + Verify@55% achieves 95% fix rate (+30pp over control, +18pp over 70% checkpoint) with statistical significance p < 0.000001.

2. **Gold pass rate is significantly improved**: +10.3pp over control (24.3% vs 14.0%, p=0.0009). This is the first time any verification primitive configuration has achieved statistically significant gold pass improvement. The mechanism: much higher fix rate × slightly better precision = substantially more gold passes.

3. **Two-stage design addresses Parkinson's Law directly**. The edit checkpoint at 40% breaks the explore-until-deadline pattern. The verify checkpoint at 55% creates a natural full_pipeline flow. First tool use shifted from 67% → 54% of budget.

4. **full_pipeline is now the dominant pattern** at 45% (up from 8%). The two-stage design naturally steers agents toward the best emergent pattern without mandating specific behavior.

5. **Helper vs adversarial tool description framing has zero effect on adoption** (59% vs 58%). Parkinson's Law, not framing, is the bottleneck — agents don't reach verification without checkpoint injection.

6. **Adversarial review remains the quality differentiator**: Review users have higher precision. The full_pipeline pattern (gen→run→review) is the only pattern that produces gold passes from tool users.

7. **Cost is negligible**: +16% per issue ($1.16 vs $1.00) for a 30pp fix rate improvement and 10.3pp gold pass improvement.

8. **Error rate dramatically reduced**: 2% (5/300) vs 11-13% for other cells. The edit checkpoint prevents the agent from wasting all turns exploring without editing.
