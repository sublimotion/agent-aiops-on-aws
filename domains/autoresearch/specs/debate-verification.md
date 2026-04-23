# Autoresearch Spec: Collaborative Debate Verification

## Status: DRAFT

## Overview

Test whether a three-agent debate protocol (Advocate + Challenger + Judge) achieves higher recall than v009 unanimous consensus while maintaining precision > 0.85. The key insight: v009's unanimous-reject protocol has no advocate for correct patches — it only asks "find the bug." Debate gives the correct side a voice, potentially recovering false negatives that unanimous rejection misses.

**Core hypothesis**: Three Haiku calls in debate format (Advocate argues patch is correct, Challenger argues it's wrong, Judge decides) achieves recall > 0.30 at precision > 0.85, compared to v009 unanimous consensus (0.92 precision, 0.14 recall / 7.7% coverage).

**Motivation**: Our recall ceiling is the single biggest limitation. v009 only speaks on 7.7% of patches (37/483). SVG consensus has better recall (52.8%) but requires a full regeneration pipeline. Debate is cheap (~$0.024/patch for 3 Haiku calls) and attacks the recall problem from a different angle than either v009 or SVG. ColMAD (arXiv:2510.20963) found collaborative debate outperforms competitive debate by ~19% on error detection. Haize Labs' Verdict framework (arXiv:2502.18018) uses debate as a core primitive for judge scaling.

**Risk**: T5 finding (self-critique paradox) — sycophancy flipped verdicts when the verifier was challenged within the same conversation. The Judge in this protocol must be structurally separate. MAS-ProVe (arXiv:2602.03053) found multi-agent verification "does not consistently improve performance and frequently exhibits high variance." Pilot before full run.

**Depends on**: verifier-reward (v009 baseline on n=483, adversarial rubric prompt)

## Components

### 1. Compute
- **Platform**: EC2 or local — API calls to Bedrock
- **Instance Type**: Any (CPU-only, API-bound)
- **GPUs**: None required

### 2. Codebase
- **Source**: New scripts in blueprint directory
- **Fixed files**:
  - v009 adversarial rubric prompt: `blueprints/verifier-reward/scripts/rubrics/v009.md`
  - SWE-bench Verified data: 483 instances with gold labels
  - SWE-bench Lite predictions: `blueprints/verification-primitives-swebench/results/predictions_lite.jsonl`
- **Agent-editable files**:
  - `blueprints/debate-verification/scripts/debate_eval.py` — orchestrates 3-agent debate
  - `blueprints/debate-verification/scripts/prompts/advocate.md` — Advocate system prompt
  - `blueprints/debate-verification/scripts/prompts/challenger.md` — Challenger system prompt
  - `blueprints/debate-verification/scripts/prompts/judge.md` — Judge system prompt
- **Agent instructions**: N/A

### 3. Experiment Protocol
- **Metric**: Precision at recall > 0.30 (higher is better)
- **Secondary**: AUC, F1, ECE, cost per verdict
- **Time budget**: Pilot 2-3 hours, full run 6-8 hours (API latency bound)
- **Loop structure**: Prompt design → pilot (n=50) → analyze → adjust → full run (n=483) if pilot shows promise
- **Termination**: Full evaluation on SWE-bench Verified with statistical comparison to v009
- **Logging**: `blueprints/debate-verification/results/` — per-instance JSONL with all three agent outputs + final verdict

### 4. Networking
- **Access**: Bedrock API (Haiku 4.5)
- **Model**: Claude Haiku 4.5 via Bedrock (all three agents)
- **Concurrency**: 10 parallel debates to manage API rate limits
- **Estimated API calls**: Pilot: 150 calls (50 × 3). Full: 1,449 calls (483 × 3).

### 5. Storage
- **Data**: SWE-bench Verified instances (existing)
- **Results**: `blueprints/debate-verification/results/`

## Debate Protocol Design

### Agent Roles

**Advocate** (argues patch is correct):
```
You are reviewing a code patch that was submitted to fix a reported issue.
Your role: argue that this patch correctly addresses the issue.

Identify specific evidence:
1. Which lines in the patch directly address the reported problem
2. Why the approach is technically sound
3. What edge cases the patch handles correctly

Be specific. Cite line numbers. If the patch has weaknesses, acknowledge them but argue they don't affect correctness for the reported issue.
```

**Challenger** (argues patch is wrong — adapted from v009):
```
Assume this patch is wrong. Your job is to find the specific bug.

Evaluate against these criteria:
1. Problem alignment: Does the patch address the actual reported issue, or a different problem?
2. Logic correctness: Is the implementation logically sound for the stated fix?
3. Completeness: Does the patch handle all cases mentioned in the issue?
4. Scope safety: Does the patch modify only what's necessary, or does it risk regressions?

If you find a concrete bug, state it precisely. If the patch looks correct, say so — but look hard first.
```

**Judge** (reads both arguments, decides):
```
Two reviewers have examined a code patch. One argues it is correct, one argues it is wrong.
Read both arguments carefully.

Your task: determine whether the patch correctly fixes the reported issue.

Rules:
- Focus on the strength of specific claims, not rhetorical quality
- A concrete, verifiable bug claim outweighs a general "looks correct" argument
- If the Challenger identifies a real bug (wrong logic, missed edge case, regression risk), rule INCORRECT
- If the Advocate's evidence is specific and the Challenger's objections are generic or hypothetical, rule CORRECT
- If uncertain, rule UNCERTAIN (do not force a verdict)

Output: CORRECT, INCORRECT, or UNCERTAIN with a one-paragraph justification.
```

### Structural Separation (Critical)

Each agent runs as a **separate API call** with no shared conversation context:
- Advocate sees: issue description + patch diff
- Challenger sees: issue description + patch diff (same input, different prompt)
- Judge sees: issue description + patch diff + Advocate argument + Challenger argument

The Judge never sees the generation context, agent trajectory, or internal reasoning from the coding agent. This prevents the sycophancy failure mode from T5.

### Variants to Test

| Variant | Description | Cost/patch |
|---------|-------------|-----------|
| **Base debate** | Advocate + Challenger + Judge (single round) | $0.024 |
| **2-round debate** | Add rebuttal round (Advocate responds to Challenger, vice versa, Judge re-decides) | $0.056 |
| **Ensemble debate** | 3 independent debates, majority vote on Judge verdicts | $0.072 |
| **Asymmetric** | Advocate=Haiku, Challenger=Sonnet (stronger adversary), Judge=Haiku | $0.035 |

Pilot tests base debate. Extend to variants only if base shows promise.

## Experiment Phases

### Phase 1: Prompt Engineering on Known-Outcome Issues (n=10)

1. Select 5 known-correct and 5 known-incorrect patches from dev set
2. Run debate protocol, manually inspect all three agent outputs
3. Calibrate prompts: Are arguments specific enough? Does the Judge reason well? Does the Advocate make the Challenger's job harder on correct patches?
4. Iterate prompts 2-3 times based on failure analysis

**Cost**: ~$0.25 (10 × 3 Haiku calls × ~3 iterations)

### Phase 2: Pilot Evaluation (n=50)

1. Select 50 instances from SWE-bench Verified dev set (25 resolved, 25 unresolved)
2. Run base debate protocol
3. Compute: precision, recall, AUC, agreement rate with v009
4. Decision gate: if precision < 0.70 OR recall < 0.15, stop and analyze before proceeding
5. If promising: test 2-round and ensemble variants on same n=50

**Cost**: ~$1.20 base + ~$2.80 variants = ~$4 total

### Phase 3: Full Evaluation (n=483)

1. Run best variant from Phase 2 on all 483 SWE-bench Verified instances
2. Compute full metrics with confidence intervals (bootstrap, 1000 samples)
3. Compare head-to-head with v009 unanimous consensus
4. Analyze disagreements: where does debate succeed and v009 fail? Vice versa?
5. Compute ECE for RL-readiness assessment

**Cost**: ~$12 for base debate, ~$35 for ensemble variant

### Phase 4: Combination Analysis

1. Test debate + v009 combination: does debate recover v009 false negatives?
2. Test debate + SVG combination: does debate complement SVG's recall gaps?
3. Construct combined verifier: if(SVG says CORRECT → accept), elif(debate says CORRECT → accept with lower confidence), else(v009 says INCORRECT → reject)

## Success Criteria

1. **Minimum viable (pilot gate)**: Base debate achieves precision > 0.70 and recall > 0.15 on n=50 pilot. Justifies full run.
2. **Strong result**: On n=483, recall > 0.30 at precision > 0.85. Debate verdicts are additive to v009 (combination outperforms either alone).
3. **Publishable result**: Debate + v009 + SVG combination achieves recall > 0.50 at precision > 0.90. Reliability diagram shows good calibration (ECE < 0.1).
4. **Negative result (still valuable)**: Debate fails (precision < 0.70 or recall no better than v009). Analysis of why — Judge sycophancy? Advocate quality? Connects to T5 self-critique paradox. Documents boundary condition for debate-based verification in code.

## Non-Requirements
- Multi-turn debate beyond 2 rounds (diminishing returns expected)
- Fine-tuning any of the debate agents
- Real-time/online debate during agent execution (offline post-hoc evaluation)
- Human-in-the-loop during debate
- Debate on non-code tasks (future experiment)

## Known Limitations
- **Haiku as all three agents**: Same model family may produce correlated arguments. The Advocate and Challenger may share blind spots. Mitigation: test asymmetric variant with Sonnet as Challenger.
- **Sycophancy risk in Judge**: Despite structural separation, the Judge may defer to the more confident-sounding argument regardless of substance. Monitor for this in Phase 1.
- **Cost scales linearly**: 3 API calls per instance is 3x v009 single-call cost. Ensemble debate (9 calls) is 9x. Must justify with recall improvement.
- **MAS-ProVe caution**: Multi-agent verification showed high variance in prior work. Report variance across instances, not just mean metrics.
- **Debate doesn't access test execution**: Both Advocate and Challenger reason about the patch statically. They cannot run tests. This limits them to the same semantic evaluation surface as v009.

## Relationship to Other Specs

- **verifier-reward**: v009 adversarial rubric provides the baseline and the Challenger prompt foundation
- **svg-ece-measurement**: ECE comparison point for debate calibration
- **tiny-judge**: Debate verdicts could be a feature in the tiny judge model
- **verification-primitives**: Debate could be integrated as a fourth verification primitive in future agent experiments

## Key References

- ColMAD (arXiv:2510.20963) — collaborative debate outperforms competitive by ~19%
- MAS-ProVe (arXiv:2602.03053) — caution: multi-agent verification has high variance
- Courtroom-Style Debate (arXiv:2603.28488) — progressive evidence refinement with role-switching
- Haize Labs Verdict (arXiv:2502.18018) — debate as composable primitive in judge framework
- Scalable oversight literature (Irving et al., 2018) — debate for AI alignment verification

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
