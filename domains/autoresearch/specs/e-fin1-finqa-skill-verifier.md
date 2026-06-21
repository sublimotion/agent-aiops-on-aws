# Autoresearch Spec: E_fin1 — Cross-Domain Skill-Verifier Replication (FinQA)

## Status: DRAFT

## Overview

Test whether the adversarial skill-verifier result from the coding domain (precision 0.40 → 0.92 from confirmatory→adversarial reframing) **transfers off-coding to financial numeric reasoning**. Every datapoint behind the verification playbook is currently SWE-bench; this is the first non-coding empirical anchor.

**Core hypothesis**: the confirmatory→adversarial reframe ("assume this answer is wrong; find the breaking assumption / arithmetic error") is **domain-general** — it reproduces the core skill-verifier result on financial numeric QA with no test suite: **precision ≥ 0.80 on the confident subset, materially above a confirmatory baseline, at ~$0.03/eval.** The coding result was 2.3× (0.40 → 0.92) from adversarial framing alone.

**Why FinQA** (decided after evaluating candidates): a near-structural clone of the coding setup minus test suites — objective numeric ground truth (exact-match = the Docker-execution analog: $0 eval, no human-in-the-loop), question + table/text context from real 10-Ks, 6,251/883/1,147 split, MIT. It is the regime legal/finance actually live in: measurable correctness, no test harness. **FinanceBench was rejected** (free-text answers requiring n=2,400 hand-graded human review — wrong substrate for a cheap verifier experiment).

**Depends on**: verifier-reward (provides the v009 adversarial-ensemble pattern and the 0.92/$0.03 coding baseline this replicates).

**Source**: obsidian `Learned-Verifier-Experiment/experiments/E_fin1.md` (proposed 2026-06-21).

## Research Questions

1. **Does adversarial framing's precision lift survive a domain with no test suite, numeric (not code) outputs, and table-heavy context?** If yes → the playbook's Part-4 cross-domain projections gain their first empirical anchor and it is honestly a *general* verification playbook. If the lift shrinks → the adversarial claim is more coding-specific than stated (still a publishable, applicability-bounding result).
2. **Does the verifier-model calibration hold off-coding?** **verifier-reward T4 falsified "the rubric is model-agnostic": only Claude's v009 calibrates — Devstral precision=0.20, Nova Pro 0.14, Mistral Large rejects everything.** The lift is Claude-specific, not domain-specific. Re-test the verifier-model choice on financial QA; calibration drift (or using a non-Claude verifier to cut cost) is the primary confound — a null result may reflect verifier-model choice, not domain.
3. **Does the $0.03/eval cost ceiling hold** when table context raises input tokens?

## Components

### 1. Compute
- **Platform**: API-driven (Bedrock / Anthropic API). **No GPU required.**
- **Instance type**: laptop or any CPU host; can run as an `agent-runner` batch job.
- **GPUs**: none.

### 2. Data
- **Dataset**: FinQA — **original `czyssrs/FinQA` (GitHub, MIT)**, NOT the HF mirror `dreamerdeo/finqa` (the mirror drops the `program` field; outcome-only E_fin1 works on either, but use the original to stay compatible with the sibling E_fin2).
- **Schema (VERIFIED 2026-06-21)**: each example ships `qa.question`, `qa.program`, `qa.gold_inds`, `qa.exe_ans`. **Use `qa.exe_ans` as the numeric ground truth** (gold execution result for exact-match).
- **Sample**: n ≈ 100 FinQA dev examples.

### 3. Experiment Protocol
- **Metric (primary)**: precision on the confident subset, **adversarial vs confirmatory**. Secondary: recall, AUC, cost-per-eval.
- **Ground truth**: exact-match of agent's numeric answer against `qa.exe_ans`, with tolerance for rounding/units. Free, objective, no human review.
- **Loop structure**:
  1. Generate ~100 agent answers with a mid-tier model (numeric answer + reasoning) over FinQA dev.
  2. **Verifier A (confirmatory baseline)**: "Rate this financial answer 1–5 for correctness."
  3. **Verifier B (adversarial, the v009 analog)**: "Assume this answer is wrong. Find the specific arithmetic error or breaking assumption that makes it fail." — **4-call temperature ensemble (1@t=0.0, 3@t=0.3); unanimous = confident.**
  4. Score both verifiers' precision/recall/AUC against exact-match ground truth.
  5. Calibration check: re-test the verifier-model choice (Claude vs other) — does adversarial calibration hold off-coding?
- **Termination**: all n examples scored under both verifiers.
- **Logging**: per-example {question, agent answer, exe_ans, match, verifier-A score, verifier-B verdict×4, confident flag}; aggregate precision/recall/AUC + cost.

### 4. Networking
- **Access**: outbound HTTPS to the model API. If run as an `agent-runner` job, IRSA→Bedrock.

### 5. Storage
- **Data**: FinQA original cloned locally (small).
- **Results**: blueprint `results/` (per-example scores + aggregate table).

## Success Criteria

- [ ] **Stage-0 eval smoke test**: run exact-match (tolerance/rounding/units) on 5 FinQA dev examples with known `qa.exe_ans` and confirm it scores correctly *before* the n=100 run. (Prior: verification-primitives lost iterations to gold-eval environment breakage — Python 3.12-vs-3.14/distutils. Never assume the eval harness works.)
- [ ] ~100 FinQA dev answers generated and exact-match-scored against `qa.exe_ans`.
- [ ] Confirmatory and adversarial (4-call ensemble) verifiers both scored on the same set.
- [ ] **Primary verdict reported**: precision-on-confident-subset, adversarial vs confirmatory, with the delta vs the coding 2.3× lift.
- [ ] Verifier-model calibration check completed (does the adversarial verdict avoid defaulting to "uncertain" off-coding?).
- [ ] Cost-per-eval reported against the $0.03 ceiling.
- [ ] Carryover audit complete (below).

## Non-Requirements

- **No FinanceBench** (free-text, human-review-only — no cheap objective ground truth).
- **No test suites** — the entire point is the no-test-suite regime.
- No gold `program` execution (that is the sibling E_fin2's job; E_fin1 is outcome-scored only).
- No fine-tuning, no GPU, no RL.

## Known Limitations

- Single domain (financial QA); a positive result anchors but does not prove full cross-domain generality (legal etc. still un-anchored).
- Table-context token cost may push some evals above $0.03 — measured, not assumed.
- Verifier calibration is model-specific (coding showed only Claude calibrated); a null result may reflect verifier-model choice rather than domain.

## Carryover Audit (spec-design gate)

Before running, confirm no prior-blueprint lesson was left behind:
- [ ] Ran `carryover-auditor` against this spec — scan `verifier-reward/lessons.md` and `verification-primitives*/lessons.md` (overlapping stack: adversarial verifier ensemble, Claude calibration, $0.03 ceiling).
- [ ] Reflect the verifier-reward findings explicitly: **v009 4/4 unanimous is the ceiling**, adversarial > confirmatory is Claude-specific, recall ceiling is semantic-mismatch not noise. Carry these as the prior; the calibration check (RQ2) is the test that they hold off-coding.

---

> **Note**: Operational artifacts (lessons, results, analysis) belong in the blueprint directory `domains/autoresearch/blueprints/e-fin1-finqa-skill-verifier/`, not in this spec.
