# Autoresearch Spec: E_trace-profile — Behavioral-RF Mechanism Test (profile-first, FinQA + FinanceBench)

## Status: DRAFT

## Overview

Solidify the two standout verifier-program findings by converting a weak *domain* claim into a falsifiable *mechanism* claim, using **only data already on disk** (no new agent runs).

**The claim under test:** the behavioral RF (Phase-3, coding AUC 0.756) is, mechanically, a **trace-richness / thrash detector** — its top features are `beh_total_cost_usd` (0.42) + `beh_tokens_per_edit` (0.33) + `beh_loop_count` (0.20) = 95% of signal. Hypothesis: **it discriminates iff the traces are long and variable enough to carry a process signal.** This explains the E_fin2 FinQA null (AUC 0.569) as a *structural* property (3-op table lookups have no trace to read), not a domain property — and predicts behavior on any other domain from its trace statistics alone.

**Why profile-FIRST (the anti-confirmation-bias gate):** profile trace-richness for both domains and **pre-register the prediction BEFORE fitting any RF**. If FinanceBench traces are also short/low-variance → predict another null → the RF finding hardens to "needs rich traces, full stop." If FinanceBench traces are long/variable → that is the discriminating test, and RF-works-there confirms the mechanism. Either outcome solidifies the finding; the profiling decides which.

**Depends on**: e-fin2 (`results/features.csv` — FinQA behavioral features + `gold_pass`), e-harness3 (`results/E_*.jsonl`, `F_*.jsonl` — FinanceBench traces with `input_tokens`/`output_tokens`/`cost_usd`/`is_correct`), learned-verifier `phase3_report.md` (the coding 0.756 baseline + feature importances). All on disk.

## Findings being solidified

- **#1 Behavioral RF**: coding AUC 0.756 (rich traces) → FinQA AUC 0.569 (sparse traces). Is the difference *trace richness* or *domain*? FinanceBench is the discriminator.
- **#2 Adversarial rubric**: already shown non-transferable on BOTH axes (T4 cross-model: Devstral 0.20/Nova 0.14; E_fin1 cross-domain: 0.99× lift on FinQA). This spec *documents* #2's status as settled — no new #2 run (deferred per scope decision); the rubric leg here is limited to re-confirming the E_fin1 number against disk.

## Research Questions

1. **Trace-richness profile**: for FinQA and FinanceBench, what are the distributions of trace length / token variance / tool-or-step count, and the pass-vs-fail separation on each behavioral feature? (Descriptive, robust at small n.)
2. **Pre-registered prediction**: from the profile alone, predict the FinanceBench behavioral-RF AUC band BEFORE fitting. Record it.
3. **Mechanism test**: fit the behavioral RF on FinanceBench (existing traces). Does the realized AUC match the profile-based prediction? Does RF-signal track trace-richness across the three points (coding 0.756 / FinQA 0.569 / FinanceBench ?)?
4. **Caveat check**: E_harness3's FinanceBench traces are graded-Q&A, not full agentic 10-K-retrieval — so they may *understate* FinanceBench trace richness. Note where that bounds the conclusion.

## Components

### 1. Compute
- **Platform**: local, **python3.13** (macOS python3.14 sklearn broken — carried lesson). No GPU, no API, no new agent runs.

### 2. Data (all on disk)
- FinQA: `e-fin2-finqa-behavioral-features/results/features.csv` (beh_* features + gold_pass, n≈100).
- FinanceBench: `e-harness3-reward-regime-x-locus/results/{E,F}_{haiku,sonnet}.jsonl` (traces + is_correct, n≈150/cell).
- Coding baseline: `learned-verifier/results/phase3_report.md` (AUC 0.756, importances) — reference only.

### 3. Protocol
1. **Profile** both domains: per-feature distribution (mean/var), trace-length proxy (tokens), and pass-vs-fail effect size (e.g. standardized mean diff) per behavioral feature. Output a profile table.
2. **Pre-register**: write the predicted FinanceBench RF AUC band to `results/prediction.md` and COMMIT it before step 3 (the gate).
3. **Fit** the behavioral RF on FinanceBench (verbatim Phase-3 recipe: RF 200×depth-7, class-balanced, 5-fold stratified, pooled OOF, bootstrap CI). Report AUC + CI.
4. **Compare**: plot/table the three points (coding/FinQA/FinanceBench) AUC vs a trace-richness scalar. Does RF-signal track richness?
5. **Honesty checks**: report n and CI width for every AUC (flag underpower — at n≈150 the CI is ~±0.07, so frame as "descriptive separation" not "significant"); note the graded-Q&A caveat (RQ4).

### 4. Storage
- `results/`: profile table, `prediction.md` (pre-registered, committed before fit), rf results + CIs, the three-point richness-vs-AUC comparison, verdict.

## Success Criteria

- [ ] Trace-richness profile table for FinQA + FinanceBench (distributions + pass/fail separation per feature).
- [ ] **`prediction.md` written AND committed BEFORE any RF fit** (the anti-confirmation gate — verifiable in git history).
- [ ] Behavioral RF fit on FinanceBench with AUC + bootstrap CI + explicit n and CI-width.
- [ ] Three-point comparison: does behavioral-RF AUC track trace-richness (coding 0.756 → FinQA 0.569 → FinanceBench ?)?
- [ ] Verdict on the mechanism claim: is the RF a trace-richness detector (richness predicts AUC) or is FinanceBench another null regardless (→ finding hardens to domain-general "doesn't transfer off coding")?
- [ ] Underpower + graded-Q&A caveats stated; no significance claim the n can't support.

## Non-Requirements

- **No new agent runs / no regeneration** — existing traces only.
- **No new #2 rubric run** — #2's non-transfer is already settled (T4 + E_fin1); this only re-confirms the E_fin1 number from disk.
- **No GPU, no fine-tuning.**
- **No fresh agentic FinanceBench** (the richer-traces version) — deferred; noted as the follow-up if the profile says E_harness3's traces are too thin to decide.

## Known Limitations

- **n is small** (FinQA ≈100, FinanceBench cells ≈150): the RF AUC CI is wide (~±0.07). This experiment can establish *descriptive separation* and *directional tracking* of richness→AUC; it CANNOT make a powered significance claim. State this, don't bury it.
- **E_harness3's FinanceBench traces are graded-Q&A, not agentic 10-K-retrieval** — they may understate FinanceBench's true trace richness. If the profile shows them nearly as sparse as FinQA, the mechanism test is *inconclusive* (need the fresh agentic run), not negative. This is the honest failure mode and must be reported as such.
- Profile-first mitigates but does not eliminate confirmation bias; the committed-prediction gate is the check.

## Carryover Audit (spec-design gate)

- [ ] Carry priors: **phase3 RF = 0.756, top-3 beh features = 95% importance** (the mechanism being tested); **E_fin2 FinQA = 0.569** (the sparse-trace point); **E6 cross-model fails 0.36/0.41** (richness/threshold non-transfer); **python3.13 not 3.14** (sklearn). **E_fin1 scoring-artifact lesson** — when re-confirming #2, audit "wrong" answers for parse artifacts before trusting the differential.

---

> **Note**: Operational artifacts → `domains/autoresearch/blueprints/e-trace-profile-mechanism/`.
