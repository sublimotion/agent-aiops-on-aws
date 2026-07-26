# Verification / Learned-Verifier Program — Retrospective

**Date:** 2026-06-21 · **Scope:** 11 autoresearch blueprints in the verifier/verification lineage · **Purpose:** an honest, numbers-grounded account of what the program established, what it did not, and which claims must not be over-cited later.

> **Read this before citing any result as "we built a learned verifier."** We did not. This document exists because three of the later experiments were narrated more confidently than their numbers support, and the single most likely future error is treating coding-/model-specific results as general.

## TL;DR

- **The strongest, most defensible result is verification-as-scaffolding**, not a learned verifier: prompting a coding agent to run/review its own work raised gold-pass +10.3pp (p=0.0009, n=300). Solid, but **Sonnet-only, SWE-bench-only**, and it is agent prompting, not a learned signal.
- **The "learned verifier" results (behavioral RF, adversarial rubric) are real numbers that do not transfer.** Both fail across model *and* across domain in every instance tested.
- **The one upgrade we earned:** the behavioral RF is mechanistically a **trace-richness / verbosity detector** — it works iff traces are long and variable, regardless of domain. That hardens a null into a usable principle.
- **Net:** the *infrastructure* progressed (a detached agent runtime that runs experiments cheaply). The *science* mostly established, with increasing rigor, that a transferable learned verifier does not exist in any form we tested.

## What stands (numbers from result files on disk)

| Finding | Headline number | n | Transfers? | Status |
|---------|-----------------|---|-----------|--------|
| **Verification-as-scaffolding** (verification-primitives) | gold-pass 24.3% vs 14% control, **+10.3pp, p=0.0009**; tool-users 69.5% vs non-users 18.8% (p=0.0001) | 300 | within Sonnet only; **off-coding UNTESTED** | **Solid, narrow.** The program's best result — and it's prompting, not a verifier. |
| **Behavioral RF = trace-richness detector** (learned-verifier phase3 + E_fin2 + E_trace-profile) | coding AUC **0.756** → FinanceBench 0.743/0.729 (n=300) → FinQA **0.569**; AUC tracks trace richness | 100–300 | **mechanism transfers** (richness predicts AUC); the *signal* needs rich traces | **Solid as a mechanism, bounded to the verbosity axis.** See below. |
| **v009 adversarial rubric** (verifier-reward) | precision **0.92** [0.79, 0.97] | 483 | **NO** — both axes | Solid number, **non-transferable.** Claude-grades-Claude. |

## What does NOT stand / does not transfer (also from disk)

| Claim tested | Result | Verdict |
|--------------|--------|---------|
| Verifier transfers across **models** | E6: Claude→Qwen AUC **0.363**, Qwen→Claude **0.410** (both below chance); Nebius 67K-scale replication 0.43/0.46 | **Replicated NEGATIVE.** The most-reproduced finding in the program is that transfer fails. |
| Rubric transfers across **verifier models** | T4: Claude 0.92, Devstral **0.20**, Nova **0.14**, Mistral rejects everything | Claude-specific artifact |
| Rubric transfers across **patch source** | T6c: precision 1.00 (Claude patches) → **0.20** (Devstral patches) | Patch-source-specific |
| Adversarial reframe transfers across **domain** | E_fin1: coding 2.3× lift → FinQA **0.99×** (AUC 0.565 ≈ chance) | Coding-specific |
| Behavioral features transfer across **domain** (raw) | E_fin2: FinQA AUC **0.569**, CI [0.43, 0.71] crosses chance | Null — *but see mechanism upgrade* |
| Flywheel RF transfers **OOD** | SWE_Rebench→SWE_Smith worse than base rate; retrain acc **32%** | OOD generalization breaks |
| Harness authoring: external > self **locus** | E_harness2 + E_harness3: gap inert across verifiable/withheld/consensus reward; 1/6 cells sig (uncorrected) | Null; near-tautological |

## The one real conceptual upgrade

E_trace-profile (pre-registered: `prediction.md` committed before the fit, verifiable at git `5e63a3f`) converted the E_fin2 domain-null into a **mechanism claim**:

> The behavioral RF's discriminative power is a function of **trace richness** (length × variability), not of domain. Its top features (`total_cost_usd`, `tokens_per_edit`, `loop_count` = 95% of importance in coding) read **process pathology** — thrashing, looping, verbose floundering. Where traces are rich enough to express that pathology, the RF separates pass from fail; where they are sparse (FinQA's 1–2-op derivations), there is nothing to read.

Evidence: AUC tracks an n-independent richness ordering (coding 0.756 > FinanceBench ~0.74 at n=300 > FinQA 0.569), and effect sizes (Cohen's d on the separating feature) follow the same order.

**Bounds — do not over-cite this either:**
1. **Verbosity axis only.** FinanceBench traces here were single-call graded Q&A; the `loop_count` / `tokens_per_edit` *thrash* features had no substrate. Only answer-length/verbosity was tested. The full thrash-detector claim needs a fresh **agentic** FinanceBench run (deferred follow-up).
2. **Underpowered at the cell level.** Per-cell n=150 AUCs were noise-dominated (mean 0.58 ≈ FinQA); the ~0.74 signal only resolved at n=300 within-model. Report effect sizes, not per-cell AUCs, as the primary evidence.
3. Single corpus family, two models — directional, not a powered law.

## Statistical-honesty notes (the over-claiming risks)

- **The cross-domain experiments (E_fin*, E_harness*) are underpowered.** Typical CIs are ±0.05–0.09 at n=100–150; they can establish *descriptive* separation and *directional* tracking but **cannot reject a small real effect**. Several "clean nulls" are "absence of evidence," not "evidence of absence." Do not cite them as proof an effect is zero.
- **E_harness3 had 1/6 cells significant with no multiple-comparison correction.** Under correction it is all null. It was narrated as a "decisive refutation"; it is an underpowered null + one marginal cell.
- **Three experiments confirmed near-tautologies** (you cannot improve against a signal you cannot observe; authoring locus is inert without an asymmetry available to the authoring act). True, but largely derivable a priori — low information gain.
- **The agents are persuasive narrators.** E_harness2 "synthesized a unifying monotonic law" that E_harness3 then refuted. Treat in-run agent syntheses as hypotheses, not findings.

## The honest one-line synthesis

Every "verification" result in this program decomposes into **(coding harness) × (model capability) × (rubric/feature calibration)**, and the calibration layer — where all the apparent "learning" lives — **does not transfer across model, patch-source, or domain in a single instance tested.** The only robustly transferring intervention is the crudest (tell the agent to check its own work), and the only general *mechanism* we established is deflationary (the behavioral RF is a trace-richness detector). "Learned verifier" overstates what exists.

## What progress actually looks like from here

A next experiment is worth running only if it is: (a) **powered** — pre-registered effect size, n large enough that the CI can *reject* a meaningful effect; (b) on a question with a **genuinely uncertain prior** (not "can you improve without a signal"); and ideally (c) a **positive** prediction that would surprise us. Candidates that meet the bar:
- **Does verification-as-scaffolding (the one survivor) transfer off-coding?** Untested, real prior, a positive would be informative.
- **Does the behavioral RF's *thrash* axis (not just verbosity) carry signal on agentic FinanceBench?** The deferred leg that would complete the trace-richness mechanism.

If a proposed experiment is another adverse-condition null at n≈100, it is not progress — it is confirmation. Stop and keep the runtime.

## Source files (for verification)

- `learned-verifier/results/phase3_report.md` (RF 0.756, importances), `e6_cross_model_report.md` (transfer 0.363/0.410), `e6_nebius_rf_report.md` (67K replication), `enew_report.md` (Simpson's paradox), `e_new6_exploit_report.md`
- `verifier-reward/results/progress.md` (v009 0.92, T4/T6c transfer failures)
- `verification-primitives/results/progress.md` + `verification-primitives-swebench/results/progress.md` (+10.3pp scaffolding, 58.3% production)
- `verification-flywheel/results/SUMMARY.md` (10× cost, OOD break)
- `e-fin1/`, `e-fin2/`, `e-harness1/`, `e-harness2/`, `e-harness3/`, `e-trace-profile-mechanism/` results dirs
