# Autoresearch Spec: E_fin2 — Behavioral-Feature Existence Off-Coding (FinQA reasoning programs)

## Status: DRAFT

## Overview

Test whether **behavioral/process features** — the playbook's most distinctive claim (the free $0 behavioral RandomForest, **AUC 0.756** in coding, repo: `learned-verifier/results/phase3_report.md` `selected_4__RandomForest`) — exist and discriminate **off-coding**. The Phase-3 RF's four features (repo-verified: `beh_total_cost_usd` 0.420, `beh_tokens_per_edit` 0.327, `beh_loop_count` 0.201, `svg_accepted` 0.052) are coding-trajectory-specific; if non-coding agents have no analogous process signal, **Phase 2 is coding-only and the playbook must say so.** This bounds the claim honestly.

**Core hypothesis (expected to PARTIALLY FAIL — and that is the result)**: behavioral features analogous to the Phase-3 four exist on FinQA reasoning programs but discriminate **weakly (AUC ~0.55–0.65, well below the 0.756 coding result)**, because financial derivations are short and clean and errors are single-step (one wrong cell reference or operation), not process pathologies. If so → **Phase 2 is largely coding/long-horizon-specific; skill/outcome verification (E_fin1) is the general primitive.** If features still discriminate on FinQA's short clean derivations (close to a worst case for behavioral signal), Phase 2 generalizes strongly.

**Why FinQA enables it**: the original release ships an **executable reasoning program** per example (a DSL derivation, `divide(subtract(a,b), b)`) — the financial analog of a code trajectory: a step-by-step process trace with measurable structure. FinanceBench can't support this (prose justification only, no structured process signal).

**Depends on**: **E_fin1** (shares the FinQA corpus and reuses its generated-answer sample + exact-match labels — E_fin2 adds no new generation). Also replicates verifier-reward / verification-primitives Phase-3 ordering off-coding.

**Source**: obsidian `Learned-Verifier-Experiment/experiments/E_fin2.md` (proposed 2026-06-21).

## Research Questions

1. **Is "process predicts quality" (the Phase-3 headline) universal or an artifact of long, messy coding trajectories?** FinQA's short clean derivations are a near worst-case for behavioral signal — discrimination here implies strong generality; collapse implies coding/long-horizon-only.
2. **Does the Phase-3 ordering hold off-coding?** In coding, behavioral beat the LLM signals (behavioral-only RF AUC 0.730 > v009-only 0.682, debate-only 0.682; `phase3_report.md`). Head-to-head behavioral (E_fin2) vs skill-verifier (E_fin1) vs combined on the same examples — does the ordering hold or invert?
3. **Does feature direction reverse by difficulty?** **Note: difficulty-conditioning was already tested in coding and FAILED** (`learned-verifier/results/enew_report.md`: conditioning regressed AUC 0.756→0.743; the cross-sectional reversal is about agent strategy, not difficulty, and difficulty variance was low). So *document whether the paradox appears* (expected, per that prior) — do **not** re-run conditioning as a fix; FinQA's even-lower difficulty variance makes it less likely to help, not more.

## Components

### 1. Compute
- **Platform**: API-driven + local feature extraction / scikit-learn. **No GPU.**
- **GPUs**: none.

### 2. Data
- **Dataset**: FinQA — **original `czyssrs/FinQA` (GitHub, MIT), HARD requirement** (the HF mirror `dreamerdeo/finqa` drops the `program` field this experiment needs).
- **Schema (VERIFIED 2026-06-21)**: ships `qa.program` (DSL: `subtract(`/`divide(`/… `EOF`), `qa.program_re` (nested form), `qa.gold_inds`, `qa.exe_ans`. Gate cleared.
- **Sample**: the **same generated-answer sample as E_fin1** (no new generation).

### 3. Experiment Protocol
- **Metric**: AUC of the behavioral RF vs the 0.756 coding baseline; plus the head-to-head ordering (behavioral vs skill vs combined).
- **Candidate financial-process features** (analogs of the Phase-3 four):
  - **program length** (# ops) — analog of trajectory length / cost
  - **operation diversity** (distinct DSL ops) — analog of action_pct / tool diversity
  - **intermediate-value sanity** (interim results in plausible ranges? sign flips, magnitude blowups) — analog of loop/thrash detection
  - **retry / self-revision count** (if generation traces show reconsideration) — analog of loop_count
- **Loop structure**:
  1. For each generated answer from E_fin1, extract the candidate process features from the reasoning program.
  2. Label outcome via exact-match (reuse E_fin1's labels).
  3. Fit the **same RandomForest recipe as Phase 3**; report AUC + forward-selected subset.
  4. Head-to-head: behavioral-only vs skill-verifier (E_fin1) vs combined on the same examples.
  5. Document whether the difficulty/strategy paradox appears (do not re-run conditioning as a fix — it regressed AUC in coding).
- **Termination**: RF fit + head-to-head + difficulty check complete on the E_fin1 sample.
- **Logging**: per-example feature vector + outcome label; RF AUC, forward-selected subset, head-to-head table, difficulty-conditioned breakdown.

### 4. Networking
- **Access**: none new (reuses E_fin1 generations; feature extraction + RF are local).

### 5. Storage
- **Data**: FinQA original (already cloned for E_fin1); E_fin1's generated-answer sample.
- **Results**: blueprint `results/` (feature table, RF AUC, head-to-head, difficulty breakdown).

## Success Criteria

- [ ] Process features extracted from FinQA reasoning programs for the E_fin1 sample.
- [ ] Phase-3 RF recipe fit; **AUC reported against the 0.756 coding baseline** with forward-selected subset.
- [ ] Head-to-head behavioral vs skill (E_fin1) vs combined — does the Phase-3 ordering hold off-coding?
- [ ] Difficulty-conditioning / Simpson's-Paradox check completed.
- [ ] **Scoping verdict reported**: is Phase 2 universal, or coding/long-horizon-specific?
- [ ] Carryover audit complete (below).

## Non-Requirements

- **No new agent generation** — reuse E_fin1's sample.
- **No FinanceBench** (no structured process signal).
- **No run without confirming `czyssrs/FinQA` ships `program`** (HF mirror does not — hard gate, already verified 2026-06-21).
- No fine-tuning, no GPU.

## Known Limitations

- A partial-fail (AUC 0.55–0.65) is the *expected* and still-publishable result — bounds Phase 2 rather than refuting it.
- Financial derivations are short/clean; thin process signal may reflect the domain's structure, not a flaw in behavioral verification per se.
- Shares E_fin1's sample size (n≈100) — small for RF AUC stability; report confidence intervals.

## Carryover Audit (spec-design gate)

- [ ] Ran `carryover-auditor` — scan `verifier-reward/lessons.md`, `verification-primitives*/lessons.md` (overlapping stack: Phase-3 behavioral RF, RandomForest recipe, n=300 ordering, Simpson's-Paradox/difficulty-conditioning).
- [ ] Carry the priors explicitly (repo-verified): **behavioral-only RF beats the LLM signals** (0.730 vs v009 0.682 / debate 0.682), **best RF AUC 0.756** (`selected_4`, the baseline to beat/miss), and **difficulty-conditioning was already tried in coding and regressed AUC 0.756→0.743** (`enew_report.md`) — so RQ3 documents the paradox rather than re-testing the failed fix.
- [ ] Confirm the `czyssrs/FinQA` program-field gate is reflected as a Non-Requirement hard gate (it is).

---

> **Note**: Operational artifacts belong in `domains/autoresearch/blueprints/e-fin2-finqa-behavioral-features/`, not in this spec.
