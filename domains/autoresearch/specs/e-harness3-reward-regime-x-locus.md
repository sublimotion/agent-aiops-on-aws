# Autoresearch Spec: E_harness3 — Reward-Regime × Authoring-Locus Matrix

## Status: COMPLETE (2026-06-21)

> **Result**: monotonic hypothesis **REFUTED** — locus gap is inert across all three
> regimes (verifiable −0.025/−0.008, withheld +0.000/+0.000, consensus −0.047 sig/+0.013),
> the one significant gap *reversed*. Both Stage-0 hard gates passed (leak audit AUC 0.547;
> judge AUC 1.00, near-miss rejection 1.00). The active asymmetry is the one available to the
> *authoring act*, not the one the reward *regime* supplies — weakening the reward does not
> move the E_harness2/E_fin1 boundary. See
> `domains/autoresearch/blueprints/e-harness3-reward-regime-x-locus/results/report.md`.

## Overview

E_harness2 overturned the "external authoring beats self" hypothesis: on DBBench, external-LLM-authored harness interventions added nothing over self-authored ones (L2→L3: −0.025 / −0.008). The agent's own synthesis explained why — **external-vs-self locus is inert for any agent-construction engine UNLESS the external party supplies a verification asymmetry the environment doesn't already give for free.** DBBench has a free verifiable reward (SQL executes → pass/fail), so the external judge was redundant.

E_harness3 tests that principle directly by varying **reward regime** against **authoring locus** in one matrix:

| | **self-author** | **external-author** |
|---|---|---|
| **Verifiable reward** (DBBench: SQL execution) | **A** ✓ reuse E_harness2 L2 | **C** ✓ reuse E_harness2 L3 |
| **Reward withheld** (DBBench, pass/fail hidden from the *authoring* loop) | **B** (new) | **D** (new) |
| **Consensus reward** (FinanceBench: free-text, LLM-judge vs reference) | **E** (new, gated) | **F** (new, gated) |

**Core hypothesis (monotonic prediction):** the external-author advantage **grows as verifiable reward weakens** —
- **Verifiable (A≈C):** null — confirmed by E_harness2.
- **Withheld (D>B):** re-emerges — with no reward signal, the external party is the only check available.
- **Consensus (F>E, largest):** the LLM-judge *is* the verification asymmetry; external authoring should pay the most.

If that monotonic pattern holds, it nails the law: **external authoring pays exactly in proportion to the verification asymmetry it supplies** — unifying E_harness2 (verifiable → judge redundant) and E_fin1 (verifiable finance → adversarial verifier redundant) as the same effect.

**Depends on**: E_harness2 (cells A/C + the DBBench harness, OpenAI Agents SDK + Bedrock plumbing, all reused); E_fin1 (the FinanceBench-is-messy lesson, T4 calibration-is-model-specific); verifier-reward T5.

**Source**: conversation synthesis 2026-06-21 — the verifiable-reward (RLVR-adjacent) framing of E_harness2's null.

## Terminology

"Verifiable reward" (practitioner term, cf. **RLVR** — RL from Verifiable Rewards): a cheap programmatic check yielding ground-truth reward (SQL execution, exact-match). "Consensus reward": a reference answer exists but scoring it needs judgment (free-text, expert/LLM-judged). The axis is the *cheapness/reliability of the reward signal*, not a binary.

## Research Questions

1. **D vs B**: with the DBBench verifiable reward withheld from the authoring loop, does the external LLM-author re-emerge as helpful (D>B)? This is the direct test that the *reward*, not the *locus*, was the active ingredient in E_harness2.
2. **F vs E**: in a consensus regime (FinanceBench, no programmatic oracle), where the LLM-judge supplies the only check, does external authoring pay the most (F>E, the largest gap in the matrix)?
3. **Monotonicity**: does the external-author advantage increase monotonically across regimes (verifiable → withheld → consensus)? That is the unifying-law result.
4. **Judge calibration confound (RQ for E/F)**: per E_fin1/T4, the LLM-judge's calibration is the fragile, non-transferring layer. Is F's signal real, or an artifact of judge miscalibration? (Gated in Stage 0.)

## Components

### 1. Compute
- **Platform**: API-driven (Bedrock Claude via OpenAI Agents SDK + LiteLLM). **No GPU.** Detached `agent-runner` job.
- **GPUs**: none.

### 2. Data / Code
- **DBBench** (cells A–D): reused from E_harness2's blueprint (`vendor/life_harness_dbbench.py`, the AgentBench fork). Same eval split, same 2 Bedrock worker models (haiku + sonnet).
- **FinanceBench** (cells E/F): source in Stage 0 (`patronus-ai/financebench` or the original GitHub release). Free-text answers + reference answers from 10-K filings. **No programmatic oracle** — this is the point.
- **Reuse A/C verbatim from E_harness2** — do NOT re-run them; load `e-harness2-.../results/L2_*.jsonl` and `L3_*.jsonl` as cells A and C. Re-running would waste budget and risk drift.

### 3. Experiment Protocol
- **Metric (primary)**: per-cell task success; the matrix deltas B vs D, E vs F, and the monotonic trend of (external − self) across the three regimes, with bootstrap CIs.
- **Reward mechanisms by regime**:
  - **Verifiable (A/C, DBBench)**: SQL execution pass/fail — the authoring loop sees it (as in E_harness2).
  - **Withheld (B/D, DBBench)**: the worker still executes SQL to *act*, but the **authoring step is blinded to pass/fail** — it must infer which tasks failed. **Stage-0 leak audit (HARD GATE):** confirm no success signal leaks to the author via error strings, row counts, empty results, or exception types. If the ablation isn't clean, B/D are meaningless — fix or drop.
  - **Consensus (E/F, FinanceBench)**: an LLM-judge scores the worker's free-text answer against the reference (reasoning-equivalence, not string-match). The judge's verdict is the reward the authoring loop uses.
- **Locus**: self = worker authors interventions from its own (regime-appropriate) reward signal; external = a separate Bedrock Claude instance observes + authors.
- **Hold fixed** within a domain: split, task order, worker model, intervention types — vary only reward visibility (B/D) and locus (self/external).
- **Termination**: B, D, E, F scored on both worker models; A/C loaded from E_harness2.
- **Logging**: per-task {cell, regime, locus, model, success, reward_seen, interventions_authored, tokens, cost}; matrix aggregate + deltas + CIs + monotonic-trend test.

### 4. Networking
- Bedrock (IRSA on the runtime). FinanceBench fetched in Stage 0.

### 5. Storage
- **Results**: blueprint `results/` (per-cell jsonl, matrix.json, analysis).
- Reuse E_harness2 results for A/C by reference.

## Success Criteria

- [ ] **Stage-0 SDK+Bedrock smoke** (one DBBench task end-to-end) — reuse E_harness2's proven setup.
- [ ] **Stage-0 reward-withholding leak audit (HARD GATE)**: prove the authoring loop cannot see DBBench pass/fail through side channels. Document the audit. If unclean → fix or report B/D as void.
- [ ] **Stage-0 FinanceBench gate (HARD GATE for E/F)**: dataset sourced; LLM-judge scoring validated on ~10 reference pairs for stability (agreement across temperature; no systematic miscalibration per E_fin1). If the judge is unstable, E/F are reported as "judge-confounded, inconclusive" rather than as a clean cell — and B/D still stand.
- [ ] A/C loaded from E_harness2 (not re-run); B/D/E/F measured, both worker models, bootstrap CIs.
- [ ] **Matrix verdict**: B-vs-D, E-vs-F, and the monotonicity of (external − self) across verifiable → withheld → consensus.
- [ ] Judge-calibration confound for F explicitly addressed (RQ4).
- [ ] Carryover audit complete (below).

## Non-Requirements

- **No GPU / no open-weight self-hosting** — Bedrock only.
- **No re-running E_harness2 cells A/C** — load them.
- **No new domain beyond FinanceBench** — DBBench (verifiable) + FinanceBench (consensus) span the axis; same finance domain as FinQA controls for domain when flipping reward type.
- No weight updates / fine-tuning.

## Known Limitations

- **FinanceBench was rejected by E_fin1 for exactly the reason it's used here** ("free-text, human-review-only, no cheap objective ground truth"). That non-verifiability is the consensus-regime datapoint — but it means E/F's reward is LLM-judge-mediated, and the judge's calibration (E_fin1/T4: the fragile layer) could confound F. The Stage-0 judge gate and RQ4 are the guards; if they fail, the clean result is the DBBench B/D half.
- Two domains, not a full sweep — supports the monotonic law, doesn't prove it. A no-reward (open-ended generation) regime would complete the axis; out of scope here.
- DBBench reward-withholding is an artificial ablation; a genuinely reward-free task would be cleaner but lacks a success metric to even score the experiment. The withheld-from-authoring design is the faithful compromise — Stage-0 leak audit is what makes it valid.
- OpenAI Agents SDK is GPT-native on Bedrock; E_harness2 proved it works, but watch for SDK-on-Bedrock artifacts (the L0/L1 replication gate logic carries over).

## Carryover Audit (spec-design gate)

- [x] Ran `carryover-auditor` (Stage 0, pre-run) — scanned `e-harness2-.../`, `e-fin1-.../`, `verifier-reward`, `verification-primitives*`. 6 gaps (2 P0, 3 P1, 1 P2); both P0s designed out before any cell ran (see `blueprints/.../results/report.md` §Carryover audit outcome).
- [x] Carry priors explicitly: E_harness2 A≈C is the extended result; E_fin1 FinanceBench-is-messy drove the judge gate; T4 → RQ4 (judge confound excluded: AUC 1.00); T5 → self-author ceiling. (Ran on python3.12; no sklearn in the run/analysis path — pure-Python bootstrap.)
- [x] Both Stage-0 hard gates implemented as blocking and **PASSED**: reward-leak audit (`leak_audit.py`, strongest channel AUC 0.547 ≪ 0.90) + FinanceBench judge gate (`judge_gate.py`, AUC 1.00, near-miss rejection 1.00, strengthened beyond stability per carryover P0-2).

---

> **Note**: Operational artifacts belong in `domains/autoresearch/blueprints/e-harness3-reward-regime-x-locus/`, not in this spec.
