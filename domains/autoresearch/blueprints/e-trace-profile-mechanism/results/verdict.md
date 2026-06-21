# E_trace-profile — VERDICT: is the behavioral RF a trace-richness detector?

**Status: SUPPORTED (directionally), on the single-call VERBOSITY axis — NOT
null-regardless, NOT inconclusive-too-sparse, but BOUNDED below the full
thrash/loop mechanism claim. Honest read: richness predicts behavioral-RF
separability; the headline AUC ranking rests on an n that FinQA can't be
matched against, so report effect sizes as the primary evidence.**

Recipe: verbatim Phase-3 RF (200×depth-7, class-balanced, seed 42, nan→−999,
5-fold stratified pooled OOF, bootstrap 95% CI ×2000), python3.13. Features
pre-committed: `[input_tokens, output_tokens, cost_usd, tokens_ratio,
jit_state_chars, jit_notes_total]`; `judge_*` excluded as label leakage. Fit
per cell (FinQA-matched single-model design) and pooled-within-model (n=300).

Anti-confirmation gate honored: `prediction.md` + `profile.json` committed at
**5e63a3f BEFORE** `rf_financebench.py` ran (verifiable in git history).

## The three-point comparison

| Point | n | out_tok mean | out_tok CV | best behavioral \|d\| | RF AUC | 95% CI | CI width |
|---|---:|---:|---:|---:|---:|---|---:|
| **coding (Phase-3)** | 300 | multi-edit | — | top-3 = 95% imp | **0.756** | — | — |
| FinQA | 100 | 148 | 0.29 | 0.57 | **0.569** | [0.43, 0.71] | 0.28 |
| FB E_haiku (per-cell) | 150 | 333 | 0.36 | 0.66 | 0.455 | [0.33, 0.57] | 0.24 |
| FB E_sonnet (per-cell) | 150 | 272 | 0.58 | 0.92 | 0.611 | [0.49, 0.74] | 0.25 |
| FB F_haiku (per-cell) | 150 | 336 | 0.37 | 0.79 | 0.654 | [0.55, 0.75] | 0.21 |
| FB F_sonnet (per-cell) | 150 | 268 | 0.57 | 1.04 | 0.604 | [0.44, 0.76] | 0.32 |
| **FB haiku (n=300 within-model)** | 300 | — | — | — | **0.743** | [0.68, 0.81] | 0.13 |
| **FB sonnet (n=300 within-model)** | 300 | — | — | — | **0.729** | [0.64, 0.81] | 0.17 |

## How the realized result compares to the pre-registered prediction

Pre-registered per-cell band: **0.62–0.72 (≈0.67)**, expecting `0.569 < FB < 0.756`.

- **Per-cell (n=150, the n-matched-to-FinQA unit): PARTIAL hit.** 1/4 cleanly in
  band (F_haiku 0.654); 2/4 just below (E_sonnet 0.611, F_sonnet 0.604); 1/4 a
  clear miss (E_haiku 0.455). Mean ≈ 0.58 — statistically **indistinguishable
  from FinQA's 0.569**; every per-cell CI straddles or nearly straddles 0.5.
  My point band was optimistic at n=150: CV shrinkage + tiny minority (18–33
  fails/cell) leaves the per-cell AUC noise-dominated. Per-cell \|d\| doesn't even
  rank the per-cell AUCs (sonnet has the largest \|d\| but mid AUC) — direct proof
  the per-cell estimate is noise, not signal resolution.
- **n=300 within-model: prediction VINDICATED in direction, point estimate
  HIGHER than band.** AUC 0.743 / 0.729 with CIs that **exclude 0.5** — clearly
  above FinQA and at/near the coding 0.756. The point estimate *moved up* from
  the per-cell mean (~0.58 → ~0.74), not merely tightened, i.e. n=150 was
  underpowered and n=300 resolves true signal ~0.74. E/F pooling is clean
  (cell-id→label AUC 0.46/0.52, no E-vs-F confound), so this is real
  richness resolution, not a pooling artifact. The RF also beats its single best
  feature (haiku: output_tokens-only 0.63 → full-6 0.74), so it is genuinely
  combining process features, not just reading length.

## Does behavioral-RF signal track trace-richness? YES (by effect size + n=300 fit)

The **n-independent** richness evidence is unambiguous and was locked in the
committed profile:

- **Richness ordering** (output-token scale + variance): `coding > FinanceBench
  (270–336 tok, CV .36–.58) > FinQA (148 tok, CV .29)`.
- **Separation ordering** (strongest behavioral feature, Cohen's d): `coding >
  FinanceBench (|d| .66–1.04) > FinQA (|d| ≤ .57)`.

Both orderings agree and place FinanceBench **between** FinQA and coding —
exactly the monotone the trace-richness mechanism predicts. The separating
feature is `output_tokens` / `tokens_ratio` with a **negative** sign (passing
answers are SHORTER; wrong answers ramble) — the *same direction* as coding's
thrash story (more output ⇒ more floundering ⇒ fail). The RF AUC confirms this
once n is adequate (n=300 → 0.73–0.74).

## Why this is NOT the "null-regardless" branch, and NOT "inconclusive-too-sparse"

- **Not null-regardless:** a null would require FB to collapse to ~0.50–0.59
  *despite* richer traces. Instead FB reaches 0.73–0.74 at n=300 and its
  effect sizes scale up with richness. Richness clearly drives separability.
- **Not inconclusive-too-sparse:** the spec's escape hatch was "if FinanceBench
  is nearly as sparse as FinQA, the test can't decide." The profile refutes
  that premise — FB output tokens are ~2× FinQA's with ~1.5–2× the CV, and
  separation is strictly larger. There WAS enough trace to read.

## CRITICAL caveats (do not overclaim)

1. **Underpower.** At n=150 the per-cell CI width is ~0.21–0.32 (wider than the
   spec's ~±0.07 rule of thumb because the base rate is 0.78–0.88, leaving only
   18–33 minority/cell). The n=300 CIs are ±0.07–0.08. **All AUCs here are
   DESCRIPTIVE separation, not significance claims.** The clean statement that
   "FB beats FinQA in RF AUC" rests on the **n=300** fit; FinQA has only 100
   examples and **cannot be n-matched**, so the fairest n-matched comparison
   (per-cell 150 vs FinQA 100) shows the two **overlapping**. The robust,
   n-independent evidence is the **effect-size ordering**, which is why it leads.

2. **Verbosity axis only — the loop/edit substrate is UNTESTED.** FinanceBench
   (e-harness3) traces are **single-call graded Q&A, NOT agentic 10-K
   retrieval.** The coding mechanism's top features `beh_loop_count` and
   `beh_tokens_per_edit` have **no substrate** here: `jit_state_chars` /
   `jit_notes_total` are near-constant (\|d\| ≤ 0.35, ~0 RF importance). So this
   experiment confirms only that **richness on the answer-length/verbosity axis**
   tracks AUC. The full *thrash/loop-detector* claim (the 95%-importance trio)
   remains untested and requires a **fresh agentic FinanceBench run** (deferred
   per scope). Conclusion is bounded to the verbosity axis.

3. **Single corpus, single harness family**, two models. Directional finding,
   not a powered law.

## Bottom line for the playbook

The behavioral RF is, on the evidence here, a **trace-richness / verbosity
detector**: its discriminative power scales with how long and variable the
trace is. FinQA's null (0.569) is a **structural** consequence of sparse 1–2-op
traces, not a domain property — FinanceBench, with richer single-call traces,
recovers separation (effect sizes ↑; n=300 AUC ≈ 0.74). This **hardens** the
E_fin2 finding from "doesn't transfer off coding" to "**needs a trace rich
enough to carry a process signal; given one, it works.**" The unresolved leg —
whether the *loop/edit-thrash* features (not just verbosity) transfer — needs
an agentic FinanceBench run and is the recommended follow-up.
