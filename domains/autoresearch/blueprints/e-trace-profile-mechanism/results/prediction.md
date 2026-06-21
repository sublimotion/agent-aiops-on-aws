# PRE-REGISTERED PREDICTION — FinanceBench behavioral-RF AUC

**Written from the trace-richness profile ALONE. Committed to git BEFORE any RF
fit on FinanceBench (the anti-confirmation gate). Do not edit after the fit.**

Profile source: `results/profile.json` (this commit). Recipe to be used in the
fit: verbatim Phase-3 — `RandomForestClassifier(n_estimators=200, max_depth=7,
class_weight="balanced", random_state=42)`, `nan→-999`, 5-fold stratified CV,
pooled OOF probabilities, bootstrap 95% CI (2000). Fit **per cell** (single
model) — the clean analog of the FinQA single-model (Haiku, n=100) design, and
to avoid a cost_usd model-detector confound when pooling Haiku+Sonnet (~10×
price gap). Behavioral feature set (pre-committed):
`[input_tokens, output_tokens, cost_usd, tokens_ratio, jit_state_chars,
jit_notes_total]`. `judge_conf` / `judge_verdict` EXCLUDED — `is_correct`
derives from the judge, so those are label leakage, not process signal.

## What the profile shows (the basis for this prediction)

| Domain / cell | n | base rate | out_tok mean | out_tok CV | best behavioral \|d\| | implied univariate AUC* |
|---|---:|---:|---:|---:|---:|---:|
| Coding (Phase-3 ref) | 300 | 0.583 | multi-edit agentic | — | top-3 = 95% imp | **0.756 (realized)** |
| FB E_sonnet | 150 | 0.867 | 271.7 | 0.578 | output_tokens −0.92 | ~0.74 |
| FB F_sonnet_v-haiku | 150 | 0.880 | 268.3 | 0.566 | output_tokens −1.04 | ~0.77 |
| FB F_haiku_v-haiku | 150 | 0.780 | 336.0 | 0.367 | output_tokens/ratio −0.79 | ~0.71 |
| FB E_haiku | 150 | 0.827 | 332.6 | 0.361 | output_tokens −0.66 | ~0.68 |
| FinQA | 100 | 0.710 | 147.9 | 0.292 | reasoning_words −0.57 | ~0.65 |

*implied univariate AUC = Φ(\|d\|/√2), normal approximation, in-sample — an
upper-ish reference, not the CV RF number.

**Richness ordering (output-token scale + variance):**
`coding  >  FinanceBench (270–336 tok, CV 0.36–0.58)  >  FinQA (148 tok, CV 0.29)`.

**Separation ordering (strongest behavioral feature):**
`coding  >  FinanceBench (|d| 0.66–1.04)  >  FinQA (|d| ≤ 0.57)`.

Both orderings agree: FinanceBench sits **between** FinQA and coding. The
separating feature is `output_tokens` (and its collinear partner `tokens_ratio`)
with a **negative** sign — passing answers are SHORTER; wrong answers ramble.
This is the *same single-call verbosity axis* FinQA showed (FinQA
`beh_output_tokens` d=−0.54), but with **more dynamic range** because
FinanceBench answers are ~2× longer and ~1.5–2× more variable.

## Prediction (the gate)

**If the behavioral RF is a trace-richness detector** (richer/more-variable
traces ⇒ more separable process signal ⇒ higher AUC), then because FinanceBench
is richer than FinQA on every scalar, the realized per-cell RF AUC should land
**above the FinQA 0.569 point and below the coding 0.756**, i.e. the monotone
`0.569 < FB < 0.756` ordering holds.

- **Predicted per-cell RF CV AUC band: 0.62 – 0.72** (point estimate ≈ **0.67**).
  - Sonnet cells (stronger \|d\|, higher CV) predicted higher within band: **0.65 – 0.74**.
  - Haiku cells (weaker \|d\|) predicted lower within band: **0.60 – 0.70**.
- Expect CV AUC **below** the implied-univariate numbers (CV shrinkage; the two
  informative features are collinear, so RF gains little beyond the single best;
  high base rate ⇒ only ~18–33 minority/cell ⇒ noisy folds).
- **CI width:** expect ≈ ±0.08 – ±0.12 per cell (n=150, small minority). State as
  **descriptive separation, NOT a significance claim.**

### Falsifiable core (decide the mechanism)

1. **Tracks richness** ⇒ all/most FB cells land in 0.62–0.72, clearly above
   FinQA's 0.569 point. Verdict: RF is a trace-richness detector on the
   single-call verbosity axis.
2. **Null-regardless** ⇒ FB AUC collapses back to ~0.50–0.59 (inside FinQA's CI)
   *despite* richer traces. Verdict: richness does NOT drive it; finding hardens
   to "behavioral RF doesn't transfer off coding, full stop."
3. **Over-performs** ⇒ FB AUC ≥ 0.75 (≥ coding). Would contradict the
   richness-monotone story (single-call < multi-edit in richness) and demand a
   confound check (base-rate leakage, model-detector).

## CRITICAL caveat bounding any positive verdict

FinanceBench (e-harness3) traces are **single-call graded Q&A, NOT agentic 10-K
retrieval**. There is **no edit/loop/revision substrate** — `jit_*` features are
near-constant (\|d\| ≤ 0.35) and contribute nothing. So a positive result
confirms only that **richness on the verbosity/length axis** tracks AUC. The
coding mechanism's *loop/edit-thrash* substrate (its top features
`beh_loop_count`, `beh_tokens_per_edit`) remains **untested** here and would
require a fresh agentic FinanceBench run (deferred per scope). The conclusion is
therefore bounded to the verbosity axis, not the full thrash-detector claim.
