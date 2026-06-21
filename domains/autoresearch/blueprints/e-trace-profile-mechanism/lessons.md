# Lessons — e-trace-profile-mechanism

Data-only mechanism test (local, python3.13). No GPU/API/new runs.

## Outcome

Behavioral RF is a **trace-richness / verbosity detector** (SUPPORTED, bounded
to the verbosity axis). FinQA null (0.569) is structural (sparse 1–2-op traces),
not domain. FinanceBench — richer single-call traces — recovers separation.

## Findings

- **Profile decided the branch, before any fit.** FinanceBench output tokens
  ~2× FinQA (270–336 vs 148) with ~1.5–2× the CV (.36–.58 vs .29), and
  pass/fail Cohen's d up to 1.04 vs FinQA's ≤0.57. Both richness and separation
  order `coding > FinanceBench > FinQA`. This refuted the spec's
  "inconclusive-too-sparse" escape hatch up front.
- **Separating feature = `output_tokens`/`tokens_ratio`, NEGATIVE sign**: passing
  answers are SHORTER, wrong answers ramble. Same direction as coding's thrash
  story, on a single-call verbosity axis.
- **n=150 per-cell is noise-dominated** (base rate 0.78–0.88 ⇒ only 18–33
  minority/cell; CI width 0.21–0.32). Per-cell mean AUC ≈0.58 ≈ FinQA 0.569;
  per-cell \|d\| did NOT rank per-cell AUC → proof the per-cell estimate is noise.
- **n=300 within-model resolves the signal**: haiku 0.743 / sonnet 0.729, CIs
  EXCLUDE 0.5, at/near coding 0.756. Point estimate *moved up* (0.58→0.74), not
  just tightened ⇒ n=150 underpowered. E/F pooling clean (cellid→label AUC
  0.46/0.52). RF beats single best feature (haiku out_tok-only 0.63 → full 0.74).

## Methodology lessons (carry forward)

- **Confound: pooling models inflates a cost_usd model-detector.** Haiku vs
  Sonnet pricing differs ~10×, so `cost_usd` becomes a model tag. Pooled-all hit
  0.768 but is CONFOUNDED. Fit per-model (or per-cell). Always check whether a
  "process" feature is secretly an identity feature when pooling heterogeneous
  sources.
- **Exclude judge-derived fields as leakage.** `is_correct` derives from
  `judge_conf`/`judge_verdict`; including them = circular. Dropped.
- **Pre-registered point bands are easy to set too tight at small n.** Band was
  0.62–0.72; per-cell mostly fell below due to CV shrinkage. The directional
  prediction (`FinQA < FB < coding`) held; the point band did not at n=150.
  Lead with effect-size ordering (n-independent) when n can't be matched
  (FinQA=100 cannot match FB cells).
- **python3.13 mandatory** (macOS python3.14 sklearn broken).

## Bounds / follow-up

- Confirms richness→AUC only on the **verbosity/length axis**. The coding
  mechanism's top features (`beh_loop_count`, `beh_tokens_per_edit`) have **no
  substrate** here — `jit_*` near-constant (\|d\|≤0.35). Full thrash-detector
  claim UNTESTED. Follow-up: fresh **agentic FinanceBench** (10-K retrieval,
  multi-step) to test the loop/edit features, not just verbosity.
