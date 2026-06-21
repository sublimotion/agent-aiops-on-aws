# E_harness3 — Reward-Regime × Authoring-Locus Matrix

**Domains**: DBBench (AgentBench SQL, verifiable + withheld) · FinanceBench (free-text, consensus/LLM-judge) ·
**Driver**: Bedrock Claude via `converse` CLI (native tool-use) · **Workers**: Haiku 4.5, Sonnet 4.6 ·
**External author / judge**: Haiku 4.5 · **n**: DBBench 120 (paired), FinanceBench 150 (paired) · **Date**: 2026-06-21

## The matrix — Pass@1 and the locus gap `(external − self)`

| Regime (reward) | worker | self | external | **gap (ext − self)** 95% CI |
|---|---|---|---|---|
| **Verifiable** (DBBench, SQL exec) — *A/C, loaded from E_harness2* | Haiku | 0.833 | 0.808 | **−0.025** [−0.083, +0.025] |
| | Sonnet | 0.867 | 0.858 | **−0.008** [−0.067, +0.042] |
| **Withheld** (DBBench, reward blinded to author) — *B/D, new* | Haiku | 0.775 | 0.775 | **+0.000** [−0.042, +0.042] |
| | Sonnet | 0.808 | 0.808 | **+0.000** [−0.050, +0.050] |
| **Consensus** (FinanceBench, LLM-judge) — *E/F, new* | Haiku | 0.827 | 0.780 | **−0.047** [−0.093, **−0.007**] |
| | Sonnet | 0.867 | 0.880 | **+0.013** [−0.020, +0.047] |

## Headline — the monotonic law is REFUTED (a clean null, weakly reversed)

The core hypothesis was a **monotonic** locus gap: ≈0 verifiable → positive withheld (D>B) →
largest positive consensus (F>E). **It does not hold.**

- **Verifiable (A≈C)**: null, as E_harness2 found. ✓ (the only confirmed prediction)
- **Withheld (D−B)**: **exactly 0.000 on BOTH workers** — the external author did **not**
  re-emerge as helpful. RQ1's central prediction (the *reward*, not the *locus*, was the
  active ingredient → removing the reward revives the locus effect) is **not supported**.
- **Consensus (F−E)**: **−0.047 on Haiku, CI excludes 0 — external authoring HURTS**; +0.013
  on Sonnet (null). The predicted *largest positive* gap is the one cell where the gap goes
  *significantly negative*. RQ2 is **refuted**.
- **Monotonicity (RQ3)**: Haiku gaps run −0.025 → 0.000 → −0.047 (**non-monotonic**); Sonnet
  −0.008 → 0.000 → +0.013 (technically non-decreasing but every step CI crosses 0). No
  step-difference CI excludes 0 on either worker. **The unifying monotonic law is not
  observed.**

**One-line result**: across three reward regimes spanning verifiable → withheld → consensus,
**authoring locus stays inert** (every gap within ±0.05, the single significant one *reversed*).
The external-author advantage does **not** grow as the verifiable reward weakens. E_harness2's
null is *wider* than its own explanation predicted: it survives even when you remove the reward
and even when an LLM-judge supplies a genuine verification asymmetry.

## Why the law fails — the asymmetry the regime supplies is not the asymmetry the *author* needs

The hypothesis assumed "weaker verifiable reward → the external party becomes the only check →
external authoring pays." Two mechanisms break that chain, and both reproduce across workers:

1. **Withholding the reward does not create an authoring asymmetry — it removes information
   from BOTH authors equally.** Self (B) and external (D) read the *same* reward-blinded
   trajectory; neither can see pass/fail (leak audit: strongest channel AUC 0.547 ≪ 0.90).
   So they remain the *same model tier reading the same trace* — exactly the E_harness2 /
   E_fin1 condition under which locus is inert. The gap is **0.000**, not positive: the most
   precisely-null cell in the whole matrix. A vivid tell: on Sonnet the **self** author
   self-gated to author just **6** interventions while the **external** author wrote **87** —
   a 14× volume gap that moved pooled accuracy by **zero**. Authoring *volume* and *locus* are
   both inert without an asymmetry.

2. **The consensus regime's asymmetry lives in the JUDGE, not in the AUTHOR.** The LLM-judge
   *is* a real verification asymmetry (it discriminates: AUC 1.00, near-miss-numeric rejection
   1.00, far stronger than E_fin1's reference-FREE FinQA judge at AUC 0.565 — because here the
   judge grades against the gold reference). But that asymmetry is consumed by *scoring the
   worker's answer*, i.e. by producing the reward. The **authoring** step that follows still
   reflects on a trajectory to write a general rule — and that is no easier for an external
   observer than for the worker. So the consensus regime adds an asymmetry to the *reward
   channel* without adding one to the *authoring channel*, and locus stays inert (Sonnet) or
   the external author's known failure mode dominates (Haiku, below).

3. **Where locus moves at all, it moves the WRONG way (Haiku consensus, −0.047, sig).** This is
   the E_harness2 RQ3 mechanism reproduced in a new domain: the external author, with no stake
   in the next attempt, drifts toward generic / over-applied advice (it authored 33 notes to
   self's 26), and a same-tier external observer with no asymmetry adds noise, not signal. On a
   near-ceiling worker that noise is a net *cost*. Sonnet absorbs it (stronger worker, +0.013
   ns); Haiku does not (−0.047 sig).

## RQ4 — judge calibration is NOT the confound here (the consensus result is real)

Per E_fin1/T4 the worry was that F's signal is an artifact of judge miscalibration. The Stage-0
judge gate (`judge_gate.json`) rules this out: the reference-based judge **discriminates** —
AUC 1.00 on labeled pairs, **rejects 100% of on-topic ×1.4 near-miss numeric errors** (the exact
E_fin1 "engaged-but-not-discriminating" failure mode), stable across temperature (1.00), mean
verdict confidence 0.95 with *separated* pass rates (E 0.827 vs F 0.780 on Haiku). The judge is
not rubber-stamping. So the F−E gap is a real *authoring*-locus effect, not a judge artifact.
(The disambiguator vs E_fin1: that judge was reference-FREE and had to re-derive the answer;
this one grades against the gold reference, which is where its asymmetry comes from — and it
still doesn't help the *author*, only the *reward*.)

## Stage-0 hard gates (both PASS)

- **Reward-withholding leak audit** (`leak_audit.json`) — 240 real trajectories. Structural
  gate: withheld digest exposes no reward field (no `Gold answer:` line / `(WRONG)` tag),
  positive-control-validated against the visible digest (36/36 failures). Empirical
  separability: strongest single digest channel recovers reward at **AUC 0.547 < 0.90** →
  informative-but-not-readable. Author invoked on **every** task (reward-independent schedule;
  D `author_calls=120/120`) so the invocation pattern leaks nothing. **B/D is a clean (bounded)
  ablation.**
- **FinanceBench judge gate** (`judge_gate.json`) — AUC 1.00, near-miss rejection 1.00,
  temperature stability 1.00 → **E/F reward signal usable; RQ4 confound excluded.**

## Carryover audit outcome

The pre-run `carryover-auditor` raised 2 P0s; both were designed out before any cell ran:
- **P0-1 (reward leak via invocation pattern)**: the killer leak in E_harness2 was that the
  author fired *only on failures* — the call itself is the reward. Fixed: withheld cells invoke
  the author on every task and self-gate via `{"skip":true}`. Verified `author_calls=120`.
- **P0-2 (stable-but-blind judge)**: the gate measures discrimination (AUC) + a near-miss
  numeric probe, not just temperature stability. The judge passed the *hard* probe, so E/F are
  reported as a clean cell, not "judge-confounded."
- P1 (multi-tool-turn Bedrock fix present — reused verbatim; FinanceBench worker is single-call
  so it can't bite there), P1 (ceiling/headroom confound — see Threats), P2 (python — 3.12 here,
  no sklearn in the run path) all handled.

## Threats to validity

- **Strong-worker ceiling (the dominant caveat).** All cells floor 0.78–0.88; the harness-/
  authoring-addressable error budget is thin (E_harness2's SELECT-class story). A null on a
  near-ceiling worker bounds the claim to that regime. Withholding *lowered* the DBBench floor
  (A→B: 0.833→0.775 Haiku) — i.e. B/D do have *more* headroom than A/C — and the gap was still
  exactly 0.000, which makes the withheld null **stronger**, not ceiling-masked. FinanceBench
  floors similarly (0.78–0.88), so headroom does not differ enough across regimes to confound
  the (small) gaps; but a weaker backbone could still surface effects we cannot see here.
- **Consensus uses a reference-based judge.** Its asymmetry comes from grading against the gold
  answer. A reference-free judge (E_fin1's setting) would be weaker and might change F — but
  that would *strengthen* the conclusion (even less asymmetry for the author to inherit).
- **Two domains, three regimes — supports, doesn't prove.** A genuinely reward-free generation
  regime (no success metric) is out of scope; the withheld-from-authoring design is the faithful
  compromise the leak audit validates.
- **DBBench SELECT-only oracle** (inherited from E_harness2): mutation guards untested; L1-class
  effects are a lower bound. Not load-bearing for the locus comparison (held fixed across cells).

## Verdict

1. **A≈C** (verifiable): null. ✓ confirmed (E_harness2).
2. **D=B** (withheld): locus gap **exactly 0.000** on both workers — removing the verifiable
   reward did **not** revive the external author. RQ1 refuted.
3. **F vs E** (consensus): **−0.047 (Haiku, sig) / +0.013 (Sonnet, ns)** — external authoring is
   if anything *worse*, never the predicted largest *positive* gap. RQ2 refuted.
4. **Monotonicity** (RQ3): **not observed** — no monotone trend, the one significant gap reversed.
5. **RQ4**: judge discriminates (AUC 1.00, near-miss 1.00) — F is not a calibration artifact.

**Law correction.** E_harness2 proposed: *external authoring pays in proportion to the
verification asymmetry the regime supplies.* E_harness3 falsifies the "in proportion to the
**regime's** asymmetry" half. The asymmetry that matters is the one available to the **authoring
act itself** — checking a candidate against a cheaper oracle. Reward regime governs the *reward*
channel; it does not hand the *author* an oracle. Trajectory-reflection authoring has no such
oracle in any of the three regimes, so **locus is inert across all of them** — and a same-tier
external author with no asymmetry adds noise, which costs a near-ceiling weak worker (Haiku
consensus). The boundary E_fin1 and E_harness2 drew for *verification* holds for *authoring*, and
**weakening the reward does not move that boundary** — only an author-side oracle would.

## Cost

API-only, no GPU. Four new cells (B/D/E/F × 2 workers) + gates ≈ **$8.5** total on Bedrock
Haiku/Sonnet. A/C loaded from E_harness2 (not re-run). DBBench cells dominate cost (multi-turn
episodes + authoring on every withheld task); FinanceBench cells are single-call worker + judge.
