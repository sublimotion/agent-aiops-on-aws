# E_fin1 — FinQA Skill-Verifier Replication: Analysis

**Run date**: 2026-06-21 · **Verifier model**: Claude Haiku 4.5 (Bedrock, us-east-2)
· **Eval set**: 100 FinQA dev examples (seed=42, all numeric) · **Agent model**: Haiku 4.5

## Headline verdict: NULL RESULT — the adversarial lift does NOT transfer to FinQA

The coding-domain skill-verifier result (precision **0.40 → 0.92**, a **2.3× lift**
from confirmatory→adversarial reframing) **does not replicate** on FinQA financial
numeric QA. Both verifiers sit essentially at the agent's base rate; neither
adds meaningful discrimination.

| Metric | Confirmatory | Adversarial (v009 analog, 4/4) | Coding reference |
|--------|-------------:|-------------------------------:|------------------|
| Precision on confident subset | **0.745** | **0.740** | 0.40 → 0.92 |
| Recall | 0.986 | 0.761 | — |
| **Lift (adv/conf precision)** | — | **0.99×** | **2.3×** |
| AUC | 0.629 | 0.565 | high |
| Confident-subset size | 94/100 | 73/100 | — |

- **Base rate (agent accuracy) = 0.71.** Adversarial precision (0.740) ≈ base rate ≈
  confirmatory precision (0.745). Selecting the "confident" subset barely moves
  precision above picking answers at random.
- **AUC 0.57–0.63** (0.50 = random). The verifiers carry weak signal at best; the
  adversarial ensemble's vote count is a *worse* ranker than the confirmatory 1–5
  rating (0.565 vs 0.629).
- **Lift = 0.99× (delta −0.005).** Adversarial framing produced **no precision gain**
  over the confirmatory baseline. This is the spec's "the adversarial claim is more
  coding-specific than stated" outcome (RQ1, negative branch).

### Robustness to label noise (the verifier-reward semantic-mismatch lesson)

29 answers were scored "wrong" by exact-match. An audit found **11 of 29 are
scoring artifacts, not agent errors**: 6 free-text/multi-number parse failures
(`"$22.57 billion"` vs gold `22556579533`, magnitude correct) and 5 sign/representation
mismatches (`"12.39%"` vs gold `-0.1239`, magnitude correct). This is exactly the
**recall-ceiling-is-semantic-mismatch-not-noise** lesson carried from verifier-reward.

Re-scoring with these 9 (programmatically recoverable) cases corrected:

| Metric | Confirmatory | Adversarial | Lift |
|--------|-------------:|------------:|-----:|
| Precision (corrected, base rate 0.80) | 0.830 | 0.836 | **1.01×** |

**The null differential is robust**: label noise hits both verifiers identically, so it
cannot manufacture or mask a *relative* lift. Whether base rate is 0.71 (raw) or 0.80
(corrected), the adversarial-vs-confirmatory lift is ≈1.0×.

## Why it failed to transfer (mechanism)

The adversarial rubric instructs the verifier to *independently recompute* the answer
and attack it. On FinQA this fails for a structural reason absent in coding:

- **No execution oracle for the verifier.** In the coding domain the patch either does
  or does not satisfy a hidden test; the adversarial verifier reasons about a
  *deterministic* artifact. On FinQA the verifier must itself perform the same
  table-lookup + arithmetic the agent did — and **Haiku-as-verifier makes the same
  class of mistakes Haiku-as-agent makes** (wrong cell, percent/ratio confusion). When
  agent and verifier share a failure mode, adversarial recomputation rubber-stamps the
  agent's error: **19 of 29 wrong answers got 4/4 "likely_correct"** (only 34% of true
  errors were flagged at all).
- **The adversarial verifier does engage** — it is NOT defaulting to "uncertain"
  (318/400 calls = likely_correct, 61 uncertain, 21 likely_incorrect; vote histogram
  heavily bimodal at 0 and 4). So the RQ2 "calibration drift / defaults-to-uncertain"
  failure mode is **ruled out for Claude**. The problem is not abstention; it is that
  confident adversarial recomputation is *wrong* about as often as it is right on
  numeric reasoning at this model tier.

This is a sharper claim than "domain-specific": the adversarial lift requires a
**verification asymmetry** (checking is cheaper/more reliable than generating). Coding
has it via test execution. FinQA, scored only on the final number with no per-step
oracle, **does not** — the verifier is doing the same hard task as the agent.

## Cost (RQ3): well under ceiling — the ceiling was never the constraint

| | per eval | vs $0.03 ceiling |
|--|---------:|------------------|
| Generation (1 call) | $0.00111 | — |
| Confirmatory (1 call) | $0.00125 | under |
| **Adversarial (4 calls)** | **$0.01341** | **UNDER (45% of ceiling)** |

Avg adversarial input = 5,788 tok/eval, output = 1,524 tok/eval. Under the coding-domain
Haiku pricing (0.80/4.00) it is $0.01073/eval. Table context did **not** blow the budget
— gold_inds-scoped context kept inputs small. **RQ3 answer: yes, $0.03 holds on FinQA.**
But cost is moot given the null precision result.

## Research-question answers

1. **Does adversarial framing's precision lift survive a no-test-suite, numeric,
   table-heavy domain?** **No.** Lift 0.99× (vs 2.3× coding). The Part-4 cross-domain
   projections do **not** get a positive anchor here; instead E_fin1 is an
   *applicability-bounding* result: the adversarial reframe needs a verification
   asymmetry (an execution/test oracle) that FinQA lacks.
2. **Does verifier-model calibration hold off-coding?** For **Claude**: the adversarial
   verifier engages and does not default to uncertain (so the coding "only Claude
   calibrates" story is not contradicted on the abstention axis) — but engagement ≠
   discrimination. AUC 0.57. **Cross-verifier disambiguation (RQ2, n=40): the null is
   NOT verifier-model-specific.** On the same 40-example subset, Haiku and Nova-Pro
   give the *identical* 0.92× adversarial lift (Haiku adv precision 0.607 / conf 0.658;
   Nova adv 0.577 / conf 0.625 = base rate exactly). Nova's confirmatory verifier
   accepts all 40 (precision = base rate); its adversarial engages (133/160 likely_correct,
   15 uncertain, 12 likely_incorrect) — same engaged-but-non-discriminating pattern as
   Claude. So the FinQA null is a **domain/asymmetry** property, not a verifier-model
   choice: switching to a cheaper or different model does not recover the lift.

### RQ2 cross-verifier table (n=40 subset, base rate 0.625)

| Verifier | Confirmatory prec | Adversarial prec | Lift |
|----------|------------------:|-----------------:|-----:|
| Claude Haiku 4.5 | 0.658 | 0.607 | 0.92× |
| Amazon Nova Pro | 0.625 | 0.577 | 0.92× |

Both null. Unlike the coding-domain T4 (where non-Claude models failed by *abstaining*),
here Nova fails the same way Claude does — by confidently affirming wrong answers. The
shared failure confirms the mechanism is the missing verification asymmetry, not
model calibration.
3. **Does the $0.03/eval ceiling hold?** **Yes** — $0.0134/eval, 45% of ceiling.

## Threats to validity / scope

- **Single model tier.** Verifier = agent = Haiku 4.5. A stronger verifier (Sonnet/Opus)
  that is genuinely better at financial arithmetic than the agent *could* restore the
  asymmetry and the lift. This is the highest-value follow-up (E_fin1b: Sonnet verifier
  over Haiku answers). The current result bounds the *same-tier* case.
- **Outcome-only scoring.** E_fin1 scores the final number, not the reasoning program
  (that is sibling E_fin2). A verifier with access to gold program execution is a
  different — and more coding-like — setup.
- **n=100, base rate 0.71/0.80.** High base rate compresses the achievable precision
  headroom (a confident set can be at most ~0.20 above base before running out of
  positives). The *delta* and *AUC* are the base-rate-robust readouts, and both say null.
