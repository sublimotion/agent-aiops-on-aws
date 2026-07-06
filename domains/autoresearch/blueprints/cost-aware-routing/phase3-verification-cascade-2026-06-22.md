# Phase 3 — Verification-Gated Cascade (coding/regime-B)

**Status**: PROPOSAL, 2026-06-22. Prompted by arXiv 2604.07494v1 ("Triage Framework").
Does **not** reopen the classifier-router negative result; tests a different lever in a
different regime.

## Why this exists

Phase 1/2 closed with a robust negative (`phase1-results-2026-05-28.md`): a 149M encoder
cannot predict the cheapest correct worker **from prompt text alone** on single-shot chat.
That conclusion was explicitly scoped to **regime A (single-shot chat)** and the redesign
flagged two deferred items it does not cover:

- `phase1-redesign-2026-05-28.md:140` — "Phase 2 cascade (FrugalGPT-style threshold escalation). Defer."
- `phase1-redesign-2026-05-28.md:142` — "Phase 4 SWE-bench transfer … multi-step (regime B); needs a different architecture entirely."

arXiv 2604.07494v1 sits exactly at that intersection: a **cheap-first cascade with a hard
verification gate on coding tasks**. It attacks our documented root cause — instead of
predicting tier from text, it runs the cheap tier and lets a **verifier** (test suite,
linter, type checker) catch misroutes, escalating failures to the flagship tier.

This is the one design we never tested. It is cheap to pilot, and it sidesteps the
signal-extraction limit entirely: a cascade does not need an accurate classifier, it needs
a cheap, **high-recall** verifier.

## What the paper actually is (caveats up front)

- **Proposal/protocol paper, no implementation, no results.** Authors state it "presents a
  new idea, not yet fully proven." Weigh the ideas, not the (absent) findings.
- **Core asymmetry**: clean code → cheap model; messy code → frontier reasoning. Routes a
  multi-file task by its "worst-health file touched."
- **Three stages**: (1) per-file CodeHealth sub-factors (25+) + coverage in a feature table;
  (2) single tier pick before generation {light/standard/heavy}; (3) binary verification
  gate, failures re-run on heavy and feed back to the classifier.
- **Cost model** (their Eq. 1): `r_L(c_L + f_L·c_H) + r_S(c_S + f_S·c_H) + (1−r_L−r_S)·c_H`.
  Two-tier reduces to a cost gate: cheap-tier pass rate `(1 − f_L) > c_L/c_H`.
- **Go/no-go**: 50-task pilot, both gates must pass — cost gate `(1−f_L) > c_L/c_H`
  (~20% for Haiku→Opus) AND signal gate `p̂ ≥ 0.56` (high- vs low-CodeHealth).
- **Their own cited prior work undercuts the premise**: Borg et al. 2026 found the clean/messy
  asymmetry helps *medium* LLMs but shows **"no significant difference for agentic Claude
  Code."** The asymmetry shrinks exactly at the flagship tier you'd most want to route away
  from. So **CodeHealth as the routing feature is the weakest part of the paper.**

## The split: keep the gate, drop the CodeHealth classifier

The paper bundles two independent ideas. Score them separately against our evidence:

| Idea | Verdict | Evidence |
|------|---------|----------|
| **Verification-gated cascade** (run cheap → test → escalate on fail) | **Worth piloting** | Sidesteps our signal-extraction limit. A real test suite is a far better gate than the LLM-judge from verifier-reward (v009: prec 0.92 but **recall 0.14** — low recall kills a fallback gate; ground-truth tests don't have that failure mode). |
| **CodeHealth as the routing signal** | **Skip in v1** | Borg's null on agentic Claude Code; their own H2 bar is only p̂ ≥ 0.56 (a *small* effect), unrun. This is "predict tier from features" — the same class of move that failed Phase 1/2. Don't pay for it before the gate is proven. |

The constructive version of the paper, for us, is: **a cascade does not need to predict
anything. Always start cheap, and let the verifier do the routing.** CodeHealth (or any
classifier) is only worth adding later as a way to *skip the cheap attempt* on tasks that
are obviously going to fail it — a second-order optimization on top of a working cascade.

## Pilot design (minimal, ~1 day, low cost)

**Question**: On coding tasks with a ground-truth verifier, does a cheap-first cascade beat
both always-flagship and always-cheap on **cost per *verified* success**?

**Workload**: SWE-bench Lite (300), or a 50-task pilot subset to mirror the paper's protocol
and keep cost down. We already have SWE-bench harness + gold test infrastructure from
`verification-primitives-swebench/` (175/300 = 58.3% with Claude Code) and
`verifier-reward/`.

**Tiers** (reuse the existing pool / Bedrock workers, two-tier to start):
- **Light**: a cheap coding worker (e.g. Qwen-Coder-480B / Devstral-class — already
  benchmarked in agent-harness: Devstral 24B × OpenCode hit 88% fix, 16% gold pass).
- **Heavy**: flagship (Claude Opus / Sonnet via the verifier-reward harness).

**Cascade**:
```
task → light tier generates patch → run gold/repo test suite
        ├─ tests pass → accept (paid light only)
        └─ tests fail → heavy tier regenerates → run tests → accept/reject
```

**Gate = the actual test suite**, not an LLM judge. This is the whole point: ground-truth
verification has the high recall that the judge-based gate lacked.

**Baselines**: always-light, always-heavy, random (the paper's three baselines).

**Metrics** (all already computable from the existing harnesses):
- Cost per *verified* success (primary).
- End-to-end pass rate vs always-heavy (must not drop materially).
- Escalation rate `f_L` and the cost-gate check `(1 − f_L) > c_L/c_H`.
- Over-escalation: tasks the heavy tier *also* fails (cascade paid twice for nothing).

**Go/no-go** (adapting the paper's two gates to our pool):
1. **Cost gate**: light-tier verified-pass rate `(1 − f_L) > c_L/c_H`. With Devstral-class
   ÷ Opus, `c_L/c_H` is small (well under 20%), so the bar is low — the agent-harness 16%
   gold-pass number is already in the right ballpark to clear it.
2. **No-regression gate**: cascade end-to-end pass rate ≥ always-heavy − 2pp.
3. If both pass → the cascade is the deployment. If (1) passes but (2) fails → the verifier
   is letting bad light patches through (false-accept); inspect the test suite coverage,
   don't add a classifier yet.

## What would make this fail (pre-registered)

- **Verifier false-accepts**: light patch passes the visible tests but is wrong on hidden
  behavior → cascade ships a regression always-heavy wouldn't. SWE-bench gold tests mitigate
  this (they *are* the hidden tests), but a production repo's own suite may not. This is the
  real risk and the reason the gate's quality matters more than the router's.
- **Escalation tax**: if light fails most tasks, you pay light + heavy on every escalation
  (`c_L + c_H > c_H`). The cost gate `(1−f_L) > c_L/c_H` is exactly the break-even; if light
  is weak the cascade is strictly worse than always-heavy. Measure `f_L` first on a tiny
  slice before running the full pilot.
- **The asymmetry doesn't exist at flagship** (Borg null): if the light tier just can't do
  the work, no gate helps — you escalate everything and pay the tax. This is why v1 picks a
  genuinely capable light tier (Qwen-Coder-480B/Devstral, not Haiku).

## Cost & timeline

| Item | Cost | Time |
|------|------|------|
| 50-task `f_L` probe (light tier + test suite only) | ~$5–15 | 2–3 h |
| Full 50-task pilot (light + escalation + heavy baseline) | ~$30–60 | ~1 day |
| (Optional) scale to SWE-bench Lite 300 if pilot clears both gates | ~$150–250 | ~1 day |

Reuses harnesses from `verification-primitives-swebench/` and `verifier-reward/`; no new
training, no classifier.

## Relationship to prior phases

- **Does not contradict** the Phase 1/2 negative — that was single-shot chat, prompt-text
  classification. This is coding, ground-truth verification, cascade. Orthogonal lever,
  different regime.
- **Confirms** the redesign's own regime-B note (`phase1-redesign:142`): regime B "needs a
  different architecture entirely." The cascade-with-verifier *is* that different
  architecture — and it's simpler than the RL-orchestrator alternative.
- **CodeHealth classifier** is deferred to a hypothetical Phase 3b, only if the cascade works
  and we want to skip doomed light attempts. Bring the same skepticism Phase 1/2 earned:
  predicting tier from features is the move that already failed once.
