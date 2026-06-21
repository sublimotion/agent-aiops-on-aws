---
experiment: E_harness3
model: "claude-haiku-4.5 + claude-sonnet-4.6 (workers); haiku (external author + judge)"
engine: "bedrock-api"          # native `aws bedrock-runtime converse` tool-use; no SDK/pip
hardware: "n/a"                # API-only, no GPU
gpu_arch: "n/a"
deployment_date: "2026-06-21"
outcome: "success"             # ran to completion; clean falsification of the monotonic hypothesis
# Rationale: all 4 new cells (B/D/E/F) x 2 workers scored on paired eval sets, A/C
# loaded from E_harness2, both Stage-0 hard gates PASS. The headline monotonic
# prediction was cleanly REFUTED with a result that transfers across both workers
# (locus inert in all 3 regimes; the one significant gap reversed) — a valid
# applicability-bounding / law-correcting finding, not a failure.
failure_categories: []         # no infra failure; one leak-audit logic bug found+fixed pre-run
cards_used:
  mdc: []
  gpu_infra: []
card_helped: null
benchmark:
  throughput_toks_s: null
  ttft_p50_ms: null
  concurrent_users: null
  gpu_util_pct: null
ralph_iterations: null
mdc_learn_commands: []
gpu_infra_learn_commands: []
---
# Lessons — e-harness3-reward-regime-x-locus

Reward-regime × authoring-locus matrix over Bedrock Claude. Cells A/C (verifiable,
DBBench) loaded from E_harness2; B/D (DBBench reward withheld from the author) and
E/F (FinanceBench, LLM-judge consensus reward) new. Full numbers in
`results/report.md`; matrix in `results/matrix.json`.

---

### [meta/verification]: Weakening the verifiable reward does NOT revive the external author — locus is inert across verifiable, withheld, AND consensus regimes

<!-- captured: 2026-06-21 | stage: analyze -->

The headline. The monotonic prediction (external−self gap grows ≈0 → D>B → F>E
largest as reward weakens) was **REFUTED**. Locus gaps: verifiable −0.025/−0.008,
withheld **+0.000/+0.000**, consensus **−0.047 (sig)/+0.013**. No monotone trend;
the single CI-excludes-0 gap is *reversed* (external HURTS Haiku consensus).

**Why** — the hypothesis confused two asymmetries. The reward REGIME governs the
*reward channel* (how the success signal is produced: SQL exec, blinded, or
LLM-judge). The external-author lift requires an asymmetry in the *authoring
channel* — a cheaper/more-reliable oracle the author can check a candidate
against. **No reward regime hands the AUTHOR an oracle.** Withholding removes
information from self and external EQUALLY (both read the same blinded trajectory →
gap exactly 0.000). The consensus judge IS a real asymmetry (AUC 1.00) but it is
consumed producing the reward; the authoring step that follows still reflects on a
trajectory to write a rule, which is no easier external than self. So locus stays
inert in all three regimes — extending the [[e-harness2-jit-vs-offline-authoring]]
null *wider* than its own explanation predicted.

**How to apply**: when predicting external>self for any agent-construction engine,
locate the asymmetry at the **step that engine performs**, not in the environment.
A verification asymmetry in the *reward* (a discriminating judge, a test oracle)
does NOT transfer to an *authoring/critique* step downstream of it. Ask "does THIS
engine have a cheaper check than re-deriving its own output?" — reward-regime
strength is irrelevant to that question. Unifies E_harness2 + [[e-fin1-finqa-skill-verifier]]:
external only helps under an asymmetry available to *its own* act.

---

### [meta/verification]: Authoring VOLUME is as inert as authoring LOCUS without an asymmetry — 14× more notes moved accuracy by zero

<!-- captured: 2026-06-21 | stage: analyze -->

The cleanest mechanistic tell in the matrix. Under withheld reward on Sonnet, the
SELF author self-gated to **6** interventions while the EXTERNAL author wrote **87**
(14×) — and pooled Pass@1 was **identical (97 = 97)**. On Haiku consensus the
external author wrote *more* notes (33 vs 26) and scored *worse* (−0.047, sig).

**Why**: when the author has no asymmetry (same tier, same trace, no oracle), the
interventions it writes are near-random with respect to what would actually fix the
next task. Writing more of them, or having a different agent write them, changes
quantity/provenance but not the signal-to-noise ratio — and on a near-ceiling weak
worker (Haiku) the extra generic advice is net negative.

**How to apply**: do not treat "the external critic produced more/longer feedback"
as evidence it helped. Volume and locus are both confounds; only an asymmetry
predicts lift. Reproduces E_harness2's RQ3 "coherence not quantity" wrinkle as a
cross-domain regularity. Builds on [[verifier-reward]] T5 (self-critique volume
doesn't equal value).

---

### [eval/scoring]: A REFERENCE-BASED LLM-judge discriminates where E_fin1's reference-FREE one did not — gate on near-miss rejection, not stability

<!-- captured: 2026-06-21 | stage: judge gate -->

[[e-fin1-finqa-skill-verifier]] found a same-tier Claude judge on financial QA
ENGAGED but did not DISCRIMINATE (AUC 0.565, rubber-stamped 19/29 wrong). Here a
FinanceBench judge hit **AUC 1.00 and rejected 100% of on-topic ×1.4 near-miss
numeric errors**. The difference is structural, not model-choice: E_fin1's judge was
**reference-free** (had to re-derive the answer → shares the agent's failure modes);
this one **grades against the gold reference** (a genuine asymmetry: comparing two
answers is cheaper than computing one). That asymmetry is exactly why it works as a
*reward* — and exactly why it still doesn't help the downstream *author* (which has
no reference for the general rule it must write).

**How to apply**: an LLM-judge's usability hinges on whether it has a reference/oracle
to check against, not on its model tier. Reference-based grading ≫ reference-free
adversarial verification for the SAME model. And gate it correctly: temperature
stability is necessary-not-sufficient (a judge that always says "correct" is maximally
stable) — the discriminating test is **AUC on labeled pairs + an on-topic near-miss
probe** (perturb numeric golds, confirm rejection). The off-topic-wrong floor test
alone passes blind judges. Codify as a standard Stage-0 check for any LLM-judge eval.

---

### [meta/verification]: Reward-withholding ablations leak through the INVOCATION schedule, not just the prompt text — fire the author on every task

<!-- captured: 2026-06-21 | stage: leak audit -->

To "withhold pass/fail from the authoring loop", scrubbing the gold label and the
"(WRONG)" tag from the digest is necessary but NOT sufficient. E_harness2 invoked the
author **only on failures** — so *whether the author is called at all* perfectly
encodes the reward, regardless of digest content. The carryover-auditor flagged this
P0; fix: under withholding, invoke the author on **every** task on a
reward-independent schedule and have it self-gate (`{"skip":true}`). Verified by
`author_calls=120/120` on cells D. Empirical separability then confirmed the residual:
strongest single digest channel (n_errors/finish_reason) recovers reward at AUC 0.547
≪ 0.90 — informative but not a deterministic read.

**How to apply**: when ablating a signal from an agent loop, audit the CONTROL FLOW
(which calls fire, in what order, how often), not only the payloads. A conditional
that branches on the hidden variable leaks it perfectly even with a sanitized prompt.
Also: a god's-eye substring match of gold values against the digest FALSE-POSITIVES
(a correct task's committed answer legitimately equals the gold) — the right leak test
is structural (no reward FIELD) + empirical (is reward recoverable?), validated with a
positive control. Reuses the [[verification-primitives]] "never assume the eval harness
works" + [[e-fin1-finqa-skill-verifier]] "audit your label noise" discipline.

---

### [ops/cost]: Cross-domain locus matrix (4 new cells × 2 workers + 2 gates) ran under $9 on Bedrock

<!-- captured: 2026-06-21 | stage: analyze -->

B/D/E/F × {haiku,sonnet} ≈ $8.5 (DBBench multi-turn episodes + author-on-every-task
dominate; FinanceBench single-call worker+judge cheap). A/C loaded from E_harness2,
not re-run (the spec's reuse discipline — saved a full DBBench re-run and avoided
drift; regenerated eval set matched A/C task_ids 120/120 so B/D pair cleanly). Two
HARD GATES cost ~$0.08. Exponential backoff on throttling (E_fin1/E_harness2 lesson)
held across ~660 sequential episodes + ~600 authoring/judge calls with no lost work.
Cost was never the binding constraint — near-ceiling worker headroom was.
