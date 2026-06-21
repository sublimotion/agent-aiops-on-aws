# E_fin2 — Behavioral-Feature Existence Off-Coding (FinQA reasoning programs)

**STATUS: COMPLETE — PARTIAL-FAIL as predicted (publishable scoping result).**

Behavioral/process features discriminate **weakly** on FinQA's short, clean
financial derivations. The Phase-3 behavioral RandomForest — the playbook's
most distinctive "process predicts quality" claim (**AUC 0.756** in coding) —
**does not transfer**: the agent-trajectory behavioral RF lands at **AUC 0.569
(95% CI [0.430, 0.709], crosses 0.5)**, inside the spec's predicted 0.55–0.65
partial-fail band and ~0.19 below the coding baseline.

**Scoping verdict: Phase 2 (behavioral verification) is largely
coding/long-horizon-specific. Skill/outcome verification is the general
primitive — but on FinQA neither primitive works (E_fin1's skill verifier also
sat at base rate, AUC 0.57).**

---

## Setup (data-only, local — no GPU, no API generation)

- **Corpus**: the **same 100 FinQA dev examples as E_fin1** (czyssrs/FinQA,
  seed=42), re-joined by `id` to the original `dev.json` to recover
  `qa.program` / `qa.steps`. **Join gate: 100/100 examples carry a non-empty
  `program`** (Stage-0 confirmation).
- **Label**: E_fin1 exact-match (`qa.exe_ans`). Base rate **71/100 pass**.
- **RF recipe (verbatim Phase-3)**: `RandomForestClassifier(n_estimators=200,
  max_depth=7, class_weight="balanced", random_state=42)`; `nan → -999`; 5-fold
  stratified CV, pooled out-of-fold probabilities; bootstrap 95% CI (2000).
- **Env**: `python3.13` + sklearn 1.8.0 (macOS python3.14 sklearn is broken —
  carried from E_harness1).

### Two feature families (kept separate by design)

| Family | What it measures | Phase-3 analog |
|--------|------------------|----------------|
| **`beh_*`** agent-trajectory (9 feats) | the agent's own run: output/input tokens, cost, reasoning words, tokens-per-word, self-revision marker count, number-mentions, abstain flag, latency | **direct** analog of the Phase-3 four (cost, tokens_per_edit, loop_count, svg_accepted) |
| **`prog_*`** gold-program structure (8 feats) | the *task's* gold DSL derivation: op count, op diversity, multi-step chaining, const use, table-op, intermediate-value sign-flip / magnitude-blowup / max-abs | task-side **difficulty proxy**, NOT agent behavior |

The distinction matters: the Phase-3 result was about *the agent's process*.
FinQA's gold program is fixed per task, so `prog_*` cannot be a behavioral
verifier — it is reported as the difficulty axis and for the paradox check.

---

## Results

### 1. RF AUC vs the 0.756 coding baseline

| Model (fixed feature set) | AUC | 95% CI | P@R≥30% | n |
|---|---:|---|---:|---:|
| **Coding baseline** (Phase-3 `selected_4`) | **0.756** | — | 0.966 | 300 |
| **behavioral_all** (9 agent-trajectory) | **0.569** | [0.430, 0.709] | — | 100 |
| behavioral_4 (cost / tok-per-word / revision / abstain) | 0.462 | — | — | 100 |
| program_structural (8 gold-program) | 0.408 | [0.282, 0.527] | — | 100 |
| program + behavioral | 0.561 | — | — | 100 |

- Every **fixed** feature set is at or below chance after CV. The best one
  (behavioral_all, 0.569) has a CI that **straddles 0.5** — not distinguishable
  from no signal at n=100.
- The "behavioral_4" subset chosen to *mirror* the Phase-3 selected_4 shape
  (cost, tokens-per-unit-work, a loop/revision count, a binary state flag)
  scores **0.462** — the analogs that carried 0.756 in coding carry nothing here.

### 2. Univariate feature direction (why it fails)

The strongest single signals are **weak and several point the "wrong" way**
vs the coding intuition:

| Feature | pass mean | fail mean | single-feature AUC | note |
|---|---:|---:|---:|---|
| beh_latency_ms | 1921.8 | 2093.7 | 0.366 | passes are *faster*, but barely |
| beh_cost_usd | 0.0011 | 0.0012 | 0.392 | cost ≈ flat; coding's #1 feature is inert here |
| beh_tokens_per_word | 2.495 | 2.385 | 0.603 | weak best behavioral signal |
| prog_magnitude_blowup | 7244 | 353 | 0.595 | noisy difficulty signal |
| prog_has_chain | 0.493 | 0.310 | 0.591 | multi-step tasks pass *more* (see §4) |
| beh_output_tokens | 141.3 | 164.0 | 0.414 | **inverse**: wrong answers ramble more |
| beh_revision_count | 0.000 | 0.103 | 0.466 | self-revision near-absent (98/100 = 0) |
| beh_abstain | 0.000 | 0.034 | 0.483 | abstention near-absent (1/100) |

**Mechanism**: financial derivations are 1–2 ops (56% single-op). The
loop/thrash/self-revision signatures the Phase-3 RF reads are **structurally
absent** — `beh_revision_count` and `beh_abstain` are ~0 with zero RF
importance. Cost is flat because every answer is one short Haiku call. There is
no long, messy trajectory for a behavioral verifier to mine.

### 3. Forward selection (reported, but flagged as small-n overfit)

Greedy forward selection over all 17 process features picks
`[beh_output_tokens, prog_op_count, prog_op_diversity]` → **OOF AUC 0.790**.

**This is not a real 0.79.** Selecting features on the same n=100 pooled-OOF
AUC being reported is optimistically biased, and two of the three picks are
gold-program (difficulty) features, not agent behavior. The honest read is the
**fixed-set** result above: no pre-committed behavioral feature set clears 0.6.
We report the forward-selected number for parity with the Phase-3 protocol and
explicitly do **not** claim it as the behavioral-verifier AUC.

### 4. Head-to-head: behavioral vs skill-verifier (E_fin1) vs combined

| Signal | AUC |
|---|---:|
| behavioral_only (9 agent-trajectory) | 0.569 |
| program_only (gold-program) | 0.408 |
| **skill_verifier_only** (E_fin1 conf+adv RF) | **0.557** |
| behavioral + skill | 0.547 |
| all combined | 0.545 |
| raw E_fin1 adv `likely_correct` count (single signal) | 0.565 |

**The Phase-3 ordering does not hold — it collapses.** In coding, behavioral
(0.730) **beat** the LLM signals (v009 0.682, debate 0.682). On FinQA, every
signal sits at **0.55–0.57**, statistically indistinguishable from each other
*and* from base-rate guessing. Combining them does not help (0.545–0.547);
there is no complementary signal to fuse. This **reproduces E_fin1's null** from
a different angle: behavioral verification is no better than the skill verifier
off-coding because **neither has anything to read** on a one-step numeric task.

### 5. Difficulty / strategy paradox — DOCUMENTED, not re-run

Per the carryover prior (`enew_report.md`: difficulty-conditioning regressed
coding AUC 0.756→0.743), RQ3 **documents** whether the paradox appears; we do
**not** fit a difficulty-conditioned RF as a fix.

Stratifying by gold-program op-count (`≤1` vs `≥2`):

| Bucket | n | pass rate |
|---|---:|---:|
| easy (1 op) | 56 | 0.643 |
| hard (≥2 ops) | 44 | 0.795 |

- **No Simpson's paradox / sign reversal detected** on any behavioral feature
  (`paradox_detected: false`). Every pass−fail delta keeps the same sign in both
  buckets.
- Counter-intuitively, **harder (multi-step) tasks pass *more* (0.80 vs 0.64)**.
  Single-op questions are dominated by table-lookup/units ambiguity (the
  E_fin1 label-noise finding), not arithmetic difficulty — so op-count is a poor
  difficulty axis here. This is exactly why FinQA has **even lower true
  difficulty variance** than coding, making conditioning even less likely to
  help (as the spec predicted). The absence of a paradox is consistent with the
  coding prior that the cross-sectional reversal was about agent *strategy* on
  long horizons, which FinQA's one-shot answers don't have.

---

## Verdict

1. **Is "process predicts quality" universal? NO — it is
   coding/long-horizon-specific.** On FinQA's short clean derivations (a
   near-worst-case for behavioral signal, as designed) the behavioral RF
   collapses from 0.756 to **0.569 (CI crosses 0.5)**. The features the
   Phase-3 RF most relied on (cost, loop/revision counts) are **structurally
   absent** when the agent does one short call.

2. **Does the Phase-3 ordering hold? NO — it flattens.** Behavioral, skill, and
   combined all sit at 0.55–0.57; behavioral does **not** beat the LLM
   verifier, and fusion adds nothing.

3. **Difficulty paradox? Not present** (documented, not re-run). FinQA's
   difficulty variance is too low and one-shot, consistent with the coding
   prior that the reversal is a long-horizon-strategy artifact.

**Bounding statement for the playbook**: *Behavioral/process verification (the
free $0 RandomForest) is a long-horizon / multi-step phenomenon. It requires a
trajectory with measurable process pathologies (loops, cost blowups, repeated
edits) to read. On short, single-derivation tasks it has no substrate and
degrades to chance. **Skill/outcome verification (E_fin1) is the more general
primitive — but it too needs a verification asymmetry that FinQA's same-tier
numeric reasoning denies.** Off-coding, on short numeric tasks, neither the
behavioral nor the skill primitive discriminates; the honest scope of Phase 2 is
long-horizon agentic work.*

## Caveats

- **n=100**, base rate 0.71 → wide CIs; the headline behavioral AUC's CI
  includes 0.5. Treat all point estimates as imprecise; the *direction*
  (collapse to chance) is the robust finding, mirrored by E_fin1.
- Single model (Haiku), single domain (FinQA). FinQA was chosen because it is
  the only finance corpus shipping an executable process trace; the
  short-derivation structure that kills behavioral signal is intrinsic to the
  domain, not an artifact of the sample.
- Forward-selected 0.79 is in-sample-selected and difficulty-feature-driven;
  not a behavioral-verifier result.
