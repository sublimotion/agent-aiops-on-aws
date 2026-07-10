---
experiment: E_harness2
model: "claude-haiku-4.5 + claude-sonnet-4.6 (workers); haiku (L3 external author)"
engine: "bedrock-api"          # native `aws bedrock-runtime converse` tool-use; no SDK/pip
hardware: "n/a"                # API-only, no GPU
gpu_arch: "n/a"
deployment_date: "2026-06-21"
outcome: "success"             # ran to completion; clean result incl. a reproduced null
# Rationale: all 4 layers x 2 workers scored on the same verified eval set; the
# headline L2->L3 prediction (external>self) produced a clean NULL that transfers
# across both workers — a scientifically valid applicability-bounding finding.
failure_categories: []         # one in-loop bug found+fixed (multi-tool-turn toolResult); no infra failure
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

learn_commands: []
---
# Lessons — e-harness2-jit-vs-offline-authoring

Layered ablation on DBBench (Life-Harness SQL env) over Bedrock Claude:
L0 bare → L1 offline-frozen harness → L2 JIT-self → L3 JIT-external.
Full numbers in `results/report.md`; raw in `results/analysis.json`.

---

### [meta/verification]: External harness AUTHORING does not beat self-authoring without a verification asymmetry — the T5 law does NOT extend, and the null reproduces E_fin1

<!-- captured: 2026-06-21 | stage: analyze -->

The headline. Prediction (from verifier-reward **T5**: self-critique in generation
HURTS, 54%→30%) was **L3 > L2** — an external verifier authoring interventions
should beat the worker authoring its own. **Result: NULL, weakly reversed.**
L2→L3 = −0.025 (Haiku) / −0.008 (Sonnet), both CIs crossing 0. L1→L2 also null
(+0.008 both) — runtime JIT matches the frozen offline harness.

**Why** — same root cause as the [[e-fin1-finqa-skill-verifier]] null: the
external>self lift requires a **verification asymmetry** (checking cheaper/more
reliable than generating; coding has it via test execution). Harness *authoring*
is not checking against an oracle — it is reflecting on a failed trajectory to
write a general rule. Self and external authors are the **same model tier reading
the same trajectory**, so they produce near-identical interventions and locus is
inert. T5's asymmetry (verifier reasons about an artifact with a deterministic
external oracle) is absent in trajectory-reflection authoring, exactly as it was
absent in FinQA outcome-only verification.

**How to apply**: before predicting external > self for ANY agent-construction
engine (verifier, harness author, critic), ask "does the external party have a
cheaper/more-reliable check than re-deriving the answer?" If author and worker
are the same tier reading the same trace with no external oracle, expect locus to
be inert (≈0 delta). This extends the asymmetry boundary from *verification* to
*authoring* — both obey one law: external only helps under an asymmetry.

---

### [meta/transfer]: The authoring-locus null is STRUCTURAL — it transfers across both worker tiers with the same signature

<!-- captured: 2026-06-21 | stage: analyze -->

RQ4. L2→L3 ≈ 0 on BOTH Haiku and Sonnet (−0.025 / −0.008, same sign, same
within-noise magnitude). No worker on which external authoring wins. This is the
[[e-fin1-finqa-skill-verifier]] / [[e-fin2-finqa-behavioral-features]] transfer
law on the meta-primitive: a null that reproduces across two unrelated backbones
with the same signature is a domain/asymmetry property, not a model-choice
confound. (T4 "calibration is model-specific" predicted the magnitude might need
re-fit per model; here even that is unnecessary — the absence of an asymmetry is
model-invariant.)

---

### [eval/harness-trust]: The Life-Harness DBBench gain is REAL but lives in ONE failure class — verify per-type before trusting (or distrusting) a pooled replication gate

<!-- captured: 2026-06-21 | stage: L1 gate -->

Pooled L0→L1 was +5.0pp (Haiku) / +7.5pp (Sonnet) — BELOW the spec's +12–16pp
gate band. A naive read would STOP ("setup broken"). The **per-type breakdown
saved the gate**: the entire harness gain is in the **SELECT** class
(0.07→0.71 Haiku, 0.00→0.64 Sonnet, +64pp), near-flat elsewhere (every other type
near ceiling). A trajectory audit proved it mechanistically faithful, NOT a
scoring artifact: the bare worker writes CORRECT SQL but submits a reformatted
answer (names only, or "label: value" strings) instead of the required
all-columns tuple-repr — a pure output-contract failure that H5+H2 fix. This IS
"Adapting the Interface, Not the Model."

**Why the pooled number shrank**: our Bedrock workers floor at ~78% vs
Life-Harness's 18 backbones at ~48.4% bare DBBench — harness-addressable failures
are a thin slice on a strong backbone. **How to apply**: a replication gate on
pooled accuracy can false-negative on a stronger model than the original paper's;
always break the gate down by the failure class the intervention TARGETS, and
audit ≥5 fix-trajectories, before calling a replication failed. (Reuses the
[[e-fin1-finqa-skill-verifier]] "audit your label noise / verify-before-assert"
discipline, applied to a replication gate rather than a scorer.)

---

### [eval/harness-trust]: Exclude the unverifiable oracle, don't fake it — DBBench mutations need a MySQL hash the official code leaves unimplemented for SQLite

<!-- captured: 2026-06-21 | stage: setup -->

200/300 DBBench standard tasks are INSERT/UPDATE/DELETE, scored by MySQL
`md5(group_concat(...))` table-hashing. The official Life-Harness task code
**explicitly leaves this unimplemented for SQLite** (`task.py:607`:
"Table hash calculation for SQLite not implemented"). Hand-rolling a SQLite hash
would have been an UNVERIFIED oracle. Per verification-primitives "never assume
the eval harness works", we instead used the SELECT-family path (`label` +
vendored `DBResultProcessor`) AND kept only tasks whose GOLD sql, run in our
SQLite env, reproduces the gold label (120/120 verified in Stage 0). Also: the
shipped inline `table.rows` in `standard.jsonl` is a PREVIEW (gold SQL reproduced
the label only ~33% of the time); the FULL tables live in `db_out_new.jsonl`
(~67% — the rest are MySQL-dialect gold-SQL that won't run in SQLite). Always
verify the oracle reproduces gold on a known-good set before trusting it as a
baseline.

**How to apply**: when an env's ground-truth oracle has an unimplemented/unportable
path, scope the eval to the verifiable subset and SAY SO (it bounds the claim),
rather than faking the oracle. Lower-bounds the L1 effect (mutation-specific H2
guards untested) but keeps every reported number trustworthy.

---

### [ops/bedrock]: Native `converse` tool-use needs a toolResult for EVERY toolUse id in the assistant turn — multi-tool turns crash mid-run

<!-- captured: 2026-06-21 | stage: L2/L3 run -->

A run died with `ValidationException ... Expected toolResult blocks at
messages.N.content for the following Ids: ...` when the model emitted TWO tool_use
blocks in one turn but the loop only answered the first. Bedrock requires a
`toolResult` for each `toolUseId` before the next turn. Fix: act on the first
tool_use, append filler toolResults ("execute only ONE tool per turn") for the
rest. The L0/L1 runs predated the bug surfacing and finished clean (it crashes,
it doesn't mis-score), so they stayed valid; L2/L3 were re-run fresh after the
fix. Also fixed resume to rebuild the JIT store from prior authored interventions
so cross-task accumulation survives an interrupt (carryover: checkpoint long runs).

**How to apply**: any native-Bedrock agent loop (no SDK) MUST answer all tool_use
ids per turn. The OpenAI Agents SDK would have masked this — it's the cost of the
SDK-on-Bedrock substitution the spec flagged. Builds on the [[agent-harness]]
"SDK-on-non-native-backend has tool-call edge cases" lesson.

---

### [ops/cost]: API-only Life-Harness-style ablation is cheap — full 4-layer × 2-model run < $2

<!-- captured: 2026-06-21 | stage: analyze -->

8 runs × 120 tasks (≈3,800 multi-turn episodes + ~150 authoring calls) cost under
$2 total on Bedrock Haiku/Sonnet. JIT authoring (L2/L3) added < $0.01 per run.
DBBench is genuinely the cheapest Life-Harness env; cost was never the binding
constraint (discrimination/ceiling was). Exponential backoff on throttling
(E_fin1 lesson) held across ~3,800 sequential episodes with no lost work.
