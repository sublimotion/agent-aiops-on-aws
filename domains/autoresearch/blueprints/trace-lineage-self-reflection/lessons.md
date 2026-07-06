# trace-lineage-self-reflection — pilot lessons

Local pilot (2026-07-06, before any cluster fan-out). Model: `us.anthropic.claude-sonnet-4-6` via Bedrock. Goal of the pilot: confirm the harness works end-to-end and the effect is *measurable* before scaling.

## What works (validated end-to-end)

1. **Full pipeline runs**: `gen_task.py` (seeded coupled-site repo) → headless `claude -p --output-format stream-json` under an arm's Stop hook → `grade.py` (mechanical grep oracle) → result row. Confirmed on control + reflect arms.
2. **`decision: block` continues headless agents** — the spec's one documented unknown. Smoke test: a Stop hook returning `{"decision":"block","reason":...}` under `--max-turns 5` made the agent keep working and act on the injected reason. **Reflect arms are feasible headless.** (Settles spec falsification-adjacent risk.)
3. **stream-json adapter works** (`lineage.py --stream-json`) — parses agent-runner `run.log` shape (skips git/npm noise lines, extracts tool_use file events). Prerequisite for the cluster path.
4. **Value-drift detection works and is precise** — new `detect_value_drift` in `lineage.py`: extracts the token the agent changed (from Edit `old_string`/`new_string`, snapped to word boundaries) and flags untouched files still holding the old token. On the pilot: **0 false positives on a completed task, correct 2/2 true positives on an incomplete one.** Clears spec falsification #4 (precision ≥ 0.7) on seeded tasks.

## What we FIXED mid-pilot (bugs the pilot caught)

- **Model ID**: bare `claude-sonnet-4-6` is rejected under Bedrock; must use the inference-profile ID `us.anthropic.claude-sonnet-4-6`. (Env already routes Opus via a full ARN.)
- **stdin**: headless `claude -p` needs `</dev/null` or it waits 3s for stdin. Added to `run_cell.sh`.
- **Coupling mismatch (the important one)**: `lineage.py`'s original drift was *reference*-coupling (file B mentions file A's name). The seeded tasks couple on a shared *value* token. The detector saw no drift where the oracle correctly saw stale files. **Fixed by adding value-drift** and defaulting the experiment hook to `--no-reference-drift` (value-only), because...
- **Reference-drift false-positives on completed tasks**: "edited README at step 9, config at step 10, README mentions config" fired even though the task was 100% correct. It injected 6 times / wasted ~8 turns on a non-issue. Value-drift does not have this problem. Experiment uses value-only.

## The blocking finding (why v1 as-designed has NO headroom)

**Sonnet 4.6 does not naturally drift on these seeded value-propagation tasks.** Control-arm completion:
- K=3 short: 1.0
- K=6 short: 1.0
- K=8 **long tier** (filler + compaction pressure, 120 max-turns): **1.0, 0 stale**

The "change X everywhere" instruction + grep-ability makes propagation too easy for a frontier model. **The ceiling effect is total** — if control is already 1.0, reflect cannot show a lift (nothing to fix), and the whole reflect-vs-control comparison collapses to noise. This is exactly the trinity-coordinator saturation pattern (`project_trinity_saturation_cost_pivot`): strong model + easy task → no headroom → intervention invisible.

**Implication: the experiment cannot run as-designed and produce signal.** Before any cluster fan-out, the task design must be made hard enough that the *control* arm naturally leaves drift. Candidate levers (untested):
1. **Weaker model** (Haiku) — the spec's v2 model axis, pulled forward. Cheapest test of "is there drift to catch at all".
2. **Genuinely long trajectories** — bury the coupled sites under substantial unrelated work so compaction actually evicts the early edits (current "long tier" filler was too light to induce forgetting).
3. **Non-obvious coupling** — sites that don't all contain the literal token (e.g. a value derived/formatted differently), so grep-and-replace doesn't trivially catch them. Risk: must keep the *oracle* mechanical.
4. **No-explicit-instruction framing** — don't tell the agent "change it everywhere"; make the propagation a consequence of a task, so missing a site is natural.

Recommended next step: a **headroom-finding sweep** (Haiku × long × K=8, control only) to locate a task/model regime where control drifts 20-60%. Only if such a regime exists does the reflect experiment have anything to measure. If no regime drifts, the honest conclusion is "for value-propagation on capable models, the drift-reflection layer has no correctness headroom" — itself a valid (if deflating) finding, consistent with verifier-reward's saturation results.

### Haiku result (headroom sweep, run 2026-07-06)

Ran the recommended sweep immediately. **Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) also hits 1.0, 0/8 stale** at K=8 long tier — just slower (46 turns vs Sonnet's 26). So the ceiling is not Sonnet-specific; **even a cheap model fully propagates** this task. (First attempt returned completion 0.0 in 1 turn — that was a bad model ID: the suffix `-v1:0` is required for Haiku on this Bedrock, and the 400 error aborts to 1 turn. Not drift.)

**Decisive conclusion:** the seeded value-propagation task has **no drift headroom on either model tier**. The lever "weaker model" is exhausted. Remaining headroom levers, in order of likelihood to actually induce drift:
1. **Much longer trajectories with real forgetting** — the current "long tier" filler (read+summarize each file) is far too light to force compaction to evict the coupled sites. Need trajectories long enough that the early edit is genuinely out of context when the later site is written. This is the *point* of the experiment (drift grows with autonomy) and the pilot never reached it — the tasks completed in 26-46 turns, nowhere near compaction.
2. **Implicit coupling** — sites where the token isn't literally identical (derived/formatted), so "grep and replace all" doesn't trivially solve it. Must keep the oracle mechanical (e.g. oracle knows the derived form).
3. **Distractor load** — many uncoupled near-miss values so the agent must discriminate which to change.

**The pilot's real lesson about the experiment:** the hard part isn't building the detector or the arms (both work) — it's **inducing genuine drift**. A task a capable model solves in <50 turns will never drift. The experiment's validity depends entirely on constructing a trajectory long/hard enough that the *control* agent forgets. Until a drifting regime is demonstrated, fanning out to the cluster would burn spend measuring a null effect. This mirrors trinity-coordinator: when accuracy saturates, there is no signal on the accuracy axis.

## Drift-headroom research (2026-07-06) — the blocker is now solvable

Two research passes for "where is drift already documented, so we reuse real headroom instead of synthesizing it." Root cause of our null result, confirmed: **frontier models saturate per-step reliability (~1.0); drift only appears under a knob that COMPOUNDS across steps.** Our task had no compounding. Documented drift takes two shapes:
- **Composition drift** — per-edit correct, the SET fails. PyMigBench: per-change 89-94% but end-to-end unit-test pass 64% (GPT-4o) / 36% (Llama-3.1). MQuAKE: single-edit recall 88-96% → multi-hop propagation 7-8%.
- **Horizon drift** — success ≈ p^H. arXiv:2509.09677 (ICLR 2026): ~100% single-step accuracy still collapses <50% within ~15 dependent steps; **self-conditioning** (injecting the model's own prior errors into context) worsens it; not fixed by scale; thinking models immune. τ-bench: pass^1<50% → pass^8<25%.

### Recommended drift-induction protocol (adopt this)
**arXiv:2509.09677 self-conditioning setup** — `github.com/long-horizon-execution/measuring-execution` (public code + HF dataset). Run a **non-thinking model** over a **long dependent-step budget (100+)** with **controlled injection of its own prior errors**. Two dials (horizon length × injected-error rate) put a control agent reliably in the 30-70% band. The injected-error mechanism IS the inconsistency the detector must catch — clean stress test. Mechanically graded (key-value execution).

### Substrate menu for reuse (all mechanically graded — no LLM judge)
| Substrate | Fit | Headroom | Grading |
|-----------|-----|----------|---------|
| arXiv:2509.09677 self-conditioning | purest horizon-drift, manufactures on demand | tunable 30-70% | key-value exact |
| PyMigBench (arXiv:2504.13272) | REAL code coupled-site propagation | ~36% fail (94%→64%) | unit tests (supply harness) |
| τ-bench (arXiv:2406.12045) | built-in `pass^k` drift metric, loop maps 1:1 | pass^8<25% | DB-state (no judge) |
| Aider refactor benchmark | fast partial/lazy-edit detector, no Docker | mid-tier 45-63% | AST node-count |
| MQuAKE (arXiv:2305.14795) | knowledge analog of coupled-site | ~92% fail | exact/alias match |

Skip LongMemEval / LoCoMo — LLM-judge graded (violates the mechanical-oracle discipline; verifier-reward recall-ceiling lesson).

### Recommended two-track redesign for v2
- **Track A (prove the mechanism):** adopt 2509.09677 self-conditioning — reliably drifting control arm; detector must catch the injected errors; reflect-vs-control lift is measurable here.
- **Track B (prove it's real):** PyMigBench — ecologically-valid code coupled-site propagation; value-drift mode already fits.

Correction: spec previously cited arXiv:2602.06413 for long-horizon decay — that is theory-only with no code. Empirical paper is 2509.09677. Spec fixed.

## Status

Harness: BUILT + VALIDATED. Experiment: BLOCKED on headroom, but **path unblocked** — adopt a documented-drift substrate (2509.09677 self-conditioning as primary, PyMigBench as the real-code track) instead of the saturated synthetic task. v2 redesign captured above; not yet built. Do not fan out to cluster until the control arm demonstrably drifts on the chosen substrate.
