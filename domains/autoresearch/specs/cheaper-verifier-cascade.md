# Autoresearch Spec: Cheaper-Verifier Cascade — make the Trinity-validated cascade Pareto-dominant

## Status: DRAFT (2026-06-29) — the pragmatic next run

## Overview

The clean n=40 cascade result (Trinity, lessons #33) established: a **verifier-gated escalation cascade** (cheap model solves → verifier judges ACCEPT/REJECT → escalate to a stronger model on REJECT) reaches **0.900 pass@1, +7.5pt over best-static** — escalation genuinely rescues failures. **But it cost \$0.0178/prob, *more* than always-Opus (\$0.0128)**, because the verifier was Opus judging every cheap solve (~58 frontier calls dominated the bill). So the cascade won on accuracy and lost on cost — **not Pareto-dominant with a frontier verifier.**

This experiment changes **exactly one variable: the verifier model.** Hypothesis: a *cheap* verifier recovers the +7.5pt accuracy lift at a fraction of the verify cost, flipping the cascade to **Pareto-dominant** (≥ best-static accuracy at < best-static cost). This is the single highest-ROI follow-up on the board and the **deployable** form — no trained router, no GPU-resident model, just API calls + an escalation rule.

### Why this and not the router

- It's what Trinity **validated and takes further** (the cascade lift is real; only the verifier cost is the problem). One variable to change.
- It's the **operational artifact** you'd actually ship — the mechanism-selection framework's case-2 answer. The CMA-ES/content-addressed router is a deferred case-3 research direction (`content-addressed-router.md`).
- It's **cheap and decisive**: reuses `clean_router_eval.py`; ~a few hundred API calls; ~\$20.

## Findings carried in

- **Cascade economics** (Trinity #33): 0.900 @ \$0.0178 vs best-static 0.825 @ \$0.00063 (qwen3-235b) vs always-Opus 0.825 @ \$0.0128. Verify cost = the frontier verifier on every solve.
- **Verifier portability by type** (`docs/verifier-router-mechanism-selection.md`, RLVR catalog): cheapest-and-still-portable options, in order — deterministic checker > **PRM (transfers across domains)** > small-LLM judge > frontier judge. PRM is the cheap-and-portable sweet spot if a usable one exists for this regime; a small-LLM judge is the simplest drop-in.
- **Self-critique is weak** (verifier-reward T5): the solver judging *its own* output HURTS — so "self-verify" is a known-negative arm, include only as a floor.
- **Model-fair grading mandatory** (Trinity #31): clean harness, fenced-block extraction, never core.py.

## Research Questions

1. **Does a cheap verifier keep the lift?** With a cheap verifier (small-LLM judge, e.g. qwen3-32b or haiku) replacing Opus, does the cascade still reach ~0.90 (within ~2pt of the frontier-verifier cascade)? A cheap verifier that's a worse judge could lose accuracy (false-accepts that should have escalated, or false-rejects that waste escalation).
2. **Is it now Pareto-dominant?** cascade-cheap-verifier vs best-static and vs always-Opus: does it land at **≥ best-static accuracy AND < best-static cost**, or at least dominate always-Opus on both axes?
3. **Where's the verifier-cost sweet spot?** Sweep the verifier across tiers (cheap-judge / mid / frontier) and plot the accuracy-vs-cost frontier — the deliverable is the curve + the recommended operating point.

## Components

### 1. Compute / cost
- No GPU training. Workers + verifier via Bedrock Converse. Reuse the on-demand box only as an execution host. Pre-register a hard cost cap (~\$50); the run is ~hundreds of calls.

### 2. Protocol (reuse `clean_router_eval.py`, minimal change)
- **Fixed pool + task:** the same differentiated pool + LiveCodeBench, **same seed-42 problem set** as the n=40 clean cascade — so the only thing that moved vs the published result is the verifier. (Apples-to-apples; do NOT change pool or task — that would confound the verifier variable.)
- **Solver ladder:** unchanged — gpt-oss-120b → deepseek-v3 → opus (cheapest-strong → strongest-open → frontier).
- **Verifier arms (the swept variable):**
  - `frontier` = Opus (the published baseline, \$0.0178/prob — re-run for exact same-split comparison).
  - `cheap-judge` = a cheap LLM ACCEPT/REJECT judge (qwen3-32b-**direct** and/or claude-haiku-4-5). **CRITICAL (Trinity #19 + #43):** if the judge is a reasoning-capable model (qwen3-32b), run it in **direct/no-reasoning mode** OR give it enough `max_tokens` that the verdict survives — the existing harness calls the verifier at `max_tokens=2048`, but a reasoning verifier burns that budget *thinking* → `stopReason=max_tokens` → empty/clipped text → the verdict parses as not-ACCEPT → **silent forced escalation on every hard problem**, inflating cost exactly where the cheap verifier was meant to save it (and corrupting the Pareto verdict). Raise the verifier `max_tokens` (≥4096) and/or force direct mode; the diagnostic (below) must report per-candidate `stopReason=max_tokens` / empty-verdict rate and fail a candidate that clips.
  - `mid-judge` = an intermediate (e.g. deepseek-v3 or nova-pro) for the sweep.
  - (optional floor) `self-verify` = solver judges itself — known-negative (verifier-reward T5), include only to confirm the floor.
  - (stretch, only if a usable one exists) `PRM` — a process/answer reward model scoring the solution; the catalog's cheap-and-portable option. Skip if none is readily available for single-solve code; don't build one (that's the learned-verifier blueprint's job). **If dropped in, a borrowed PRM MUST pass the same precision/recall diagnostic before its verdicts count** — learned verifiers don't transfer cross-model/regime (learned-verifier AUC 0.363); a PRM trained off a different regime could score near-random here. No un-validated borrowed verdicts.
- **Baselines on the same split:** best-static (qwen3-235b single-solve), always-Opus single-solve, oracle. pass@1 + measured \$/prob each (real Converse tokens × verified prices).
- **Grading:** clean harness, model-fair *solver* extraction, `using_closed_models=False` (arm the executor alarm — Trinity executor-hang lesson).
- **Verdict extraction (the NEW model-fair axis — Trinity #31 applied to the judge):** the verifier's ACCEPT/REJECT needs its own robust parse, not a naive `startswith`. Handle leading-token clip, casing, and the "ACCEPT appears inside the reason" false-positive; bucket anything unparseable as **malformed-verdict** (NOT silently → REJECT/escalate). Count malformed-verdict rate per candidate — a high rate is a fail signal, not a routing decision.
- **Verifier-quality diagnostic (cheap, run first):** before the full cascade, measure each candidate verifier's **ACCEPT/REJECT precision+recall against ground-truth pass/fail** on a labeled slice. **The slice MUST be disjoint from the n=40 seed-42 eval set** (separate seed / held-out ids) — measuring verifier quality on the same problems the cascade is then scored on is a train-on-test leak that over-states the cheap verifier and biases the operating-point pick. Also report each candidate's malformed-verdict + `max_tokens`-clip rate here. A verifier with poor precision/recall OR a high clip rate will route badly regardless of price — this \$2 check predicts whether the cheap verifier can work before the full run.

### 3. Storage
- Per-arm results JSON committed (pass@1, \$/prob, ladder usage, verifier precision/recall), raw rollouts retained. Artifact-durability before teardown.

## Success Criteria
1. **Pareto verdict (headline):** report whether any cheap-verifier cascade is Pareto-dominant — ≥ best-static accuracy at < best-static cost, OR dominates always-Opus on both axes. A clean "yes, cheap verifier flips it" or "no, verification has an irreducible cost floor here" are both publishable.
2. **Lift retention:** cheap-verifier cascade accuracy within ~2pt of the frontier-verifier cascade (0.90), or quantify the accuracy/cost trade.
3. **The frontier curve:** accuracy-vs-cost across verifier tiers + a recommended operating point — the deployable recommendation.

## Non-Requirements
- No trained router, no CMA-ES, no content-addressing (deferred — `content-addressed-router.md`).
- Not building a PRM (that's `learned-verifier`); only use one if readily available.
- Not changing the pool or task — the whole point is to isolate the verifier variable against the published n=40 cascade.

## Known Limitations
- **n=40, single task (LiveCodeBench).** Same scope as the result it extends; a cheap verifier that wins here may not generalize across domains (verifier portability caveat — PRMs transfer better than judges, per the catalog).
- **Cheap-judge quality is the risk.** If the cheap verifier can't tell good solves from bad (low precision/recall in the diagnostic), the cascade either over-escalates (cost back up) or under-escalates (accuracy down). The diagnostic predicts this for \$2 before the full spend.
- **Low-headroom caveat stands.** If best-static already ties the pool (our compressed-tier finding), even a Pareto-dominant cascade may not beat "just ship qwen3-235b" — report the cascade vs the single cheap model honestly.

## Carryover Audit (spec-design gate)
- **Trinity #31 (fair grading)** — clean harness only, never core.py; applied to BOTH solver extraction AND the new verdict-extraction axis.
- **Trinity #19 + #43 (reasoning-token clip / dual-mode)** — a reasoning verifier (qwen3-32b) at low `max_tokens` clips its ACCEPT/REJECT → silent forced escalation. Run the judge in direct mode and/or raise verifier `max_tokens` ≥4096; diagnostic reports clip rate. THE most dangerous confound here.
- **Trinity executor-hang** — `using_closed_models=False` to arm the timeout alarm.
- **Shared-split** — identical seed-42 eval set across all verifier arms + baselines; only the verifier changes (cost-aware-routing apples-to-apples). Diagnostic slice DISJOINT from the eval set (no train-on-test leak).
- **verifier-reward T5 (self-critique hurts)** — self-verify is a known-negative floor, not a candidate.
- **Verified-price snapshot** — committed price snapshot, every arm priced (cost-aware-routing verified-prices). *(Spawn-isolation cost-aggregation #20/#22 does NOT apply — `clean_router_eval.py` is single-process, inline token accounting, no mp.Pool.)*
- **learned-verifier AUC 0.363** — a borrowed PRM must pass the precision/recall diagnostic before counting; no un-validated cross-regime verdicts.
- **Mechanism-selection framework** — this is the case-2 cheap-verifier lever the framework explicitly names; report the cascade-vs-single-model ROI per its checklist.

## Relationship to other specs
- **`docs/verifier-router-mechanism-selection.md`** — names this exact lever ("a cheaper verifier is the clear next lever — same accuracy lift, far less verify cost").
- **`domains/autoresearch/blueprints/trinity-coordinator/`** — the n=40 cascade result this extends; reuses `clean_router_eval.py`, lessons #31–33.
- **`content-addressed-router.md`** — the deferred case-3 router direction; this is the pragmatic case-2 run instead.
- **`domains/autoresearch/blueprints/learned-verifier/`** — the PRM/cheap-verifier source if the stretch arm is run.
