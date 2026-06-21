# Autoresearch Spec: E_harness1 — Harness × Behavioral-Verifier Interaction

## Status: DRAFT

## Overview

Does improving the harness **destroy** the failure signatures the Phase-3 behavioral RandomForest reads — making harness-engineering and behavioral-verification **substitutes** — or do failures **relocate** to new signatures, making them **complements**? This is a critical, currently-unstated applicability caveat for the playbook: if a better harness removes looping/thrashing, the RF's discriminative signal could collapse precisely as teams adopt better harnesses.

**Why now**: the RF's top features (repo-verified, `learned-verifier/results/phase3_report.md`) are `beh_total_cost_usd` (0.420, thrashing/budget exhaustion), `beh_tokens_per_edit` (0.327), `beh_loop_count` (0.201, stuck detection) — exactly the failure modes a Trajectory-Regulation harness layer is built to suppress. We can test this **on data already in the vault**: the verification-primitives work is itself a harness manipulation (0% unguided → 83% tool adoption via the two-stage checkpoint; the 70%-checkpoint variant reached 59%), and the agent-harness eval ran Devstral-24B across multiple harnesses. So we have paired pre/post-harness-improvement trajectories **without running any new agent**.

> **Data-availability caveat (resolve in Stage 0):** the obsidian note assumed a clean "7-harness eval"; the repo's `agent-harness/lessons.md` records only **3 harnesses actually ran** (SERA, LangGraph, Aider) — 4 were blocked (SWE-agent, OpenHands, Claude Code, OpenCode). The cross-harness partition below must use the harnesses that have trajectories on disk, not an assumed 7. Confirm the actual count before designing the partition.

**Core hypotheses (mutually exclusive, both reportable)**:
- **H-substitute**: post-improvement, behavioral-feature distributions compress (less looping, lower cost variance) and RF AUC drops toward chance → substitutes; **Phase 2 weakens under good harnesses.** Playbook caveat: "behavioral verification has highest ROI on weak/unimproved harnesses."
- **H-complement**: RF AUC holds because failures relocate to new signatures (e.g. confident-wrong with clean trajectories) → complements; the verifier survives harness evolution but **the feature set must be re-selected per harness generation.**

**Depends on**: existing internal corpora (verification-primitives, agent-harness multi-harness eval, Phase-3 labeled trajectories). **No new generation.** Companion to E_fin2 — both probe behavioral-signal durability (E_harness1 across harness *quality*, E_fin2 across *domain*).

**Source**: obsidian `Learned-Verifier-Experiment/experiments/E_harness1.md` (proposed 2026-06-21). Uses Life-Harness / SIA as a *lens*, not a reproduction target.

## Research Questions

1. **Is the behavioral RF's discrimination a property of *bad harnesses* (it reads the thrashing weak harnesses produce) or of *failure itself* (failures always leave a process trace, just different ones per harness)?**
2. **Does the optimal feature set change across harnesses?** (Prior repo evidence that feature *boundaries* don't transfer across scaffolds — `learned-verifier/results/enew_report.md` cross-scaffold results — suggests yes.) How much of the RF is reading harness quality vs generation quality (cf. the pivot-analysis finding that adopting verification tooling at all was the dominant lever, risk-diff ≈ +46.3pp — itself a harness property)?

> **Note on cited experiment IDs:** the obsidian source referenced `E_new12` (cross-scaffold transfer 0.43/0.46) and `E1` (+46.3pp); **those IDs are vault-only and do not exist as experiments in the repo.** Re-anchored above to the repo's actual artifacts (`enew_report.md` cross-scaffold results; the pivot-analysis +46.3pp risk-diff). Confirm the exact repo figures in Stage 0 before quoting them in results.
3. **Failure relocation**: for outputs wrong *despite* a clean (improved-harness) trajectory, do any features still flag them, or are they invisible to behavioral signal?

## Components

### 1. Compute
- **Platform**: local scikit-learn / analysis. **No GPU, no API generation.**
- **GPUs**: none.

### 2. Data (all internal, pre-existing)
- verification-primitives corpus — `ignore`/no-verification vs `full_pipeline` (0% unguided → 83% two-stage; 70%-checkpoint variant 59%).
- agent-harness eval — Devstral-24B across the harnesses that actually ran (SERA, LangGraph, Aider; verify count in Stage 0).
- Phase-3 labeled trajectories (n=300) for RF re-fit per condition.

### 3. Experiment Protocol
- **Metric**: per-condition RF AUC (does discrimination survive harness improvement?) + behavioral-feature distribution shift (mean/variance compression).
- **Loop structure**:
  1. Partition the existing trajectory corpus into harness conditions: (a) `ignore`/no-verification vs `full_pipeline`; (b) across the harnesses with trajectories on disk (3 ran: SERA/LangGraph/Aider — confirm).
  2. Per condition, recompute the 4 behavioral feature distributions (cost, edit%, loop_count, tool-adoption) — measure compression/shift (mean, variance) post-improvement.
  3. Re-train / re-evaluate the Phase-3 RF **within each harness condition**; compare AUC.
  4. **Failure-relocation probe**: for wrong-output-despite-clean-trajectory cases, do any features still flag them (H-complement test)?
  5. Forward-select features **per harness condition** — does the optimal set change across harnesses?
- **Termination**: per-condition distributions, RF AUCs, relocation probe, and per-condition feature selection complete.
- **Logging**: per-condition feature-distribution shift table; per-condition RF AUC; relocation-probe results; per-condition forward-selected feature sets.

### 4. Networking
- **Access**: none (all local on existing data).

### 5. Storage
- **Data**: existing verification-primitives + agent-harness + Phase-3 corpora (locate in their blueprint `results/`).
- **Results**: blueprint `results/` (shift table, per-condition AUC, relocation analysis).

## Success Criteria

- [ ] Trajectory corpus partitioned into harness conditions (ignore vs full_pipeline; per-harness across those that ran).
- [ ] 4 behavioral feature distributions recomputed per condition with mean/variance shift reported.
- [ ] Phase-3 RF re-fit + **AUC compared within each harness condition**.
- [ ] Failure-relocation probe completed (wrong-despite-clean-trajectory cases).
- [ ] Per-condition forward feature selection — does the optimal set change across harnesses?
- [ ] **Verdict reported: substitutes (AUC collapses post-improvement) vs complements (AUC holds, features relocate)**, with the playbook caveat naming where on the harness-quality axis behavioral verification has ROI.
- [ ] Carryover audit complete (below).

## Non-Requirements

- **Do NOT reproduce Life-Harness** (no tau-bench/AgentBench sweep, no 18-backbone harness evolution) — it is the lens, not the target.
- **Do NOT run new agent trajectories** — use existing paired pre/post-harness data.
- No fine-tuning, no GPU, no API generation.

## Known Limitations

- Both outcomes (substitute / complement) are publishable; this is a bounding experiment, not a win/lose one.
- "Harness quality" is operationalized via the within-vault splits (checkpoint nudge, multi-harness eval) — a coarse proxy for the full harness-quality axis.
- n=300 Phase-3 set further partitioned by condition → small per-condition samples; report AUC confidence intervals and avoid over-reading thin cells.

## Carryover Audit (spec-design gate)

- [ ] Ran `carryover-auditor` — scan `verification-primitives*/lessons.md`, `learned-verifier/results/*`, and `agent-harness`/`agent-swarm` lessons (overlapping stack: behavioral RF, the multi-harness eval, checkpoint nudge, cross-scaffold transfer).
- [ ] Carry the priors explicitly (repo-verified, NOT the obsidian-only IDs): **cross-scaffold feature boundaries don't transfer** (`enew_report.md`) — this is why RQ2 expects the feature set to change per harness; **adopting verification tooling at all was the dominant lever** (pivot-analysis risk-diff ≈ +46.3pp) — a harness property, motivating "how much of the RF is harness vs generation quality." Confirm the referenced corpora (verification-primitives, agent-harness — only 3 harnesses ran, Phase-3 n=300) actually exist in their blueprint `results/` and re-quote exact figures before committing to the partition design.

---

> **Note**: Operational artifacts belong in `domains/autoresearch/blueprints/e-harness1-harness-behavioral-interaction/`, not in this spec.
