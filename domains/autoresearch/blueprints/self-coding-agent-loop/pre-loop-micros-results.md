# Pre-Loop Micros — Consolidated Results

**Spec**: `domains/autoresearch/specs/pre-loop-micros.md`
**Status**: COMPLETE (+ follow-up E_transfer)
**Date**: 2026-05-09
**Cost**: ~$1-2 (well under $50 budget)

**Follow-up**: After reviewing E_env's per-cell AUCs (same model / different scaffold cells had wildly different AUCs, 0.536 to 0.727), we added E_transfer — an out-of-sample test of the Phase 3 RF on OpenHands and Qwen3.5×OpenCode. E_transfer produced the V1b-relevant numbers the three original micros missed, and triggered the `self-coding-agent-loop.md` edits described at the bottom of this file.

## TL;DR

| Experiment | Decision | Action |
|---|---|---|
| E_env | `proceed_single_rf` (Δ AUC = **-0.144**) | No per-pipeline ensemble needed — per-cell RFs are *worse* than the pooled RF. |
| E_attr | Informative despite sparse simpsons cell | Bake v009_fail+rf_pass cell (n=141, gold=72.3%) into drift monitoring. |
| E_constraint_agent | `negative_result` (P@R≥0.30 = **0.286**) | v009 remains the only V1b candidate for non-Claude traces. |

Net effect on `self-coding-agent-loop.md`:

- V1 gate: **unchanged** — proceed with single-RF verifier (E_env does not recommend ensemble).
- V1b gate: **unchanged** — no viable constraint-verifier backup. If V1b fails on Qwen3.5 traces, Loop 1 recalibration is the only path.
- Loop 1 drift monitoring: **add disagreement-attribution signal** from E_attr (see below).

## E_env: Environment Variance Reduction Test

**Output**: `learned-verifier/docs/e_env_report.md`, `learned-verifier/docs/e_env_report.json`

Pooled `claude_opencode_300` (n=300) + `e6_openhands_features.csv` (n=2,098) + `e6_nebius_features.csv` (n=1,000 stratified cap) onto the 6-feature intersection (`total_cost_usd`, `tokens_per_edit`, `loop_count`, `_n_edits`, `_n_reads`, `_n_bash`).

Baseline pooled RF: **AUC = 0.784**.

| Cell | n | AUC |
|---|---|---|
| nebius::qwen3-30b | 1000 | 0.655 |
| claude_opencode::claude-sonnet | 300 | 0.727 |
| openhands::deepseek | 300 | 0.734 |
| openhands::gpt-4o | 300 | 0.425 |
| openhands::gpt-4o-mini | 300 | 0.612 |
| openhands::llama-70b | 300 | 0.783 |
| openhands::qwen-72b | 300 | 0.584 |
| openhands::claude-haiku | 299 | 0.662 |
| openhands::claude-sonnet | 299 | 0.536 |

Sample-weighted per-cell AUC: **0.639**. Pooled baseline beats per-cell ensemble by +0.144 AUC.

**Interpretation**: on the feature subset Loop 1 would actually deploy, pooling *helps* — a single RF generalizes better than per-cell RFs in this regime. The E6 result (AUC 0.801 for per-family routing) does not replicate under the common-feature-intersection constraint. This is still a useful finding: the routing gain in E6 came from model-specific features unavailable to Loop 1 at deployment time.

**Known caveat**: the 6-feature intersection drops all v009, debate, svg signals and most behavioral features. Phase 3's 4-feature `selected_4` AUC of 0.756 is not comparable. E_env tests pooled-vs-routed under the *available-features* constraint, which is the correct Loop 1 baseline.

**Spec update**: `self-coding-agent-loop.md` §V1 and §Loop 1 architecture — no change. Single-RF design is correct.

## E_attr: Verifier-Disagreement Attribution

**Output**: `learned-verifier/docs/e_attr_report.md`

2×2 (v009_verdict × rf_verdict) on Phase 3 n=300:

| | RF pass | RF fail |
|---|---|---|
| **v009 pass** | 24 (agree_pass, gold=87.5%) | **4** (simpsons, gold=75.0%) |
| **v009 fail** | **141** (adversarial, gold=72.3%) | 131 (agree_fail, gold=37.4%) |

v009 is very conservative (only 28/300 pass) so the simpsons cell is essentially empty (n=4, not useful). But the **adversarial cell is large and informative**:

- Adversarial cell (v009_fail, RF_pass, n=141) has gold pass rate **72.3%** — 14pp above the 58.3% baseline.
- Agree_fail cell (n=131) has gold rate **37.4%** — 21pp below baseline.
- Gap between adversarial and agree_fail = **34.9pp**.

In other words, the RF **corrects** v009's over-rejection in 102 of 141 cases. The existing script's decision ("inconclusive") is mechanical — it required both disagreement cells to have n≥10. Substantively the adversarial cell's behavior is informative: RF_pass adds strong signal on top of v009_fail.

**Feature profile difference** (agree_fail vs adversarial):

| Feature | agree_fail (median) | adversarial (median) |
|---|---|---|
| `beh_total_cost_usd` | 0.425 | 0.262 |
| `beh_loop_count` | 18.0 | 13.0 |
| `beh_tokens_per_edit` | 523k | 559k |

Adversarial-cell agents are *cheaper and shorter-running*. v009 rejects them stylistically; they usually pass anyway.

**Spec update**: `self-coding-agent-loop.md` §Loop 1 Phase 2 drift monitoring should add `disagreement_pattern` as a signal — specifically, `v009_fail AND rf_pass` is an "override v009" region, not a noise cell.

## E_constraint_agent: Per-Instance Constraints on Agent Patches

**Output**: `learned-verifier/docs/e_constraint_agent_report.md`, `learned-verifier/docs/e_constraint_agent_report.json`

Data substitution: Qwen3.5 × SERA patches not materialized; substituted Qwen3.5-397B-FP8 × OpenCode (n=43, diffs in `verifier-reward/results/diffs/qwen35_opencode/`). Spec intent (non-Claude agent patches as v009 alternative) preserved.

| Metric | Value |
|---|---|
| n | 43 |
| Gold pass rate | 0.093 (4/43) |
| Constraint verifier AUC | 0.558 |
| Precision @ recall ≥ 0.30 | **0.286** |
| Mean score on gold-pass (n=4) | 0.500 |
| Mean score on gold-fail (n=39) | 0.409 |

Decision rule thresholds: 0.85 → backup verifier, 0.80-0.85 → ensemble, <0.80 → negative. Outcome: **negative_result**.

**Interpretation**: constraint verifier gives 3x base-rate precision (0.286 vs 0.093) but nowhere near the 0.80 threshold. Two of four gold-pass patches (django-10924, django-11001) score 0.0 — Haiku evaluation flagged them as "style-only changes" because the Qwen3.5 diffs were terse reformattings rather than substantive fixes. The other two (seaborn-3010, pytest-11143) score 1.0. Separation mean 0.091 is below noise floor at n=43.

**Spec update**: `self-coding-agent-loop.md` V1b gate unchanged. If v009 fails V1b on Qwen3.5 traces, **Loop 1 recalibration is the only path** — no pre-built alternative.

## E_transfer: Phase 3 RF Cross-Pipeline Transfer (Follow-up)

**Output**: `learned-verifier/docs/e_transfer_report.md`, `learned-verifier/docs/e_transfer_report.json`
**Script**: `learned-verifier/experiments/e_transfer_cross_pipeline.py`

Train on `claude_opencode_300` (Phase 3, in-sample 5-fold AUC=0.727 on 6-feature intersection), evaluate out-of-sample on 7 OpenHands cells + Qwen3.5×OpenCode.

| Transfer cell | n | pos_rate | AUC | prob_std |
|---|---|---|---|---|
| openhands::claude-sonnet (same model, diff scaffold) | 299 | 0.435 | **0.486** | 0.018 |
| openhands::claude-haiku | 299 | 0.288 | 0.411 | 0.036 |
| openhands::gpt-4o | 300 | 0.213 | 0.554 | 0.065 |
| openhands::gpt-4o-mini | 300 | 0.077 | 0.357 | 0.069 |
| openhands::deepseek | 300 | 0.077 | 0.335 | 0.061 |
| openhands::qwen-72b | 300 | 0.077 | 0.444 | 0.074 |
| openhands::llama-70b | 300 | 0.107 | 0.728 | 0.074 |
| qwen35_opencode (diff-only proxies) | 43 | 0.093 | 0.538 | 0.011 |

**Key finding**: the RF's probability output becomes near-constant (std 0.018-0.07) on OOD traces — it falls off the training distribution into flat leaf regions. Even same-model/different-scaffold (claude-sonnet on OpenHands) degrades from 0.727 in-sample to 0.486 OOD.

**Implication for V1b**: do not attempt V1b validation on the Claude-trained RF. Run `FlywheelBootstrap` with ~200 labels from the target distribution first (new requirement: V1b').

## Updates to `self-coding-agent-loop.md` (applied 2026-05-09)

1. **Validation table (§Starting Point)**: row M marked DONE; new row M2 added for the transfer test.
2. **V1b prerequisites (§Experimental Design)**: explicitly warn that V1b will fail on the Claude-trained RF; add V1b' (mandatory FlywheelBootstrap on target distribution before V1b evaluation); note no constraint-verifier backup exists.
3. **V1b validation flow chart**: replaced "block C/D/E if v009 precision < 0.70" with "bootstrap then validate" — matches the two-step V1b'/V1b sequence.
4. **Drift monitor (§Loop 1 YAML)**: added `secondary_signal` block — `v009_fail AND rf_pass` rate baseline 0.47 with ±0.10 alarm threshold. Framed as an "override" cell, not an adversarial-patch cell (corrects the spec's original hypothesis).
5. **Claude-First Traces insight (§Loop 1)**: promoted `bootstrap_new_distribution` from "edge case" to "default path for new deployment targets", citing M2 numbers.

**Not changed**:
- §V1 (Verifier transfers to SWE-ReBench): the precision-on-100-tasks gate stands. M2 only shows the Claude-trained RF doesn't transfer zero-shot — it doesn't invalidate the V1 gate itself.
- Per-pipeline ensemble language: E_env showed the routing gain in E6 came from model-specific features, not architecture. The Loop 1 YAML already uses a single RF; no structural change needed.

## Cost Summary

| Line item | Estimated | Actual |
|---|---|---|
| E_env | $0 | $0 |
| E_attr | $0 | $0 |
| E_constraint_agent extraction (43 × 1 Haiku call) | $2.50 | ~$1 |
| E_constraint_agent evaluation (~215 Haiku calls) | $7.50 | ~$1 |
| Buffer | $15 | $0 |
| **Total** | **~$25** | **~$2** |

Haiku 4.5 pricing is much lower than the spec's $0.05/extraction estimate.

## Artifacts

- `learned-verifier/experiments/e_env_environment_variance.py`
- `learned-verifier/experiments/e_attr_verifier_disagreement.py`
- `learned-verifier/experiments/e_constraint_agent_pilot.py`
- `learned-verifier/docs/e_env_report.md` + `.json`
- `learned-verifier/docs/e_attr_report.md`
- `learned-verifier/docs/e_constraint_agent_report.md` + `.json`
- `learned-verifier/data/features/e_constraint_agent_constraints.jsonl`
- `learned-verifier/data/features/e_constraint_agent_evaluations.jsonl`
- `learned-verifier/data/features/e_constraint_agent_issues.json`
- `learned-verifier/data/features/combined_with_rf.csv` (Phase 3 + rf_prob column for E_attr)
