# Pre-Loop Micros — Progress Tracker

Spec: `domains/autoresearch/specs/pre-loop-micros.md`
Started: 2026-05-09
Completed: 2026-05-09 (iteration 1)
Mode: ralph-loop, option A (all three experiments)

## Iteration Log

### Iteration 1 (2026-05-09)
- Scaffolded blueprint dir `blueprints/self-coding-agent-loop/`
- Inspected feature column intersection across 3 CSVs:
  - `combined_features.csv` — `beh_*` prefix, 85 cols
  - `e6_openhands_features.csv` — no prefix, 12 cols
  - `e6_nebius_features.csv` — no prefix, 12 cols
  - **Intersection**: `total_cost_usd`, `tokens_per_edit`, `loop_count`, `_n_edits`, `_n_reads`, `_n_bash` (6 features)
  - Phase 3's 4th feature `svg_accepted` is not in e6 datasets (noted as known limitation in spec)
- Ran all three experiments end-to-end:
  - **E_env**: pooled n=3,398 (Claude 300 + OpenHands 2,098 + Nebius cap 1,000). Pooled AUC 0.784, per-cell weighted AUC 0.639, **Δ = -0.144 → `proceed_single_rf`**.
  - **E_attr**: Phase 3 Claude×OpenCode n=300. 2×2 disagreement matrix. Adversarial cell (v009_fail∩rf_pass) n=141, gold=72.3% vs agree_fail gold=37.4% → 34.9pp gap → attribution signal is real, script's "inconclusive" label was mechanical.
  - **E_constraint_agent**: substituted Qwen3.5×OpenCode (n=43, SERA diffs not materialized). AUC=0.558, P@R≥0.30=0.286 → **`negative_result`**. Total Bedrock spend ~$2.
- Wrote consolidated report `pre-loop-micros-results.md`.

## Status

| Experiment | Status | Output |
|---|---|---|
| E_env | ✅ COMPLETE | `docs/e_env_report.md` |
| E_attr | ✅ COMPLETE | `docs/e_attr_report.md` |
| E_constraint_agent | ✅ COMPLETE | `docs/e_constraint_agent_report.md` |
| Consolidated | ✅ COMPLETE | `pre-loop-micros-results.md` |

## Decisions Made

- **Feature set for E_env**: 6-feature intersection. No `svg_accepted`, no v009/debate signals.
- **Nebius downsampling**: cap at n=1000 (stratified by gold_pass) so one cell doesn't dominate pooled RF.
- **Cell identifier**: `(source, model_family)` — source ∈ {claude_opencode, openhands, nebius}, model_family from per-row metadata.
- **E_constraint_agent data substitution**: Qwen3.5×SERA patches not materialized → use Qwen3.5×OpenCode (n=43). Documented in consolidated results.
- **Python 3.12 for sklearn**: 3.14 in repo `.venv` has a broken sklearn. Used system `/opt/homebrew/bin/python3.12` for all experiment scripts.
- **Haiku 4.5 inference profile**: must use `us.anthropic.claude-haiku-4-5-20251001-v1:0` (on-demand throughput not supported for the bare model ID).

## Completion Promise

All three experiments ran to completion, wrote JSON+Markdown reports, and a consolidated writeup. Spec-level success criteria (§277-284) are met:

- ✅ Each experiment completed and produced a decision.
- ✅ Consolidated results flow to `pre-loop-micros-results.md`.
- ✅ Total cost under $50 (actual: ~$2).
- ✅ Total duration under 2 weeks (actual: single session).

No further ralph iterations needed. Safe to mark `/ralph-loop:cancel-ralph`.
