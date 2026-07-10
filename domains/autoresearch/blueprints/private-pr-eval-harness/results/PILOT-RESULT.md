# Private-PR Eval Harness — Pilot Result (Phase 0–3, n=10)

**Question:** does a contamination-free private-PR benchmark yield a different (harder/OOD)
result than the likely-contaminated SWE-bench-Lite?

**Setup:** 10 recent pydantic PRs (2025-08→2026-07, post model-cutoff), solution stripped, git
history sealed (fix commit unreachable), held-out tests injected from the fix commit, scored by
pytest — no LLM judge. Two closed-model cells via Bedrock + Claude Code headless.

## Result

| cell | pydantic OOD | SWE-bench-Lite | gap | empty patches | pass\|edited |
|------|------|------|------|------|------|
| Opus 4.8   | 6/10 (60%) | 58.3% | +2pp | 3 | **6/7 = 86%** |
| Sonnet 4.6 | 6/10 (60%) | 58.3% | +2pp | 1 | 6/9 = 67% |

Per-tier (both cells identical): low 4/5, medium 2/5.

## Interpretation

1. **No measurable contamination gap.** Fresh sealed pydantic ≈ Lite for both models. Pre-registered
   negative-result branch: contamination is not visibly distorting Lite rankings here. Lite still usable.

2. **Empty-patch confound is the real story.** Opus's aggregate 60% is dragged down by 3 no-edit
   runs (over-exploration within 30 turns), not OOD difficulty — conditional on editing, Opus is
   86%. Cross-model pass-rate comparison on this eval is currently a *harness artifact*. Fix: inject
   edit-checkpoint pressure (verification-primitives two-stage) before comparing models.

3. **Instrument validated end-to-end.** All 3 gates pass; gold patches score green; the harness
   honestly surfaces both no-edit fails and genuine test fails. This is the deliverable.

## Caveats
- **n=10 → CI ≈ ±30pp.** +2pp is noise. Not a publishable contamination number.
- pydantic ≠ Lite repos → "OOD" conflates contamination with codebase difficulty.
- Single harness (Claude Code), 30-turn budget — edit rate not yet tuned.

## Files
- `pred-{opus,sonnet}.jsonl` — candidate patches; `scored-{opus,sonnet}.json` — verdicts.
- `tasks.jsonl` (50 clean tasks), `validate-batch.json` (9/10 gold-patch sanity), `generation-plan.json`.

## Next (larger, budgeted)
- Scale n (50+), add edit-checkpoint pressure to remove the empty-patch confound, add an open-model
  cell (GLM-5.2) once an endpoint is warm, and run the same models on a Lite subset *through this
  harness* for a true apples-to-apples contamination delta (same repos would need a second target).
