# Agent Harness Autoresearch

Systematic investigation of how agent scaffolding affects coding task performance. Three phases: turn degradation analysis, multi-harness comparison, and model finetuning (future).

## What This Is

The model weights are fixed (Devstral Small 2 24B FP8). We vary the scaffolding around the model to understand what actually drives SWE-bench performance.

Inspired by the "harness problem" research: a single tool format change gave Grok a 10x improvement (6.7% → 68.3%). The harness is often a bigger lever than the model itself.

## Phases

### Phase 1: Turn Degradation Analysis
**Question**: Are late turns helping or hurting? Where does quality degrade?

Run 6 turn-budget configurations (10/15/20/30 turns + restart + compaction) on the same 50-issue subset. Track per-turn fix quality, repetition rate, and context pollution.

### Phase 2: Multi-Harness Comparison
**Question**: Which scaffolding architecture extracts the most capability?

Run 7 harnesses (SERA, Claude Code, OpenHands, SWE-agent, Aider, OpenCode, LangGraph) against the same 50-issue subset via Bedrock endpoint. Produces a harness leaderboard.

### Phase 3: Model Finetuning (future, not executing)
**Question**: After harness optimization plateaus, can targeted LoRA finetuning compound further?

Deferred until Phases 1-2 converge. Documented in spec for future reference.

## Baseline

From SERA Phase 1 (devstral-sera blueprint):
- **17.7% pass rate** on SWE-bench Lite (53/300 tests pass, 28 SVG accepted)
- **82% fix generation** (246/300 issues get a fix)
- **29.6 avg turns** (nearly all exhaust 30-turn budget)

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Agent loop instructions (Phase 1 + Phase 2) |
| `lessons.md` | Operational lessons (append-only) |
| `results/` | Experiment logs (per-phase JSONL files) |

## References

- [The Harness Problem](http://blog.can.ac/2026/02/12/the-harness-problem/) — hashline edit tool, 10x Grok improvement
- [SERA devstral-sera lessons](../../gpu-serving/blueprints/devstral-sera/lessons.md) — Phase 1 baseline
- [ISO-Bench: Coding Agents on Real-World Inference Optimization](https://arxiv.org/abs/2502.00091) — scaffolding matters as much as model quality
