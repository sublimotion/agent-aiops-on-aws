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

Run multiple harnesses against the same 50-issue subset. Phase 2a tested SERA, LangGraph, Aider. Phase 2b added hashline editing (oh-my-pi style), PiAgent (str_replace control), DeepAgents, OpenCode (via custom vLLM provider), and Claude Code (via patched vLLM Anthropic Messages API).

### Phase 3: Model Finetuning (future, not executing)
**Question**: After harness optimization plateaus, can targeted LoRA finetuning compound further?

Deferred until Phases 1-2 converge. Documented in spec for future reference.

## Baseline

From SERA Phase 1 (devstral-sera blueprint):
- **17.7% pass rate** on SWE-bench Lite (53/300 tests pass, 28 SVG accepted)
- **82% fix generation** (246/300 issues get a fix)
- **29.6 avg turns** (nearly all exhaust 30-turn budget)

## Results

See **[RESULTS.md](RESULTS.md)** for the full experiment results, including:
- Phase 1 turn degradation table (6 configs)
- Phase 2a multi-harness leaderboard (verified pass rates)
- Phase 2b hashline vs str_replace comparison
- Ensemble analysis (30% union pass rate with 6 harnesses, OpenCode best pass rate at 22%, Claude Code best precision at 53%)
- Failure analysis and Phase 3 finetuning recommendations

## Key Files

| File | Purpose |
|------|---------|
| `RESULTS.md` | Experiment results and analysis |
| `program.md` | Agent loop instructions (Phase 1 + Phase 2) |
| `lessons.md` | Operational lessons and debugging notes |
| `results-report.html` | Interactive Chart.js visualization |
| `results/` | Raw JSONL experiment data |
| `scripts/adapters/langgraph_agent.py` | LangGraph ReAct agent (str_replace) |
| `scripts/adapters/hashline_agent.py` | Hashline ReAct agent (LINE:HASH editing) |
| `scripts/adapters/deepagents_agent.py` | DeepAgents adapter |
| `scripts/multi_harness_eval.py` | Phase 2 orchestrator |
| `scripts/adapters/run_*.sh` | Shell adapter wrappers |

## References

- [The Harness Problem](http://blog.can.ac/2026/02/12/the-harness-problem/) — hashline edit tool, 10x Grok improvement
- [SERA devstral-sera lessons](../../gpu-serving/blueprints/devstral-sera/lessons.md) — Phase 1 baseline
- [ISO-Bench: Coding Agents on Real-World Inference Optimization](https://arxiv.org/abs/2502.00091) — scaffolding matters as much as model quality
