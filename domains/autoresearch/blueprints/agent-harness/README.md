# Agent Harness Autoresearch

Autonomous optimization of a coding agent's scaffolding — system prompts, tool definitions, turn strategy, and context management — targeting higher SWE-bench pass rates without changing the model.

## What This Is

The model weights are fixed (Devstral Small 2 24B FP8). Claude Code autonomously iterates on the harness around the model: how it's prompted, what tools it has, how turns are managed, and how context is compressed. Each experiment runs 50 SWE-bench Lite issues and measures pass rate.

Inspired by the "harness problem" research: a single tool format change gave Grok a 10x improvement (6.7% → 68.3%). The harness is often a bigger lever than the model itself.

## Quick Start

```bash
# 1. Ensure vLLM is serving Devstral on port 9000
# 2. SSH to g7e instance
ssh -i ~/.ssh/g7e-bench.pem ec2-user@35.94.217.100
cd /mnt/nvme/sera-scripts

# 3. Launch Claude Code with program.md as context
```

## Baseline

From SERA Phase 1 (devstral-sera blueprint):
- **17.7% pass rate** on SWE-bench Lite (53/300 tests pass, 28 SVG accepted)
- **82% fix generation** (246/300 issues get a fix)
- **29.6 avg turns** (nearly all exhaust 30-turn budget)

## Optimization Dimensions

| Dimension | Expected Leverage | Examples |
|-----------|------------------|----------|
| System prompt | High | Step-by-step debugging, repo-specific hints |
| Tool design | High | Edit granularity, search tools, output truncation |
| Turn strategy | Medium | Backtracking, loop detection, early pivot |
| Context management | Medium | Truncation policy, file summarization |
| Sampling | Low | Temperature scheduling per turn type |
| Repo adaptation | Medium | Django vs pytest vs sympy patterns |

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Agent loop instructions |
| `lessons.md` | Operational lessons (append-only) |
| `results/experiments.jsonl` | Structured experiment log |

## References

- [The Harness Problem](http://blog.can.ac/2026/02/12/the-harness-problem/) — hashline edit tool, 10x Grok improvement
- [SERA devstral-sera lessons](../../gpu-serving/blueprints/devstral-sera/lessons.md) — Phase 1 baseline
- [ISO-Bench: Coding Agents on Real-World Inference Optimization](https://arxiv.org/abs/2502.00091) — scaffolding matters as much as model quality
