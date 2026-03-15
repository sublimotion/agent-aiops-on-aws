# Agent Harness Experiment Results

**Date**: 2026-03-14
**Model**: Devstral Small 2 24B FP8
**Infrastructure**: 4x RTX PRO 6000 Blackwell (g7e.24xlarge), vLLM v0.16.0
**Benchmark**: SWE-bench Lite, 50-issue subset (seed 42, stratified by 11 repos)

---

## Phase 1: Turn Degradation Analysis

**Goal**: Determine how turn count affects fix quality and whether restart/compaction strategies help.

### Results

| Config | Turn Budget | Strategy | Fix Rate | Edit Rate | Avg 1st Edit Turn | Repeat Rate |
|--------|-------------|----------|----------|-----------|-------------------|-------------|
| A | 10 | baseline | 36% (18/50) | 42% (21/50) | 6.5 | 15.7% |
| B | 15 | baseline | 48% (24/50) | 56% (28/50) | 9.4 | 23.6% |
| C | 20 | baseline | 52% (26/50) | 64% (32/50) | 11.9 | 29.3% |
| **D** | **30** | **baseline** | **60% (30/50)** | **78% (39/50)** | **17.5** | **37.3%** |
| E | 30 | restart at turn 15 | 56% (28/50) | 76% (38/50) | 18.2 | 37.9% |
| F | 30 | compact at turn 15 | 40% (20/50) | 64% (32/50) | 16.5 | 41.3% |

Fix rate = generated a diff. Edit rate = called `edit_file` at least once.

### Key Findings

**Parkinson's Law for Agents**: The model consistently delays its first edit to the final third of the turn budget, regardless of budget size. With 10 turns, first edit at 6.5 (65%). With 30 turns, first edit at 17.5 (58%). The model fills available time with reading and searching.

**Diminishing returns**: Fix rate improves from 36% (10 turns) to 60% (30 turns), but marginal gains decrease: +12pp for 10->15, +4pp for 15->20, +8pp for 20->30.

**Compaction actively hurts**: Config F (compact at turn 15) produces the worst fix rate at 40% — worse even than Config A with only 10 turns. Compaction strips exploration context, causing the model to re-read files it already examined. The 41.3% repeat rate (highest of all configs) confirms the model loses orientation after compaction.

**Restart is neutral**: Config E (restart at turn 15) achieves 56% fix rate vs Config D's 60%. The fresh context doesn't reduce looping — repeat rate is nearly identical (37.9% vs 37.3%).

**Best config**: D (30 turns, no intervention) — highest fix rate (60%) and edit rate (78%).

---

## Phase 2: Multi-Harness Comparison

**Goal**: Determine which scaffolding architecture extracts the most capability from a fixed model.

### Harnesses Tested

| Harness | Architecture | Status |
|---------|-------------|--------|
| SERA | Custom Python agent loop, turn pressure, auto-compaction | Tested |
| LangGraph | LangChain ChatOpenAI + structured tool calling, ReAct loop | Tested |
| Aider | Edit-focused, `--edit-format diff`, single-turn | Tested |
| SWE-agent | ACI (Agent-Computer Interface) | Blocked (`togetherunidiff` dep broken) |
| OpenHands | CodeAct, Jupyter + bash sandbox | Blocked (`e2b` dep conflict) |
| Claude Code | CLI agent, auto-compaction | Blocked (needs Anthropic API key) |
| OpenCode | Minimal CLI agent | Blocked (Go binary N/A) |

### Phase 2a Results (Verified Pass Rate)

Pass rate verified by applying agent patches + gold `test_patch` from SWE-bench, then running FAIL_TO_PASS tests. Evaluated repos: Django (all), pytest (partial), sympy (partial). Other repos could not be tested without Docker.

| Rank | Harness | Pass Rate | Fix Rate | Edit Rate | Avg 1st Edit | Avg Turns | Avg Latency |
|------|---------|-----------|----------|-----------|-------------|-----------|-------------|
| 1 | **SERA** | **8/50 (16%)** | 23/50 (46%) | 24/50 (48%) | 13.5 | 29.1 | 87.6s |
| 2 | **LangGraph** | **7/50 (14%)** | 31/50 (62%) | 32/50 (64%) | 19.7 | 28.4 | 50.2s |
| 3 | Aider | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | n/a | n/a | 5.9s |

### Phase 2b Results: Hashline vs str_replace

**Goal**: Test the "harness problem" hypothesis — does hash-anchored line editing (LINE:HASH format from oh-my-pi) outperform str_replace for Devstral Small 2 24B?

| Rank | Harness | Pass Rate | Patches | Precision | Edit Format |
|------|---------|-----------|---------|-----------|-------------|
| 1 | **PiAgent** (str_replace) | **8/50 (16%)** | 39 | 20.5% | str_replace |
| 2 | **Hashline** (ohmypi) | **7/50 (14%)** | 38 | 18.4% | LINE:HASH |
| - | DeepAgents | DNF | - | - | (recursion limit) |

**Hashline format**: `read_file` returns lines as `42:a3|def method(self):` — 2-char content hash per line. `edit_file` takes `start_hash="42:a3"` and `end_hash="45:f1"` instead of exact text match. Eliminates str_replace failures from whitespace/quoting mismatches.

**PiAgent**: Control for hashline — identical LangGraph ReAct agent with standard str_replace editing. Replicates Phase 2a LangGraph result as expected.

**DeepAgents**: langchain-ai/deepagents with SummarizationMiddleware + sub-agents. Failed — `create_deep_agent` hits LangGraph recursion limit (60) before completing even one issue. Sub-agent spawning consumes recursion budget too quickly.

### Ensemble (All 4 Harnesses)

| Metric | Value |
|--------|-------|
| **Union pass rate** | **14/50 (28%)** |
| SERA | 8/50 (16%) |
| PiAgent (str_replace) | 8/50 (16%) |
| LangGraph | 7/50 (14%) |
| Hashline (ohmypi) | 7/50 (14%) |

**Per-issue breakdown** (issues where at least one harness passes):

| Instance | SERA | LangGraph | Hashline | PiAgent |
|----------|------|-----------|----------|---------|
| django-11039 | PASS | - | fail | fail |
| django-11620 | PASS | - | fail | - |
| django-11815 | PASS | - | - | - |
| django-12286 | - | PASS | fail | PASS |
| django-12453 | PASS | PASS | PASS | PASS |
| django-12497 | - | PASS | fail | PASS |
| django-14238 | - | - | PASS | PASS |
| django-14608 | - | PASS | PASS | PASS |
| django-14672 | PASS | - | fail | PASS |
| django-14855 | PASS | PASS | PASS | PASS |
| pytest-11143 | PASS | PASS | - | - |
| sympy-17022 | - | - | PASS | fail |
| sympy-18835 | - | fail | PASS | fail |
| sympy-24152 | PASS | PASS | PASS | PASS |

**Unique passes** (solved by only one harness):
- SERA: django-11039, django-11620, django-11815 (3)
- Hashline: sympy-17022, sympy-18835 (2)
- LangGraph: none unique
- PiAgent: none unique

Running all 4 harnesses lifts pass rate from 14-16% to **28%** (75% improvement over best individual).

### Key Findings

**SERA has best precision, LangGraph has better recall**. SERA converts 35% of its fixes to passes (8/23) vs LangGraph's 23% (7/31). SERA generates fewer but higher-quality fixes.

**Hashline does NOT beat str_replace for weak models**. 14% vs 14-16% — statistically identical. The hashline format that produced 10x gains for Grok (6.7% → 68.3%) provides no advantage for Devstral Small 2 24B. The bottleneck is fix correctness, not edit addressing.

**Hashline solves DIFFERENT issues**. Despite identical pass rates, hashline uniquely solves sympy-17022 and sympy-18835 — issues where precise line identification helps with multi-site edits. str_replace uniquely solves other issues where exact text matching works better.

**Hashline is token-expensive**. Average 191K tokens/issue vs ~10K for LangGraph str_replace (~19x overhead). The `LINE:HASH|` prefix on every line inflates context significantly, causing more 502 errors from vLLM context overflow.

**Hashline has lowest precision**. 18.4% of patches pass vs SERA's 34.8%. Generates lots of patches (38/50) but fewer are correct — the format makes editing easier but doesn't improve fix quality.

**Ensemble is the largest optimization**. 28% union vs 14-16% individual (+75%). Each harness contributes unique passes. The 4-harness ensemble is 2x better than the best individual.

**LangGraph generates more fixes, faster**. 62% fix rate vs 46%, at 42% lower latency (50.2s vs 87.6s). LangChain's structured tool calling produces fewer malformed tool calls.

**Aider cannot drive Devstral Small 2**. The `--edit-format diff` mode produces zero fixes — the model cannot produce valid unified diffs in Aider's expected format.

**DeepAgents incompatible with vLLM**. The sub-agent spawning in `create_deep_agent` consumes LangGraph recursion budget too quickly, hitting the limit before completing any issue.

---

## Failure Analysis

Of 24 LangGraph-generated patches that failed verification:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| dep_missing | 8 | 33% | Cannot evaluate — repo needs Docker (matplotlib, astropy, etc.) |
| wrong_fix | ~10 | 42% | Patch applies, tests run, assertions fail — model edits wrong code |
| broken_fix | 3 | 12% | Patch introduces TypeError/AttributeError — syntax-level mistake |
| other | 3 | 12% | Test output truncation, ambiguous |

The dominant failure mode is **"wrong fix"** (42%): the model finds the right file and area but applies incorrect logic. Examples:
- django-12308: Returns `'a'` instead of `'"a"'` (string quoting)
- django-13660: Returns `'False'` instead of `'True'` (logic inversion)
- django-14997: Doesn't raise `IntegrityError` (missing constraint)
- django-12747: Includes extra dict entries (deletion cascade logic)

This is a trainable error class — a LoRA finetuned on successful fix trajectories could target these patterns.

---

## Limitations

- **Pass rate is a lower bound**: Only Django/pytest/sympy could be evaluated without Docker. Other repos (sphinx, scikit-learn, matplotlib, astropy, seaborn, requests, pylint) have version-specific dependencies that conflict with Python 3.11.
- **50-issue subset**: Introduces ~5% sampling variance. Results may not generalize to the full 300-issue set.
- **4 of 7 Phase 2a harnesses blocked**: SWE-agent, OpenHands, Claude Code, and OpenCode could not be tested due to dependency/platform issues.
- **DeepAgents incompatible**: Recursion limit issue prevents evaluation.
- **Letta Code not tested**: Requires Docker Letta server setup, deferred.
- **Django dominance**: Most verifiable passes are Django issues due to evaluation constraints.
- **Hashline token overhead**: 19x higher token usage may disproportionately affect weak models vs strong models where hashline has proven benefits.

---

## Conclusions

1. **30 turns baseline is optimal** — compaction and restart strategies hurt or are neutral.
2. **Harness ensemble is the largest single optimization**: 28% union vs 14-16% individual (+75% with 4 harnesses).
3. **Fix rate is not pass rate** — the correlation is weak (~18-35% conversion).
4. **Individual harness scaffolding has converged at 14-16%** for this model size.
5. **Hashline helps strong models, not weak ones** — the edit format that produced 10x gains for Grok provides no advantage for Devstral Small 2, but solves different issues (ensemble value).
6. **Phase 3 (finetuning) is now the right next step** — 42% of failures are "wrong fix" (trainable errors).
7. **Full SWE-bench evaluation requires Docker** — bare-metal eval is a lower bound covering ~50% of patches.

---

## Raw Data

- `results/phase1_{A-F}.jsonl` — 50 lines each, per-issue metrics with turn-level breakdown
- `results/phase2_{langgraph,sera,aider}.jsonl` — 50 lines each, per-issue harness metrics
- `results/phase2b_{ohmypi,piagent,deepagents}.jsonl` — Phase 2b per-issue metrics
- `results/eval_{ohmypi,piagent}.jsonl` — Gold test evaluation results for Phase 2b
- `results-report.html` — Interactive Chart.js visualization
- `lessons.md` — Operational lessons and debugging notes
