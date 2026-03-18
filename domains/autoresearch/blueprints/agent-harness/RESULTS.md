# Agent Harness Experiment Results

**Date**: 2026-03-18
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
| Claude Code | CLI agent (Anthropic SDK), auto-compaction, ~22K system prompt | Tested (via patched vLLM Anthropic API) |
| OpenCode | Vercel AI SDK (@ai-sdk/openai-compatible), built-in tools | Tested |
| Codex CLI | OpenAI Responses API, sandboxed shell + file tools | Tested (via streaming proxy + 4 vLLM patches) |

### Phase 2a Results (Verified Pass Rate)

Pass rate verified by applying agent patches + gold `test_patch` from SWE-bench, then running FAIL_TO_PASS tests. Evaluated repos: Django (all), pytest (partial), sympy (partial). Other repos could not be tested without Docker.

| Rank | Harness | Pass Rate | Fix Rate | Edit Rate | Avg 1st Edit | Avg Turns | Avg Latency |
|------|---------|-----------|----------|-----------|-------------|-----------|-------------|
| 1 | **SERA** | **8/50 (16%)** | 23/50 (46%) | 24/50 (48%) | 13.5 | 29.1 | 87.6s |
| 2 | **LangGraph** | **7/50 (14%)** | 31/50 (62%) | 32/50 (64%) | 19.7 | 28.4 | 50.2s |
| 3 | Aider | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | n/a | n/a | 5.9s |

### Phase 2b Results: Hashline vs str_replace + OpenCode

**Goal**: Test the "harness problem" hypothesis — does hash-anchored line editing (LINE:HASH format from oh-my-pi) outperform str_replace for Devstral Small 2 24B? Also test OpenCode (Vercel AI SDK) as a full-featured CLI agent harness.

| Rank | Harness | Pass Rate | Fix Rate | Patches | Precision | Edit Format |
|------|---------|-----------|----------|---------|-----------|-------------|
| 1 | **OpenCode** | **11/50 (22%)** | 44/50 (88%) | 44 | 25.0% | str_replace (built-in) |
| 2 | **Claude Code** | **10/50 (20%)** | 19/50 (38%) | 26 | **52.6%** | str_replace (built-in) |
| 3 | **PiAgent** (str_replace) | **8/50 (16%)** | 39/50 (78%) | 39 | 20.5% | str_replace |
| 4 | **Hashline** (ohmypi) | **7/50 (14%)** | 38/50 (76%) | 38 | 18.4% | LINE:HASH |
| 5 | **Codex CLI** | **9/50 (18%)** | 48/50 (96%) | 48 | 18.8% | shell commands |
| - | DeepAgents | DNF | - | - | - | (recursion limit) |

**OpenCode**: CLI agent using Vercel AI SDK with `@ai-sdk/openai-compatible` custom provider pointing at local vLLM. Uses its own built-in tool set (glob, read, edit, bash, grep, write) rather than custom tools. Required 64K context (vs 32K for other harnesses) due to ~11K system prompt + ~20 built-in tool definitions. Connected via `opencode.json` config: `{"provider":{"vllm":{"npm":"@ai-sdk/openai-compatible","options":{"baseURL":"http://localhost:8080/v1"}}}}`.

**Hashline format**: `read_file` returns lines as `42:a3|def method(self):` — 2-char content hash per line. `edit_file` takes `start_hash="42:a3"` and `end_hash="45:f1"` instead of exact text match. Eliminates str_replace failures from whitespace/quoting mismatches.

**PiAgent**: Control for hashline — identical LangGraph ReAct agent with standard str_replace editing. Replicates Phase 2a LangGraph result as expected.

**Claude Code**: CLI agent using Anthropic SDK, pointed at vLLM via patched Anthropic Messages API (`/v1/messages`). Three bugs in vLLM v0.16.0 had to be fixed: (1) `tool_use_id` field not parsed from `tool_result` blocks; (2) streaming first chunk drops tool arguments (opening `{"`); (3) Mistral requires 9-char alnum tool_call_ids vs Anthropic `toolu_*` format. Required 131K context (TP2) — Claude Code requests 32K `max_tokens` + ~22K system prompt + conversation. Best precision at 52.6% (10/19 fixes pass) but lowest fix rate (38%) — Claude Code's system prompt makes Devstral conservative, generating fewer but higher-quality patches.

**Codex CLI**: OpenAI's coding agent using the Responses API (`/v1/responses`). Connected to vLLM via a streaming-to-non-streaming proxy (vLLM v0.16.0's Responses API streaming path doesn't parse Mistral tool calls). Required 4 vLLM patches: (1) `developer` role → `system` in mistral_common; (2) `input_text` content arrays → plain strings; (3) strip extra `type: "message"` fields; (4) filter non-function tools (Codex sends `web_search` type). The proxy also normalizes multi-turn input items for vLLM compatibility. Codex uses `exec_command` (shell), `write_stdin`, `update_plan`, `spawn_agent` tools — edits are done via shell commands (`cat >`, `sed`, heredocs) rather than structured edit tools. Highest fix rate (96%) but lowest precision (19%) and highest token consumption (1.4M avg/issue, ~140x more than OpenCode).

**DeepAgents**: langchain-ai/deepagents with SummarizationMiddleware + sub-agents. Failed — `create_deep_agent` hits LangGraph recursion limit (60) before completing even one issue. Sub-agent spawning consumes recursion budget too quickly.

### Ensemble (All 7 Harnesses)

| Metric | Value |
|--------|-------|
| **Union pass rate** | **16/50 (32%)** |
| **OpenCode** | **11/50 (22%)** |
| **Claude Code** | **10/50 (20%)** |
| **Codex CLI** | **9/50 (18%)** |
| SERA | 8/50 (16%) |
| PiAgent (str_replace) | 8/50 (16%) |
| LangGraph | 7/50 (14%) |
| Hashline (ohmypi) | 7/50 (14%) |

**Per-issue breakdown** (issues where at least one harness passes):

| Instance | SERA | LangGraph | Hashline | PiAgent | OpenCode | Claude Code | Codex |
|----------|------|-----------|----------|---------|----------|-------------|-------|
| django-11039 | PASS | - | fail | fail | PASS | PASS | fail |
| django-11620 | PASS | - | fail | - | PASS | PASS | PASS |
| django-11630 | - | - | - | - | - | - | **PASS** |
| django-11815 | PASS | - | - | - | PASS | PASS | fail |
| django-12286 | - | PASS | fail | PASS | PASS | PASS | PASS |
| django-12453 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| django-12497 | - | PASS | fail | PASS | - | - | PASS |
| django-12747 | - | - | - | - | - | - | **PASS** |
| django-14238 | - | - | PASS | PASS | PASS | PASS | fail |
| django-14608 | - | PASS | PASS | PASS | PASS | - | PASS |
| django-14672 | PASS | - | fail | PASS | PASS | PASS | PASS |
| django-14855 | PASS | PASS | PASS | PASS | - | - | PASS |
| django-15400 | - | - | - | - | - | PASS | fail |
| pytest-11143 | PASS | PASS | - | - | PASS | - | fail |
| sympy-17022 | - | - | PASS | fail | PASS | PASS | fail |
| sympy-18835 | - | fail | PASS | fail | - | - | fail |
| sympy-24152 | PASS | PASS | PASS | PASS | PASS | PASS | fail |

**Unique passes** (solved by only one harness):
- Hashline: sympy-18835 (1)
- Claude Code: django-15400 (1)
- Codex CLI: django-11630, django-12747 (2)

Codex adds 2 new passes to bring the 7-harness ensemble from 14/50 to 16/50 (32%).

### Key Findings

**OpenCode is the best single harness by pass rate** at 22% with 88% fix rate. Its built-in tool set (glob, read, edit, bash, grep, write) + structured system prompt produces the most fixes. The 88% fix rate is 42% higher than LangGraph (62%) and nearly double SERA (46%).

**Codex CLI has the highest fix rate** at 96% (48/50) but only 18% pass rate (9/50). Codex uses shell commands (`cat >`, `sed`, heredocs) for edits rather than structured edit tools. Its Responses API format is stateless — the full conversation resends each turn, consuming ~1.4M tokens/issue (140x more than OpenCode). Despite the low precision (19%), Codex contributes 2 unique passes (django-11630, django-12747) not solved by any other harness.

**Claude Code has the best precision** at 52.6% (10/19 fixes pass) — double OpenCode's 25% and far ahead of SERA's 35%. Claude Code's ~22K system prompt makes Devstral conservative: it generates fewer patches (38% fix rate, lowest of all working harnesses) but those patches are much more likely to be correct.

**Claude Code requires 131K context (TP2)**. Claude Code requests 32K `max_tokens` by default (not configurable). Combined with its ~22K system prompt + growing conversation, 65K context overflows after ~5 turns. Running with TP2 and 131K `max_model_len` resolves this but halves throughput.

**Claude Code needs 3 vLLM patches**. vLLM v0.16.0's Anthropic Messages API (`/v1/messages`) has bugs: (1) `tool_use_id` not parsed; (2) streaming drops first tool arguments chunk; (3) Mistral 9-char tool_call_id mismatch. All fixable with ~30 lines of patches.

**OpenCode needs 64K context**. Its ~11K system prompt + ~20 tool definitions consume a third of a 32K context window, causing token-limit errors. Running with 64K `max_model_len` (single GPU) resolves this but limits throughput to sequential evaluation.

**OpenCode reformats aggressively**. Patches include quote-style changes (`'` → `"`) and line-wrapping reformats alongside the functional fix. This noise doesn't break correctness but inflates patch size.

**Hashline does NOT beat str_replace for weak models**. 14% vs 14-16% — statistically identical. The hashline format that produced 10x gains for Grok (6.7% → 68.3%) provides no advantage for Devstral Small 2 24B. The bottleneck is fix correctness, not edit addressing.

**Hashline solves DIFFERENT issues**. sympy-18835 is the only issue uniquely solved by hashline (the sole unique pass across all 5 harnesses).

**Hashline is token-expensive**. Average 191K tokens/issue vs ~10K for LangGraph str_replace (~19x overhead). The `LINE:HASH|` prefix on every line inflates context significantly.

**Ensemble ceiling is 32%**. 7-harness union = 16/50. Codex adds 2 unique passes (django-11630, django-12747) over the 6-harness ensemble (14/50). Diminishing returns persist but Codex's unique passes show shell-based editing can solve issues that structured edit tools miss.

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
- **2 of 7 Phase 2a harnesses blocked**: SWE-agent and OpenHands could not be tested due to dependency/platform issues. OpenCode was unblocked via custom `@ai-sdk/openai-compatible` provider. Claude Code was unblocked by patching vLLM's Anthropic Messages API.
- **DeepAgents incompatible**: Recursion limit issue prevents evaluation.
- **Letta Code not tested**: Requires Docker Letta server setup, deferred.
- **Django dominance**: Most verifiable passes are Django issues due to evaluation constraints.
- **Hashline token overhead**: 19x higher token usage may disproportionately affect weak models vs strong models where hashline has proven benefits.

---

## Conclusions

1. **OpenCode is the best single harness by pass rate** at 22% — Claude Code is 2nd at 20%, Codex CLI 3rd at 18%. All three use built-in tool sets rather than custom adapters.
2. **Claude Code has the best precision** at 52.6% — its conservative approach (38% fix rate) produces the highest-quality patches. Codex has the highest fix rate (96%) but lowest precision (19%).
3. **30 turns baseline is optimal** — compaction and restart strategies hurt or are neutral.
4. **Ensemble ceiling is 32%** with 7 harnesses (16/50). Each additional harness adds 1-2 new passes. Diminishing returns are clear.
5. **Fix rate is not pass rate** — conversion ranges from 19% (Codex) to 53% (Claude Code). Harness design affects precision as much as recall.
6. **Shell-based editing (Codex) solves different issues** — Codex's 2 unique passes (django-11630, django-12747) are not covered by any structured edit tool harness.
7. **Hashline helps strong models, not weak ones** — the edit format that produced 10x gains for Grok provides no advantage for Devstral Small 2.
8. **Phase 3 (finetuning) is now the right next step** — 42% of failures are "wrong fix" (trainable errors).
9. **Full SWE-bench evaluation requires Docker** — bare-metal eval is a lower bound covering ~50% of patches.

---

## Raw Data

- `results/phase1_{A-F}.jsonl` — 50 lines each, per-issue metrics with turn-level breakdown
- `results/phase2_{langgraph,sera,aider}.jsonl` — 50 lines each, per-issue harness metrics
- `results/phase2b_{ohmypi,piagent,deepagents,opencode,claude_code,codex}.jsonl` — Phase 2b per-issue metrics
- `results/eval_{ohmypi,piagent,opencode,claude_code,codex}.jsonl` — Gold test evaluation results for Phase 2b
- `results-report.html` — Interactive Chart.js visualization
- `lessons.md` — Operational lessons and debugging notes
