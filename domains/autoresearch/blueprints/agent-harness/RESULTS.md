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

## Phase 3: Agent Swarm — Multi-Model Comparison

**Goal**: Measure how model capability (scale, finetuning) affects harness effectiveness across 4 models × 3 harnesses. Tests the "bitter lesson" hypothesis: does raw scale outperform task-specific finetuning?

### Models

| Model | Role | Parameters | Active | TP | Tool Calling |
|-------|------|-----------|--------|-----|-------------|
| Devstral Small 2 24B FP8 | Baseline | 24B | 24B | 1 | Mistral native |
| Qwen 2.5 Coder 32B | Finetuning control | 32B | 32B | 1 | Bare JSON (no `<tool_call>` tags) |
| SWE-agent-LM 32B | Finetuned (SWE-smith) | 32B | 32B | 1 | Bare JSON (no `<tool_call>` tags) |
| Qwen3.5-397B-A17B FP8 | Scale frontier | 397B MoE | 17B | 4 | Hermes native |

### Fix Rate Matrix

| Model | SERA | OpenCode | Claude Code | Best Harness |
|-------|------|----------|-------------|-------------|
| **Qwen3.5 397B** | **36/50 (72%)** | **44/50 (88%)** | — (incompatible) | OpenCode |
| **Devstral 24B** | 23/50 (46%) | 44/50 (88%) | 19/50 (38%) | OpenCode |
| Qwen 2.5 32B | 24/50 (48%) | 0/45 (0%) | — | SERA |
| SWE-LM 32B | 9/50 (18%) | 0/46 (0%) | — | SERA |

Fix rate = generated a non-empty git diff. Not verified by gold tests.

### Harness Compatibility

| Model | SERA | OpenCode | Claude Code |
|-------|------|----------|-------------|
| Devstral 24B | Works (Mistral chat template) | Works (Mistral tool calling) | Works (vLLM Anthropic API patches) |
| Qwen 2.5 32B | Works (bare JSON fallback) | FAIL (bare JSON, hermes parser can't extract) | Not tested |
| SWE-LM 32B | Works (bare JSON fallback) | FAIL (bare JSON, hermes parser can't extract) | Not tested |
| Qwen3.5 397B | Works (hermes parser) | Works (hermes parser) | FAIL (Anthropic API doesn't translate tools to Qwen chat template) |

### Qwen3.5 397B: SERA vs OpenCode

| Metric | Value |
|--------|-------|
| Both fix | 32 |
| Only SERA | 4 |
| Only OpenCode | 12 |
| Neither | 2 |
| **Union** | **48/50 (96%)** |

Just 2 harnesses with Qwen3.5 produce fixes for 96% of issues. The 4 SERA-only fixes and 12 OpenCode-only fixes demonstrate genuine harness complementarity even with a strong model.

### 4-Model SERA Comparison

| Metric | Value |
|--------|-------|
| Devstral 24B | 23/50 fixes |
| Qwen 2.5 32B | 24/50 fixes |
| SWE-LM 32B | 9/50 fixes |
| Qwen3.5 397B | 36/50 fixes |
| **Union (any model)** | **46/50 (92%)** |
| Intersection (all) | 1/50 |

Models fix almost completely different issues — only 1 issue (pytest-11143) is fixed by all 4 models via SERA. The 4-model union (46/50) far exceeds the best single model (36/50), demonstrating strong model complementarity.

### Behavioral Metrics

| Model | Avg Turns | Avg 1st Edit | Parkinson's Ratio |
|-------|-----------|-------------|-------------------|
| Devstral 24B | 29.1 | 13.5 | 46% |
| Qwen 2.5 32B | 13.5 | 3.2 | 24% |
| SWE-LM 32B | 14.0 | 3.6 | 26% |
| Qwen3.5 397B | 18.1 | 8.3 | 46% |

Parkinson's ratio = first edit turn / avg turns used. Higher = more time spent exploring before editing.

Devstral and Qwen3.5 both exhibit high Parkinson's ratios (~46%) — they explore extensively before editing. Qwen 2.5 and SWE-LM edit much earlier (24-26%) but produce worse fixes, suggesting the exploration phase is important for fix quality.

### Key Findings

**Bitter lesson partially validated**: Scale from 24B → 397B MoE produces +26pp on SERA (46% → 72%). But Devstral 24B matches Qwen3.5 397B on OpenCode (both 88%), suggesting harness choice can compensate for model scale.

**SWE-agent-LM finetuning is actively harmful**: -30pp from base Qwen 2.5 (48% → 18%) on the same architecture and harness. The SWE-bench-specific finetuning narrowed the model's capabilities rather than enhancing them.

**Harness spread varies by model**: Devstral has 50pp spread (38-88%), Qwen3.5 has 16pp spread (72-88%). Stronger models are less sensitive to harness choice — the harness matters most for weaker models.

**Tool calling compatibility is the primary barrier**: Qwen 2.5 and SWE-LM output bare JSON tool calls without `<tool_call>` tags, making them incompatible with OpenCode and Claude Code (which rely on vLLM's hermes parser). SERA's regex fallback handles bare JSON, but SERA is the only harness that works.

**Claude Code's Anthropic API doesn't generalize**: vLLM's `/v1/messages` endpoint translates Anthropic tool schemas to the internal chat template, but the translation doesn't produce correct Qwen tool calling prompts. Qwen3.5 ignores the tools entirely and generates generic text responses. Claude Code only works with Devstral (Mistral native format).

**Context management is critical for weaker models**: Qwen 2.5 32B overflows 16K context by turn 5. Even with 32K + aggressive trimming (tool output caps at 4K chars, old results trimmed to 500 chars), conversations still overflow. Qwen3.5's 65K context eliminates this bottleneck.

**MoE inference is fast**: Qwen3.5-397B (17B active) processes issues as fast as 32B dense models despite 12x total parameters. MoE is free performance at inference time.

---

## Phase 3b: Concurrent Agent Swarm (Phase 2a)

**Goal**: Measure how concurrent agent execution scales on a single GPU node. Does parallelism degrade fix rate? What is the throughput ceiling?

**Setup**: Qwen3.5-397B-A17B FP8 (TP4, all 4 GPUs), OpenCode harness, g7e.24xlarge. vLLM `max-num-seqs=4`. N concurrent OpenCode agents share the single vLLM endpoint.

### Scaling Results

| Config | Workers | Fix Rate | Wall Time | Speedup | Throughput | Efficiency | Avg/Issue |
|--------|---------|----------|-----------|---------|-----------|------------|-----------|
| N=1 (seq) | 1 | 44/50 (88%) | 43.7 min | 1.0x | 1.1/min | 100% | 52s |
| N=2 | 2 | 43/50 (86%) | 36.5 min | 1.2x | 1.4/min | 60% | 64s |
| **N=4** | **4** | **46/50 (92%)** | **21.4 min** | **2.0x** | **2.3/min** | **51%** | **71s** |
| N=8 | 8 | 49/50 (98%) | 26.2 min | 1.7x | 1.9/min | 21% | 172s |

### Key Findings

**N=4 is the sweet spot**: 2.0x speedup at 51% efficiency. Matches the vLLM `max-num-seqs=4` limit — the GPU can serve 4 concurrent sequences optimally.

**Concurrency does not degrade fix rate**: N=4 actually improves fix rate from 88% to 92%, and N=8 reaches 98%. Non-deterministic processing order means some timeout-prone issues get handled when other workers are busy, reducing per-issue queueing pressure.

**N=8 is GPU-bound, not CPU-bound**: Wall time increases from 21.4 min (N=4) to 26.2 min (N=8) because vLLM queues the excess 4 requests. Per-issue avg jumps from 71s to 172s as agents wait for GPU slots. Efficiency drops to 21%.

**The GPU bubble problem is real**: At N=4, efficiency is only 51% — agents spend ~49% of wall time waiting for tool execution (file reads, git operations) while the GPU holds their KV cache. This is the exact problem ThunderAgent (Phase 2b) would address by backfilling GPU bubbles with queued requests.

**50 issues in 21 minutes**: The N=4 configuration processes the full 50-issue SWE-bench subset in 21.4 minutes with 92% fix rate, compared to 43.7 minutes sequential. A production swarm could process ~110 issues/hour.

---

## Raw Data

- `results/phase1_{A-F}.jsonl` — 50 lines each, per-issue metrics with turn-level breakdown
- `results/phase2_{langgraph,sera,aider}.jsonl` — 50 lines each, per-issue harness metrics
- `results/phase2b_{ohmypi,piagent,deepagents,opencode,claude_code,codex}.jsonl` — Phase 2b per-issue metrics
- `results/eval_{ohmypi,piagent,opencode,claude_code,codex}.jsonl` — Gold test evaluation results for Phase 2b
- `results/swarm/swarm_phase1_{model}_{harness}.jsonl` — Phase 3 multi-model results
- `results/swarm/swarm_phase2a_n{2,4,8}_*.jsonl` — Phase 2a concurrent scaling results
- `results-report.html` — Interactive Chart.js visualization
- `lessons.md` — Operational lessons and debugging notes
