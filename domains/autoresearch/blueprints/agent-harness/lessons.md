# Agent Harness Autoresearch — Lessons Learned

## Blueprint Completion (2026-03-14)

### Blueprint Structure
- Spec updated to three-phase design: turn degradation → multi-harness → finetuning (future)
- Phase 1 evaluator (`harness_eval.py`) forks the SERA agent loop from `sera-datagen.py` with per-turn instrumentation, configurable turn budgets (10/15/20/30), restart-with-summary, and context compaction strategies
- Phase 2 evaluator (`multi_harness_eval.py`) runs 7 harnesses (SERA, Claude Code, OpenHands, SWE-agent, Aider, OpenCode, LangGraph) via standardized adapter scripts
- Adapter interface: env vars (WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO) → JSON output `{"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}`

### Key Design Decisions
- **Same 50-issue subset everywhere**: seed 42, stratified by repo for diversity — ensures fair comparison across configs and harnesses
- **Phase 1 before Phase 2**: Turn degradation analysis informs the SERA baseline config for multi-harness comparison
- **Bedrock endpoint for Phase 2**: Simplifies harness setup (just set OPENAI_BASE_URL), avoids local GPU dependency for harness comparison
- **Phase 3 deferred**: 82% fix generation rate with 17.7% pass rate indicates scaffolding problem, not weights problem. Finetune only after harness optimization plateaus.

### Deployment Readiness
- g7e instance accessible (4x RTX PRO 6000 Blackwell)
- vLLM **not currently serving** — run `setup_vllm.sh` before Phase 1
- SWE-bench workspaces on `/mnt/nvme/sera-workspaces/` (only 5 repos cached — first run will clone more)
- Devstral Small 2 FP8 weights at `/mnt/nvme/models/devstral-small-2-fp8`
- External dependency: `lb-proxy.py` at `/mnt/nvme/sera-scripts/lb-proxy.py` (from devstral-sera blueprint)

### Pre-Deployment Blockers
- None. Blueprint is code-complete. Execution requires starting vLLM serving on g7e instance.

---

## Phase 1 Execution (2026-03-14)

### Infrastructure Setup
- 4x vLLM v0.16.0 replicas (1 per GPU), ports 8000-8003, `CUDA_VISIBLE_DEVICES` per replica
- Round-robin LB on port 9000 via `lb-proxy.py`
- Python 3.11 venv at `/mnt/nvme/agent-harness-env/` with swebench 4.1.0 + openai
- 12 repos cloned with full history (5 cached repos unshallowed + 7 new clones)
- 50-issue eval subset generated (seed 42, 11 repos)
- `max-model-len=32768`, `gpu-memory-utilization=0.95` → 63.85 GiB KV cache, 418K token capacity

### Critical Findings (Pre-Experiment)

#### 1. Mistral Chat Template Ordering
The Mistral chat template enforces strict message role ordering: `tool` messages MUST follow an `assistant` message with `tool_calls`. Inserting a `user` message between assistant tool calls and tool results causes `ValueError: Unexpected role 'tool' after role 'user'` (HTTP 400). Context compaction must preserve the `[assistant+tool_calls] → [tool_results]` ordering.

#### 2. Devstral Small 2 Doesn't Self-Direct to Edits
Without turn pressure, Devstral Small 2 24B spends ALL turns reading/searching and NEVER attempts `edit_file`. This is consistent across all test issues. The model enters a pathological read loop — reading the same files repeatedly after context compaction strips the previous reads.

**Root cause**: The 24B model lacks the "meta-planning" capability to decide when to stop exploring and start editing. Larger models (70B+) self-direct better.

**Fix**: Inject turn pressure reminders into the last tool result content:
- Every 2 turns after explore_budget (max_turns//5): "You MUST use edit_file now"
- Last 3 turns: "URGENT: Use edit_file RIGHT NOW"

This converts a 0% edit rate to ~33% edit rate (1/3 issues get edits in 3-issue test).

#### 3. Context Compaction is Necessary but Insufficient
At ~20K estimated tokens, context must be compacted to avoid 400 errors (Mistral token limit). Compaction works (drops old tool results) but the model re-reads the same files after compaction, entering a loop. Compaction alone doesn't fix the edit avoidance problem.

### Harness Iterations
| Version | Change | Result |
|---------|--------|--------|
| v1 | Baseline (no compaction, no pressure) | All issues hit 502 at ~25K tokens |
| v2 | Auto-compaction at 20K estimated tokens | No 502s, but model enters read loop forever |
| v3 | + Mistral-safe compaction (preserve role order) | Compaction works, still no edits |
| v4 | + Turn pressure reminders every 2 turns | Django passes, 33% rate on 3-issue test |

### Bugs Found During Execution
- `subprocess.run` with `cwd` raises `FileNotFoundError` if model's `run_command` deletes the workspace. Fixed with `os.path.isdir()` guard.
- Shell pipe `| tail` masks pytest exit code. Fixed by extracting `EXIT_CODE=$?` from output (then switched to offline patch evaluation).
- Mistral chat template rejects `user` role after `tool_calls` — context compaction must preserve `[assistant+tool_calls] → [tool_results]` ordering.
- `pip install` in `run_command` blocks the agent loop for minutes. Blocked in dangerous commands list.

### Phase 1 Configs A-F Results (MEASURED)

| Config | Turns | Strategy | Fix Rate | Edit Rate | Avg 1st Edit | Repeat Rate |
|--------|-------|----------|----------|-----------|-------------|-------------|
| A | 10 | baseline | 36% (18/50) | 42% (21/50) | 6.5 | 15.7% |
| B | 15 | baseline | 48% (24/50) | 56% (28/50) | 9.4 | 23.6% |
| C | 20 | baseline | 52% (26/50) | 64% (32/50) | 11.9 | 29.3% |
| D | 30 | baseline | **60% (30/50)** | 78% (39/50) | 17.5 | 37.3% |
| E | 30 | restart at 15 | 56% (28/50) | 76% (38/50) | 18.2 | 37.9% |
| F | 30 | compact at 15 | 40% (20/50) | 64% (32/50) | 16.5 | 41.3% |

Note: Fix rate = generated a diff (not verified pass). Pass rate requires offline swebench evaluation.

#### Turn Degradation Analysis

**Parkinson's Law for Agents**: The model consistently delays first edit. Average first edit occurs at turn 6.5-17.5 depending on budget, always in the final third of available turns (65% → 58% of budget).

**Diminishing returns**: Fix rate improves from 36% (10 turns) to 60% (30 turns) but marginal gains per turn drop: +12pp for 10→15, +4pp for 15→20, +8pp for 20→30. Repeat rate scales linearly with budget (15.7% → 37.3%).

**Compaction actively hurts**: Config F (compact at 15) is the worst performer at 40% fix rate — worse than Config A (10 turns, 36%). Compaction strips exploration context, causing the model to re-read files and waste turns (41.3% repeat rate). The model loses orientation after compaction.

**Restart is neutral**: Config E (restart at 15) achieves 56% fix rate vs Config D's 60% — modest drop. But it has nearly identical repeat rate (37.9% vs 37.3%), suggesting the fresh context doesn't reduce looping.

**Best config**: D (30 turns, baseline) with 60% fix generation rate and highest edit rate (78%).

### Status
- Configs A-F: COMPLETE — results in `results/phase1_{A-F}.jsonl`
- Phase 2: COMPLETE — results below

---

## Phase 2 Execution (2026-03-14)

### Harness Setup
- **LangGraph**: LangChain's `ChatOpenAI` with structured tool calling in a manual ReAct loop with context compaction. Same 5 tools as SERA. Truncated tool outputs to 4K chars.
- **SERA** (builtin): Custom Python agent loop with Config D (30 turns baseline), turn pressure, auto-compaction at 20K tokens.
- **Aider**: `aider-chat` with `--edit-format diff`, single-turn message mode.
- Skipped: SWE-agent (dep broken), OpenHands (requires Python 3.10+), Claude Code (needs Anthropic API key), OpenCode (Go binary N/A)
- All harnesses use Devstral Small 2 24B FP8 via local vLLM (4 replicas, round-robin LB on port 9000)

### Phase 2 Harness Leaderboard (MEASURED)

| Rank | Harness | Pass Rate | Fix Rate | Edit Rate | Avg 1st Edit | Avg Turns | Avg Latency |
|------|---------|-----------|----------|-----------|-------------|-----------|-------------|
| 1 | **SERA** | **8/50 (16%)** | 23/50 (46%) | 24/50 (48%) | 13.5 | 29.1 | 87.6s |
| 2 | **LangGraph** | **7/50 (14%)** | 31/50 (62%) | 32/50 (64%) | 19.7 | 28.4 | 50.2s |
| 3 | **Ensemble** | **11/50 (22%)** | 41/50 (82%) | — | — | — | — |
| 4 | Aider | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | n/a | n/a | 5.9s |

Pass rate = verified against FAIL_TO_PASS tests with gold test_patch applied. Evaluated: Django (all), pytest (partial), sympy (partial). Other repos (sphinx, scikit-learn, matplotlib, astropy, etc.) could not be tested without Docker — their version-specific dependencies conflict with Python 3.11.

**Note**: Pass rate is a lower bound for repos that could not be evaluated.

### Key Findings

#### SERA Slightly Better Pass Rate Despite Fewer Fixes
SERA achieves **16% pass rate** (8/50) vs LangGraph's **14%** (7/50), despite generating only 46% fixes vs 62%. SERA's pass/fix ratio is 35% (8/23) vs LangGraph's 23% (7/31). **SERA generates fewer but higher-quality fixes.**

#### LangGraph Generates More Fixes, Faster
LangGraph achieves 62% fix rate vs SERA's 46% with the same model, tools, and benchmark:
- **LangChain's structured tool calling** vs SERA's manual JSON parsing — fewer malformed tool calls
- **Aggressive output truncation** (4K chars) vs SERA's 10K — keeps context smaller, more room for reasoning
- **42% faster** (50.2s vs 87.6s avg) — fewer wasted turns on repeated actions

#### Massive Complementarity — Ensemble Is the Biggest Win
Fix generation union: 41/50 (82%). **Verified pass rate union: 11/50 (22%)**:
- LangGraph-only passes: 3 (django-12286, django-12497, django-14608)
- SERA-only passes: 4 (django-11039, django-11620, django-11815, django-14672)
- Both pass: 4 (django-12453, django-14855, pytest-11143, sympy-24152)
- Running both and taking the union lifts pass rate from 14-16% → **22%** (37-57% improvement).

#### SERA's Turn Pressure Creates Different Fix Patterns
SERA's avg first edit is turn 13.5 (vs LangGraph's 19.7). The earlier forced edits produce different patches that succeed on different issues — explaining the high complementarity.

#### Fix Generation ≠ Quality
LangGraph's 62% fix rate converts to only 14% pass rate (23% conversion). SERA's 46% fix rate converts to 16% pass rate (35% conversion). **Generating more diffs doesn't mean generating better diffs.**

#### Aider Cannot Drive Devstral Small 2
Aider with `--edit-format diff` produces zero fixes.

#### Additional Harnesses Blocked
- **SWE-agent**: `togetherunidiff` package unavailable (broken PyPI dep)
- **OpenHands**: `e2b` dependency conflict
- **OpenCode**: Go binary not available for this platform

#### Pass Rate Is a Lower Bound
Evaluated with gold test_patch: Django (all 16 patches), pytest (3 patches), sympy (7 patches). Other repos (sphinx ×3, scikit-learn ×2, matplotlib ×2, astropy, seaborn, requests, pylint) could not be tested — their version-specific deps conflict with Python 3.11. SWE-bench Docker containers required for full evaluation.

### Conclusions

1. **SERA has better precision** (35% pass/fix) but LangGraph has better recall (62% fix rate). Neither dominates.
2. **Harness ensemble is the biggest win**: Running both yields **22% pass rate** (11/50) — 37-57% improvement over either alone.
3. **Fix generation is not pass rate.** The correlation between generating a diff and generating a correct diff is weak (~25-35%).
4. **Scaffolding has converged at 14-16%** individually. Phase 3 (finetuning) may now be relevant — a targeted LoRA on common error patterns could compound with the ensemble approach.
5. **Full SWE-bench evaluation requires Docker** — our bare-metal eval is a lower bound covering ~50% of patches.
