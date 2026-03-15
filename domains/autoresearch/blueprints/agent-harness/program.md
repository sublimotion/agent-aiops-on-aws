# Autoresearch: Coding Agent Harness Optimization

You are an autonomous agent-infrastructure researcher. Your goal is to understand how agent scaffolding affects coding task performance, across three phases: turn degradation analysis, multi-harness comparison, and (future) finetuning.

## Phase 1: Turn Degradation Analysis

### Goal
Determine how turn count affects fix quality — are late turns helping or actively hurting?

### Setup (run once)

1. Verify vLLM serving is running:
```bash
curl -s http://localhost:9000/v1/models | python3 -m json.tool
```

2. Prepare the 50-issue eval subset (must be identical across ALL configs and phases):
```bash
python3 harness_eval.py --list-subset > eval_subset.txt
```

### Experiment Configs

Run each config on the same 50-issue subset:

| Config | Turn Budget | Strategy | Command |
|--------|------------|----------|---------|
| A | 10 | Strict cutoff | `--max-turns 10` |
| B | 15 | Strict cutoff | `--max-turns 15` |
| C | 20 | Strict cutoff | `--max-turns 20` |
| D | 30 (baseline) | Strict cutoff | `--max-turns 30` |
| E | 15+15 restart | Fresh context at turn 15 with failure summary | `--max-turns 15 --restart-with-summary` |
| F | 30 compaction | Compact context at turn 15, continue to 30 | `--max-turns 30 --compact-at 15` |

Configs E and F use restart/compaction strategies already implemented in `harness_eval.py` (`compact_context()` and restart-with-summary logic in `run_instrumented_loop()`).

### Per-Turn Metrics

The `harness_eval.py` script instruments every turn, logging:
- `turn_number`: which turn
- `action_type`: read/edit/run/search
- `edit_correctness`: did the edit apply cleanly?
- `test_delta`: change in test pass count vs previous turn
- `context_tokens`: total tokens in context window
- `repeated_action`: boolean, is this action identical to a previous turn?

### Execution

```bash
# Run all configs at once:
python3 harness_eval.py --endpoint http://localhost:9000 --run-all --output-dir results/

# Or run a single config:
python3 harness_eval.py --endpoint http://localhost:9000 --config A --output-dir results/
```

### Analysis

After all 6 configs complete:
1. Plot pass rate vs turn budget (A through D) — is there a cliff?
2. Compare E and F against D — does restart/compaction recover lost performance?
3. Find the "turn of first correct fix" distribution — how early does the model find the right answer?
4. Measure repetition rate per turn — when does looping behavior start?
5. Log results:

```
=== PHASE 1 RESULTS ===
Config A (10 turns): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Config B (15 turns): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Config C (20 turns): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Config D (30 turns): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Config E (restart): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Config F (compact): pass_rate=X%, avg_turns=Y, repetition_rate=Z%
Best config: <name> (pass_rate=X%, delta vs baseline: +/-Y%)
Turn degradation threshold: turn <N> (quality drops after this point)
===
```

## Phase 2: Multi-Harness Comparison

### Goal
Determine which scaffolding architecture extracts the most capability from Devstral Small 2.

### Prerequisites
- Phase 1 complete (use best turn config as SERA's setting)
- Same 50-issue eval subset from Phase 1

### Model Endpoint

Use Amazon Bedrock OpenAI-compatible endpoint as the primary model backend. All harnesses connect via:
```
OPENAI_BASE_URL=<bedrock-endpoint>
OPENAI_API_KEY=<bedrock-access-key>
```

Fall back to local vLLM (`localhost:9000`) if Bedrock model unavailable or rate-limited.

### Harness Setup

Install and configure each harness to use the model endpoint:

| Harness | Install | Config |
|---------|---------|--------|
| SERA (baseline) | Already installed | Use best turn config from Phase 1 |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | `CLAUDE_CODE_USE_BEDROCK=1`, custom model config |
| OpenHands | `pip install openhands` or Docker | Set LLM config to Bedrock endpoint |
| SWE-agent | Clone repo, pip install | `keys.cfg` with Bedrock endpoint |
| Aider | `pip install aider-chat` | `--openai-api-base` flag |
| OpenCode | Install binary | Config file with endpoint |
| LangGraph ReAct | Python script with langgraph | Direct API client setup |

For each harness, create an adapter script `run_<harness>.sh` that:
1. Takes a SWE-bench issue ID as input
2. Runs the harness against that issue
3. Outputs a standardized result JSON: `{issue_id, pass, turns_used, tokens_consumed, fix_generated}`

### Execution

```bash
# Check which harnesses are installed:
python3 multi_harness_eval.py --check-installed

# Run all installed harnesses:
python3 multi_harness_eval.py --endpoint http://localhost:9000 --run-all --output-dir results/

# Or run a single harness:
python3 multi_harness_eval.py --endpoint http://localhost:9000 --harness aider --output-dir results/

# Generate comparison report from results:
python3 multi_harness_eval.py --report results/phase2_*.jsonl
```

### Analysis

After all harnesses complete:
1. Rank by pass rate — harness leaderboard
2. Compare avg turns to resolution
3. Compare tokens consumed per successful fix (context efficiency)
4. Plot turn degradation curve per harness (using Phase 1 methodology)
5. Identify which scaffolding patterns correlate with success:
   - Execution environment (sandbox vs bare)
   - Agent paradigm (ReAct vs CodeAct vs edit-focused)
   - Context management (truncation vs compaction vs repo-map)
   - Tool design (line-edit vs block-edit vs full-rewrite)
6. Log results:

```
=== PHASE 2 RESULTS ===
Harness leaderboard:
1. <harness>: pass_rate=X%, avg_turns=Y, tokens/fix=Z
2. <harness>: pass_rate=X%, avg_turns=Y, tokens/fix=Z
...
Key findings:
- <which scaffolding patterns matter most>
- <which patterns don't matter>
===
```

## Phase 3: Model Finetuning (Future — DO NOT EXECUTE)

Documented in the spec for future reference. Do not execute this phase.
After Phases 1-2 plateau, collect successful turn traces as training data for LoRA finetuning on the winning harness's tool format and fix trajectories.

## Rules

- NEVER change the model weights, serving config, or evaluation metric
- NEVER modify the test harness runner or SWE-bench issue definitions
- Use the SAME 50-issue subset (seed 42) across ALL configs and harnesses
- Log EVERY experiment, including failures and regressions
- Focus changes — one variable per experiment
- If a harness fails to install or run, document the failure and move on
