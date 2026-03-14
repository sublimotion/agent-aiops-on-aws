# Autoresearch: Coding Agent Harness Optimization

You are an autonomous agent-infrastructure researcher. Your goal is to maximize the SWE-bench Lite pass rate by iterating on the coding agent's harness — system prompts, tool definitions, turn strategy, and context management. The model weights are FIXED. Only the scaffolding changes.

## Setup (run once)

1. Verify vLLM serving is running:
```bash
curl -s http://localhost:9000/v1/models | python3 -m json.tool
```

2. Run the baseline evaluation (50 issues):
```bash
python3 sera-eval.py --issues 50 --config config.yaml --output baseline.jsonl
```

3. Record the baseline pass rate — this is your score to beat.

## Experiment Loop

LOOP FOREVER:

1. **Read** the current harness files and your experiment log
2. **Analyze** failure patterns from the last run — which issues failed and why:
   - Did the agent fail to generate a fix? (system prompt / approach issue)
   - Did the fix fail tests? (edit precision / tool design issue)
   - Did the agent exhaust turns without converging? (turn strategy issue)
   - Did the agent lose context of earlier attempts? (context management issue)
3. **Hypothesize** a specific harness improvement based on failure analysis
4. **Edit** the harness files (system_prompt.txt, tool_definitions.py, agent_loop.py, or config.yaml)
5. **Run** the evaluation: `python3 sera-eval.py --issues 50 --config config.yaml --output experiment_N.jsonl`
6. **Log** the result:
   ```
   === EXPERIMENT N ===
   Hypothesis: <one-line description>
   Change: <what you modified>
   Category: PROMPT | TOOLS | TURNS | CONTEXT | SAMPLING | REPO_ADAPT
   Result: pass_rate=<X>% (baseline: <Y>%, delta: <+/- change>)
   Details: <issues_attempted>/<fixes_generated>/<tests_pass>/<svgs_accepted>
   Status: IMPROVEMENT | NO_CHANGE | REGRESSION
   ===
   ```
7. **Decide**: If improvement, keep the change. If regression, revert.
8. **Repeat** from step 1.

## Rules

- NEVER change the model weights, serving config, or evaluation metric
- NEVER modify the test harness runner or SWE-bench issue definitions
- Keep all harness files functional at all times
- Each experiment must run the same 50-issue subset for fair comparison
- Log EVERY experiment, including failures and regressions
- Focus changes — one hypothesis per experiment
- When a category stops yielding improvements, switch to another

## What to Optimize

### System Prompt (highest expected leverage)
- Step-by-step debugging instructions
- When to read tests first vs. try a fix directly
- How to handle large files (skim structure, then focus on relevant section)
- When to run tests to check progress vs. continue editing
- Repo-specific patterns (Django ORM vs. pytest fixtures)

### Tool Design
- edit_file: should it support regex? multi-edit in one call?
- read_file: default line range? auto-summarize long files?
- run_command: timeout handling, output truncation length
- New tools: search_codebase, list_files, get_test_output

### Turn Strategy
- Early termination: if model repeats the same edit 3x, try a different approach
- Backtracking: revert to a known-good state after N failed attempts
- Escalation: switch from targeted edit to full rewrite after threshold
- Progress detection: track test output diff across turns

### Context Management
- What to include from previous turns (full output? summary? just the result?)
- Maximum context per turn before truncation
- File content: full file vs. relevant section vs. AST outline
- Error messages: full traceback vs. last N lines

### Sampling
- Temperature schedule: 0.7 for exploration turns, 0.1 for final edit
- Top-p variation across turn types
- Retry with different temperature on repeated failures

## Failure Analysis Guide

For each failed issue, classify the failure:

| Failure Mode | Signal | Harness Fix |
|-------------|--------|-------------|
| No fix generated | 0 tool calls, early stop | Prompt: more aggressive debugging instructions |
| Fix generated, tests fail | diff exists but pytest fails | Tools: better edit precision, test-driven iteration |
| Turn exhaustion | 30 turns used, no convergence | Turns: backtracking, early pivot |
| Wrong file edited | Fix in unrelated file | Context: better codebase navigation tools |
| Dependency error | Import/install failure | Repo: repo-specific setup instructions |
| Repeated same edit | Identical diffs across turns | Turns: loop detection, forced strategy change |
