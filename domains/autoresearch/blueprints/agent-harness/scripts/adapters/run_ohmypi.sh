#!/usr/bin/env bash
# Adapter: oh-my-pi style — hash-anchored LINE:HASH edits (the "harness problem" innovation)
# Same LangGraph ReAct loop as langgraph, but with hashline edit_file tool.
# read_file returns "42:a3|content", edit_file takes start_hash="42:a3" end_hash="45:f1".
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

python3 "$SCRIPT_DIR/hashline_agent.py" \
    --workspace "$WORKSPACE" --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" --test-cmd "$TEST_CMD" --max-turns 30
