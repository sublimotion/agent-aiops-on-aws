#!/usr/bin/env bash
# Adapter: Pi Agent control — same LangGraph ReAct loop with str_replace edit
# Control for hashline comparison (identical to langgraph adapter)
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

python3 "$SCRIPT_DIR/langgraph_agent.py" \
    --workspace "$WORKSPACE" --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" --test-cmd "$TEST_CMD" --max-turns 30
