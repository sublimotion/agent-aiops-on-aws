#!/usr/bin/env bash
# Adapter: LangGraph ReAct
# Runs a LangGraph ReAct agent against a single SWE-bench issue.
# Delegates to the Python script langgraph_agent.py in the same directory.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

python3 "$SCRIPT_DIR/langgraph_agent.py" \
    --workspace "$WORKSPACE" \
    --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" \
    --test-cmd "$TEST_CMD" \
    --max-turns 30
