#!/usr/bin/env bash
# Adapter: OpenCode (anomalyco/opencode)
# Terminal coding agent with 75+ LLM providers, LSP, multi-session.
# Uses the OpenCode SDK for programmatic headless invocation.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

node "$SCRIPT_DIR/opencode_agent.mjs" \
    --workspace "$WORKSPACE" \
    --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" \
    --test-cmd "$TEST_CMD" \
    --max-turns 30
