#!/usr/bin/env bash
# Adapter: oh-my-pi (can1357/oh-my-pi)
# Fork of Pi with hash-anchored LINE#ID edit tool (the "harness problem" innovation).
# Also has LSP integration, ast_grep, Python kernel, sub-agents, context compaction.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

node "$SCRIPT_DIR/ohmypi_agent.mjs" \
    --workspace "$WORKSPACE" \
    --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" \
    --test-cmd "$TEST_CMD" \
    --max-turns 30
