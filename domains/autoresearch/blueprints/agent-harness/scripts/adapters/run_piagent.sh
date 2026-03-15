#!/usr/bin/env bash
# Adapter: Pi Agent (badlogic/pi-mono)
# Base framework that oh-my-pi forks from. Uses str_replace-style editing
# (no hashline). Useful as a control to isolate hashline vs base scaffolding.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

node "$SCRIPT_DIR/piagent_agent.mjs" \
    --workspace "$WORKSPACE" \
    --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" \
    --test-cmd "$TEST_CMD" \
    --max-turns 30
