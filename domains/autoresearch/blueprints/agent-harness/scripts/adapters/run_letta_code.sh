#!/usr/bin/env bash
# Adapter: Letta Code (letta-ai/letta-code) — self-hosted
# Memory-first coding agent with persistent memory, skill learning, and sleeptime.
# Runs against a self-hosted Letta server (Docker) which forwards to local vLLM.
#
# Prerequisites:
#   docker run -d -p 8283:8283 \
#     -e OPENAI_API_BASE=$ENDPOINT/v1 \
#     -e OPENAI_API_KEY=dummy \
#     letta/letta-server
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LETTA_BASE_URL="${LETTA_BASE_URL:-http://localhost:8283}"
export OPENAI_BASE_URL="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

node "$SCRIPT_DIR/letta_agent.mjs" \
    --workspace "$WORKSPACE" \
    --model "$MODEL" \
    --issue "$PROBLEM_STATEMENT" \
    --test-cmd "$TEST_CMD"
