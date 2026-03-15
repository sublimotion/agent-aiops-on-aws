#!/usr/bin/env bash
# Adapter: OpenCode style — str_replace edits via AI SDK runtime
# NOTE: Blocked — Vercel AI SDK @ai-sdk/openai-compatible does not work with
# vLLM (tool args come through as undefined, uses Responses API not Chat
# Completions). Kept for reference; use langgraph_agent.py instead.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

echo '{"pass":false,"turns":0,"tokens":0,"fix_generated":false,"error":"OpenCode adapter blocked: AI SDK incompatible with vLLM"}'
