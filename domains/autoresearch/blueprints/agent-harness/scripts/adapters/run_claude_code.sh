#!/usr/bin/env bash
# Adapter: Claude Code
# Runs Claude Code CLI against a single SWE-bench issue.
#
# Env vars (set by multi_harness_eval.py):
#   WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
#
# Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}

set -euo pipefail

cd "$WORKSPACE"

# Claude Code uses ANTHROPIC_API_KEY by default. For Bedrock:
#   export CLAUDE_CODE_USE_BEDROCK=1
# For local vLLM via OpenAI-compatible:
#   export OPENAI_BASE_URL="$ENDPOINT/v1"
#   export OPENAI_API_KEY="dummy"

PROMPT="You are working in repository ${REPO}, checked out at the correct commit.

Fix this issue:
${PROBLEM_STATEMENT}

After fixing, verify with: ${TEST_CMD}

Use edit_file for targeted changes. Do not add unrelated changes."

# Run Claude Code in non-interactive mode with the prompt
# Capture output to parse turn count
OUTPUT_FILE=$(mktemp)

claude --print \
    --model "$MODEL" \
    --max-turns 30 \
    --prompt "$PROMPT" \
    > "$OUTPUT_FILE" 2>&1 || true

# Check if tests pass
TEST_OUTPUT=$(bash -c "cd $WORKSPACE && $TEST_CMD" 2>&1 || true)

PASS=false
if echo "$TEST_OUTPUT" | grep -q "passed" && ! echo "$TEST_OUTPUT" | grep -q "failed"; then
    PASS=true
fi
if echo "$TEST_OUTPUT" | grep -qE "^OK$|^OK " && ! echo "$TEST_OUTPUT" | grep -qi "fail"; then
    PASS=true
fi

# Check for fix
FIX_GENERATED=false
if [ -n "$(git diff)" ]; then
    FIX_GENERATED=true
fi

# Count turns (approximate: count tool use blocks in output)
TURNS=$(grep -c "tool_use\|Tool:" "$OUTPUT_FILE" 2>/dev/null || echo "0")

rm -f "$OUTPUT_FILE"

echo "{\"pass\": $PASS, \"turns\": $TURNS, \"tokens\": 0, \"fix_generated\": $FIX_GENERATED}"
