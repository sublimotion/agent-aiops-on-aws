#!/usr/bin/env bash
# Adapter: OpenHands
# Runs OpenHands CodeAct agent against a single SWE-bench issue.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

cd "$WORKSPACE"

OUTPUT_DIR=$(mktemp -d)

# OpenHands supports OpenAI-compatible endpoints
export LLM_BASE_URL="$ENDPOINT/v1"
export LLM_API_KEY="${OPENAI_API_KEY:-dummy}"
export LLM_MODEL="openai/$MODEL"

# Run OpenHands in headless mode
python3 -m openhands.core.main \
    --task "$PROBLEM_STATEMENT

Fix the issue. Then verify with: $TEST_CMD" \
    --workspace-dir "$WORKSPACE" \
    --max-iterations 30 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 || true

# Check tests
TEST_OUTPUT=$(bash -c "cd $WORKSPACE && $TEST_CMD" 2>&1 || true)

PASS=false
if echo "$TEST_OUTPUT" | grep -q "passed" && ! echo "$TEST_OUTPUT" | grep -q "failed"; then
    PASS=true
fi
if echo "$TEST_OUTPUT" | grep -qE "^OK$|^OK " && ! echo "$TEST_OUTPUT" | grep -qi "fail"; then
    PASS=true
fi

FIX_GENERATED=false
if [ -n "$(git diff)" ]; then
    FIX_GENERATED=true
fi

# Parse trajectory for turn count
TURNS=0
if [ -f "$OUTPUT_DIR/output.jsonl" ]; then
    TURNS=$(wc -l < "$OUTPUT_DIR/output.jsonl" 2>/dev/null || echo "0")
fi

rm -rf "$OUTPUT_DIR"
echo "{\"pass\": $PASS, \"turns\": $TURNS, \"tokens\": 0, \"fix_generated\": $FIX_GENERATED}"
