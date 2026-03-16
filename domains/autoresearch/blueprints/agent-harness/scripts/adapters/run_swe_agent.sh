#!/usr/bin/env bash
# Adapter: SWE-agent
# Runs SWE-agent against a single SWE-bench issue.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

cd "$WORKSPACE"

# SWE-agent config for OpenAI-compatible endpoint
export OPENAI_API_BASE="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

OUTPUT_DIR=$(mktemp -d)

# Run SWE-agent
sweagent run \
    --model_name "$MODEL" \
    --data.text "$PROBLEM_STATEMENT" \
    --repo_path "$WORKSPACE" \
    --output_dir "$OUTPUT_DIR" \
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
if [ -f "$OUTPUT_DIR/trajectory.json" ]; then
    TURNS=$(python3 -c "import json; t=json.load(open('$OUTPUT_DIR/trajectory.json')); print(len(t.get('trajectory', [])))" 2>/dev/null || echo "0")
fi

rm -rf "$OUTPUT_DIR"
echo "{\"pass\": $PASS, \"turns\": $TURNS, \"tokens\": 0, \"fix_generated\": $FIX_GENERATED}"
