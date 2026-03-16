#!/usr/bin/env bash
# Adapter: Aider
# Runs Aider against a single SWE-bench issue.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

cd "$WORKSPACE"

export OPENAI_API_BASE="$ENDPOINT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export AIDER_NO_AUTO_UPGRADE=1

OUTPUT_FILE=$(mktemp)

# Run Aider in no-git, message mode with diff edit format
# diff format works better with Devstral than whole format
aider \
    --model "openai/$MODEL" \
    --openai-api-base "$ENDPOINT/v1" \
    --no-git \
    --yes \
    --edit-format diff \
    --no-show-model-warnings \
    --message "Fix this issue in the repository:

${PROBLEM_STATEMENT}

After fixing, the following test command should pass: ${TEST_CMD}

Make targeted changes only." \
    > "$OUTPUT_FILE" 2>&1 || true

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

# Count edits from Aider output (grep -c outputs "0" on no match but exits 1,
# so || echo "0" would append a second "0" — use subshell to avoid double output)
TURNS=$(grep -c "Applied edit\|wrote\|Updated" "$OUTPUT_FILE" 2>/dev/null) || TURNS=0

# Token usage from Aider output
TOKENS=$(grep -oP 'Tokens: [\d,]+' "$OUTPUT_FILE" 2>/dev/null | tail -1 | tr -cd '0-9') || TOKENS=0

rm -f "$OUTPUT_FILE"
echo "{\"pass\": $PASS, \"turns\": $TURNS, \"tokens\": ${TOKENS:-0}, \"fix_generated\": $FIX_GENERATED}"
