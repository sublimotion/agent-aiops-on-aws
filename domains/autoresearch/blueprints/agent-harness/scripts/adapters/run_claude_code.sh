#!/usr/bin/env bash
# Adapter: Claude Code with vLLM via patched Anthropic Messages API
# Uses temporary env vars to point Claude Code at local vLLM.
# Requires vLLM v0.16.0 with 3 patches (tool_use_id, streaming args, id remap).
# System prompt ~22K tokens — needs 64K context.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

PROMPT="Fix this bug in the repository at ${WORKSPACE}.

${PROBLEM_STATEMENT}

IMPORTANT: Read the relevant source file, then use the Edit tool to fix the bug. Do not read more than 3 files before making your edit. Be precise — change only what is needed."

# Run Claude Code with vLLM backend via temporary env vars
OUTPUT_FILE="/tmp/claude_code_${ISSUE_ID}.json"

ANTHROPIC_BASE_URL="${ENDPOINT}" \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL="${MODEL}" \
ANTHROPIC_DEFAULT_SONNET_MODEL="${MODEL}" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}" \
timeout 540 claude \
    --print \
    --output-format json \
    --dangerously-skip-permissions \
    --max-turns 30 \
    -p "$WORKSPACE" \
    "$PROMPT" > "$OUTPUT_FILE" 2>/tmp/claude_code_stderr_${ISSUE_ID}.log || true

# Parse Claude Code JSON output for metrics
python3 << 'PYEOF'
import json, sys, os, subprocess

issue_id = os.environ.get("ISSUE_ID", "unknown")
output_file = f"/tmp/claude_code_{issue_id}.json"
workspace = os.environ.get("WORKSPACE", "")

turns = 0
input_tokens = 0
output_tokens = 0
tool_calls = 0
has_edit = False
has_error = False
error_msg = ""
stop_reason = ""

try:
    with open(output_file) as f:
        data = json.load(f)

    turns = data.get("num_turns", 0)
    stop_reason = data.get("stop_reason", "")
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Check for errors
    if data.get("is_error"):
        has_error = True
        errors = data.get("errors", [])
        error_msg = errors[0] if errors else data.get("subtype", "unknown")

    # Approximate tool_calls from cost — each turn is roughly one API call
    tool_calls = max(0, turns - 1)  # first turn is usually text, rest are tool calls

except (FileNotFoundError, json.JSONDecodeError) as e:
    has_error = True
    error_msg = str(e)
    # Try to read stderr
    try:
        with open(f"/tmp/claude_code_stderr_{issue_id}.log") as f:
            stderr = f.read().strip()[-200:]
            if stderr:
                error_msg = stderr
    except:
        pass

# Check if a diff was generated
diff_result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=workspace)
fix_generated = len(diff_result.stdout.strip()) > 0

result = {
    "pass": False,
    "turns": turns,
    "tokens": input_tokens + output_tokens,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "tool_calls": tool_calls,
    "fix_generated": fix_generated,
    "has_edit": fix_generated,  # if there's a diff, an edit was made
    "stop_reason": stop_reason,
}
if has_error:
    result["error"] = error_msg

print(json.dumps(result))
PYEOF
