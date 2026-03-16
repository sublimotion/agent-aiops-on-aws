#!/usr/bin/env bash
# Adapter: OpenCode with vLLM via @ai-sdk/openai-compatible custom provider
# OpenCode uses its own tool set (glob, read, edit, bash, grep, write).
# We configure a custom "vllm" provider in opencode.json pointing at localhost.
# Requires 64K context (OpenCode system prompt ~11K + ~20 tool defs).
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

# Ensure opencode.json exists in workspace with vLLM provider config
cat > "$WORKSPACE/opencode.json" << OCEOF
{
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM Local",
      "options": {
        "baseURL": "${ENDPOINT}/v1"
      },
      "models": {
        "${MODEL}": {
          "name": "Devstral Small 2 24B",
          "limit": {
            "context": 65000,
            "output": 4000
          }
        }
      }
    }
  }
}
OCEOF

# Build the prompt — compact to save context (OpenCode system prompt is ~11K tokens)
PROMPT="Fix this bug in the repo at ${WORKSPACE}.

${PROBLEM_STATEMENT}

IMPORTANT: Edit the file immediately after reading the relevant code. Do not read more than 2 files before making your edit. Use the edit tool, not bash, to modify files."

# Run opencode with timeout, capture JSON events
OPENCODE_API_KEY=dummy timeout 540 opencode run \
    --model "vllm/${MODEL}" \
    --format json \
    --dir "$WORKSPACE" \
    "$PROMPT" > /tmp/opencode_events_${ISSUE_ID}.json 2>/tmp/opencode_stderr_${ISSUE_ID}.log || true

# Parse events to extract metrics
python3 << 'PYEOF'
import json, sys, os

issue_id = os.environ.get("ISSUE_ID", "unknown")
events_file = f"/tmp/opencode_events_{issue_id}.json"

total_tokens = 0
total_input = 0
total_output = 0
turns = 0
tool_calls = 0
has_edit = False
has_error = False
error_msg = ""

try:
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = evt.get("type", "")

            if evt_type == "step_finish":
                part = evt.get("part", {})
                tokens = part.get("tokens", {})
                total_tokens += tokens.get("total", 0)
                total_input += tokens.get("input", 0)
                total_output += tokens.get("output", 0)
                turns += 1

            elif evt_type == "tool_use":
                part = evt.get("part", {})
                state = part.get("state", {})
                tool_name = part.get("tool", "")
                tool_calls += 1
                if tool_name == "edit":
                    has_edit = True

            elif evt_type == "error":
                has_error = True
                error_msg = evt.get("error", {}).get("data", {}).get("message", "unknown error")

except FileNotFoundError:
    has_error = True
    error_msg = "no events file"

# Check if a diff was generated
workspace = os.environ.get("WORKSPACE", "")
import subprocess
diff_result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=workspace)
fix_generated = len(diff_result.stdout.strip()) > 0

result = {
    "pass": False,  # always False from adapter; gold eval determines pass
    "turns": turns,
    "tokens": total_tokens,
    "input_tokens": total_input,
    "output_tokens": total_output,
    "tool_calls": tool_calls,
    "fix_generated": fix_generated,
    "has_edit": has_edit,
}
if has_error:
    result["error"] = error_msg

print(json.dumps(result))
PYEOF
