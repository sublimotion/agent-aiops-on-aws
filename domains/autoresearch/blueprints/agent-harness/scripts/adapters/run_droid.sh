#!/usr/bin/env bash
# Adapter: Factory Droid CLI with vLLM via generic-chat-completion-api
# Droid supports custom model endpoints via ~/.factory/settings.json.
# We configure it to point at the local vLLM instance.
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

# Configure Droid to use local vLLM
FACTORY_CONFIG_DIR="${HOME}/.factory"
mkdir -p "$FACTORY_CONFIG_DIR"

cat > "$FACTORY_CONFIG_DIR/settings.json" << CFGEOF
{
  "customModels": [{
    "displayName": "vLLM Local",
    "provider": "generic-chat-completion-api",
    "baseUrl": "${ENDPOINT}/v1",
    "apiKey": "${OPENAI_API_KEY:-dummy}",
    "model": "${MODEL}",
    "maxOutputTokens": 2048
  }]
}
CFGEOF

PROMPT="Fix this bug in the repository.

${PROBLEM_STATEMENT}

IMPORTANT: Read the relevant source file, then edit it to fix the bug. Do not read more than 3 files before making your edit. Be precise — change only what is needed."

# Use TRAJECTORY_FILE from multi_harness_eval.py if set, else /tmp
TRAJ_FILE="${TRAJECTORY_FILE:-/tmp/droid_traj_${ISSUE_ID}.jsonl}"
OUTPUT_FILE="/tmp/droid_output_${ISSUE_ID}.json"
STDERR_FILE="/tmp/droid_stderr_${ISSUE_ID}.log"
DIFF_FILE="/tmp/droid_diff_${ISSUE_ID}.diff"

# Run Droid in non-interactive exec mode
# --skip-permissions-unsafe: no approval prompts
# --auto high: max autonomy
# -o json: structured JSON output
# -m: select our custom vLLM model
timeout 540 droid exec \
    --cwd "$WORKSPACE" \
    -m "custom:${MODEL}" \
    --auto high \
    -o json \
    "$PROMPT" > "$OUTPUT_FILE" 2>"$STDERR_FILE" || true

# Save git diff before parsing (workspace may get cleaned up after)
git -C "$WORKSPACE" diff > "$DIFF_FILE" 2>/dev/null || true

# Parse Droid JSON output and write trajectory
python3 << 'PYEOF'
import json, os, subprocess, shutil

issue_id = os.environ.get("ISSUE_ID", "unknown")
output_file = f"/tmp/droid_output_{issue_id}.json"
workspace = os.environ.get("WORKSPACE", "")
traj_file = os.environ.get("TRAJECTORY_FILE", f"/tmp/droid_traj_{issue_id}.jsonl")
diff_file = f"/tmp/droid_diff_{issue_id}.diff"

total_input = 0
total_output = 0
turns = 0
tool_calls = 0
has_edit = False
has_error = False
error_msg = ""
trajectory_events = []

try:
    with open(output_file, "rb") as f:
        raw = f.read()
    # Strip non-printable chars (droid emits BEL 0x07 before JSON)
    content = raw.decode("utf-8", errors="ignore").strip()
    content = "".join(c for c in content if c.isprintable() or c in "\n\r\t")

    if not content:
        has_error = True
        error_msg = "empty output"
    else:
        # Droid -o json may emit JSONL events or a single JSON object
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Collect every event for trajectory
            trajectory_events.append(evt)

            # Handle event-stream format
            evt_type = evt.get("type", "")

            if "usage" in evt:
                usage = evt["usage"] if isinstance(evt["usage"], dict) else {}
                total_input += usage.get("input_tokens", usage.get("prompt_tokens", 0))
                total_output += usage.get("output_tokens", usage.get("completion_tokens", 0))

            if evt_type in ("tool_call", "function_call", "command"):
                tool_calls += 1

            if evt_type in ("turn", "message", "response"):
                turns += 1

            # Detect edits from tool calls
            tool_name = evt.get("tool", evt.get("name", ""))
            if tool_name in ("write_file", "edit_file", "apply_diff", "create_file",
                             "str_replace", "insert", "patch"):
                has_edit = True

            # Detect edits from shell commands
            cmd = evt.get("command", evt.get("input", ""))
            if isinstance(cmd, str) and any(k in cmd for k in
                    ["sed ", "patch ", "cat >", "tee ", "echo ", "> "]):
                has_edit = True

            if evt_type == "error":
                has_error = True
                error_msg = evt.get("message", str(evt))[:300]

except FileNotFoundError:
    has_error = True
    error_msg = "no output file"
    try:
        with open(f"/tmp/droid_stderr_{issue_id}.log") as f:
            stderr = f.read().strip()[-300:]
            if stderr:
                error_msg = stderr
    except Exception:
        pass

# Write trajectory JSONL (raw droid events + diff)
try:
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    with open(traj_file, "w") as tf:
        for evt in trajectory_events:
            tf.write(json.dumps(evt) + "\n")
        # Append diff as final event
        if os.path.isfile(diff_file) and os.path.getsize(diff_file) > 0:
            with open(diff_file) as df:
                diff_content = df.read()
            tf.write(json.dumps({"type": "diff", "content": diff_content}) + "\n")
except Exception:
    pass

# Check if a diff was generated
diff_result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=workspace)
fix_generated = len(diff_result.stdout.strip()) > 0

if fix_generated:
    has_edit = True

# Ensure at least 1 turn if we got output
if turns == 0 and (fix_generated or tool_calls > 0):
    turns = 1

total_tokens = total_input + total_output
result = {
    "pass": False,  # adapter never decides pass; gold eval does
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
