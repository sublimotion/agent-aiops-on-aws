#!/usr/bin/env bash
# Adapter: OpenAI Codex CLI with vLLM via Responses API proxy
# Codex uses its own tool set (shell, file read/write, apply_diff).
# Requires a streaming-to-non-streaming proxy for vLLM (tool call parsing
# only works in the non-streaming path of vLLM v0.16.0's Responses API).
#
# Env vars: WORKSPACE, ENDPOINT, MODEL, ISSUE_ID, PROBLEM_STATEMENT, TEST_CMD, REPO
# Output: JSON on last line

set -euo pipefail

# Configure Codex to use local vLLM via config.toml
CODEX_CONFIG_DIR="${HOME}/.codex"
mkdir -p "$CODEX_CONFIG_DIR"

cat > "$CODEX_CONFIG_DIR/config.toml" << CFGEOF
model = "${MODEL}"
openai_base_url = "${ENDPOINT}/v1"
CFGEOF

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

PROMPT="Fix this bug in the repository.

${PROBLEM_STATEMENT}

IMPORTANT: Read the relevant source file, then edit it to fix the bug. Do not read more than 3 files before making your edit. Be precise — change only what is needed."

OUTPUT_FILE="/tmp/codex_events_${ISSUE_ID}.jsonl"
STDERR_FILE="/tmp/codex_stderr_${ISSUE_ID}.log"

# Run Codex in non-interactive exec mode with JSON output
# --dangerously-bypass-approvals-and-sandbox: no prompts, no sandbox (harness workspaces are disposable)
# --json: JSONL events on stdout
timeout 540 codex exec \
    -C "$WORKSPACE" \
    -m "$MODEL" \
    --dangerously-bypass-approvals-and-sandbox \
    --json \
    "$PROMPT" > "$OUTPUT_FILE" 2>"$STDERR_FILE" || true

# Parse Codex CLI JSONL events (type-field format, not camelCase keyed)
python3 << 'PYEOF'
import json, os, subprocess

issue_id = os.environ.get("ISSUE_ID", "unknown")
events_file = f"/tmp/codex_events_{issue_id}.jsonl"
workspace = os.environ.get("WORKSPACE", "")

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

            if evt_type == "turn.completed":
                usage = evt.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                turns += 1

            elif evt_type == "item.started":
                item = evt.get("item", {})
                item_type = item.get("type", "")
                if item_type in ("command_execution", "function_call"):
                    tool_calls += 1

            elif evt_type == "item.completed":
                item = evt.get("item", {})
                item_type = item.get("type", "")
                if item_type == "command_execution":
                    cmd = item.get("command", "")
                    if any(k in cmd for k in ["sed ", "patch ", "cat >", "tee ", "echo ", "> ", "write_file", "apply_diff"]):
                        has_edit = True
                elif item_type in ("file_write", "apply_diff"):
                    has_edit = True
                elif item_type == "function_call":
                    name = item.get("name", "")
                    if name in ("write_file", "apply_diff", "create_file"):
                        has_edit = True

            elif evt_type == "turn.failed":
                has_error = True
                error_data = evt.get("error", {})
                error_msg = error_data.get("message", str(error_data))[:300]

            elif evt_type == "error":
                msg = evt.get("message", "")
                # Ignore benign model metadata warnings
                if "metadata" not in msg.lower():
                    has_error = True
                    error_msg = msg[:300]

except FileNotFoundError:
    has_error = True
    error_msg = "no events file"
    try:
        with open(f"/tmp/codex_stderr_{issue_id}.log") as f:
            stderr = f.read().strip()[-300:]
            if stderr:
                error_msg = stderr
    except:
        pass

# Check if a diff was generated
diff_result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=workspace)
fix_generated = len(diff_result.stdout.strip()) > 0

if fix_generated:
    has_edit = True

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
