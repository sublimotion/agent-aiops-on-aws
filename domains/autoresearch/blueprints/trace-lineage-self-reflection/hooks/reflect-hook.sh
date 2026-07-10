#!/bin/bash
# Reflect Stop-hook for the trace-lineage experiment — the reflect-* arms.
#
# Runs the drift detector on THIS session's trace and, if drift is found,
# returns decision:block so the agent KEEPS WORKING with the flags injected.
# Framing depends on $DRIFT_ARM:
#   reflect-informational : neutral "review whether these are still consistent"
#   reflect-mandatory     : coercive "you MUST reconcile each before finishing"
#   advisory              : systemMessage only, NO block (surface, don't act)
#   control               : do nothing (hook effectively off)
#
# Never crashes the stop: any internal failure exits 0 silently. To avoid an
# infinite block loop, blocks at most $DRIFT_MAX_BLOCKS times per session
# (tracked in a per-session counter file).

set -uo pipefail
trap 'exit 0' ERR

ARM="${DRIFT_ARM:-advisory}"
[ "$ARM" = "control" ] && exit 0

command -v jq >/dev/null 2>&1 || exit 0
command -v python3 >/dev/null 2>&1 || exit 0

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
# lineage.py lives in the ai-infra blueprint; allow override for portability.
TOOL="${DRIFT_TOOL:-$ROOT/domains/ai-infra/blueprints/trace-effectiveness/lineage.py}"
[ -f "$TOOL" ] || exit 0

HOOK_INPUT=$(cat)
TRANSCRIPT=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')
SESSION=$(echo "$HOOK_INPUT" | jq -r '.session_id // "nosess"')
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

# Block budget — prevent an unbounded reflect loop.
MAX_BLOCKS="${DRIFT_MAX_BLOCKS:-3}"
CNT_FILE="/tmp/drift-blocks-${SESSION}.cnt"
COUNT=$(cat "$CNT_FILE" 2>/dev/null || echo 0)

TMP=$(mktemp) || exit 0
# Local sessions are projects-JSONL; on agent-runner the transcript is stream-json.
FMT_FLAG=""
case "$TRANSCRIPT" in
  *.log|*run.log) FMT_FLAG="--stream-json" ;;
esac
# This experiment couples on shared VALUES (config-propagation tasks). Reference-
# drift ("edited A before B where B mentions A") is noisy here and false-positives
# on completed tasks — use value-drift only. Override with DRIFT_MODE=all.
DRIFT_FLAGS="--no-reference-drift"
[ "${DRIFT_MODE:-value}" = "all" ] && DRIFT_FLAGS=""
python3 "$TOOL" --session "$TRANSCRIPT" $FMT_FLAG $DRIFT_FLAGS --json "$TMP" >/dev/null 2>&1 || { rm -f "$TMP"; exit 0; }

NFLAGS=$(jq -r '(.drift_flags // []) | length' "$TMP" 2>/dev/null || echo 0)
if [ "${NFLAGS:-0}" -eq 0 ]; then rm -f "$TMP"; exit 0; fi

# Build the flag list (top 8; the flag is a POINTER, never the correct value —
# reward-hacking guard: it names which file drifted, not what to write).
FLAGS=$(jq -r '
  (.drift_flags // [])[:8][]
  | "  - " + (.path | sub(".*/";"")) + " references "
    + ( [ .references_changed[] | sub(".*/";"") ] | .[:3] | join(", ") )
    + " (changed after; not revisited)"
' "$TMP" 2>/dev/null)
rm -f "$TMP"

if [ "$ARM" = "advisory" ]; then
  # Surface only — do not re-enter the loop.
  jq -cn --arg m "Consistency drift detected (advisory):"$'\n'"$FLAGS" \
     '{systemMessage:$m, suppressOutput:true}'
  exit 0
fi

# reflect arms: block (keep working) with framed reason, until block budget spent.
if [ "${COUNT:-0}" -ge "$MAX_BLOCKS" ]; then exit 0; fi
echo $((COUNT + 1)) > "$CNT_FILE"

if [ "$ARM" = "reflect-mandatory" ]; then
  REASON="You MUST reconcile each of the following before finishing. Every listed file references something that changed and was not revisited — you are required to open each one and fix it or state explicitly why it is already correct:"$'\n'"$FLAGS"
else  # reflect-informational (default reflect framing)
  REASON="Before finishing, some files may be worth a look: each references something that changed later in this session and was not revisited. Review whether they are still consistent; update any that need it, or continue if they are fine:"$'\n'"$FLAGS"
fi

jq -cn --arg r "$REASON" '{decision:"block", reason:$r}'
exit 0
