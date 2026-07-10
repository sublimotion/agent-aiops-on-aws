#!/bin/bash
# Lineage Drift Check — Stop hook that surfaces "you edited X, forgot to update Y".
#
# Runs the trace-effectiveness lineage tool on THIS session's transcript at stop
# time and, if any consistency-drift flags are found, shows them to the user as a
# systemMessage. Never blocks the stop; any failure exits 0 silently.
#
# Drift = a file was mutated, references another file that changed LATER in the
# session, and was never revisited. See domains/ai-infra/blueprints/trace-effectiveness/.

set -uo pipefail
trap 'exit 0' ERR

command -v jq >/dev/null 2>&1 || exit 0
command -v python3 >/dev/null 2>&1 || exit 0

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
TOOL="$ROOT/domains/ai-infra/blueprints/trace-effectiveness/lineage.py"
[ -f "$TOOL" ] || exit 0

HOOK_INPUT=$(cat)
TRANSCRIPT=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

TMP=$(mktemp) || exit 0
python3 "$TOOL" --session "$TRANSCRIPT" --json "$TMP" >/dev/null 2>&1 || { rm -f "$TMP"; exit 0; }

# Build a compact "B → (references) A" summary from the drift flags.
SUMMARY=$(jq -r '
  (.drift_flags // []) as $all
  | ($all | length) as $n
  | if $n == 0 then empty
    else
      "Consistency drift — \($n) file(s) may be stale (edited something they reference, never revisited):\n" +
      ( [ $all[:6][]
          | "  • " + (.path | sub(".*/";"")) + " → references "
            + ( [ .references_changed[] | sub(".*/";"") ] | .[:3] | join(", ") )
            + " (changed after)"
        ] | join("\n") )
      + (if $n > 6 then "\n  … and \($n - 6) more (run lineage.py for the full list)" else "" end)
    end
' "$TMP" 2>/dev/null)
rm -f "$TMP"

[ -n "$SUMMARY" ] || exit 0

jq -cn --arg msg "$SUMMARY" '{systemMessage: $msg, suppressOutput: true}'
exit 0
