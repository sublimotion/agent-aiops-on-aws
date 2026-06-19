#!/bin/bash
# Ralph Drift Guard — companion Stop hook for the ralph-loop plugin.
#
# Contract (read this before editing):
#   • This hook NEVER blocks the stop. It only ever (a) appends telemetry and
#     (b) on a tripped rule, DELETES the ralph state file so the plugin's own
#     Stop hook chooses to exit on the next evaluation.
#   • Because it never emits decision:block, hook ordering vs the plugin hook
#     only affects promptness (worst case: loop stops one iteration late),
#     never correctness.
#   • Any internal failure exits 0 silently. A guard that crashes the Stop
#     event is worse than no guard.
#
# Signals (Tier 1 — mechanical, always on):
#   STALL        N consecutive iterations with an identical working-tree
#                fingerprint AND no new commit → busy but making no progress.
#   OSCILLATION  A working-tree fingerprint repeats after changing (A→B→A) ≥K
#                times → doing and undoing the same work.
#   REPETITION   The last assistant text block is byte-identical across R
#                iterations → the model is restating, not advancing.
#   BUDGET       Wall-clock since started_at exceeds max_minutes.
#
# Signals (Tier 2 — coherence, opt-in via judge.enabled):
#   COHERENCE    A fast model rates whether this iteration's stated claims are
#                consistent with the actual artifacts (git diff, files, tests).
#                Advisory: only trips on a SUSTAINED low score, never one read.
#
# State files (all .local, gitignored):
#   .claude/ralph-loop.local.md       owned by the plugin (we only delete it)
#   .claude/ralph-drift.config.json   optional tuning (defaults below if absent)
#   .claude/ralph-metrics.local.jsonl append-only telemetry, one row/iteration
#   .claude/ralph-drift.local.md       written when a rule trips (the verdict)

set -uo pipefail

STATE_FILE=".claude/ralph-loop.local.md"
CONFIG_FILE=".claude/ralph-drift.config.json"
METRICS_FILE=".claude/ralph-metrics.local.jsonl"
REPORT_FILE=".claude/ralph-drift.local.md"
JUDGE_SCRIPT=".claude/hooks/ralph-coherence-judge.sh"

# Never let an unexpected error block the stop event.
trap 'exit 0' ERR

# --- 0. Only engage when a ralph loop owned by THIS session is active --------
[[ -f "$STATE_FILE" ]] || exit 0
command -v jq >/dev/null 2>&1 || exit 0

HOOK_INPUT=$(cat)
HOOK_SESSION=$(echo "$HOOK_INPUT" | jq -r '.session_id // ""')
TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path // ""')

FM=$(sed -n '/^---$/,/^---$/{ /^---$/d; p; }' "$STATE_FILE")
STATE_SESSION=$(echo "$FM" | grep '^session_id:' | sed 's/session_id: *//' || true)
ITERATION=$(echo "$FM" | grep '^iteration:' | sed 's/iteration: *//' || echo "")
STARTED_AT=$(echo "$FM" | grep '^started_at:' | sed 's/started_at: *//; s/"//g' || echo "")

# Respect the plugin's session isolation: don't act on another session's loop.
if [[ -n "$STATE_SESSION" && -n "$HOOK_SESSION" && "$STATE_SESSION" != "$HOOK_SESSION" ]]; then
  exit 0
fi
[[ "$ITERATION" =~ ^[0-9]+$ ]] || exit 0

# --- 1. Load config (defaults are deliberately conservative) -----------------
cfg() { # cfg <jq-path> <default>
  local v=""
  [[ -f "$CONFIG_FILE" ]] && v=$(jq -r "$1 // empty" "$CONFIG_FILE" 2>/dev/null)
  [[ -n "$v" ]] && echo "$v" || echo "$2"
}
STALL_LIMIT=$(cfg '.stall_iterations' 3)
OSC_LIMIT=$(cfg '.oscillation_repeats' 2)
REPEAT_LIMIT=$(cfg '.repetition_iterations' 3)
MAX_MINUTES=$(cfg '.max_minutes' 0)            # 0 = no wall-clock cap
JUDGE_ENABLED=$(cfg '.judge.enabled' false)
JUDGE_MIN_SCORE=$(cfg '.judge.min_score' 4)    # 1-10 scale
JUDGE_SUSTAIN=$(cfg '.judge.sustain_iterations' 2)

# --- 2. Compute this iteration's signals -------------------------------------
# Working-tree fingerprint: porcelain status + diff shortstat + HEAD. Captures
# "did anything in the repo actually change" without being noisy about content.
HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
TREE_RAW=$( { git status --porcelain 2>/dev/null; git diff --shortstat 2>/dev/null; echo "$HEAD_SHA"; } )
TREE_FP=$(printf '%s' "$TREE_RAW" | shasum 2>/dev/null | awk '{print $1}')
CHANGED_FILES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

# Last assistant text block (same extraction the plugin uses), hashed.
LAST_TEXT=""
if [[ -f "$TRANSCRIPT_PATH" ]]; then
  LAST_TEXT=$(grep '"role":"assistant"' "$TRANSCRIPT_PATH" 2>/dev/null | tail -n 100 \
    | jq -rs 'map(.message.content[]? | select(.type=="text") | .text) | last // ""' 2>/dev/null || echo "")
fi
TEXT_FP=$(printf '%s' "$LAST_TEXT" | shasum 2>/dev/null | awk '{print $1}')

NOW_EPOCH=$(date -u +%s)
ELAPSED_MIN=0
if [[ -n "$STARTED_AT" ]]; then
  # GNU date and BSD date differ; try both, ignore failure.
  START_EPOCH=$(date -u -d "$STARTED_AT" +%s 2>/dev/null || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$STARTED_AT" +%s 2>/dev/null || echo "")
  [[ -n "$START_EPOCH" ]] && ELAPSED_MIN=$(( (NOW_EPOCH - START_EPOCH) / 60 ))
fi

# --- 3. Append telemetry row -------------------------------------------------
jq -nc \
  --argjson iter "$ITERATION" \
  --arg tree "$TREE_FP" \
  --arg text "$TEXT_FP" \
  --argjson files "$CHANGED_FILES" \
  --arg head "$HEAD_SHA" \
  --argjson elapsed "$ELAPSED_MIN" \
  --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{ts:$ts, iter:$iter, tree_fp:$tree, text_fp:$text, changed_files:$files, head:$head, elapsed_min:$elapsed}' \
  >> "$METRICS_FILE" 2>/dev/null || true

# --- 4. Evaluate Tier 1 rules over the telemetry history ---------------------
TRIP_REASON=""

# STALL: last STALL_LIMIT rows share one tree_fp and one HEAD (no commit either).
if [[ "$STALL_LIMIT" -gt 0 ]]; then
  RECENT_TREES=$(tail -n "$STALL_LIMIT" "$METRICS_FILE" 2>/dev/null | jq -r '.tree_fp' | sort -u | wc -l | tr -d ' ')
  RECENT_HEADS=$(tail -n "$STALL_LIMIT" "$METRICS_FILE" 2>/dev/null | jq -r '.head' | sort -u | wc -l | tr -d ' ')
  RECENT_COUNT=$(tail -n "$STALL_LIMIT" "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$RECENT_COUNT" -ge "$STALL_LIMIT" && "$RECENT_TREES" -eq 1 && "$RECENT_HEADS" -eq 1 ]]; then
    TRIP_REASON="STALL: working tree unchanged and no new commit across $STALL_LIMIT consecutive iterations (busy but no progress)."
  fi
fi

# OSCILLATION: a tree_fp recurs after changing, ≥ OSC_LIMIT times (A→B→A→B).
if [[ -z "$TRIP_REASON" && "$OSC_LIMIT" -gt 0 ]]; then
  MAX_RECUR=$(jq -rs '
    [ .[].tree_fp ] as $fps
    | ($fps | length) as $n
    | reduce range(0;$n) as $i ({}; .[$fps[$i]] += 1)
    | [ to_entries[] | select(.value > 1) | .value ] | (max // 1)
  ' "$METRICS_FILE" 2>/dev/null || echo 1)
  DISTINCT=$(jq -rs '[ .[].tree_fp ] | unique | length' "$METRICS_FILE" 2>/dev/null || echo 1)
  if [[ "$MAX_RECUR" -gt "$OSC_LIMIT" && "$DISTINCT" -gt 1 ]]; then
    TRIP_REASON="OSCILLATION: a prior working-tree state recurred $MAX_RECUR times (doing and undoing the same change)."
  fi
fi

# REPETITION: identical final assistant text across REPEAT_LIMIT iterations.
if [[ -z "$TRIP_REASON" && "$REPEAT_LIMIT" -gt 0 && -n "$LAST_TEXT" ]]; then
  RECENT_TEXTS=$(tail -n "$REPEAT_LIMIT" "$METRICS_FILE" 2>/dev/null | jq -r '.text_fp' | sort -u | wc -l | tr -d ' ')
  RECENT_TCOUNT=$(tail -n "$REPEAT_LIMIT" "$METRICS_FILE" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$RECENT_TCOUNT" -ge "$REPEAT_LIMIT" && "$RECENT_TEXTS" -eq 1 ]]; then
    TRIP_REASON="REPETITION: the model's final message was byte-identical across $REPEAT_LIMIT iterations (restating, not advancing)."
  fi
fi

# BUDGET: wall-clock cap.
if [[ -z "$TRIP_REASON" && "$MAX_MINUTES" -gt 0 && "$ELAPSED_MIN" -ge "$MAX_MINUTES" ]]; then
  TRIP_REASON="BUDGET: wall-clock $ELAPSED_MIN min reached the $MAX_MINUTES min cap."
fi

# --- 5. Tier 2: coherence judge (opt-in, advisory, sustained-only) -----------
if [[ -z "$TRIP_REASON" && "$JUDGE_ENABLED" == "true" && -x "$JUDGE_SCRIPT" ]]; then
  SCORE=$("$JUDGE_SCRIPT" "$TRANSCRIPT_PATH" 2>/dev/null || echo "")
  if [[ "$SCORE" =~ ^[0-9]+$ ]]; then
    # Record the score in a parallel file keyed by iteration.
    echo "{\"iter\":$ITERATION,\"coherence\":$SCORE}" >> "${METRICS_FILE%.jsonl}.coherence.jsonl" 2>/dev/null || true
    LOW_RUN=$(tail -n "$JUDGE_SUSTAIN" "${METRICS_FILE%.jsonl}.coherence.jsonl" 2>/dev/null \
      | jq -r --argjson min "$JUDGE_MIN_SCORE" 'select(.coherence < $min) | .iter' | wc -l | tr -d ' ')
    if [[ "$LOW_RUN" -ge "$JUDGE_SUSTAIN" ]]; then
      TRIP_REASON="COHERENCE: judge scored < $JUDGE_MIN_SCORE/10 for $JUDGE_SUSTAIN consecutive iterations (claims drifting from artifacts)."
    fi
  fi
fi

# --- 6. Act on a trip: write verdict, remove state file, surface a message ----
if [[ -n "$TRIP_REASON" ]]; then
  {
    echo "# Ralph Drift Guard — loop halted"
    echo ""
    echo "- **When:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- **Iteration:** $ITERATION"
    echo "- **Reason:** $TRIP_REASON"
    echo ""
    echo "Telemetry: \`$METRICS_FILE\`  •  Re-arm with \`/ralph-loop ...\`."
  } > "$REPORT_FILE" 2>/dev/null || true

  rm -f "$STATE_FILE" 2>/dev/null || true

  jq -n --arg msg "🛑 Ralph drift guard tripped — $TRIP_REASON Loop will end. See $REPORT_FILE" \
    '{systemMessage: $msg}'
  exit 0
fi

# Healthy iteration: stay silent so we never interfere with the plugin's block.
exit 0
