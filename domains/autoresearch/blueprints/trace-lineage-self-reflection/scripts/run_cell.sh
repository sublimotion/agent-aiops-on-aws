#!/bin/bash
# Run ONE experiment cell: generate a seeded task, run a headless Claude Code
# agent under a given ARM's Stop-hook, grade with the mechanical oracle, emit a
# result row. Local harness (matches what agent-runner does headless), so the
# pilot can run on a workstation before the cluster fan-out.
#
# Usage:
#   run_cell.sh --arm reflect-informational --seed 0 --k 3 --tier short \
#               [--model claude-sonnet-4-6] [--max-turns 60] [--out results/]
#
# ARM in {control, advisory, reflect-informational, reflect-mandatory}.
# Requires: claude CLI on PATH; python3; jq.
set -uo pipefail

ARM="advisory"; SEED=0; K=3; TIER="short"
MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}"; MAX_TURNS=60
BP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$BP_DIR/results"

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --max-turns) MAX_TURNS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

command -v claude >/dev/null 2>&1 || { echo "claude CLI not on PATH" >&2; exit 3; }
mkdir -p "$OUT"
RUN_ID="${ARM}-s${SEED}-k${K}-${TIER}"
TASK_DIR="$(mktemp -d "/tmp/dl-${RUN_ID}-XXXX")"

# 1. Seed the task
python3 "$BP_DIR/scripts/gen_task.py" --out "$TASK_DIR" --seed "$SEED" --k "$K" >/dev/null
PROMPT=$(python3 -c "import json;print(json.load(open('$TASK_DIR/task.json'))['prompt'])")

# 2. Tier: long tier prepends filler work to grow context toward compaction.
if [ "$TIER" = "long" ]; then
  PROMPT="First, read every file in this repo and briefly summarize each one to build context. Then: $PROMPT"
  MAX_TURNS=$((MAX_TURNS > 120 ? MAX_TURNS : 120))
fi

# 3. Write a per-run settings.json wiring THIS arm's Stop hook into the task repo.
mkdir -p "$TASK_DIR/.claude"
HOOK="$BP_DIR/hooks/reflect-hook.sh"
cat > "$TASK_DIR/.claude/settings.json" <<JSON
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command",
        "command": "DRIFT_ARM=$ARM DRIFT_TOOL='$BP_DIR/../../../ai-infra/blueprints/trace-effectiveness/lineage.py' bash '$HOOK'",
        "timeout": 45 } ] }
    ]
  }
}
JSON

LOG="$OUT/run-${RUN_ID}.log"
# 4. Run the agent headless from inside the task repo (stream-json → log).
( cd "$TASK_DIR" && ANTHROPIC_MODEL="$MODEL" \
    claude -p "$PROMPT" --output-format stream-json --verbose \
    --max-turns "$MAX_TURNS" \
    --allowedTools Bash,Read,Write,Edit,Glob,Grep >"$LOG" 2>&1 </dev/null ) || true

# 5. Grade with the mechanical oracle.
VERDICT="$OUT/verdict-${RUN_ID}.json"
python3 "$BP_DIR/scripts/grade.py" --task "$TASK_DIR" --json "$VERDICT" >/dev/null

# 6. Emit a result row (arm/tier/k joined to the grade + turn count from the log).
TURNS=$(grep -c '"type":"assistant"' "$LOG" 2>/dev/null || echo 0)
python3 - "$VERDICT" "$ARM" "$TIER" "$MODEL" "$TURNS" "$LOG" <<'PY' >> "$OUT/results.jsonl"
import json, sys
v = json.load(open(sys.argv[1]))
row = {"arm": sys.argv[2], "tier": sys.argv[3], "model": sys.argv[4],
       "assistant_turns": int(sys.argv[5]), "log": sys.argv[6],
       **{k: v[k] for k in ("name","seed","k","consistency_completion",
                            "updated_sites","stale_sites","acted_on_stale")}}
print(json.dumps(row))
PY
echo "[$RUN_ID] completion=$(python3 -c "import json;print(json.load(open('$VERDICT'))['consistency_completion'])") turns=$TURNS  (task kept at $TASK_DIR)"
