#!/usr/bin/env bash
# Sync experiment data from g7e and inventory what we have.
#
# Usage:
#   ./sync_and_recover.sh              # Inventory local data only
#   ./sync_and_recover.sh --sync       # Sync from g7e + inventory
#   ./sync_and_recover.sh --sync --recover-diffs  # Also re-extract patch diffs
#
# Prerequisites:
#   SSH key at ~/.ssh/g7e-bench.pem
#   g7e instance running at the IP in G7E_HOST

set -euo pipefail

G7E_HOST="${G7E_HOST:-35.94.217.100}"
G7E_KEY="${G7E_KEY:-$HOME/.ssh/g7e-bench.pem}"
G7E_USER="${G7E_USER:-ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HARNESS_RESULTS="$SCRIPT_DIR/../../agent-harness/results"
SWARM_RESULTS="$SCRIPT_DIR/../results"

SYNC=false
RECOVER_DIFFS=false

for arg in "$@"; do
    case "$arg" in
        --sync) SYNC=true ;;
        --recover-diffs) RECOVER_DIFFS=true ;;
        --help|-h)
            echo "Usage: $0 [--sync] [--recover-diffs]"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Sync from g7e
# ---------------------------------------------------------------------------

if $SYNC; then
    echo "=== Syncing from g7e ($G7E_HOST) ==="

    # Test connectivity
    if ! ssh -i "$G7E_KEY" -o ConnectTimeout=5 "$G7E_USER@$G7E_HOST" "echo ok" &>/dev/null; then
        echo "ERROR: Cannot connect to g7e at $G7E_HOST"
        echo "  Check: is the instance running? Is the IP correct?"
        exit 1
    fi

    # Sync agent-harness results
    echo "--- Syncing agent-harness results ---"
    mkdir -p "$HARNESS_RESULTS"
    rsync -avz --progress \
        -e "ssh -i $G7E_KEY" \
        "$G7E_USER@$G7E_HOST:/mnt/nvme/agent-harness/results/" \
        "$HARNESS_RESULTS/" \
        2>/dev/null || echo "WARN: agent-harness sync failed (may not exist on remote)"

    # Sync agent-harness trajectories (if they exist)
    rsync -avz --progress \
        -e "ssh -i $G7E_KEY" \
        "$G7E_USER@$G7E_HOST:/mnt/nvme/agent-harness/results/trajectories/" \
        "$HARNESS_RESULTS/trajectories/" \
        2>/dev/null || echo "INFO: No trajectories dir on remote yet"

    # Sync swarm results
    echo "--- Syncing swarm results ---"
    mkdir -p "$SWARM_RESULTS"
    rsync -avz --progress \
        -e "ssh -i $G7E_KEY" \
        "$G7E_USER@$G7E_HOST:/mnt/nvme/agent-harness/results/swarm/" \
        "$SWARM_RESULTS/" \
        2>/dev/null || echo "WARN: swarm results sync failed (may not exist on remote)"

    # Sync swarm trajectories
    rsync -avz --progress \
        -e "ssh -i $G7E_KEY" \
        "$G7E_USER@$G7E_HOST:/mnt/nvme/agent-harness/results/swarm/trajectories/" \
        "$SWARM_RESULTS/trajectories/" \
        2>/dev/null || echo "INFO: No swarm trajectories on remote yet"

    # Sync logs for debugging
    echo "--- Syncing logs ---"
    mkdir -p "$SWARM_RESULTS/logs"
    rsync -avz --progress \
        -e "ssh -i $G7E_KEY" \
        "$G7E_USER@$G7E_HOST:/mnt/nvme/agent-harness/swarm_*.log" \
        "$SWARM_RESULTS/logs/" \
        2>/dev/null || echo "INFO: No swarm logs on remote"

    echo ""
fi

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

echo "=== DATA INVENTORY ==="
echo ""

echo "--- Agent Harness Results ($HARNESS_RESULTS) ---"
if [ -d "$HARNESS_RESULTS" ]; then
    total_lines=0
    total_files=0
    for f in "$HARNESS_RESULTS"/*.jsonl; do
        [ -f "$f" ] || continue
        lines=$(wc -l < "$f")
        total_lines=$((total_lines + lines))
        total_files=$((total_files + 1))
        # Check for patch_diff field
        has_diff=$(head -1 "$f" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('YES' if d.get('patch_diff') else 'NO')" 2>/dev/null || echo "ERR")
        printf "  %-45s %3d lines  patch_diff: %s\n" "$(basename "$f")" "$lines" "$has_diff"
    done
    echo "  TOTAL: $total_files files, $total_lines lines"
else
    echo "  (directory not found)"
fi

echo ""
echo "--- Trajectories ($HARNESS_RESULTS/trajectories) ---"
if [ -d "$HARNESS_RESULTS/trajectories" ]; then
    traj_count=$(find "$HARNESS_RESULTS/trajectories" -name "*.json" | wc -l)
    traj_size=$(du -sh "$HARNESS_RESULTS/trajectories" 2>/dev/null | cut -f1)
    echo "  $traj_count trajectory files, $traj_size total"
else
    echo "  (not yet captured — run instrumented eval to generate)"
fi

echo ""
echo "--- Swarm Results ($SWARM_RESULTS) ---"
if [ -d "$SWARM_RESULTS" ]; then
    for f in "$SWARM_RESULTS"/*.jsonl; do
        [ -f "$f" ] || continue
        lines=$(wc -l < "$f")
        has_diff=$(head -1 "$f" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('YES' if d.get('patch_diff') else 'NO')" 2>/dev/null || echo "ERR")
        printf "  %-55s %3d lines  patch_diff: %s\n" "$(basename "$f")" "$lines" "$has_diff"
    done
else
    echo "  (directory not found — swarm experiment not yet synced)"
fi

echo ""
echo "--- Swarm Trajectories ($SWARM_RESULTS/trajectories) ---"
if [ -d "$SWARM_RESULTS/trajectories" ]; then
    traj_count=$(find "$SWARM_RESULTS/trajectories" -name "*.json" | wc -l)
    traj_size=$(du -sh "$SWARM_RESULTS/trajectories" 2>/dev/null | cut -f1)
    echo "  $traj_count trajectory files, $traj_size total"
else
    echo "  (not yet captured)"
fi

# ---------------------------------------------------------------------------
# Data gap analysis
# ---------------------------------------------------------------------------

echo ""
echo "=== DATA GAPS ==="

# Check which Phase 2b harnesses have results
echo ""
echo "Phase 2b harnesses:"
for harness in sera langgraph aider ohmypi piagent opencode claude_code codex deepagents; do
    f="$HARNESS_RESULTS/phase2b_${harness}.jsonl"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $harness: $lines results"
    else
        # Check phase2_ prefix too
        f2="$HARNESS_RESULTS/phase2_${harness}.jsonl"
        if [ -f "$f2" ]; then
            lines=$(wc -l < "$f2")
            echo "  $harness: $lines results (phase2 format)"
        else
            echo "  $harness: MISSING"
        fi
    fi
done

# Check eval files
echo ""
echo "Gold test evaluation files:"
for harness in sera langgraph ohmypi piagent opencode claude_code codex; do
    f="$HARNESS_RESULTS/eval_${harness}.jsonl"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $harness: $lines evaluations"
    else
        echo "  $harness: MISSING"
    fi
done

# Count total labeled examples for verifier training
echo ""
echo "=== VERIFIER TRAINING DATA SUMMARY ==="
total_pos=0
total_neg=0
for f in "$HARNESS_RESULTS"/*.jsonl "$SWARM_RESULTS"/*.jsonl; do
    [ -f "$f" ] || continue
    pos=$(python3 -c "
import sys, json
count = 0
for line in open('$f'):
    d = json.loads(line)
    if d.get('tests_pass') or d.get('pass'):
        count += 1
print(count)
" 2>/dev/null || echo 0)
    neg=$(python3 -c "
import sys, json
count = 0
for line in open('$f'):
    d = json.loads(line)
    fg = d.get('fix_generated', False)
    tp = d.get('tests_pass', d.get('pass', False))
    if fg and not tp:
        count += 1
print(count)
" 2>/dev/null || echo 0)
    total_pos=$((total_pos + pos))
    total_neg=$((total_neg + neg))
done
echo "  Positive (pass): $total_pos"
echo "  Negative (fix but fail): $total_neg"
echo "  Total labeled: $((total_pos + total_neg))"
echo ""
echo "  Has patch_diff: $(grep -rl 'patch_diff' "$HARNESS_RESULTS" "$SWARM_RESULTS" 2>/dev/null | wc -l) files"
echo "  Has transcript: $(find "$HARNESS_RESULTS/trajectories" "$SWARM_RESULTS/trajectories" -name "*.json" 2>/dev/null | wc -l) files"

# ---------------------------------------------------------------------------
# Recover patch diffs (optional)
# ---------------------------------------------------------------------------

if $RECOVER_DIFFS; then
    echo ""
    echo "=== RECOVERING PATCH DIFFS ==="
    echo "Checking if workspaces still exist on g7e..."

    if $SYNC; then
        workspace_count=$(ssh -i "$G7E_KEY" "$G7E_USER@$G7E_HOST" \
            "ls -d /mnt/nvme/sera-workspaces/*/ /mnt/nvme/swarm-workspaces/*/ 2>/dev/null | wc -l" \
            2>/dev/null || echo 0)
        echo "  Found $workspace_count workspace directories on g7e"

        if [ "$workspace_count" -gt 0 ]; then
            echo "  To recover diffs, run on g7e:"
            echo '    for ws in /mnt/nvme/sera-workspaces/*/; do'
            echo '      name=$(basename "$ws")'
            echo '      diff=$(cd "$ws" && git diff 2>/dev/null)'
            echo '      if [ -n "$diff" ]; then'
            echo '        echo "$diff" > "/mnt/nvme/agent-harness/results/diffs/${name}.diff"'
            echo '      fi'
            echo '    done'
        fi
    else
        echo "  (skipped — use --sync to check remote)"
    fi
fi

echo ""
echo "=== DONE ==="
