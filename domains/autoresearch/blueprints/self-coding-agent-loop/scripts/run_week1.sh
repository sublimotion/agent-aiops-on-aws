#!/bin/bash
# Week 1 orchestrator: Gen0 OpenHands re-baseline → V1b_bootstrap → V1b_validate.
# Runs inside the p4de instance (invoked from user_data or manually via SSH).
# All outputs land in /mnt/nvme/self-coding-agent-loop/runs/week1/ with continuous S3 sync.

set -euo pipefail

WORK_DIR="/mnt/nvme/self-coding-agent-loop"
RUNS_DIR="$WORK_DIR/runs/week1"
BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
REPO="$WORK_DIR/agent-aiops-on-aws"
BLUEPRINT="$REPO/domains/autoresearch/blueprints/self-coding-agent-loop"

mkdir -p "$RUNS_DIR"/{gen0_rebaseline,v1b_bootstrap,v1b_validate,logs}
cd "$WORK_DIR"
source venv/bin/activate
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUNS_DIR/logs/week1.log"; }

snap() {
    aws s3 sync "$RUNS_DIR/" "$BUCKET/runs/week1/" --only-show-errors || true
}

# ========== Step 1: Gen0 OpenHands re-baseline ==========
log "Step 1/3: Gen0 OpenHands re-baseline (SWE-bench Lite 300)"
if [[ -f "$RUNS_DIR/gen0_rebaseline/summary.json" ]]; then
    log "  already done, skipping"
else
    python3 "$BLUEPRINT/scripts/gen0_rebaseline.py" \
        --adapter "$WORK_DIR/checkpoints/gen0" \
        --base-model Qwen/Qwen3.5-27B \
        --harness openhands \
        --harness-version v0.54 \
        --dataset swebench-lite-300 \
        --output-dir "$RUNS_DIR/gen0_rebaseline" \
        2>&1 | tee "$RUNS_DIR/logs/gen0_rebaseline.log"
    snap
fi

# ========== Step 2: V1b_bootstrap ==========
log "Step 2/3: V1b_bootstrap — FlywheelBootstrap on Qwen3.5-27B × OpenHands target distribution"
if [[ -f "$RUNS_DIR/v1b_bootstrap/rf_recalibrated.pkl" ]]; then
    log "  already done, skipping"
else
    python3 "$BLUEPRINT/scripts/v1b_bootstrap.py" \
        --pool "$BLUEPRINT/data/v1b_bootstrap_pool.jsonl" \
        --output-dir "$RUNS_DIR/v1b_bootstrap" \
        --n-labels 200 \
        2>&1 | tee "$RUNS_DIR/logs/v1b_bootstrap.log"
    snap
fi

# ========== Step 3: V1b_validate ==========
log "Step 3/3: V1b_validate — RF precision on held-out target-distribution calibration set"
python3 "$BLUEPRINT/scripts/v1b_validate.py" \
    --rf "$RUNS_DIR/v1b_bootstrap/rf_recalibrated.pkl" \
    --calibration "$BLUEPRINT/data/v1b_bootstrap_pool.jsonl" \
    --output-dir "$RUNS_DIR/v1b_validate" \
    2>&1 | tee "$RUNS_DIR/logs/v1b_validate.log"
snap

# ========== Summary ==========
log "Week 1 complete. Summary:"
for f in gen0_rebaseline/summary.json v1b_bootstrap/summary.json v1b_validate/summary.json; do
    if [[ -f "$RUNS_DIR/$f" ]]; then
        echo "--- $f ---"
        cat "$RUNS_DIR/$f"
    fi
done

snap
log "Final S3 sync done. Next: Arm A iteration 1 (scripts/run_arm_a_iter.sh N=1)"
