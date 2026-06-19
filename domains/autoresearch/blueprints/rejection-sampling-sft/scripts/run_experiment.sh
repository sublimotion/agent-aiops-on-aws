#!/bin/bash
# Spot-safe experiment orchestrator.
# Syncs all data/results/checkpoints to S3 periodically.
# Can resume from any point after spot termination.
#
# Usage:
#   bash run_experiment.sh                # run full pipeline
#   bash run_experiment.sh --phase 2      # resume from training phase
#   bash run_experiment.sh --config B     # run single training config

set -euo pipefail

WORK_DIR="/mnt/nvme/rejection-sampling-sft"
VENV="$WORK_DIR/venv"
S3_BUCKET="s3://agent-aiops-artifacts"
S3_PREFIX="rejection-sampling-sft"
SCRIPTS_DIR="$WORK_DIR/scripts"
N_TRACES=20000
SPLIT="SWE_Rebench"
PHASE="${1:---all}"
CONFIG="${2:-all}"

source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

# S3 sync helper
sync_all() {
    echo "[$(date +%H:%M:%S)] Syncing to S3..."
    aws s3 sync "$WORK_DIR/data/" "$S3_BUCKET/$S3_PREFIX/data/" --quiet 2>/dev/null || true
    aws s3 sync "$WORK_DIR/results/" "$S3_BUCKET/$S3_PREFIX/results/" --quiet 2>/dev/null || true
    aws s3 sync "$WORK_DIR/models/" "$S3_BUCKET/$S3_PREFIX/models/" --quiet 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] S3 sync complete"
}

# Periodic sync in background (every 10 minutes)
start_sync_daemon() {
    while true; do
        sleep 600
        sync_all
    done &
    SYNC_PID=$!
    echo "Started S3 sync daemon (PID=$SYNC_PID, every 10 min)"
    trap "kill $SYNC_PID 2>/dev/null; sync_all; echo 'Final sync done'" EXIT
}

# Try to restore from S3 if local data is missing
restore_from_s3() {
    echo "Checking for existing data on S3..."
    aws s3 sync "$S3_BUCKET/$S3_PREFIX/data/" "$WORK_DIR/data/" --quiet 2>/dev/null || true
    aws s3 sync "$S3_BUCKET/$S3_PREFIX/results/" "$WORK_DIR/results/" --quiet 2>/dev/null || true
    echo "S3 restore complete"
}

# ============================================================
# PHASE 1: Filter trajectories
# ============================================================
run_phase1() {
    echo "============================================================"
    echo "PHASE 1: FILTER TRAJECTORIES"
    echo "============================================================"

    SCORES_FILE="$WORK_DIR/data/scores_${SPLIT}_${N_TRACES}.jsonl"
    if [ -f "$SCORES_FILE" ]; then
        N_LINES=$(wc -l < "$SCORES_FILE")
        echo "Scores file exists ($N_LINES lines). Skipping Phase 1."
        return
    fi

    python3 "$SCRIPTS_DIR/filter_trajectories.py" \
        --n-traces "$N_TRACES" \
        --split "$SPLIT" \
        --rf-only

    sync_all
    echo "Phase 1 complete."
}

# ============================================================
# PHASE 2: SFT Training
# ============================================================
run_training() {
    local config=$1
    echo "============================================================"
    echo "PHASE 2: TRAINING CONFIG $config"
    echo "============================================================"

    RESULTS_FILE="$WORK_DIR/results/training_${config}.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Training results exist for config $config. Skipping."
        return
    fi

    # Check for existing checkpoint to resume from
    RESUME_ARG=""
    CKPT_DIR="$WORK_DIR/models/run_${config}_$(echo $MODEL | tr '/' '_')"
    if [ -d "$CKPT_DIR" ]; then
        LATEST_CKPT=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Resuming from $LATEST_CKPT"
            RESUME_ARG="--resume-from $LATEST_CKPT"
        fi
    fi

    python3 "$SCRIPTS_DIR/train_sft.py" \
        --config "$config" \
        --model "$MODEL" \
        --split "$SPLIT" \
        --n-traces "$N_TRACES" \
        --lora-r 16 \
        --lora-alpha 32 \
        --lr 2e-5 \
        --epochs 1 \
        --batch-size 2 \
        --grad-accum 8 \
        --max-seq-length 8192 \
        --save-steps 500 \
        --s3-sync \
        $RESUME_ARG

    sync_all
    echo "Training config $config complete."
}

# ============================================================
# MAIN
# ============================================================

MODEL="Qwen/Qwen3.5-Coder-32B-Instruct"
mkdir -p "$WORK_DIR"/{data,results,models,scripts}

restore_from_s3
start_sync_daemon

case "$PHASE" in
    --all)
        run_phase1
        for cfg in none a b c d; do
            run_training "$cfg"
        done
        ;;
    --phase)
        if [ "$CONFIG" = "1" ]; then
            run_phase1
        elif [ "$CONFIG" = "2" ]; then
            for cfg in none a b c d; do
                run_training "$cfg"
            done
        fi
        ;;
    --config)
        run_training "$CONFIG"
        ;;
    *)
        echo "Usage: $0 [--all | --phase 1|2 | --config none|a|b|c|d]"
        exit 1
        ;;
esac

echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
sync_all
