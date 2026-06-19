#!/bin/bash
# Arm A orchestrator (runs on p4de, after Week 1 V1b passes).
set -euo pipefail

WORK_DIR="/mnt/nvme/self-coding-agent-loop"
RUNS_DIR="$WORK_DIR/runs/arm_a"
BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
REPO="$WORK_DIR/agent-aiops-on-aws"
BLUEPRINT="$REPO/domains/autoresearch/blueprints/self-coding-agent-loop"

mkdir -p "$RUNS_DIR"
cd "$WORK_DIR"
source venv/bin/activate

echo "[$(date -u +%H:%M:%S)] Starting Arm A (iterative STaR, gold-filtered Nebius)"
python3 "$BLUEPRINT/scripts/train_arm_a.py" \
    --base-model Qwen/Qwen3.5-27B \
    --gen0-adapter "$WORK_DIR/checkpoints/gen0" \
    --train-set "$BLUEPRINT/data/arm_a_train.jsonl" \
    --output-dir "$RUNS_DIR" \
    --max-iterations 3 \
    --n-samples 4000 \
    2>&1 | tee "$RUNS_DIR/arm_a.log"

# Final S3 sync (term handler does this on reclaim; do it explicitly on clean exit too)
aws s3 sync "$RUNS_DIR/" "$BUCKET/runs/arm_a/" --only-show-errors
echo "[$(date -u +%H:%M:%S)] Arm A complete. Results in $RUNS_DIR"
