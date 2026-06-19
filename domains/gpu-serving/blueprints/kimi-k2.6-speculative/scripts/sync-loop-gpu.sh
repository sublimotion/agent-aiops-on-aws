#!/usr/bin/env bash
# sync-loop-gpu.sh — Periodic S3 sync of on-node artifacts for spot-reclaim safety.
# Run on the GPU node in the background:  nohup ./sync-loop-gpu.sh > /var/log/sync-loop.log 2>&1 &
#
# Syncs:
#   /mnt/nvme/results/        → s3://.../kimi-k2.6-speculative/gpu-node/results/
#   /mnt/nvme/logs/           → s3://.../kimi-k2.6-speculative/gpu-node/logs/
#   /var/log/kimi-bootstrap.log + container logs → s3://.../kimi-k2.6-speculative/gpu-node/boot/
#
# Interval: 60s default (override via SYNC_INTERVAL env)

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
RESULTS_BUCKET="${RESULTS_BUCKET:-s3://kimi-k2-bench-results-20260216163240701700000007}"
S3_PREFIX="${S3_PREFIX:-kimi-k2.6-speculative/gpu-node}"
INTERVAL="${SYNC_INTERVAL:-60}"

mkdir -p /mnt/nvme/{results,logs}

while true; do
  ts=$(date -u +%FT%TZ)

  # Per-container logs (best effort; skip stopped containers)
  for c in $(sudo docker ps --format '{{.Names}}' 2>/dev/null); do
    sudo docker logs --tail 2000 "$c" > "/mnt/nvme/logs/${c}.log" 2>&1 || true
  done

  # Bootstrap + kernel logs
  sudo cp -f /var/log/kimi-bootstrap.log /mnt/nvme/logs/ 2>/dev/null || true
  sudo dmesg -T | tail -n 2000 > /mnt/nvme/logs/dmesg.log 2>/dev/null || true
  nvidia-smi > /mnt/nvme/logs/nvidia-smi.log 2>&1 || true

  # nvidia-smi dmon last 60s sample
  nvidia-smi dmon -s u -c 5 >> /mnt/nvme/logs/nvidia-dmon.log 2>/dev/null || true

  # S3 sync
  aws s3 sync /mnt/nvme/results/ "${RESULTS_BUCKET}/${S3_PREFIX}/results/" \
    --region "$REGION" --only-show-errors --exclude '*/tmp/*' 2>>/mnt/nvme/logs/sync-errors.log || true
  aws s3 sync /mnt/nvme/logs/ "${RESULTS_BUCKET}/${S3_PREFIX}/logs/" \
    --region "$REGION" --only-show-errors 2>>/mnt/nvme/logs/sync-errors.log || true

  echo "[$ts] synced results+logs"
  sleep "$INTERVAL"
done
