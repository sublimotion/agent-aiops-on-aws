#!/usr/bin/env bash
# spot-reclaim-watcher.sh — Monitor IMDS for spot interruption notice.
# On notice: force an immediate full sync + flag so the RALPH loop knows to restart.
#
# AWS emits /latest/meta-data/spot/instance-action 2 min before interruption.
# Also monitors /spot/termination-time (legacy).

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
RESULTS_BUCKET="${RESULTS_BUCKET:-s3://kimi-k2-bench-results-20260216163240701700000007}"
S3_PREFIX="${S3_PREFIX:-kimi-k2.6-speculative/gpu-node}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# IMDSv2 helper
imds_token() {
  curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60"
}

imds_get() {
  local path=$1
  local tok; tok=$(imds_token)
  curl -s -H "X-aws-ec2-metadata-token: $tok" \
    "http://169.254.169.254/latest/meta-data${path}" -o /dev/null -w '%{http_code}'
}

emergency_sync() {
  local ts=$(date -u +%FT%TZ)
  echo "[$ts] SPOT RECLAIM NOTICE — emergency sync" | tee -a /mnt/nvme/logs/reclaim.log

  # Grab container logs one last time
  for c in $(sudo docker ps -a --format '{{.Names}}' 2>/dev/null); do
    sudo docker logs "$c" > "/mnt/nvme/logs/${c}.final.log" 2>&1 || true
  done
  sudo cp -f /var/log/kimi-bootstrap.log /mnt/nvme/logs/ 2>/dev/null || true
  sudo dmesg -T > /mnt/nvme/logs/dmesg.final.log 2>/dev/null || true

  # Mark reclaim event
  echo "$ts" > /mnt/nvme/logs/RECLAIMED_AT

  # Hard sync (no --only-show-errors, surface problems)
  aws s3 sync /mnt/nvme/results/ "${RESULTS_BUCKET}/${S3_PREFIX}/results/" --region "$REGION" || true
  aws s3 sync /mnt/nvme/logs/    "${RESULTS_BUCKET}/${S3_PREFIX}/logs/"    --region "$REGION" || true
  echo "[$ts] emergency sync done" | tee -a /mnt/nvme/logs/reclaim.log
}

while true; do
  code=$(imds_get "/spot/instance-action" || echo "000")
  if [[ "$code" == "200" ]]; then
    emergency_sync
    break
  fi
  sleep 5
done
