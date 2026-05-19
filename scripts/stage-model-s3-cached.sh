#!/usr/bin/env bash
# Reusable model staging utility — S3-cached HuggingFace download.
#
# Pattern (extracted from qwen3-next/scripts/stage-model.sh + deepseek-v4-flash):
#   1. Check S3 cache; if hit, sync down (~5x faster than HF)
#   2. On cache miss, download from HF via hf_transfer
#   3. After successful HF download, mirror to S3 for spot-reclaim resilience
#
# This script generates the K8s Job manifest for any blueprint. Source it in your
# blueprint's stage-model.sh, OR run directly:
#
#   ./scripts/stage-model-s3-cached.sh <hf-model-id> <s3-bucket> <local-path> <region>
#
# Example:
#   ./scripts/stage-model-s3-cached.sh \
#     deepseek-ai/DeepSeek-V4-Flash \
#     qn-sglang-models-20260303161715850900000007 \
#     /mnt/nvme/models/DeepSeek-V4-Flash \
#     us-west-2
#
# Prerequisites on the K8s side:
#   - GPU node IAM role has s3:GetObject, s3:PutObject, s3:ListBucket on the bucket
#   - Bucket exists in the same region as the cluster
#   - Pod has hostPath mount at /mnt/nvme
set -euo pipefail

MODEL_ID="${1:?Usage: $0 <hf-model-id> <s3-bucket> <local-path> <region>}"
S3_BUCKET="${2:?S3 bucket required}"
LOCAL_PATH="${3:?Local path required (e.g. /mnt/nvme/models/MyModel)}"
REGION="${4:-us-west-2}"

S3_KEY=$(basename "$LOCAL_PATH")
S3_PATH="s3://${S3_BUCKET}/models/${S3_KEY}"

cat <<EOF
=== Model staging plan ===
HF model:   $MODEL_ID
S3 cache:   $S3_PATH
Local NVMe: $LOCAL_PATH
Region:     $REGION
EOF

# Generate the inline shell command body — same logic shared between K8s Job
# and direct EC2 SSM execution.
cat <<SCRIPT
set -euo pipefail

S3_PATH="$S3_PATH"
LOCAL_PATH="$LOCAL_PATH"
REGION="$REGION"
MODEL_ID="$MODEL_ID"

mkdir -p "\$(dirname "\$LOCAL_PATH")"

# Phase 1: try S3 cache
if aws s3 ls "\$S3_PATH/config.json" --region "\$REGION" >/dev/null 2>&1; then
  echo "=== S3 cache HIT — syncing from \$S3_PATH ==="
  time aws s3 sync "\$S3_PATH" "\$LOCAL_PATH" --region "\$REGION" --no-progress
  echo "=== S3 sync complete ==="
  du -sh "\$LOCAL_PATH"
  exit 0
fi

# Phase 2: HF cold download
echo "=== S3 cache MISS — downloading from HuggingFace ==="
yum install -y python3-pip python3 >/dev/null 2>&1 || \
  apt-get update -qq && apt-get install -y -qq python3 python3-pip >/dev/null 2>&1
pip3 install -q "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_HF_TRANSFER_PARALLELISM=8

time hf download "\$MODEL_ID" \
  --local-dir "\$LOCAL_PATH" \
  --exclude "*.md" "*.txt" "original/*"

echo "=== HF download complete ==="
du -sh "\$LOCAL_PATH"

# Phase 3: persist to S3 for next time
echo "=== Persisting to S3 cache: \$S3_PATH ==="
time aws s3 sync "\$LOCAL_PATH" "\$S3_PATH" --region "\$REGION" --size-only --no-progress
echo "=== S3 cache populated for future runs ==="
SCRIPT
