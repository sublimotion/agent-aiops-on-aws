#!/usr/bin/env bash
# Build the customer vLLM image on the GPU node
# Run this AFTER the node joins the cluster and has internet access (via NAT)
#
# Usage (on GPU node via SSM or SSH):
#   ./build-customer-image.sh
#
# Or pull Dockerfile from S3 first:
#   aws s3 cp s3://qwen3-next-bench-results-20260223212318268900000007/custbench/docker/Dockerfile.vllm-customer .
#   ./build-customer-image.sh
set -euo pipefail

CTR="${CONTAINER_RUNTIME:-nerdctl}"
ECR_REGISTRY="615299764834.dkr.ecr.us-east-2.amazonaws.com"
ECR_REPO="qwen3-next-custbench"
IMAGE_TAG="latest"
LOCAL_TAG="${ECR_REPO}:${IMAGE_TAG}"
ECR_TAG="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/../docker/Dockerfile.vllm-customer"

# Fall back to current directory if running from S3 download
if [ ! -f "$DOCKERFILE" ]; then
  DOCKERFILE="./Dockerfile.vllm-customer"
fi

if [ ! -f "$DOCKERFILE" ]; then
  echo "ERROR: Cannot find Dockerfile.vllm-customer"
  echo "Either run from the blueprint directory or download from S3:"
  echo "  aws s3 cp s3://qwen3-next-bench-results-20260223212318268900000007/custbench/docker/Dockerfile.vllm-customer ."
  exit 1
fi

echo "=== Building Customer vLLM Image ==="
echo "Dockerfile: $DOCKERFILE"
echo "Local tag:  $LOCAL_TAG"
echo "ECR tag:    $ECR_TAG"
echo ""

# Build
$CTR build -f "$DOCKERFILE" -t "$LOCAL_TAG" "$(dirname "$DOCKERFILE")"

echo ""
echo "=== Build Complete ==="
echo "Local image: $LOCAL_TAG"

# Push to ECR if aws cli is available
if command -v aws &>/dev/null; then
  echo ""
  echo "=== Pushing to ECR ==="
  aws ecr get-login-password --region us-east-2 | $CTR login --username AWS --password-stdin "$ECR_REGISTRY"
  $CTR tag "$LOCAL_TAG" "$ECR_TAG"
  $CTR push "$ECR_TAG"
  echo "Pushed: $ECR_TAG"
else
  echo "aws CLI not found — skipping ECR push. Image available locally as $LOCAL_TAG"
fi
