#!/usr/bin/env bash
# Build SGLang+LMCache image and push to ECR
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
REPO_NAME="glm5-sglang-lmcache"
TAG="${IMAGE_TAG:-latest}"

echo "=== Building and pushing SGLang+LMCache image ==="

# Create ECR repo if needed
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION"

# Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Build
docker build -t "${ECR_REGISTRY}/${REPO_NAME}:${TAG}" -f docker/Dockerfile.sglang-lmcache docker/

# Push
docker push "${ECR_REGISTRY}/${REPO_NAME}:${TAG}"

echo ""
echo "Image: ${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
echo "Set sglang_image = \"${ECR_REGISTRY}/${REPO_NAME}:${TAG}\" in tfvars"
