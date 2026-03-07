#!/usr/bin/env bash
# Pull official vLLM GLM-5 image and push to ECR
# Image includes DeepGEMM, GLM-5 model architecture, and tool-call parser (glm47)
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
REPO_NAME="glm5-vllm-lmcache"
SOURCE_IMAGE="${SOURCE_IMAGE:-vllm/vllm-openai:glm5}"
TAG="${IMAGE_TAG:-glm5}"

echo "=== Pushing vLLM GLM-5 image to ECR ==="

# Create ECR repo if needed
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION"

# Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Pull, tag, push
docker pull "$SOURCE_IMAGE"
docker tag "$SOURCE_IMAGE" "${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
docker push "${ECR_REGISTRY}/${REPO_NAME}:${TAG}"

echo ""
echo "Image: ${ECR_REGISTRY}/${REPO_NAME}:${TAG}"
echo "Set vllm_image = \"${ECR_REGISTRY}/${REPO_NAME}:${TAG}\" in tfvars"
