#!/usr/bin/env bash
# Builds and pushes custom SGLang container to private ECR.
# Must run from a host with internet access before deploying in closed network.
#
# Usage: ./scripts/copy-images-to-ecr.sh [region]
# The ECR URI is output by `terraform output sglang_ecr_uri`

set -euo pipefail
REGION="${1:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/../docker"

echo "=== Building and pushing containers to ECR ==="
echo "Account: ${ACCOUNT_ID}"
echo "Region:  ${REGION}"
echo "ECR URI: ${ECR_URI}"
echo ""

# Authenticate with ECR
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

# Build SGLang GLM-5 container
echo ""
echo "--- Building sglang-glm5 ---"
SGLANG_TAG="${ECR_URI}/glm5/sglang-glm5:v0.5.2"
docker build --platform linux/amd64 \
    -f "${DOCKER_DIR}/Dockerfile.sglang-glm5" \
    -t "${SGLANG_TAG}" \
    "${DOCKER_DIR}"

echo "Pushing ${SGLANG_TAG}..."
docker push "${SGLANG_TAG}"

echo ""
echo "=== Complete ==="
echo "SGLang GLM-5: ${SGLANG_TAG}"
