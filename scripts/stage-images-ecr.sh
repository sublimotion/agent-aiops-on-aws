#!/usr/bin/env bash
# stage-images-ecr.sh
# Pre-builds and mirrors all container images to private ECR for air-gapped p5e deployment.
#
# Prerequisites:
#   - Docker running locally
#   - AWS CLI configured with ECR push permissions
#   - Internet access (run from a connected machine)
#
# Usage:
#   ./scripts/stage-images-ecr.sh <aws-account-id> <aws-region>
#
# Example:
#   ./scripts/stage-images-ecr.sh 615299764834 us-east-2

set -euo pipefail

ACCOUNT_ID="${1:?Usage: $0 <aws-account-id> <aws-region>}"
REGION="${2:?Usage: $0 <aws-account-id> <aws-region>}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ============================================================================
# Image manifest: all pre-built images for kimi-k2.5 benchmark configs
# Format: source_image -> ecr_repo_name
# ============================================================================

# --- Base infrastructure images ---
declare -A IMAGES=(
  # vLLM inference engine (cu130-nightly for CUDA 13.0 / driver 580.x on p5e)
  ["vllm/vllm-openai:cu130-nightly"]="vllm-openai"

  # NVIDIA device plugin (EKS GPU scheduling)
  ["nvcr.io/nvidia/k8s-device-plugin:v0.14.5"]="nvidia-k8s-device-plugin"

  # FSx CSI driver + sidecars
  ["public.ecr.aws/fsx-csi-driver/aws-fsx-csi-driver:v1.2.0"]="fsx-csi-driver"
  ["public.ecr.aws/eks-distro/kubernetes-csi/livenessprobe:v2.12.0-eks-1-29-5"]="csi-livenessprobe"
  ["public.ecr.aws/eks-distro/kubernetes-csi/node-driver-registrar:v2.10.0-eks-1-29-5"]="csi-node-driver-registrar"
  ["public.ecr.aws/eks-distro/kubernetes-csi/external-provisioner:v4.0.0-eks-1-29-5"]="csi-external-provisioner"
  ["public.ecr.aws/eks-distro/kubernetes-csi/external-resizer:v1.10.0-eks-1-29-5"]="csi-external-resizer"

  # AWS EFA device plugin (p5e EFA support)
  ["public.ecr.aws/eks/aws-efa-k8s-device-plugin:v0.5.7"]="aws-efa-k8s-device-plugin"

  # Prometheus monitoring
  ["quay.io/prometheus/prometheus:v2.54.1"]="prometheus"
  ["quay.io/prometheus-operator/prometheus-config-reloader:v0.76.0"]="prometheus-config-reloader"

  # NVIDIA DCGM exporter (GPU telemetry)
  ["nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"]="dcgm-exporter"
)

# kube-prometheus-stack additional images (chart v62.7.0 defaults)
declare -A KUBE_PROM_IMAGES=(
  ["quay.io/prometheus-operator/prometheus-operator:v0.76.0"]="prometheus-operator"
  ["registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.13.0"]="kube-state-metrics"
  ["quay.io/prometheus/alertmanager:v0.27.0"]="alertmanager"
  ["quay.io/prometheus/node-exporter:v1.8.2"]="node-exporter"
  ["docker.io/grafana/grafana:11.2.0"]="grafana"
  ["quay.io/kiwigrid/k8s-sidecar:1.27.5"]="grafana-sidecar"
)

# --- Config A: vLLM + LMCache ---
declare -A LMCACHE_IMAGES=(
  ["lmcache/vllm-openai:latest"]="lmcache-vllm-openai"
  ["lmcache/lmstack-router:benchmark"]="lmcache-router"
)

# --- Shared support: event plane + distributed config ---
declare -A SUPPORT_IMAGES=(
  ["nats:2.10"]="nats"
  ["bitnami/etcd:3.5"]="etcd"
)

echo "============================================"
echo "ECR Image Staging for Air-Gapped Deployment"
echo "============================================"
echo "Account:  ${ACCOUNT_ID}"
echo "Region:   ${REGION}"
echo "Registry: ${ECR_REGISTRY}"
echo ""

# ============================================================================
# Step 1: Authenticate to ECR
# ============================================================================
echo ">>> Authenticating to ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"
echo ""

# ============================================================================
# Step 2: Create ECR repositories (idempotent)
# ============================================================================
create_repo() {
  local repo_name="$1"
  aws ecr describe-repositories --repository-names "${repo_name}" --region "${REGION}" > /dev/null 2>&1 || \
    aws ecr create-repository \
      --repository-name "${repo_name}" \
      --region "${REGION}" \
      --image-scanning-configuration scanOnPush=true \
      --encryption-configuration encryptionType=AES256 \
      --output text --query 'repository.repositoryUri'
}

echo ">>> Creating ECR repositories..."
for repo_name in "${IMAGES[@]}" "${KUBE_PROM_IMAGES[@]}" "${LMCACHE_IMAGES[@]}" "${SUPPORT_IMAGES[@]}"; do
  create_repo "${repo_name}"
done
# Custom build repos (built separately via Dockerfiles)
create_repo "vllm-mooncake"
create_repo "dynamo-kvbm"
echo ""

# ============================================================================
# Step 3: Pull, tag, and push each image
# ============================================================================
mirror_image() {
  local source="$1"
  local repo_name="$2"
  local tag="${source##*:}"
  local ecr_target="${ECR_REGISTRY}/${repo_name}:${tag}"

  echo "  [pull]  ${source}"
  if ! docker pull --platform linux/amd64 "${source}"; then
    echo "  [ERROR] Failed to pull ${source}"
    return 1
  fi

  echo "  [tag]   ${ecr_target}"
  docker tag "${source}" "${ecr_target}"

  echo "  [push]  ${ecr_target}"
  docker push "${ecr_target}"

  # Record digest for pinning
  local digest
  digest=$(docker inspect --format='{{index .RepoDigests 0}}' "${source}" 2>/dev/null | cut -d@ -f2)
  echo "  [digest] ${digest:-unknown}"

  # Cleanup local copy to save disk space
  echo "  [clean] Removing local images"
  docker rmi "${source}" "${ecr_target}" 2>/dev/null || true
  echo ""
}

# Mirror an image that uses :latest — pin to digest after pull
mirror_image_pin_digest() {
  local source="$1"
  local repo_name="$2"
  local tag="${source##*:}"
  local ecr_target="${ECR_REGISTRY}/${repo_name}:${tag}"

  echo "  [pull]  ${source} (will pin digest)"
  if ! docker pull --platform linux/amd64 "${source}"; then
    echo "  [ERROR] Failed to pull ${source}"
    return 1
  fi

  # Get the digest for reproducibility
  local digest
  digest=$(docker inspect --format='{{index .RepoDigests 0}}' "${source}" 2>/dev/null | cut -d@ -f2)
  echo "  [digest] ${digest:-unknown}"
  if [ -n "${digest}" ] && [ "${digest}" != "unknown" ]; then
    echo "  [pin]   Pinned ${repo_name}:${tag} to ${digest}"
  fi

  echo "  [tag]   ${ecr_target}"
  docker tag "${source}" "${ecr_target}"

  echo "  [push]  ${ecr_target}"
  docker push "${ecr_target}"

  # Cleanup local copy
  echo "  [clean] Removing local images"
  docker rmi "${source}" "${ecr_target}" 2>/dev/null || true
  echo ""
}

# Attempt nvcr.io pull without authentication first
pull_nvcr_image() {
  local source="$1"
  echo "  [nvcr]  Attempting unauthenticated pull: ${source}"
  if docker pull --platform linux/amd64 "${source}" 2>/dev/null; then
    return 0
  fi
  echo "  [nvcr]  Unauthenticated pull failed. Checking for NGC credentials..."
  if [ -n "${NGC_API_KEY:-}" ]; then
    echo "  [nvcr]  Authenticating to nvcr.io with NGC_API_KEY..."
    echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
    docker pull --platform linux/amd64 "${source}"
  else
    echo "  [ERROR] NGC_API_KEY not set. Set it to pull from nvcr.io:"
    echo "          export NGC_API_KEY=<your-ngc-api-key>"
    return 1
  fi
}

echo ">>> Mirroring core infrastructure images..."
for source in "${!IMAGES[@]}"; do
  repo_name="${IMAGES[${source}]}"
  # Use nvcr.io-aware pull for NVIDIA images
  if [[ "${source}" == nvcr.io/* ]]; then
    pull_nvcr_image "${source}"
    tag="${source##*:}"
    ecr_target="${ECR_REGISTRY}/${repo_name}:${tag}"
    docker tag "${source}" "${ecr_target}"
    docker push "${ecr_target}"
    echo "  [digest] $(docker inspect --format='{{index .RepoDigests 0}}' "${source}" 2>/dev/null | cut -d@ -f2)"
    docker rmi "${source}" "${ecr_target}" 2>/dev/null || true
    echo ""
  else
    mirror_image "${source}" "${repo_name}"
  fi
done

echo ">>> Mirroring kube-prometheus-stack images..."
for source in "${!KUBE_PROM_IMAGES[@]}"; do
  mirror_image "${source}" "${KUBE_PROM_IMAGES[${source}]}"
done

echo ">>> Mirroring LMCache images (Config A)..."
for source in "${!LMCACHE_IMAGES[@]}"; do
  mirror_image_pin_digest "${source}" "${LMCACHE_IMAGES[${source}]}"
done

echo ">>> Mirroring shared support images (NATS, etcd)..."
for source in "${!SUPPORT_IMAGES[@]}"; do
  mirror_image "${source}" "${SUPPORT_IMAGES[${source}]}"
done

# ============================================================================
# Step 4: Build and push custom images
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Config B: vLLM + Mooncake (custom build) ---
MOONCAKE_DOCKERFILE="${SCRIPT_DIR}/../blueprints/kimi-k2.5/docker/Dockerfile.vllm-mooncake"
if [ -f "${MOONCAKE_DOCKERFILE}" ]; then
  echo ">>> Building vLLM + Mooncake custom image (Config B)..."
  MOONCAKE_TAG="${ECR_REGISTRY}/vllm-mooncake:cu130"

  docker build \
    --platform linux/amd64 \
    -f "${MOONCAKE_DOCKERFILE}" \
    -t "${MOONCAKE_TAG}" \
    "$(dirname "${MOONCAKE_DOCKERFILE}")"

  echo "  [push] ${MOONCAKE_TAG}"
  docker push "${MOONCAKE_TAG}"

  docker rmi "${MOONCAKE_TAG}" 2>/dev/null || true
  echo ""
else
  echo "  [skip] Mooncake Dockerfile not found at ${MOONCAKE_DOCKERFILE}"
fi

# --- Config C: Dynamo KVBM (custom build) ---
DYNAMO_DOCKERFILE="${SCRIPT_DIR}/../blueprints/vllm-kv-benchmark/docker/Dockerfile.dynamo-kvbm"
if [ -f "${DYNAMO_DOCKERFILE}" ]; then
  echo ">>> Building Dynamo KVBM custom image (Config C)..."
  DYNAMO_TAG="${ECR_REGISTRY}/dynamo-kvbm:v0.9.0"

  docker build \
    --platform linux/amd64 \
    -f "${DYNAMO_DOCKERFILE}" \
    -t "${DYNAMO_TAG}" \
    "$(dirname "${DYNAMO_DOCKERFILE}")"

  echo "  [push] ${DYNAMO_TAG}"
  docker push "${DYNAMO_TAG}"

  docker rmi "${DYNAMO_TAG}" 2>/dev/null || true
  echo ""
else
  echo "  [skip] Dynamo Dockerfile not found at ${DYNAMO_DOCKERFILE}"
fi

# ============================================================================
# Step 5: Print Terraform variable overrides
# ============================================================================
echo "============================================"
echo "Staging complete. Use these Terraform overrides:"
echo "============================================"
echo ""
cat <<EOF
# terraform.tfvars (kimi-k2.5)
vllm_image                 = "${ECR_REGISTRY}/vllm-openai:cu130-nightly"
nvidia_device_plugin_image = "${ECR_REGISTRY}/nvidia-k8s-device-plugin:v0.14.5"

# Config A: LMCache
lmcache_vllm_image         = "${ECR_REGISTRY}/lmcache-vllm-openai:latest"
lmcache_router_image       = "${ECR_REGISTRY}/lmcache-router:benchmark"

# Config B: Mooncake
vllm_mooncake_image        = "${ECR_REGISTRY}/vllm-mooncake:cu130"

# Config C: Dynamo
dynamo_kvbm_image          = "${ECR_REGISTRY}/dynamo-kvbm:v0.9.0"

# Shared support
nats_image                 = "${ECR_REGISTRY}/nats:2.10"
etcd_image                 = "${ECR_REGISTRY}/etcd:3.5"

# For Helm chart image overrides, update main.tf or pass via values:
# FSx CSI:       ${ECR_REGISTRY}/fsx-csi-driver:v1.2.0
# CSI sidecars:  ${ECR_REGISTRY}/csi-livenessprobe:v2.12.0-eks-1-29-5
#                ${ECR_REGISTRY}/csi-node-driver-registrar:v2.10.0-eks-1-29-5
#                ${ECR_REGISTRY}/csi-external-provisioner:v4.0.0-eks-1-29-5
#                ${ECR_REGISTRY}/csi-external-resizer:v1.10.0-eks-1-29-5
# EFA plugin:    ${ECR_REGISTRY}/aws-efa-k8s-device-plugin:v0.5.7
# Prometheus:    ${ECR_REGISTRY}/prometheus:v2.54.1
# Config reload: ${ECR_REGISTRY}/prometheus-config-reloader:v0.76.0
# DCGM:          ${ECR_REGISTRY}/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
# Grafana:       ${ECR_REGISTRY}/grafana:11.2.0
EOF

echo ""
echo ">>> All images staged to ${ECR_REGISTRY}"
