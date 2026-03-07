#!/usr/bin/env bash
# Config S2: SGLang HiCache + NVMe L3 (cascading eviction GPU→CPU→NVMe)
# Extends Phase 1 with file-backed L3 tier on local NVMe RAID0
#
# Model: Kimi K2.5 (1T params, native INT4)
# SGLang auto-detects quantization from model config.json.
#
# Usage: bash configs/sglang-hicache-nvme.sh [PORT]
# Requires: CUDA 13.0 driver (580+). NVMe RAID0 mounted at /mnt/nvme.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest-cu130}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"
NVME_CACHE_PATH="/mnt/nvme/kv-cache"

echo "=== Config S2: SGLang HiCache + NVMe L3 ==="
echo "Model:      ${MODEL_PATH}"
echo "Port:       ${PORT}"
echo "Image:      ${SGLANG_IMAGE}"
echo "NVMe cache: ${NVME_CACHE_PATH}"
echo "Runtime:    ${CTR}"

# Ensure NVMe cache directory exists
mkdir -p "${NVME_CACHE_PATH}"

${CTR} run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx:ro \
  -v /mnt/nvme:/mnt/nvme \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 \
  --trust-remote-code \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-storage-backend file \
  --hicache-storage-backend-extra-config '{"path": "/mnt/nvme/kv-cache"}' \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 32768 \
  --port "${PORT}"
