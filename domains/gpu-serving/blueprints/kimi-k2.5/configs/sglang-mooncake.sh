#!/usr/bin/env bash
# Config S3: SGLang HiCache + Mooncake L3 (cascading eviction GPU→CPU→Mooncake via RDMA)
# Uses mooncake-transfer-engine as HiCache L3 storage backend
#
# Model: Kimi K2.5 (1T params, native INT4)
# SGLang auto-detects quantization from model config.json.
#
# Usage: bash configs/sglang-mooncake.sh [PORT]
# Requires: CUDA 13.0 driver (580+). EFA/RDMA interfaces. Mooncake metadata server.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
SGLANG_MOONCAKE_IMAGE="${SGLANG_MOONCAKE_IMAGE:-sglang-mooncake:cu130}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-localhost:2379}"

echo "=== Config S3: SGLang HiCache + Mooncake L3 ==="
echo "Model:            ${MODEL_PATH}"
echo "Port:             ${PORT}"
echo "Image:            ${SGLANG_MOONCAKE_IMAGE}"
echo "Mooncake proto:   ${MOONCAKE_PROTOCOL}"
echo "Mooncake meta:    ${MOONCAKE_TE_META_DATA_SERVER}"
echo "Runtime:          ${CTR}"

${CTR} run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx:ro \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /dev/infiniband:/dev/infiniband \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL}" \
  -e MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER}" \
  "${SGLANG_MOONCAKE_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 \
  --trust-remote-code \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy timeout \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 32768 \
  --port "${PORT}"
