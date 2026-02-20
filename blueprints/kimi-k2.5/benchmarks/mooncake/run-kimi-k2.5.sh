#!/usr/bin/env bash
# Config B: vLLM + Mooncake Transfer Engine (tiered VRAM→DRAM→NVMe→FSx via RDMA)
# Uses mooncake-transfer-engine for tiered KV cache with GDS and RDMA support
#
# Model: Kimi K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
# No --quantization flag needed — weights are pre-quantized in safetensor shards.
# vLLM auto-detects quantization from model config.json.
#
# Usage: ./run-kimi-k2.5.sh [--port PORT]
# Requires: CUDA 13.0 driver (580+). Mooncake image must be built with cu130.

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
MOONCAKE_IMAGE="${MOONCAKE_IMAGE:-vllm-mooncake:cu130}"
MOONCAKE_FSX_PATH="/mnt/fsx/kv-cache/mooncake"
MOONCAKE_NVME_PATH="/mnt/nvme/mooncake"

echo "=== Config B: vLLM + Mooncake ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${MOONCAKE_IMAGE}"
echo "FSx path:  ${MOONCAKE_FSX_PATH}"
echo "NVMe path: ${MOONCAKE_NVME_PATH}"

# Ensure cache directories exist
mkdir -p "${MOONCAKE_FSX_PATH}" "${MOONCAKE_NVME_PATH}"

# Start vLLM with Mooncake transfer engine
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx \
  -v /mnt/nvme:/mnt/nvme \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e MOONCAKE_ENABLE_GDS=1 \
  -e MOONCAKE_STORAGE_TIERS="vram,dram,nvme,fsx" \
  -e MOONCAKE_NVME_PATH="${MOONCAKE_NVME_PATH}" \
  -e MOONCAKE_FSX_PATH="${MOONCAKE_FSX_PATH}" \
  -e CUFILE_ENV_PATH_JSON=/etc/cufile.json \
  "${MOONCAKE_IMAGE}" \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --enforce-eager \
  --max-model-len 32768 \
  --swap-space 32 \
  --gpu-memory-utilization 0.85 \
  --port "${PORT}" \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --disable-log-requests
