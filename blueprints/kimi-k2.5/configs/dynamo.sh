#!/usr/bin/env bash
# Config C: NVIDIA Dynamo with KVBM + GDS (Tiered KV Cache to FSx Lustre)
# Uses dynamo-run with ai-dynamo-vllm backend and NIXL GDS_MT transfers
# Ref: https://github.com/ai-dynamo/dynamo
#
# Model: Kimi K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
# No --quantization flag needed — weights are pre-quantized in safetensor shards.
# vLLM auto-detects quantization from model config.json.
#
# Usage: ./run-kimi-k2.5.sh [--build] [--port PORT]
# Requires: GDS support, FSx Lustre mounted at /mnt/fsx.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.
#
# Build the image first with --build, then run:
#   ./run-kimi-k2.5.sh --build
#   ./run-kimi-k2.5.sh --port 8000

set -euo pipefail

# Parse arguments
BUILD=false
PORT="8000"
while [[ $# -gt 0 ]]; do
  case $1 in
    --build) BUILD=true; shift ;;
    --port) PORT="$2"; shift 2 ;;
    *) PORT="$1"; shift ;;
  esac
done

MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
DYNAMO_IMAGE="${DYNAMO_IMAGE:-dynamo-kvbm:latest}"
DYNAMO_CACHE_DIR="${DYNAMO_CACHE_DIR:-/mnt/fsx/kv-cache/dynamo}"
DOCKERFILE_DIR="${DOCKERFILE_DIR:-$(dirname "$0")/../docker}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

# ============================================
# Build phase (optional)
# ============================================
if [ "$BUILD" = true ]; then
  echo "=== Building Dynamo KVBM Docker image ==="
  echo "Dockerfile: ${DOCKERFILE_DIR}/Dockerfile.dynamo-kvbm"
  echo "Image tag:  ${DYNAMO_IMAGE}"

  ${CTR} build \
    -f "${DOCKERFILE_DIR}/Dockerfile.dynamo-kvbm" \
    -t "${DYNAMO_IMAGE}" \
    "${DOCKERFILE_DIR}"

  echo "Build complete: ${DYNAMO_IMAGE}"
  echo ""
fi

# ============================================
# Run phase
# ============================================
echo "=== Config C: NVIDIA Dynamo + KVBM (GDS) ==="
echo "Model:      ${MODEL_PATH}"
echo "Port:       ${PORT}"
echo "Image:      ${DYNAMO_IMAGE}"
echo "Cache dir:  ${DYNAMO_CACHE_DIR}"
echo "Runtime:    ${CTR}"

# Ensure cache directory exists
mkdir -p "${DYNAMO_CACHE_DIR}"

# Dynamo KVBM environment variables:
#   DYN_KVBM_CPU_CACHE_GB:   Host DRAM tier (G2) — 128GB of ~2TB available
#   DYN_KVBM_DISK_CACHE_GB:  Disk tier (G3) — 500GB on FSx Lustre
#   DYN_KVBM_DISK_CACHE_DIR: FSx path for persistent, shareable cache
#   DYN_KVBM_NIXL_BACKEND_GDS_MT: Multi-threaded GDS backend via NIXL
#   DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER: Disable frequency filter for benchmarking
#
# See docs/DYNAMO_KV_CACHE_GDS.md for tier architecture details.

echo "Starting NVIDIA Dynamo with KVBM..."
${CTR} run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v "$(dirname "$0")/kimi-k2.5-gds.yaml:/etc/dynamo/kimi-k2.5-gds.yaml:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e DYNAMO_EVENT_PLANE=zmq \
  -e CUFILE_ENV_PATH_JSON=/etc/cufile.json \
  -e NIXL_LOG_LEVEL=INFO \
  -e DYN_KVBM_CPU_CACHE_GB=128 \
  -e DYN_KVBM_DISK_CACHE_GB=500 \
  -e DYN_KVBM_DISK_CACHE_DIR="${DYNAMO_CACHE_DIR}" \
  -e DYN_KVBM_NIXL_BACKEND_GDS_MT=true \
  -e DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true \
  "${DYNAMO_IMAGE}" \
  dynamo-run \
    --config /etc/dynamo/kimi-k2.5-gds.yaml \
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
