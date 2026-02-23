#!/usr/bin/env bash
# Config A': vLLM + LMCache with POSIX I/O (CPU bounce, no GDS)
# Uses LMCacheConnectorV1 but with POSIX backend instead of cuFile/GDS.
# KV cache writes go GPU → CPU DRAM → FSx Lustre (2-hop path).
#
# Model: Kimi K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
# No --quantization flag needed — weights are pre-quantized in safetensor shards.
# vLLM auto-detects quantization from model config.json.
#
# IMPORTANT: Use `nightly` (CUDA 12.9), NOT `cu130-nightly`.
# LMCache must be built from source against the installed torch version.
# See lessons.md #13-#15 for details.
#
# Usage: ./run-kimi-k2.5.sh [--port PORT]
# Requires: FSx Lustre mounted at /mnt/fsx.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.
#
# Difference from lmcache/run-kimi-k2.5.sh:
#   - LMCACHE_USE_EXPERIMENTAL=False (no cuFile GDS)
#   - LMCACHE_LOCAL_CPU=True (CPU-bounce path)
#   - No cuFile symlink setup needed
#   - Expected throughput: ~1-3 GB/s (vs 9+ GB/s with GDS)

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
LMCACHE_IMAGE="${LMCACHE_IMAGE:-vllm/vllm-openai:nightly}"
LMCACHE_FSX_PATH="/mnt/fsx/kv-cache/lmcache"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== Config A': vLLM + LMCache (POSIX, no GDS) ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${LMCACHE_IMAGE}"
echo "FSx path:  ${LMCACHE_FSX_PATH}"
echo "Runtime:   ${CTR}"

# Ensure cache directory exists
mkdir -p "${LMCACHE_FSX_PATH}"

# Start vLLM with LMCache POSIX backend
# No cuFile setup needed — uses CPU-bounce path
echo "Starting vLLM + LMCache (POSIX)..."
${CTR} run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx \
  -v /mnt/nvme:/mnt/nvme:ro \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e LMCACHE_USE_EXPERIMENTAL=False \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_LOCAL_DISK="file://${LMCACHE_FSX_PATH}" \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE=100.0 \
  "${LMCACHE_IMAGE}" \
  bash -c '
    echo "Installing CUDA dev headers..."
    apt-get update -qq && apt-get install -y -qq libcusparse-dev-12-9 libcublas-dev-12-9 libcusolver-dev-12-9 2>/dev/null

    echo "Building LMCache from source..."
    TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 pip uninstall lmcache -y -q
    pip install lmcache --no-binary lmcache --no-deps --no-build-isolation -q 2>&1 | tail -3
    echo "LMCache built successfully"

    exec python3 -m vllm.entrypoints.openai.api_server \
      --model '"${MODEL_PATH}"' \
      --tensor-parallel-size 8 \
      --enable-prefix-caching \
      --enforce-eager \
      --max-model-len 32768 \
      --swap-space 32 \
      --gpu-memory-utilization 0.85 \
      --port '"${PORT}"' \
      --trust-remote-code \
      --tool-call-parser kimi_k2 \
      --reasoning-parser kimi_k2 \
      --disable-log-requests \
      --kv-transfer-config '"'"'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'"'"'
  '
