#!/usr/bin/env bash
# Config A: vLLM + LMCache with GDS (GPU Direct Storage to FSx Lustre)
# Uses LMCacheConnectorV1 experimental backend with cuFile for GPU-direct I/O
# Ref: https://docs.lmcache.ai/kv_cache/gds.html
#
# Model: Kimi K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
# No --quantization flag needed — weights are pre-quantized in safetensor shards.
# vLLM auto-detects quantization from model config.json.
#
# IMPORTANT: Use `nightly` (CUDA 12.9), NOT `cu130-nightly`.
# LMCache must be built from source against the installed torch version.
# The cu130 image lacks CUDA dev headers needed for the build.
# See lessons.md #13-#15 for details.
#
# Usage: ./run-kimi-k2.5.sh [--port PORT]
# Requires: cuFile/GDS support, FSx Lustre mounted at /mnt/fsx.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
LMCACHE_IMAGE="${LMCACHE_IMAGE:-vllm/vllm-openai:nightly}"
LMCACHE_GDS_PATH="/mnt/fsx/kv-cache/lmcache"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== Config A: vLLM + LMCache (GDS) ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${LMCACHE_IMAGE}"
echo "GDS path:  ${LMCACHE_GDS_PATH}"
echo "Runtime:   ${CTR}"

# Ensure cache directory exists
mkdir -p "${LMCACHE_GDS_PATH}"

# Start vLLM with LMCache GDS backend
# The entrypoint script:
# 1. Creates cuFile symlinks (libcufile.so.0 → libcufile.so)
# 2. Installs CUDA dev headers (needed for LMCache source build)
# 3. Builds LMCache from source against installed torch (~3-4 min)
# 4. Starts vLLM with LMCacheConnectorV1
echo "Starting vLLM + LMCache (GDS)..."
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
  -e LMCACHE_USE_EXPERIMENTAL=True \
  -e LMCACHE_GDS_PATH="${LMCACHE_GDS_PATH}" \
  -e LMCACHE_CUFILE_BUFFER_SIZE=8192 \
  -e LMCACHE_LOCAL_CPU=False \
  -e CUFILE_ENV_PATH_JSON=/etc/cufile.json \
  "${LMCACHE_IMAGE}" \
  bash -c '
    echo "Setting up cuFile symlinks..."
    CUFILE_DIR=$(python3 -c "import nvidia.cufile; import os; print(os.path.dirname(nvidia.cufile.__file__) + \"/lib\")" 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages/nvidia/cufile/lib)
    ln -sf ${CUFILE_DIR}/libcufile.so.0 ${CUFILE_DIR}/libcufile.so
    ln -sf ${CUFILE_DIR}/libcufile_rdma.so.1 ${CUFILE_DIR}/libcufile_rdma.so
    echo ${CUFILE_DIR} > /etc/ld.so.conf.d/cufile.conf && ldconfig

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
