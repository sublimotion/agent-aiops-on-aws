#!/usr/bin/env bash
# vLLM GLM-5 + LMCache GDS backend (GPU Direct Storage → FSx Lustre)
# Uses official vllm/vllm-openai:glm5 image with tool calling + reasoning
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:glm5}"
LMCACHE_GDS_PATH="/mnt/fsx/kv-cache/lmcache"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== vLLM GLM-5 + LMCache (GDS) ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${VLLM_IMAGE}"
echo "GDS path:  ${LMCACHE_GDS_PATH}"

mkdir -p "${LMCACHE_GDS_PATH}"

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
  "${VLLM_IMAGE}" \
  bash -c '
    echo "Setting up cuFile symlinks..."
    CUFILE_DIR=$(python3 -c "import nvidia.cufile; import os; print(os.path.dirname(nvidia.cufile.__file__) + \"/lib\")" 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages/nvidia/cufile/lib)
    if [ -d "${CUFILE_DIR}" ]; then
      ln -sf ${CUFILE_DIR}/libcufile.so.0 ${CUFILE_DIR}/libcufile.so
      ln -sf ${CUFILE_DIR}/libcufile_rdma.so.1 ${CUFILE_DIR}/libcufile_rdma.so
      echo ${CUFILE_DIR} > /etc/ld.so.conf.d/cufile.conf && ldconfig
    fi

    exec python3 -m vllm.entrypoints.openai.api_server \
      --model '"${MODEL_PATH}"' \
      --served-model-name glm5 \
      --tensor-parallel-size 8 \
      --enable-prefix-caching \
      --max-model-len 131072 \
      --swap-space 32 \
      --gpu-memory-utilization 0.85 \
      --speculative-config.method mtp \
      --speculative-config.num_speculative_tokens 1 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --port '"${PORT}"' \
      --disable-log-requests \
      --kv-transfer-config '"'"'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'"'"'
  '
