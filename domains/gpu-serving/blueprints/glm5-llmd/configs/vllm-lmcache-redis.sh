#!/usr/bin/env bash
# vLLM GLM-5 + LMCache with Redis L2 (cross-replica KV cache sharing)
# Uses official vllm/vllm-openai:glm5 image with tool calling + reasoning
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:glm5}"
REDIS_URL="${REDIS_URL:-redis://redis-glm5:6379}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== vLLM GLM-5 + LMCache (Redis L2) ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${VLLM_IMAGE}"
echo "Redis:     ${REDIS_URL}"

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
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=400 \
  -e LMCACHE_REMOTE_URL="${REDIS_URL}" \
  "${VLLM_IMAGE}" \
  bash -c '
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
