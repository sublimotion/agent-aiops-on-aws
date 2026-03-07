#!/usr/bin/env bash
# vLLM GLM-5 baseline — no LMCache, prefix caching only
# Uses official vllm/vllm-openai:glm5 image with tool calling + MTP
# Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:glm5}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== vLLM GLM-5 Baseline (no LMCache) ==="
echo "Model:     ${MODEL_PATH}"
echo "Port:      ${PORT}"
echo "Image:     ${VLLM_IMAGE}"

${CTR} run --rm -d --name vllm-glm5-baseline \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  "${VLLM_IMAGE}" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
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
    --port "${PORT}" \
    --host 0.0.0.0 \
    --disable-log-requests
