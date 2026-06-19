#!/usr/bin/env bash
# serve-vllm-eagle3.sh — Start vLLM with EAGLE3 speculative decode (Phase 2).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../configs/vllm-eagle3.env}"
source "$ENV_FILE"

NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

IMAGE="${IMAGE:-voipmonitor/vllm:cu130-mtp-tuned-v3-20260423}"
CONTAINER_NAME="vllm-k26-eagle3"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

exec docker run -d --name "$CONTAINER_NAME" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/nvme:/mnt/nvme \
  "$IMAGE" \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --speculative-model "$DRAFT_PATH" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
