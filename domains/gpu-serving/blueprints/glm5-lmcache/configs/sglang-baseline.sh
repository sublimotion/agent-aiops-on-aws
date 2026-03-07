#!/usr/bin/env bash
# SGLang baseline — RadixAttention prefix caching only, no LMCache
# Run via nerdctl on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:glm5-blackwell}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"

$CTR run --rm -d --name sglang-glm5-baseline \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 8 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --served-model-name glm5 \
  --port 30000 \
  --host 0.0.0.0
