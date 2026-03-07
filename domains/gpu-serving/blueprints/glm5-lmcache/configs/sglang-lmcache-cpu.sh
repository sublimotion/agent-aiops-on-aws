#!/usr/bin/env bash
# SGLang + LMCache CPU offload (L1 CPU memory, 400 GB)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-glm5-sglang-lmcache:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"

$CTR run --rm -d --name sglang-glm5-lmcache-cpu \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=400 \
  -e LMCACHE_USE_LAYERWISE=True \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 8 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85 \
  --enable-lmcache \
  --served-model-name glm5 \
  --port 30000
