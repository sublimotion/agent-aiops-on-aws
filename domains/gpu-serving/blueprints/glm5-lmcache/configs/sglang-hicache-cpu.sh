#!/usr/bin/env bash
# SGLang HiCache CPU offload — native hierarchical KV cache to host memory
# Works with NSA/MLA attention (unlike LMCache which is blocked)
# Run via nerdctl on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:glm5-blackwell}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"

# HiCache config: 100 GB per TP rank = 800 GB total host KV cache
# Must be > device KV pool size (~82 GB/rank for GLM-5 at mem_fraction_static=0.90)
HICACHE_SIZE="${HICACHE_SIZE:-100}"

$CTR run --rm -d --name sglang-glm5-hicache \
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
  --host 0.0.0.0 \
  --enable-hierarchical-cache \
  --hicache-size "$HICACHE_SIZE" \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel
