#!/usr/bin/env bash
# SGLang GLM-5 FP8 on H200 (p5e.48xlarge, 8x H200 141GB)
# FA3 default on Hopper, 256 max requests
# Run via nerdctl/docker on GPU node for local smoke testing
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsys/sglang:v0.5.2-cu124}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"

$CTR run --rm -d --name sglang-glm5-h200 \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name glm-5-fp8 \
  --port 30000
