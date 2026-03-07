#!/usr/bin/env bash
# SGLang GLM-5 FP8 on H200 with EAGLE speculative decoding
# Requires EAGLE draft model at /mnt/nvme/models/GLM-5-EAGLE
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsys/sglang:v0.5.2-cu124}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/GLM-5-FP8}"
EAGLE_MODEL="${EAGLE_PATH:-/mnt/nvme/models/GLM-5-EAGLE}"

$CTR run --rm -d --name sglang-glm5-eagle \
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
  --speculative-eagle-path "$EAGLE_MODEL" \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 5 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name glm-5-fp8 \
  --port 30000
