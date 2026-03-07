#!/usr/bin/env bash
# SGLang tp4-x1 baseline for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: tp4-x1 (no MTP) — single replica, TP=4 (parallelism comparison)
# GPUs 4-7 idle; in production a second replica runs on them.
# Run via nerdctl/docker on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.2-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name sglang-qwen3-tp4 \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --served-model-name qwen3-next \
  --port 30000
