#!/usr/bin/env bash
# SGLang tp8-x1 baseline for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: tp8-x1 (no MTP) — single replica, TP=8
# Note: SGLang FP8 may work at TP=8 (different quantization kernel).
#       If it fails with block_k errors like vLLM, switch to TP=4.
# Run via nerdctl/docker on GPU node for P0 smoke testing
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.2-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name sglang-qwen3-tp8 \
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
  --mem-fraction-static 0.90 \
  --served-model-name qwen3-next \
  --port 30000
