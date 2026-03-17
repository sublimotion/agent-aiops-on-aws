#!/usr/bin/env bash
# SGLang TP=8 baseline for Qwen3-Next FP8 on g7e.48xlarge (8x RTX PRO 6000, PCIe)
# Config: S1-48xl — full 8-GPU, single replica
# g7e.48xlarge uses PCIe interconnect (same as g7e.24xl — RTX PRO 6000 has no NVLink)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run -d --name sglang-qwen3-tp8 \
  --gpus 8 --ipc=host --network=host --privileged \
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
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --attention-backend triton \
  --disable-cuda-graph \
  --fp8-gemm-backend cutlass \
  --host 0.0.0.0 \
  --port 30000
