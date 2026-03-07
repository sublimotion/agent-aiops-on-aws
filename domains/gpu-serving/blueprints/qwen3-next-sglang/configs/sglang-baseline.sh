#!/usr/bin/env bash
# SGLang TP=4 baseline for Qwen3-Next FP8 on g7e.24xlarge (4x RTX PRO 6000 Blackwell)
# Config: S0/S1 — baseline inference, no HiCache
# Phase S0: smoke test, Phase S1: throughput baseline
#
# Required flags for Blackwell + Qwen3-Next FP8:
#   --attention-backend triton  (hybrid GDN requires triton or trtllm_mha)
#   --disable-cuda-graph        (required for hybrid attention models)
#   --fp8-gemm-backend cutlass  (DeepGemm crashes with non-ue8m0 scale format)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run -d --name sglang-qwen3-baseline \
  --gpus 4 --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 \
  --dtype bfloat16 \
  --context-length 65536 \
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
