#!/usr/bin/env bash
# vLLM tp4 extended context for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: tp4-x1 with full 262K native context length
# Purpose: P2b — extend viable context range to full native 262K
# Based on vllm-baseline.sh with one change:
#   --max-model-len 262144 (full native context, was 131072)
#
# NOTE: --cpu-offload-gb is BLOCKED on vLLM 0.16 V1 engine
#   (AssertionError in may_reinitialize_input_batch, see vllm#18298).
#   H200 has 104.5 GiB available KV cache per GPU at 0.92 utilization,
#   providing 34.6x concurrency at 262K — offloading is unnecessary.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen3_5-x86_64-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run -d --name vllm-qwen3-tp4-extctx \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
