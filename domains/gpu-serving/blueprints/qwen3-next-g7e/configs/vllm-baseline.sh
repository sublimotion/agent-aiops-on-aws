#!/usr/bin/env bash
# vLLM tp4-x1 baseline for Qwen3-Next FP8 on g7e.48xlarge (8x RTX PRO Server 6000)
# Config: tp4-x1 (no MTP) — single replica, TP=4
# Winner config from p5en benchmarks, adapted for Blackwell
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen3_5-x86_64-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-qwen3-tp4 \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
