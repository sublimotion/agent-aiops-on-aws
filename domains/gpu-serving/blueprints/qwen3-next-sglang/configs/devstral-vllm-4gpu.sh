#!/usr/bin/env bash
# vLLM 4-GPU for Devstral Small 2 (24B FP8) on g7e.24xlarge
# Config: D1-scale — TP=4 for maximum throughput comparison with Qwen3-Next
#
# Memory budget (FP8, TP=4):
#   Model weights: ~6.5 GB/GPU
#   KV cache (0.92 util): ~82 GB/GPU → ~2M tokens total capacity
#   Overhead: ~4 GB/GPU
#
# Use this config when benchmarking apples-to-apples throughput vs Qwen3-Next TP=4.
# The single-GPU config (devstral-vllm-baseline.sh) tests cost efficiency.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL="${DEVSTRAL_MODEL_PATH:-/mnt/nvme/models/devstral-small-2-fp8}"

$CTR run -d --name vllm-devstral-tp4 \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --served-model-name devstral-small-2 \
  --port 8000
