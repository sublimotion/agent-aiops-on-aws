#!/usr/bin/env bash
# vLLM single-GPU baseline for Devstral Small 2 (24B FP8) on g7e.24xlarge
# Config: D0/D1 — baseline inference on 1x RTX PRO 6000 (96 GB GDDR7)
#
# Memory budget (FP8, 1 GPU):
#   Model weights: ~26 GB
#   KV cache (0.92 util): ~62 GB → ~388K tokens capacity
#   Overhead: ~8 GB
#
# Uses only GPU 0. GPUs 1-3 available for Qwen3-Next (SGLang) comparison.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
MODEL="${DEVSTRAL_MODEL_PATH:-/mnt/nvme/models/devstral-small-2-fp8}"

$CTR run -d --name vllm-devstral-baseline \
  --gpus '"device=0"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 64 \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --served-model-name devstral-small-2 \
  --port 8000
