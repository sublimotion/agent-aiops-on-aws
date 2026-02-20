#!/usr/bin/env bash
# Config 0: Baseline vLLM with native prefix caching
# No external KV cache — uses vLLM's built-in prefix caching only
#
# Model: Kimi K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
# No --quantization flag needed — weights are pre-quantized in safetensor shards.
# vLLM auto-detects quantization from model config.json.
#
# Usage: ./run-kimi-k2.5.sh [--port PORT]
# Requires: CUDA 13.0 driver (580+). Uses cu130-nightly for PyTorch 2.10+ compat.
#
# EKS nodes use containerd (not Docker). nerdctl is the default CLI.
# Override with CONTAINER_RUNTIME=docker if running elsewhere.

set -euo pipefail

PORT="${1:-8000}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:cu130-nightly}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

echo "=== Config 0: Baseline vLLM ==="
echo "Model:   ${MODEL_PATH}"
echo "Port:    ${PORT}"
echo "Image:   ${VLLM_IMAGE}"
echo "Runtime: ${CTR}"

${CTR} run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  -v /mnt/fsx:/mnt/fsx:ro \
  -v /mnt/nvme:/mnt/nvme:ro \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  "${VLLM_IMAGE}" \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --enforce-eager \
  --max-model-len 32768 \
  --swap-space 32 \
  --gpu-memory-utilization 0.85 \
  --port "${PORT}" \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --disable-log-requests
