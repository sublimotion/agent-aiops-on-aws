#!/usr/bin/env bash
# vLLM tp8-x1 + MTP for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: tp8-x1 + MTP (speculative decoding with 2 draft tokens)
# WARNING: FP8 block_k=128 may be incompatible with TP=8 (partition size 64).
#          This config is included for completeness but may fail at launch.
#          If it fails, use vllm-tp4-mtp.sh instead.
# Run via nerdctl/docker on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen3_5-x86_64-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-qwen3-tp8-mtp \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
