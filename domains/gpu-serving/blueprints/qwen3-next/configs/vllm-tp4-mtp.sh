#!/usr/bin/env bash
# vLLM tp4-x1 + MTP for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: tp4-x1 + MTP (speculative decoding with 2 draft tokens)
# Note: FP8 block_k=128 is incompatible with TP=8; TP=4 works.
# Run via nerdctl/docker on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen3_5-x86_64-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-qwen3-tp4-mtp \
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
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
