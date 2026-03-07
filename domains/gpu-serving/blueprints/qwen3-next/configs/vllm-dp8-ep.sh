#!/usr/bin/env bash
# vLLM dp8-ep for Qwen3-Next FP8 on p5en.48xlarge (8x H200)
# Config: DP=8 + Expert Parallel (throughput-max mode)
# Note: MTP is not compatible with --data-parallel-size. No MTP variant.
# Note: max-model-len reduced to 32768 to maximize batch size at high concurrency.
# Run via nerdctl/docker on GPU node
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen3_5-x86_64-cu130}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-qwen3-dp8-ep \
  --gpus all --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --expert-placement-strategy round_robin \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}' \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 32768 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --api-server-count 8 \
  --disable-log-requests \
  --port 8000
