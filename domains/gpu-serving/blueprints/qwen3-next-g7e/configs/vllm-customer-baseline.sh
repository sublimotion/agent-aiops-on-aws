#!/usr/bin/env bash
# Customer's exact vLLM configuration for Qwen3-Next on g7e.48xlarge
# Reproduces customer's reported setup: MTP, no chunked prefill, no prefix caching
# Customer reports: ~3,612 tok/s throughput, TTFT p50 ~940ms at 1000 concurrent (on p5en)
#
# Uses customer's Dockerfile (Dockerfile.vllm-customer with model_type=qwen3-next)
# which installs vLLM nightly + transformers==4.57.1 (MTP works on this version)
#
# Build image first:
#   docker build -f docker/Dockerfile.vllm-customer --build-arg model_type=qwen3-next \
#     -t qwen3-next-customer:latest docker/
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-qwen3-next-customer:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-customer-baseline \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --tokenizer-mode auto \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-num-seqs 512 \
  --compilation_config.pass_config.enable_fi_allreduce_fusion true \
  --compilation_config.pass_config.enable_noop true \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
  --no-enable-chunked-prefill \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
