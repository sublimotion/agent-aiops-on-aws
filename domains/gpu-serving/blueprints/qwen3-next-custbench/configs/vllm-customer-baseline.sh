#!/usr/bin/env bash
# Config A: Customer's exact vLLM configuration for Qwen3-Next on p5en.48xlarge
# Reproduces customer's reported setup: MTP, no chunked prefill, no prefix caching
# Expected: ~3,612 tok/s throughput, TTFT p50 ~940ms at 1000 concurrent
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '615299764834')}"
IMAGE="${VLLM_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/qwen3-next-custbench:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-custbench-baseline \
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
