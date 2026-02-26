#!/usr/bin/env bash
# Config C: Optimized config WITHOUT MTP (isolate MTP contribution)
# Same as Config B but without --speculative-config
# Compare against Config B to measure MTP's impact on throughput and latency
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '615299764834')}"
IMAGE="${VLLM_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/qwen3-next-custbench:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-custbench-opt-nomtp \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-seqs 512 \
  --compilation_config.pass_config.enable_fi_allreduce_fusion true \
  --compilation_config.pass_config.enable_noop true \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
