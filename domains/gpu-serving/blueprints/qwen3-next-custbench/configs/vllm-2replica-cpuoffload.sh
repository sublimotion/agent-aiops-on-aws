#!/usr/bin/env bash
# Config E: 2x TP=4 replicas + CPU KV cache offload
# Launches two vLLM instances on GPUs 0-3 and 4-7 (ports 8000 and 8001)
# Both with --cpu-offload-gb 64 for expanded KV cache capacity.
# This doubles both prefill compute AND KV cache vs single replica,
# making it the best-case scenario for high-concurrency workloads.
# Use with a load balancer or round-robin across ports 8000/8001.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '615299764834')}"
IMAGE="${VLLM_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/qwen3-next-custbench:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

COMMON_ARGS=(
  --model "$MODEL"
  --tensor-parallel-size 4
  --quantization fp8
  --gpu-memory-utilization 0.92
  --max-model-len 32768
  --max-num-batched-tokens 32768
  --kv-cache-dtype fp8
  --enable-prefix-caching
  --max-num-seqs 512
  --compilation_config.pass_config.enable_fi_allreduce_fusion true
  --compilation_config.pass_config.enable_noop true
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
  --cpu-offload-gb 64
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --served-model-name qwen3-next
  --disable-log-requests
)

echo "Starting replica 1 on GPUs 0-3 (port 8000)..."
$CTR run --rm -d --name vllm-custbench-2rep-r0 \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$IMAGE" \
  "${COMMON_ARGS[@]}" \
  --port 8000

echo "Starting replica 2 on GPUs 4-7 (port 8001)..."
$CTR run --rm -d --name vllm-custbench-2rep-r1 \
  --gpus '"device=4,5,6,7"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  "$IMAGE" \
  "${COMMON_ARGS[@]}" \
  --port 8001

echo "Both replicas starting. Wait for health checks:"
echo "  curl http://localhost:8000/health"
echo "  curl http://localhost:8001/health"
