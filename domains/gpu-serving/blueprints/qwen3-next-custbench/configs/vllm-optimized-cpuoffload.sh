#!/usr/bin/env bash
# Config D: Optimized config WITH CPU KV cache offloading
# Same as Config B + --cpu-offload-gb 64 to expand KV cache capacity
# At 1000 concurrent × 10K input, GPU KV cache is ~2.2x oversubscribed.
# Offloading 64 GB per GPU to CPU DRAM (~512 GB total across 8 GPUs)
# nearly doubles effective KV capacity, reducing queue wait times.
# Trade-off: ~1-3ms per-token latency penalty from PCIe Gen5 transfers
#            vs hundreds of ms saved from reduced queuing at high concurrency.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '615299764834')}"
IMAGE="${VLLM_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/qwen3-next-custbench:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-custbench-opt-cpuoffload \
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
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
  --cpu-offload-gb 64 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
