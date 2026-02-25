#!/usr/bin/env bash
# Config B: Optimized vLLM configuration for Qwen3-Next on p5en.48xlarge
# Customer config + our recommended fixes:
#   1. --quantization fp8 (weight quantization, frees ~80GB VRAM if was BF16)
#   2. --enable-prefix-caching (58-76% TTFT reduction with shared prefixes)
#   3. Removed --no-enable-chunked-prefill (allows scheduler interleaving)
#   4. --gpu-memory-utilization 0.92 (safe on H200, more KV cache)
#   5. --tool-call-parser qwen3_coder (native Qwen3 parser)
#   6. Keep MTP speculative decoding (works on customer's vLLM version)
#   7. Keep compilation config flags (low risk)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_IMAGE:-615299764834.dkr.ecr.us-east-2.amazonaws.com/qwen3-next-custbench:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-custbench-optimized \
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
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
