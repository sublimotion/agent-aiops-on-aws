#!/usr/bin/env bash
# vLLM tp4-x1 with customer MTP config for Qwen3-Next FP8 on g7e.48xlarge
# Config: tp4-x1 with MTP speculative decoding (2 tokens), FP8 KV cache
# Mirrors customer production config for benchmarking comparison
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
  --enable-prefix-caching \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
