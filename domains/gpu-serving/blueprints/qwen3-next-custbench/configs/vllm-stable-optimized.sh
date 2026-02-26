#!/usr/bin/env bash
# Config F: Latest stable vLLM with our optimized flags
# Uses vllm/vllm-openai:latest (V1 engine architecture) instead of customer's
# v0.11.0 nightly. Tests whether upgrading to latest stable alone improves
# latency, independent of flag changes.
#
# V1 engine differences:
#   - Async scheduling (no head-of-line blocking from prefill)
#   - Improved chunked prefill (enabled by default)
#   - Better prefix caching implementation
#   - --cpu-offload-gb is BROKEN on V1 (do not add)
#
# Note: MTP speculative decoding may not work on latest stable if
# qwen3_next_mtp method isn't merged yet. If --speculative-config fails
# at startup, remove it and note the incompatibility.
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${VLLM_STABLE_IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run --rm -d --name vllm-custbench-stable-optimized \
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
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --served-model-name qwen3-next \
  --disable-log-requests \
  --port 8000
