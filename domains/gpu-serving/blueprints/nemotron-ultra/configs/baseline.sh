#!/usr/bin/env bash
# Baseline serving config for Nemotron-3-Ultra-550B-A55B-NVFP4 on p6-b300.48xlarge.
# vLLM TP4 single replica (NVIDIA's documented unit), VERBATIM from the HF model card.
# Run this directly on the GPU node for debugging outside of K8s, or as a reference for
# the flags Terraform injects. B300 sm_103 -> -cu130 image.
#
# Sampling at request time is model-mandated: temperature=1.0, top_p=0.95 (NO greedy).
# Stage 0c WARN: --enable-prefix-caching + MTP may conflict with mamba 'align' mode.
# If startup fails on a mamba/prefix-cache error, remove --enable-prefix-caching.
set -euo pipefail

MODEL_CKPT="${MODEL_CKPT:-/mnt/nvme/models/nemotron-3-ultra-nvfp4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0-cu130}"

# nerdctl on B-series bare metal (containerd runtime); use docker if available.
RUN="${RUN:-sudo nerdctl}"

$RUN run -d --name nemotron-ultra-vllm \
  --gpus '"device=0,1,2,3"' \
  --ipc=host --network=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_CKPT":/model:ro \
  -v /mnt/nvme/vllm-cache:/mnt/nvme/vllm-cache \
  -v /mnt/nvme/triton-cache:/mnt/nvme/triton-cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_COMPILE_CACHE=/mnt/nvme/vllm-cache \
  -e TRITON_CACHE_DIR=/mnt/nvme/triton-cache \
  "$IMAGE" \
  /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name nvidia/nemotron-3-ultra \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'

echo "Started nemotron-ultra-vllm. Tail logs: $RUN logs -f nemotron-ultra-vllm"
