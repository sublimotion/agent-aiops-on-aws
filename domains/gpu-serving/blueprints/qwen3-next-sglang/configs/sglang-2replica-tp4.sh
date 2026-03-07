#!/usr/bin/env bash
# SGLang 2x TP=4 replicas for Qwen3-Next FP8 on g7e.48xlarge (8x RTX PRO 6000)
# Config: S5 — production topology: 2 replicas, each using 4 GPUs
# Replica A: GPUs 0-3 on port 30000, Replica B: GPUs 4-7 on port 30001
# Use a load balancer (nginx, HAProxy) to distribute requests.
#
# On p5en custbench: 2-replica achieved 1.71x throughput (84,628 tok/s combined)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

echo "=== Starting Replica A (GPUs 0-3, port 30000) ==="
$CTR run -d --name sglang-qwen3-replica-a \
  --gpus 4 --ipc=host --network=host --privileged \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 \
  --dtype bfloat16 \
  --context-length 65536 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --attention-backend triton \
  --disable-cuda-graph \
  --fp8-gemm-backend cutlass \
  --host 0.0.0.0 \
  --port 30000

echo "=== Starting Replica B (GPUs 4-7, port 30001) ==="
$CTR run -d --name sglang-qwen3-replica-b \
  --gpus 4 --ipc=host --network=host --privileged \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 \
  --dtype bfloat16 \
  --context-length 65536 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --attention-backend triton \
  --disable-cuda-graph \
  --fp8-gemm-backend cutlass \
  --host 0.0.0.0 \
  --port 30001

echo "=== Both replicas started ==="
echo "Replica A: http://localhost:30000"
echo "Replica B: http://localhost:30001"
