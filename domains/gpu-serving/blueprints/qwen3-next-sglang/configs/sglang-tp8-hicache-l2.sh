#!/usr/bin/env bash
# SGLang TP=8 + HiCache L2 (GPU→CPU) for Qwen3-Next FP8 on g7e.48xlarge
# Config: S3b-48xl — HiCache with CPU tier on 8-GPU NVLink
# g7e.48xlarge: 768 GB VRAM + ~1.5 TB CPU RAM = massive KV headroom
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"

$CTR run -d --name sglang-qwen3-tp8-hicache \
  --gpus 8 --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx:/mnt/fsx:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --attention-backend triton \
  --fp8-gemm-backend cutlass \
  --host 0.0.0.0 \
  --port 30000
