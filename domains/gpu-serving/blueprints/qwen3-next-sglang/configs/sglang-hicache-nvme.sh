#!/usr/bin/env bash
# SGLang TP=4 + HiCache L2+L3 (GPU→CPU→NVMe) for Qwen3-Next FP8 on g7e.24xlarge
# Config: S3c — HiCache with CPU + NVMe tiers
# Phase S3: Full tiered KV cache offloading
#
# NOTE: g7e.24xlarge uses PCIe — no EFA, no GDS support.
#       NVMe L3 uses standard file I/O (kernel backend), not GDS.
#       For GDS + EFA KV offloading, use p5en.48xlarge instead.
#
# NOTE: --disable-cuda-graph required for HiCache + hybrid attention
# NOTE: NVMe mount must be read-write (-v /mnt/nvme:/mnt/nvme, no :ro)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
NVME_CACHE_PATH="/mnt/nvme/kv-cache"

# Ensure NVMe cache directory exists on host
mkdir -p "${NVME_CACHE_PATH}"

$CTR run -d --name sglang-qwen3-hicache-nvme \
  --gpus 4 --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme \
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
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-storage-backend file \
  --hicache-storage-backend-extra-config '{"path": "/mnt/nvme/kv-cache"}' \
  --tool-call-parser qwen3_coder \
  --served-model-name qwen3-next \
  --attention-backend triton \
  --fp8-gemm-backend cutlass \
  --host 0.0.0.0 \
  --port 30000
