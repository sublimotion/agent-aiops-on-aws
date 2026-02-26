#!/usr/bin/env bash
# Config for T5d: Memory-constrained (gpu-mem=0.30) + Dynamo KVBM tiered offload to FSx
# Simulates smaller-GPU scenario (e.g. g7e) on p5en by constraining VRAM.
# KV cache overflow flows through Dynamo's 4-tier hierarchy:
#   Tier 0: GPU VRAM (~22 GB KV at 0.30 util)
#   Tier 1: CPU DRAM (128 GB pinned)
#   Tier 2: FSx Lustre via NIXL GDS_MT (~500 GB)
#
# Uses dynamo-run (not vllm serve) with DynamoConnector for async KV transfers.
# DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true ensures offloading activates during benchmarks.
#
# Known issues from kimi-k2.5 Dynamo benchmarks:
#   - FSx permission: Container user cannot write to Lustre mount (UID mapping).
#     Fix: run container as root (--user 0:0) and pre-create cache dir with 777 on host.
#   - GDS buffer registration may fail in container (falls back to POSIX compat mode).
#     Fix: mount host /dev and cuFile config; --privileged helps but not sufficient alone.
#   - CPU cache OOM at 128GB (kimi lesson #25). Reduced to 64GB for safety.
#
# Requires:
#   - Dynamo KVBM Docker image (build from docker/Dockerfile.dynamo-kvbm)
#   - EFA-enabled FSx Lustre mounted at /mnt/fsx-efa on the host
#     FSx: fs-0952e4fd84eed47af (EFA=true, PERSISTENT_2, 4800 GiB, Lustre 2.15)
#     DNS: fs-0952e4fd84eed47af.fsx.us-east-2.amazonaws.com
#     Mount: falszb4v
#     Mount cmd: mount -t lustre -o flock fs-0952e4fd84eed47af.fsx.us-east-2.amazonaws.com@tcp:/falszb4v /mnt/fsx-efa
#     SG: sg-073b15d42ff3f4006 (must be attached to GPU node)
#   - Model at /mnt/nvme/models/qwen3-next-fp8 (copied from NVMe, not FSx-EFA)
set -euo pipefail
CTR="${CONTAINER_RUNTIME:-nerdctl}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '615299764834')}"
IMAGE="${DYNAMO_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/dynamo-kvbm-qwen3:latest}"
MODEL="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
DYNAMO_CACHE_DIR="${DYNAMO_CACHE_DIR:-/mnt/fsx-efa/kv-cache/qwen3-custbench}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
# Reduced from 128 to 64 per kimi lesson #25 (OOM at 128GB with large models)
KVBM_CPU_CACHE_GB="${KVBM_CPU_CACHE_GB:-64}"

echo "=== T5d: Constrained VRAM + Dynamo KVBM tiered offload to FSx ==="
echo "Model:        ${MODEL}"
echo "GPU mem util: ${GPU_MEM_UTIL}"
echo "FSx cache:    ${DYNAMO_CACHE_DIR}"
echo "CPU cache:    ${KVBM_CPU_CACHE_GB} GB"
echo "Disk cache:   500 GB"
echo "Image:        ${IMAGE}"

# Pre-create cache directory on FSx with permissive permissions.
# Kimi lesson: Dynamo container user cannot write to Lustre mount (UID mapping).
# Fix: create dir as root on host with 777 before starting container.
mkdir -p "${DYNAMO_CACHE_DIR}"
chmod 777 "${DYNAMO_CACHE_DIR}"

$CTR run --rm -d --name vllm-custbench-t5d-dynamo \
  --user 0:0 \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host --privileged \
  -v /mnt/nvme:/mnt/nvme:ro \
  -v /mnt/fsx-efa:/mnt/fsx-efa \
  -v /dev:/dev \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_TIMEOUT=1800 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e DYNAMO_EVENT_PLANE=zmq \
  -e CUFILE_ENV_PATH_JSON=/etc/cufile.json \
  -e NIXL_LOG_LEVEL=INFO \
  -e DYN_KVBM_CPU_CACHE_GB="${KVBM_CPU_CACHE_GB}" \
  -e DYN_KVBM_DISK_CACHE_GB=500 \
  -e DYN_KVBM_DISK_CACHE_DIR="${DYNAMO_CACHE_DIR}" \
  -e DYN_KVBM_NIXL_BACKEND_GDS_MT=true \
  -e DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true \
  "$IMAGE" \
  dynamo-run \
    --model "$MODEL" \
    --tensor-parallel-size 4 \
    --quantization fp8 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
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
