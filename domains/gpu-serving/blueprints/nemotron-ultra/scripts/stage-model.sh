#!/usr/bin/env bash
# Stage Nemotron-3-Ultra-550B-A55B-NVFP4 weights: HuggingFace -> S3 -> (init container) -> NVMe.
#
# Uses huggingface_hub.snapshot_download() Python API, NOT huggingface-cli
# (nemotron-super lesson #5: CLI had PATH issues in-container; also `huggingface-cli`
# was renamed to `hf` in hub v1.11+). ~352 GB (113 NVFP4 shards).
#
# MUST run on a machine with >=400 GB scratch + bandwidth (the GPU node's NVMe, or a
# build host) — NOT a laptop. The K8s init container (main.tf) then `aws s3 sync`s
# S3 -> /mnt/nvme on the GPU node.
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}"
S3_BUCKET="${S3_BUCKET:-qn-sglang-models-20260303161715850900000007}"
S3_PREFIX="${S3_PREFIX:-nemotron-3-ultra-nvfp4}"
REGION="${REGION:-us-west-2}"
LOCAL_DIR="${LOCAL_DIR:-/mnt/nvme/hf-stage/nemotron-3-ultra-nvfp4}"

echo "[stage-model] model=$MODEL_ID -> s3://$S3_BUCKET/$S3_PREFIX (region $REGION)"

mkdir -p "$LOCAL_DIR"

python3 - <<PY
from huggingface_hub import snapshot_download
import os
# NOTE: huggingface_hub v1.11+ removed `local_dir_use_symlinks` (and renamed the CLI
# huggingface-cli -> hf). Do NOT pass local_dir_use_symlinks on hub >= 1.x — it raises
# TypeError. local_dir already copies real files in v1.x. (See lessons.md hub-v1 entry.)
path = snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${LOCAL_DIR}",
    max_workers=16,
    # NVFP4 weights only — skip duplicate fp/bf16 mirrors if the repo carries them.
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.jinja", "tokenizer*", "*.model"],
)
print("snapshot_download complete:", path)
# Sanity: confirm config.json + at least one shard
assert os.path.exists(os.path.join("${LOCAL_DIR}", "config.json")), "config.json missing"
shards = [f for f in os.listdir("${LOCAL_DIR}") if f.endswith(".safetensors")]
print(f"shard count: {len(shards)}")
assert shards, "no safetensors shards downloaded"
PY

echo "[stage-model] uploading to S3 ..."
aws s3 sync "$LOCAL_DIR" "s3://${S3_BUCKET}/${S3_PREFIX}" --region "$REGION"

echo "[stage-model] verifying S3 contents ..."
aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --region "$REGION" | tail -10
SHARDS=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --region "$REGION" | grep -c '\.safetensors' || true)
echo "[stage-model] S3 shard count: $SHARDS (expect 113)"
echo "[stage-model] DONE. The serving pod's init container will sync this to /mnt/nvme."
