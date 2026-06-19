#!/usr/bin/env bash
# serve-sglang-eagle3.sh — Start SGLang with EAGLE3 speculative decode.
# Phase 1 of the speculative spec. Runs on the GPU node.
#
# Sweep knobs (override via env):
#   SPEC_NUM_STEPS=3
#   SPEC_NUM_DRAFT_TOKENS=4
#   SPEC_EAGLE_TOPK=1

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../configs/sglang-eagle3.env}"
source "$ENV_FILE"

SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"

IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.10-cu130}"
CONTAINER_NAME="sglang-k26-eagle3"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

exec docker run -d --name "$CONTAINER_NAME" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/nvme:/mnt/nvme \
  -e SGLANG_ENABLE_SPEC_V2="$SGLANG_ENABLE_SPEC_V2" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp "$TP" \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps "$SPEC_NUM_STEPS" \
    --speculative-eagle-topk "$SPEC_EAGLE_TOPK" \
    --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS" \
    --speculative-draft-model-path "$DRAFT_PATH" \
    --speculative-draft-attention-backend trtllm_mha \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000
