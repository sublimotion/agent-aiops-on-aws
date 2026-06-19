#!/usr/bin/env bash
# serve-fullstack.sh — Phase 4 full composition (EAGLE3 + prefix cache + dynamic MLA + HiCache).
# Requires configs/fullstack.env to be filled in from Phases 1-3 winning values.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../configs/fullstack.env}"
source "$ENV_FILE"

: "${ENGINE:?set ENGINE=vllm|sglang in $ENV_FILE}"
: "${SPEC_NUM_STEPS:?set from Phase 1/2}"
: "${SPEC_NUM_DRAFT_TOKENS:?set from Phase 1/2}"

if [[ "$ENGINE" == "sglang" ]]; then
  SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
  docker rm -f sglang-k26-full 2>/dev/null || true
  exec docker run -d --name sglang-k26-full \
    --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/nvme:/mnt/nvme \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    lmsysorg/sglang:v0.5.10-cu130 \
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
      --enable-hierarchical-cache \
      --hicache-size "${HICACHE_SIZE:-200}" \
      --trust-remote-code \
      --host 0.0.0.0 --port 30000
else
  docker rm -f vllm-k26-full 2>/dev/null || true
  exec docker run -d --name vllm-k26-full \
    --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/nvme:/mnt/nvme \
    "${IMAGE:-voipmonitor/vllm:cu130-mtp-tuned-v3-20260423}" \
      --model "$MODEL_PATH" \
      --tensor-parallel-size "$TP" \
      --speculative-model "$DRAFT_PATH" \
      --num-speculative-tokens "$SPEC_NUM_DRAFT_TOKENS" \
      --enable-prefix-caching \
      ${ENABLE_DYNAMIC_MLA_ROUTING:+--enable-dynamic-mla-routing} \
      --max-model-len "${MAX_MODEL_LEN:-131072}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
      --tool-call-parser kimi_k2 \
      --reasoning-parser kimi_k2 \
      --trust-remote-code \
      --host 0.0.0.0 --port 8000
fi
