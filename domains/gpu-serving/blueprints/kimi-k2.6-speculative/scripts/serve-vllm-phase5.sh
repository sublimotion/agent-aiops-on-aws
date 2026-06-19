#!/usr/bin/env bash
# serve-vllm-phase5.sh — Phase 5 frontier stack.
# Variants: compile | overlap | pp2tp4 | nvfp4 | all
# Usage:   ./serve-vllm-phase5.sh {compile|overlap|pp2tp4|nvfp4|all}

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../configs/vllm-phase5.env}"
source "$ENV_FILE"

VARIANT="${1:-all}"
CONTAINER="vllm-k26-p5-${VARIANT}"
docker rm -f "$CONTAINER" 2>/dev/null || true

COMMON_FLAGS=(
  --model "$MODEL_PATH"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --tool-call-parser kimi_k2
  --reasoning-parser kimi_k2
  --trust-remote-code
  --host 0.0.0.0 --port 8000
)

# Spec decode args (from Phase 4 winning config, defaults here)
SPEC_FLAGS=(
  --speculative-model "$DRAFT_PATH"
  --num-speculative-tokens "$NUM_SPEC_TOKENS"
)

case "$VARIANT" in
  compile)
    EXTRA=(--tensor-parallel-size "$TP"
           --compilation-config "{\"level\": ${COMPILATION_LEVEL}, \"backend\": \"inductor\"}")
    ;;
  overlap)
    # vLLM V1 async scheduler; fall back if flag unrecognized by image.
    EXTRA=(--tensor-parallel-size "$TP" --scheduler-policy async)
    ;;
  pp2tp4)
    # PP2 x TP4 — note: SPEC_FLAGS may be incompatible; caller can unset.
    EXTRA=(--tensor-parallel-size 4 --pipeline-parallel-size 2)
    ;;
  nvfp4)
    # FP4 tensor core path for verification. Requires image with FP4 kernels.
    EXTRA=(--tensor-parallel-size "$TP"
           --compilation-config "{\"level\": 3, \"backend\": \"inductor\"}"
           --kv-cache-dtype fp8)
    ;;
  all)
    EXTRA=(--tensor-parallel-size "$TP"
           --compilation-config "{\"level\": 3, \"backend\": \"inductor\"}"
           --scheduler-policy async)
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Valid: compile | overlap | pp2tp4 | nvfp4 | all" >&2
    exit 2
    ;;
esac

exec docker run -d --name "$CONTAINER" \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/nvme:/mnt/nvme \
  -e VLLM_USE_V1=1 \
  "$IMAGE" \
    "${COMMON_FLAGS[@]}" \
    "${SPEC_FLAGS[@]}" \
    "${EXTRA[@]}"
