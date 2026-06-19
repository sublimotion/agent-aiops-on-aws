#!/usr/bin/env bash
# run-phase4-fullstack.sh — SGLang EAGLE3 (winning config) + HiCache 200 GB/rank.
# Reads winning spec decode params from /mnt/nvme/results/phase-1b/WINNER.env

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WINNER=/mnt/nvme/results/phase-1b/WINNER.env
if [[ ! -f "$WINNER" ]]; then
  echo "WINNER.env missing — falling back to defaults"
  SPEC_NUM_STEPS=2
  SPEC_NUM_DRAFT_TOKENS=4
  SPEC_EAGLE_TOPK=1
else
  source "$WINNER"
fi

MODEL_PATH=/mnt/nvme/models/kimi-k26-fp8
DRAFT_PATH=/mnt/nvme/models/kimi-k26-eagle3
IMAGE=lmsysorg/sglang:v0.5.10-cu130
RESULTS=/mnt/nvme/results/phase-4
mkdir -p "$RESULTS"

sudo docker rm -f sglang-k26-full 2>/dev/null || true
sudo docker run -d --name sglang-k26-full \
  --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/nvme:/mnt/nvme \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  "$IMAGE" \
  bash -c "find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; \
    exec python3 -m sglang.launch_server \
      --model-path '$MODEL_PATH' \
      --tp 8 \
      --reasoning-parser kimi_k2 \
      --tool-call-parser kimi_k2 \
      --speculative-algorithm EAGLE3 \
      --speculative-num-steps $SPEC_NUM_STEPS \
      --speculative-num-draft-tokens $SPEC_NUM_DRAFT_TOKENS \
      --speculative-eagle-topk $SPEC_EAGLE_TOPK \
      --speculative-draft-model-path '$DRAFT_PATH' \
      --speculative-draft-attention-backend trtllm_mha \
      --enable-hierarchical-cache \
      --hicache-size 200 \
      --trust-remote-code \
      --host 0.0.0.0 --port 30000"
sleep 2
sudo docker logs sglang-k26-full > "$RESULTS/server.log" 2>&1 || true

# Wait for health
for i in {1..180}; do
  if curl -sf http://localhost:30000/health >/dev/null 2>&1; then
    echo "[health] ready after ${i}*5s"
    break
  fi
  sleep 5
done

# Run concurrency sweep
for c in 1 8 32 64 128 256 512; do
  echo "[bench] c=$c"
  python3.11 "$SCRIPT_DIR/bench-one.py" 30000 "$c" 512 256 4 "$RESULTS/c${c}.json" sglang || true
done

sudo docker logs sglang-k26-full 2>&1 | grep -E "accept|spec|hicache" | tail -100 > "$RESULTS/metrics.log"
sudo docker stop sglang-k26-full
sudo docker rm sglang-k26-full

echo "[done] Phase 4 results in $RESULTS"
