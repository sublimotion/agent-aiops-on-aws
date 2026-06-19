#!/usr/bin/env bash
# run-phase5.sh — Phase 5 frontier sub-phases on GPU node.
# 5a: default (compile + overlap + cuda graphs — all on in SGLang 0.5.10)
# 5b: --disable-cuda-graph to isolate graph benefit
# 5c: TP4 + DP2 (pipeline-parallel approximation on SGLang; K2.6 may not support)
# 5d: FP4 probe (profile only — kernels unavailable in sm_103)
# Winner params come from Phase 1b WINNER.env.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WINNER=/mnt/nvme/results/phase-1b/WINNER.env
if [[ ! -f "$WINNER" ]]; then
  SPEC_NUM_STEPS=4
  SPEC_NUM_DRAFT_TOKENS=4
  SPEC_EAGLE_TOPK=1
else
  source "$WINNER"
fi

MODEL_PATH=/mnt/nvme/models/kimi-k26-fp8
DRAFT_PATH=/mnt/nvme/models/kimi-k26-eagle3
IMAGE=lmsysorg/sglang:v0.5.10-cu130
RESULTS_BASE=/mnt/nvme/results/phase-5
mkdir -p "$RESULTS_BASE"

wait_health() {
  local port=${1:-30000} retries=300
  for i in $(seq 1 $retries); do
    curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

run_bench() {
  local results_dir=$1 port=$2
  for c in 1 8 32 64 128 256; do
    python3.11 "$SCRIPT_DIR/bench-one.py" "$port" "$c" 512 256 4 "$results_dir/c${c}.json" sglang || true
  done
}

stop_srv() {
  sudo docker stop "$1" 2>/dev/null || true
  sudo docker rm "$1" 2>/dev/null || true
}

launch_sglang() {
  local container=$1; shift
  local extra_flags=("$@")
  stop_srv "$container"
  sudo docker run -d --name "$container" \
    --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/nvme:/mnt/nvme \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    "$IMAGE" \
    bash -c "find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; \
      exec python3 -m sglang.launch_server \
        --model-path '$MODEL_PATH' \
        --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 \
        --speculative-algorithm EAGLE3 \
        --speculative-num-steps $SPEC_NUM_STEPS \
        --speculative-num-draft-tokens $SPEC_NUM_DRAFT_TOKENS \
        --speculative-eagle-topk $SPEC_EAGLE_TOPK \
        --speculative-draft-model-path '$DRAFT_PATH' \
        --speculative-draft-attention-backend trtllm_mha \
        --enable-hierarchical-cache --hicache-size 200 \
        --trust-remote-code --host 0.0.0.0 --port 30000 \
        ${extra_flags[*]}"
}

# ---- 5a: default stack (compile + overlap + cuda graphs, all on) with HiCache ----
run_5a() {
  local OUT="$RESULTS_BASE/5a-default-stack"
  mkdir -p "$OUT"
  [[ -f "$OUT/done" ]] && { echo "[5a] skip"; return 0; }
  echo "[5a] launching default stack"
  launch_sglang sglang-k26-p5a --tp 8
  if ! wait_health; then
    echo "[5a] health timeout"
    sudo docker logs sglang-k26-p5a > "$OUT/server.log" 2>&1 || true
    stop_srv sglang-k26-p5a
    return 1
  fi
  run_bench "$OUT" 30000
  sudo docker logs sglang-k26-p5a > "$OUT/server.log" 2>&1 || true
  stop_srv sglang-k26-p5a
  touch "$OUT/done"
}

# ---- 5b: --disable-cuda-graph to isolate graph benefit ----
run_5b() {
  local OUT="$RESULTS_BASE/5b-no-cuda-graph"
  mkdir -p "$OUT"
  [[ -f "$OUT/done" ]] && { echo "[5b] skip"; return 0; }
  echo "[5b] launching no-cuda-graph"
  launch_sglang sglang-k26-p5b --tp 8 --disable-cuda-graph
  if ! wait_health; then
    echo "[5b] health timeout"
    sudo docker logs sglang-k26-p5b > "$OUT/server.log" 2>&1 || true
    stop_srv sglang-k26-p5b
    return 1
  fi
  run_bench "$OUT" 30000
  sudo docker logs sglang-k26-p5b > "$OUT/server.log" 2>&1 || true
  stop_srv sglang-k26-p5b
  touch "$OUT/done"
}

# ---- 5c: TP4 + DP2 (pipeline-parallel approximation via SGLang DP) ----
run_5c() {
  local OUT="$RESULTS_BASE/5c-tp4-dp2"
  mkdir -p "$OUT"
  [[ -f "$OUT/done" ]] && { echo "[5c] skip"; return 0; }
  echo "[5c] launching tp4 dp2"
  launch_sglang sglang-k26-p5c --tp 4 --dp 2
  if ! wait_health; then
    echo "[5c] health timeout (DP+spec decode may be incompatible)"
    sudo docker logs sglang-k26-p5c > "$OUT/server.log" 2>&1 || true
    stop_srv sglang-k26-p5c
    touch "$OUT/skipped"
    return 0
  fi
  run_bench "$OUT" 30000
  sudo docker logs sglang-k26-p5c > "$OUT/server.log" 2>&1 || true
  stop_srv sglang-k26-p5c
  touch "$OUT/done"
}

# ---- 5d: NVFP4 / FP4 probe (profile only) ----
run_5d() {
  local OUT="$RESULTS_BASE/5d-fp4-probe"
  mkdir -p "$OUT"
  [[ -f "$OUT/done" ]] && return 0
  echo "[5d] fp4 capability probe"
  sudo docker run --rm --gpus all -v /mnt/nvme:/mnt/nvme "$IMAGE" \
    python3 -c "
import torch
print('cuda:', torch.version.cuda)
print('compute cap:', torch.cuda.get_device_capability(0))
for dt in ['float8_e4m3fn', 'float8_e5m2']:
    try:
        x = torch.zeros(4, 4, dtype=getattr(torch, dt), device='cuda')
        print(f'{dt}: supported')
    except Exception as e:
        print(f'{dt}:', e)
# FP4 path is sm_100+ / cutlass 3.x / triton — not directly exposed in torch 2.x
print('note: FP4 tensor cores require custom kernels (cutlass 3.x or triton fp4)')
" > "$OUT/fp4-probe.log" 2>&1 || true
  touch "$OUT/done"
}

run_5a
run_5b
run_5c
run_5d
echo "[phase-5] all sub-phases complete"
