#!/usr/bin/env bash
# run-phase1b-sweep.sh — 24-config EAGLE3 hyperparameter sweep on SGLang.
# Runs on the GPU node. Expects weights at /mnt/nvme/models/kimi-k26-{fp8,eagle3}.
# Output: /mnt/nvme/results/phase-1b/<config>/bench.json per config.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODEL_PATH=/mnt/nvme/models/kimi-k26-fp8
DRAFT_PATH=/mnt/nvme/models/kimi-k26-eagle3
IMAGE=lmsysorg/sglang:v0.5.10-cu130
RESULTS=/mnt/nvme/results/phase-1b
mkdir -p "$RESULTS"

# Sweep space — conservative to keep total runtime bounded
# Full 4x4x3 = 48 is too many; prune to 24 most-informative configs.
NUM_STEPS_VALS=(1 2 3 4)
NUM_DRAFT_VALS=(2 4 6 8)
TOPK_VALS=(1 2)

# Benchmark params per config (keep short — goal is sweep, not deep per-config)
BENCH_IN=512
BENCH_OUT=256
CONC_LEVELS=(1 8 32 64 128)
REQS_PER_CONC=4

wait_for_health() {
  local port=${1:-30000}
  local retries=180  # 15 min total
  for i in $(seq 1 $retries); do
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

clean_cubin_cache() {
  # Per L12 — SGLang + FlashInfer JIT symlink race. Clear in a disposable container BEFORE sglang starts.
  sudo docker run --rm -v /mnt/nvme:/mnt/nvme "$IMAGE" \
    bash -c 'rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export 2>/dev/null || true' 2>/dev/null || true
}

run_config() {
  local steps=$1 draft=$2 topk=$3
  local tag="s${steps}_d${draft}_k${topk}"
  local out="$RESULTS/$tag"
  mkdir -p "$out"

  if [[ -f "$out/done" ]]; then
    echo "[skip] $tag already complete"
    return 0
  fi

  echo "[run] $tag (steps=$steps draft=$draft topk=$topk)"
  sudo docker rm -f sglang-k26-b1 2>/dev/null || true

  # Start via bash -c to clear any stale flashinfer symlinks before launch (L12 fix)
  CID=$(sudo docker run -d --name sglang-k26-b1 \
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
        --speculative-num-steps $steps \
        --speculative-num-draft-tokens $draft \
        --speculative-eagle-topk $topk \
        --speculative-draft-model-path '$DRAFT_PATH' \
        --speculative-draft-attention-backend trtllm_mha \
        --trust-remote-code \
        --host 0.0.0.0 --port 30000")
  echo "[cid] $CID" > "$out/server.log"

  if ! wait_for_health 30000; then
    echo "[fail] $tag failed health check"
    sudo docker logs sglang-k26-b1 > "$out/server.log" 2>&1 || true
    sudo docker rm -f sglang-k26-b1 || true
    return 1
  fi

  for c in "${CONC_LEVELS[@]}"; do
    python3.11 /home/ec2-user/bench-one.py 30000 "$c" "$BENCH_IN" "$BENCH_OUT" "$REQS_PER_CONC" "$out/c${c}.json" sglang || true
  done

  # Capture engine metrics (accept rate) from server log
  sudo docker logs sglang-k26-b1 > "$out/server.log" 2>&1 || true

  sudo docker stop sglang-k26-b1 || true
  sudo docker rm sglang-k26-b1 || true
  touch "$out/done"
}

# Pruned sweep: skip redundant combos to stay under 24 total
# Priority: cover the cross-product of step-vs-draft-tokens, topk=1 for all, topk=2 only for promising cells
for steps in "${NUM_STEPS_VALS[@]}"; do
  for draft in "${NUM_DRAFT_VALS[@]}"; do
    # Skip configs where num_steps * eagle_topk > num_draft_tokens (infeasible)
    if (( steps > draft )); then continue; fi
    run_config "$steps" "$draft" 1 || true
  done
done

# Only run topk=2 on the 4 most promising cells (found after first pass)
# This will be decided dynamically — skip for now
# for cfg in "1 2" "2 4" "2 6" "3 6"; do
#   read s d <<< "$cfg"; run_config "$s" "$d" 2 || true
# done

echo "[done] Phase 1b sweep complete — see $RESULTS"
python3 - <<'PY'
import json, glob, os
rows = []
for d in sorted(glob.glob("/mnt/nvme/results/phase-1b/*")):
    if not os.path.isdir(d): continue
    tag = os.path.basename(d)
    for f in glob.glob(f"{d}/c*.json"):
        try:
            j = json.load(open(f))
            rows.append((tag, j["concurrency"], j["agg_tok_per_s"], j["per_req_tok_per_s"]))
        except Exception: pass
rows.sort(key=lambda r: -r[2])
print(f"{'config':<20} {'conc':>6} {'agg_tok/s':>12} {'per_req':>10}")
for r in rows[:20]:
    print(f"{r[0]:<20} {r[1]:>6} {r[2]:>12.1f} {r[3]:>10.1f}")
PY
