#!/usr/bin/env bash
# Stage 0 smoke test:
#   1. Boot vLLM serving Qwen3-0.6B
#   2. Warm with a deterministic prompt, hash first 64 token IDs
#   3. cuda-checkpoint --toggle (suspend GPU contexts)
#   4. criu dump
#   5. criu restore
#   6. cuda-checkpoint --toggle (resume)
#   7. Same prompt → same token IDs?  → Gate 1 (correctness)
#   8. Artifact size  ≤ 2× model weights  → Gate 2
#   9. Restore wall-clock < 30s for 6 GiB on local NVMe → Gate 3
set -euo pipefail

WORK=/opt/dynamo-snapshot
CKPT_DIR=/mnt/nvme/criu-checkpoint
mkdir -p "$CKPT_DIR"
source "$WORK/venv/bin/activate"

MODEL=Qwen/Qwen3-0.6B  # placeholder; spec calls for "Qwen3-0.6B class"
# If the model isn't yet on HF under that exact name, fall back to a known small
# vLLM-compatible model.
PROMPT="The capital of France is"
SEED=42
MAX_TOKENS=64

echo "=== Cold-start vLLM (PID will be checkpointed) ==="
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enable-sleep-mode \
  --port 8000 \
  > "$WORK/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"

echo "=== Wait for /v1/models ==="
for i in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM ready after ${i}s"; break
  fi
  sleep 1
done

deterministic_completion() {
  curl -fsS -X POST http://127.0.0.1:8000/v1/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0,\"seed\":$SEED,\"n\":1,\"stream\":false}"
}

echo "=== Warm + capture pre-checkpoint completion ==="
PRE_JSON=$(deterministic_completion)
echo "$PRE_JSON" > "$WORK/pre.json"

echo "=== cuda-checkpoint --toggle (suspend) ==="
/usr/local/sbin/cuda-checkpoint --toggle --pid "$VLLM_PID"

echo "=== criu dump ==="
T0=$(date +%s.%N)
sudo criu dump -t "$VLLM_PID" -D "$CKPT_DIR" --shell-job --tcp-established \
  --file-locks --leave-running -v4 -o dump.log || {
  echo "criu dump FAILED — see $CKPT_DIR/dump.log"; exit 3; }
T1=$(date +%s.%N)
DUMP_SEC=$(echo "$T1 - $T0" | bc -l)
echo "dump took ${DUMP_SEC}s"

ART_BYTES=$(du -sb "$CKPT_DIR" | awk '{print $1}')
WEIGHTS_BYTES=$(du -sb "$HOME/.cache/huggingface/hub" 2>/dev/null | awk '{print $1}')
echo "artifact bytes=$ART_BYTES weights bytes=$WEIGHTS_BYTES"

echo "=== Kill original, criu restore ==="
kill -9 "$VLLM_PID" 2>/dev/null || true
sleep 2

T2=$(date +%s.%N)
sudo criu restore -D "$CKPT_DIR" --shell-job --tcp-established --file-locks -v4 -o restore.log &
RESTORE_PID=$!
# Wait for vLLM to respond again
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    T3=$(date +%s.%N)
    RESTORE_SEC=$(echo "$T3 - $T2" | bc -l)
    echo "restore took ${RESTORE_SEC}s"; break
  fi
  sleep 1
done

echo "=== cuda-checkpoint --toggle (resume) ==="
/usr/local/sbin/cuda-checkpoint --toggle --pid "$RESTORE_PID" || true

echo "=== Post-restore completion ==="
POST_JSON=$(deterministic_completion)
echo "$POST_JSON" > "$WORK/post.json"

echo "=== Gate evaluation ==="
python - <<PY
import json, hashlib, sys
pre = json.load(open("$WORK/pre.json"))["choices"][0]["text"]
post = json.load(open("$WORK/post.json"))["choices"][0]["text"]
ph = hashlib.sha256(pre.encode()).hexdigest()
qh = hashlib.sha256(post.encode()).hexdigest()
print("pre :", ph, repr(pre[:80]))
print("post:", qh, repr(post[:80]))
gate1 = (ph == qh)
art = $ART_BYTES; wts = $WEIGHTS_BYTES or 1
gate2 = (art <= 2*wts)
gate3 = ($RESTORE_SEC < 30.0)
print(f"Gate 1 (token equality): {'PASS' if gate1 else 'FAIL'}")
print(f"Gate 2 (artifact <= 2x weights, {art}/{wts} = {art/wts:.2f}x): {'PASS' if gate2 else 'FAIL'}")
print(f"Gate 3 (restore < 30s, $RESTORE_SEC s): {'PASS' if gate3 else 'FAIL'}")
sys.exit(0 if (gate1 and gate2 and gate3) else 1)
PY
