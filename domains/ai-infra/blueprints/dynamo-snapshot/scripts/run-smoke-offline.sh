#!/usr/bin/env bash
# Orchestrator: launches smoke-offline.py, waits for READY, runs
# cuda-checkpoint + criu dump/restore, then signals resume.
set -euo pipefail

WORK=/opt/dynamo-snapshot
CKPT_DIR=/mnt/nvme/criu-checkpoint
LOG=/tmp/dynamo-snapshot/smoke.log
mkdir -p "$CKPT_DIR" /tmp/dynamo-snapshot
rm -rf "$CKPT_DIR"/* "$LOG"

source "$WORK/venv/bin/activate"
export HF_HOME=/mnt/nvme/hf-cache
mkdir -p "$HF_HOME"

# Background-launch the smoke harness, capture stdout
python "$WORK/smoke-offline.py" > "$LOG" 2>&1 &
SMOKE_PID=$!
echo "[orch] smoke pid=$SMOKE_PID"

# Wait for READY_FOR_CHECKPOINT marker
for i in $(seq 1 600); do
  if grep -q "READY_FOR_CHECKPOINT" "$LOG" 2>/dev/null; then
    echo "[orch] saw READY at t=${i}s"; break
  fi
  if ! kill -0 "$SMOKE_PID" 2>/dev/null; then
    echo "[orch] smoke died early — log tail:"; tail -40 "$LOG"; exit 2
  fi
  sleep 1
done
TARGET_PID=$(grep -oE 'pid=[0-9]+' "$LOG" | head -1 | cut -d= -f2)
echo "[orch] target pid=$TARGET_PID"

# Step 1: lock GPU state
echo "[orch] cuda-checkpoint --action lock"
T0=$(date +%s.%N)
sudo /usr/local/sbin/cuda-checkpoint --action lock --pid "$TARGET_PID" --timeout 30000

echo "[orch] cuda-checkpoint --action checkpoint"
sudo /usr/local/sbin/cuda-checkpoint --action checkpoint --pid "$TARGET_PID"

# Step 2: criu dump (leave-running so we can resume in place if dump succeeds)
echo "[orch] criu dump"
T_DUMP_S=$(date +%s.%N)
sudo criu dump -t "$TARGET_PID" -D "$CKPT_DIR" \
  --shell-job --tcp-established --file-locks --link-remap \
  --leave-running -v3 -o dump.log || {
    echo "[orch] criu dump FAILED"
    sudo tail -80 "$CKPT_DIR/dump.log" 2>/dev/null || true
    sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "$TARGET_PID" 2>/dev/null || true
    sudo /usr/local/sbin/cuda-checkpoint --action unlock  --pid "$TARGET_PID" 2>/dev/null || true
    kill -USR1 "$TARGET_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$SMOKE_PID" 2>/dev/null || true
    exit 3
  }
T_DUMP_E=$(date +%s.%N)
DUMP_SEC=$(python3 -c "print(${T_DUMP_E}-${T_DUMP_S})")
echo "[orch] dump took ${DUMP_SEC}s"

ART_BYTES=$(sudo du -sb "$CKPT_DIR" | awk '{print $1}')
WEIGHTS_BYTES=$(du -sb "$HF_HOME" 2>/dev/null | awk '{print $1}')
echo "[orch] artifact=$ART_BYTES weights=$WEIGHTS_BYTES"

# Resume original (we used --leave-running)
echo "[orch] cuda-checkpoint --action restore (in-place)"
sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "$TARGET_PID"
sudo /usr/local/sbin/cuda-checkpoint --action unlock  --pid "$TARGET_PID"

# Optionally: kill original and criu restore from disk to validate restore path.
# For Gate 3 (restore time), we need the actual restore-from-disk timing.
# Do that next:
echo "[orch] killing live process to test from-disk restore"
kill -9 "$SMOKE_PID" 2>/dev/null || true
sleep 3

echo "[orch] criu restore from $CKPT_DIR"
T_RESTORE_S=$(date +%s.%N)
sudo criu restore -D "$CKPT_DIR" --shell-job --tcp-established --file-locks \
  -v3 -o restore.log --restore-detached || {
    echo "[orch] criu restore FAILED"
    sudo tail -80 "$CKPT_DIR/restore.log" 2>/dev/null || true
    exit 4
  }
T_RESTORE_E=$(date +%s.%N)
RESTORE_SEC=$(python3 -c "print(${T_RESTORE_E}-${T_RESTORE_S})")
echo "[orch] restore took ${RESTORE_SEC}s"

# After restore, the smoke process is alive again. Send cuda-checkpoint resume
# and SIGUSR1 to make it generate the post-completion.
RESTORED_PID=$(pgrep -f smoke-offline.py | head -1)
echo "[orch] restored pid=$RESTORED_PID"

sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "$RESTORED_PID" || true
sudo /usr/local/sbin/cuda-checkpoint --action unlock  --pid "$RESTORED_PID" || true
kill -USR1 "$RESTORED_PID"

# Wait for completion
for i in $(seq 1 60); do
  if grep -qE "GATE_1_TOKEN_EQUALITY: (PASS|FAIL)" "$LOG"; then
    break
  fi
  sleep 1
done

echo "============ SMOKE LOG TAIL ============"
tail -40 "$LOG"

# Final gate computation
python3 - <<PY
import json, pathlib
WORK = pathlib.Path("/tmp/dynamo-snapshot")
pre = json.loads((WORK/"pre.json").read_text())
post = json.loads((WORK/"post.json").read_text())
g1 = pre["token_ids"] == post["token_ids"]
art = $ART_BYTES; wts = $WEIGHTS_BYTES or 1
g2 = art <= 2*wts
g3 = $RESTORE_SEC < 30.0
print(f"DUMP_SEC      = ${DUMP_SEC}")
print(f"RESTORE_SEC   = ${RESTORE_SEC}")
print(f"ARTIFACT_B    = {art}  ({art/2**30:.2f} GiB)")
print(f"WEIGHTS_B     = {wts}  ({wts/2**30:.2f} GiB)")
print(f"GATE 1 TOKEN EQUALITY:    {'PASS' if g1 else 'FAIL'}")
print(f"GATE 2 ARTIFACT <= 2x WT: {'PASS' if g2 else 'FAIL'}  ({art/wts:.2f}x)")
print(f"GATE 3 RESTORE < 30s:     {'PASS' if g3 else 'FAIL'}")
PY
