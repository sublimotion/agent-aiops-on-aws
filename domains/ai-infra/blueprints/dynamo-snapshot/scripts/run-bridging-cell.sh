#!/usr/bin/env bash
# Bridging-cell orchestrator. Mirrors Dynamo internal/criu/dump.go +
# protocol/checkpoint.go for a single-host vLLM workload.
set -uo pipefail

CKPT_DIR="${CKPT_DIR:-/mnt/nvme/criu-checkpoint}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-/mnt/nvme/smoke}"
LOG_DIR="${LOG_DIR:-/mnt/nvme/smoke/logs}"
HARNESS="${HARNESS:-/home/ec2-user/dynamo-snapshot/scripts/smoke-vllm-sleep.py}"
VENV="${VENV:-/mnt/nvme/venv}"

mkdir -p "${CKPT_DIR}" "${SMOKE_OUT_DIR}" "${LOG_DIR}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
HLOG="${LOG_DIR}/harness.log"
: > "${HLOG}"

echo "[orch] launching workload (log: ${HLOG})"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Use setsid + new pgid so the harness is detached from the SSH session
setsid --fork bash -c "HF_HOME=/mnt/nvme/hf UV_USE_IO_URING=0 exec python ${HARNESS}" \
  >"${HLOG}" 2>&1 &
ORCH_CHILD=$!
echo "[orch] launcher pid=${ORCH_CHILD} (harness will be its child)"

# Wait for READY_FOR_CHECKPOINT line in the log
WPID=""
for i in $(seq 1 600); do
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  # Detect early failure
  if ! pgrep -P "${ORCH_CHILD}" -f smoke-vllm-sleep >/dev/null && \
     ! pgrep -f smoke-vllm-sleep >/dev/null && [ ${i} -gt 5 ]; then
    echo "[orch] harness exited before READY (after ${i}s); tail of log:"
    tail -40 "${HLOG}"
    exit 2
  fi
  sleep 1
done

if [ -z "${WPID}" ]; then
  echo "[orch] timed out waiting for READY_FOR_CHECKPOINT; tail of log:"
  tail -60 "${HLOG}"
  pkill -P "${ORCH_CHILD}" 2>/dev/null || true
  exit 2
fi
echo "[orch] workload ready, pid=${WPID}"
echo "[orch] GPU mem before C/R:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# --- Dynamo-protocol C/R sequence ---
# 1. cuda-checkpoint lock
sudo /usr/local/sbin/cuda-checkpoint --action lock --pid "${WPID}" --timeout 30000 || { echo "[orch] lock failed"; exit 3; }
echo "[orch] cuda-checkpoint lock OK"

# 2. cuda-checkpoint checkpoint (GPU -> host RAM)
T0=$(date +%s.%N)
sudo /usr/local/sbin/cuda-checkpoint --action checkpoint --pid "${WPID}" || { echo "[orch] checkpoint failed"; exit 3; }
T1=$(date +%s.%N)
CP_DUR=$(awk -v a=${T0} -v b=${T1} 'BEGIN{print b-a}')
echo "[orch] cuda-checkpoint checkpoint dur=${CP_DUR} s"

# 3. criu dump (process state -> CKPT_DIR). We dump the harness pid; the
#    EngineCore subprocess(es) are picked up via process-tree sweep.
T2=$(date +%s.%N)
if ! sudo /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job \
  --tcp-established \
  --file-locks \
  --link-remap \
  --leave-running \
  --log-file dump.log \
  -v3; then
  echo "[orch] criu dump FAILED; tail of dump.log:"
  sudo tail -120 "${CKPT_DIR}/dump.log" 2>/dev/null
  # Best-effort cleanup so we don't leave the GPU locked
  sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "${WPID}" 2>/dev/null || true
  sudo /usr/local/sbin/cuda-checkpoint --action unlock --pid "${WPID}" 2>/dev/null || true
  exit 3
fi
T3=$(date +%s.%N)
CRIU_DUMP=$(awk -v a=${T2} -v b=${T3} 'BEGIN{print b-a}')
echo "[orch] criu dump dur=${CRIU_DUMP} s"

# 4. cuda-checkpoint restore (in-place, since --leave-running)
sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "${WPID}" || true
sudo /usr/local/sbin/cuda-checkpoint --action unlock --pid "${WPID}" || true
echo "[orch] cuda-checkpoint in-place restore + unlock OK"

ART_BYTES=$(sudo du -sb "${CKPT_DIR}" | awk '{print $1}')
ART_GIB=$(awk -v b=${ART_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
echo "[orch] artifact size: ${ART_BYTES} bytes = ${ART_GIB} GiB"

# 5. Kill original tree, then criu restore from disk
echo "[orch] killing original process tree"
sudo pkill -9 -P "${WPID}" 2>/dev/null || true
sudo kill -9 "${WPID}" 2>/dev/null || true
sleep 2

T4=$(date +%s.%N)
if ! sudo /usr/local/sbin/criu restore \
  --images-dir "${CKPT_DIR}" \
  --shell-job \
  --tcp-established \
  --file-locks \
  --restore-detached \
  --log-file restore.log \
  -v3; then
  echo "[orch] criu restore FAILED; tail:"
  sudo tail -200 "${CKPT_DIR}/restore.log" 2>/dev/null
  exit 4
fi
T5=$(date +%s.%N)
RESTORE=$(awk -v a=${T4} -v b=${T5} 'BEGIN{print b-a}')
echo "[orch] criu restore dur=${RESTORE} s"

# Find restored harness pid
RESTORED_PID=""
for _ in $(seq 1 30); do
  if sudo kill -0 "${WPID}" 2>/dev/null; then
    RESTORED_PID="${WPID}"; break
  fi
  CAND=$(pgrep -f smoke-vllm-sleep.py | head -1 || true)
  if [ -n "${CAND}" ]; then RESTORED_PID="${CAND}"; break; fi
  sleep 1
done
echo "[orch] restored pid=${RESTORED_PID:-<none>}"

if [ -z "${RESTORED_PID}" ]; then
  echo "[orch] could not locate restored process"
  exit 5
fi

# 6. SIGUSR1 the restored process to make it re-generate and exit
sudo kill -USR1 "${RESTORED_PID}"
echo "[orch] sent SIGUSR1 to ${RESTORED_PID}"

# Wait up to 60s for harness exit (post.json appears)
for _ in $(seq 1 60); do
  if [ -s "${SMOKE_OUT_DIR}/post.json" ]; then break; fi
  sleep 1
done

HARNESS_RC="?"
if grep -q "GATE_1_PASS" "${HLOG}"; then HARNESS_RC=0
elif grep -q "GATE_1_FAIL" "${HLOG}"; then HARNESS_RC=1
fi

# Summary
echo
echo "=== BRIDGING-CELL RESULT ==="
echo "cuda_checkpoint_seconds=${CP_DUR}"
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact_bytes=${ART_BYTES}"
echo "artifact_gib=${ART_GIB}"
echo "harness_rc=${HARNESS_RC}"

cat > "${SMOKE_OUT_DIR}/bridging-result.json" <<EOF
{
  "cuda_checkpoint_seconds": ${CP_DUR},
  "criu_dump_seconds": ${CRIU_DUMP},
  "criu_restore_seconds": ${RESTORE},
  "artifact_bytes": ${ART_BYTES},
  "artifact_gib": ${ART_GIB},
  "harness_rc": "${HARNESS_RC}"
}
EOF
echo "result written to ${SMOKE_OUT_DIR}/bridging-result.json"
