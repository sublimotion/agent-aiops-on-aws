#!/usr/bin/env bash
# Phase-3 gate runner. Wraps run-bridging-cell.sh with:
#   - seccomp-wrap on the harness (blocks io_uring like the Dynamo agent does)
#   - nvidia-smi snapshots before+after llm.sleep()
#   - Gate 2/3 evaluation against fixed thresholds
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-checkpoint
SMOKE_OUT_DIR=/mnt/nvme/smoke
LOG_DIR=/mnt/nvme/smoke/logs
HARNESS=/home/ec2-user/dynamo-scripts/smoke-vllm-sleep.py
VENV=/mnt/nvme/venv

mkdir -p "${CKPT_DIR}" "${SMOKE_OUT_DIR}" "${LOG_DIR}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
HLOG="${LOG_DIR}/harness.log"
GPULOG="${LOG_DIR}/gpu-mem.log"
: > "${HLOG}"; : > "${GPULOG}"

echo "[orch] launching workload via seccomp-wrap (log: ${HLOG})"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# setsid + new pgid; wrap with seccomp-wrap to enforce no-io_uring on the harness
# (mirrors Dynamo snapshot-agent's localhost seccomp profile).
setsid --fork bash -c "HF_HOME=/mnt/nvme/hf UV_USE_IO_URING=0 exec /usr/local/bin/seccomp-wrap ${VENV}/bin/python ${HARNESS}" \
  >"${HLOG}" 2>&1 &
ORCH_CHILD=$!
echo "[orch] launcher pid=${ORCH_CHILD}"

# Background GPU memory sampler
(
  while true; do
    ts=$(date +%s.%N)
    line=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "${ts} ${line}" >> "${GPULOG}"
    sleep 1
  done
) &
GPU_SAMPLER_PID=$!
trap "kill ${GPU_SAMPLER_PID} 2>/dev/null || true" EXIT

# Wait for READY_FOR_CHECKPOINT
WPID=""
for i in $(seq 1 600); do
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  if ! pgrep -f smoke-vllm-sleep >/dev/null && [ ${i} -gt 5 ]; then
    echo "[orch] harness exited before READY (after ${i}s); tail of log:"; tail -60 "${HLOG}"; exit 2
  fi
  sleep 1
done
if [ -z "${WPID}" ]; then echo "[orch] timed out"; tail -80 "${HLOG}"; exit 2; fi
echo "[orch] workload READY_FOR_CHECKPOINT pid=${WPID}"

echo "[orch] GPU mem after sleep() (workload reports READY only after sleep):"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
RSS_KB=$(awk '/VmRSS/ {print $2}' /proc/${WPID}/status 2>/dev/null || echo "?")
echo "[orch] harness RSS=${RSS_KB} kB"

# Dynamo C/R sequence
sudo /usr/local/sbin/cuda-checkpoint --action lock --pid "${WPID}" --timeout 30000 || { echo "[orch] lock failed"; exit 3; }
echo "[orch] cuda-checkpoint lock OK"

T0=$(date +%s.%N)
sudo /usr/local/sbin/cuda-checkpoint --action checkpoint --pid "${WPID}" || { echo "[orch] checkpoint failed"; exit 3; }
T1=$(date +%s.%N)
CP_DUR=$(awk -v a=${T0} -v b=${T1} 'BEGIN{print b-a}')
echo "[orch] cuda-checkpoint dur=${CP_DUR} s"

T2=$(date +%s.%N)
if ! sudo /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --link-remap --leave-running \
  --log-file dump.log -v3; then
  echo "[orch] criu dump FAILED; tail:"
  sudo tail -120 "${CKPT_DIR}/dump.log" 2>/dev/null
  sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "${WPID}" 2>/dev/null || true
  sudo /usr/local/sbin/cuda-checkpoint --action unlock --pid "${WPID}" 2>/dev/null || true
  exit 3
fi
T3=$(date +%s.%N)
CRIU_DUMP=$(awk -v a=${T2} -v b=${T3} 'BEGIN{print b-a}')
echo "[orch] criu dump dur=${CRIU_DUMP} s"

sudo /usr/local/sbin/cuda-checkpoint --action restore --pid "${WPID}" || true
sudo /usr/local/sbin/cuda-checkpoint --action unlock --pid "${WPID}" || true

ART_BYTES=$(sudo du -sb "${CKPT_DIR}" | awk '{print $1}')
ART_GIB=$(awk -v b=${ART_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
echo "[orch] artifact: ${ART_BYTES} bytes = ${ART_GIB} GiB"

# Kill original tree, then criu restore
echo "[orch] killing original"
sudo pkill -9 -P "${WPID}" 2>/dev/null || true
sudo kill -9 "${WPID}" 2>/dev/null || true
sleep 2

T4=$(date +%s.%N)
if ! sudo /usr/local/sbin/criu restore \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --restore-detached \
  --log-file restore.log -v3; then
  echo "[orch] criu restore FAILED; tail:"
  sudo tail -200 "${CKPT_DIR}/restore.log" 2>/dev/null
  exit 4
fi
T5=$(date +%s.%N)
RESTORE=$(awk -v a=${T4} -v b=${T5} 'BEGIN{print b-a}')
echo "[orch] criu restore dur=${RESTORE} s"

# Find restored harness
RESTORED_PID=""
for _ in $(seq 1 30); do
  if sudo kill -0 "${WPID}" 2>/dev/null; then RESTORED_PID="${WPID}"; break; fi
  CAND=$(pgrep -f smoke-vllm-sleep.py | head -1 || true)
  if [ -n "${CAND}" ]; then RESTORED_PID="${CAND}"; break; fi
  sleep 1
done
echo "[orch] restored pid=${RESTORED_PID:-<none>}"
if [ -z "${RESTORED_PID}" ]; then echo "[orch] could not locate restored process"; exit 5; fi

sudo kill -USR1 "${RESTORED_PID}"
echo "[orch] sent SIGUSR1"

for _ in $(seq 1 90); do
  if [ -s "${SMOKE_OUT_DIR}/post.json" ]; then break; fi
  sleep 1
done

HARNESS_RC="?"
if grep -q "GATE_1_PASS" "${HLOG}"; then HARNESS_RC=0
elif grep -q "GATE_1_FAIL" "${HLOG}"; then HARNESS_RC=1
fi

# Model weight size for Gate-2 ratio
WEIGHTS_BYTES=$(du -sb /mnt/nvme/hf/models--Qwen--Qwen3-0.6B 2>/dev/null | awk '{print $1}')
WEIGHTS_GIB=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
RATIO=$(awk -v a=${ART_BYTES} -v b=${WEIGHTS_BYTES} 'BEGIN{if (b>0) printf "%.3f", a/b; else print "NA"}')

echo
echo "=== BRIDGING CELL — GATE RESULTS ==="
echo "cuda_checkpoint_seconds=${CP_DUR}"
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact_bytes=${ART_BYTES} (${ART_GIB} GiB)"
echo "weights_bytes=${WEIGHTS_BYTES} (${WEIGHTS_GIB} GiB)"
echo "artifact_to_weights_ratio=${RATIO}"
echo "harness_rc=${HARNESS_RC}  (0=GATE_1_PASS, 1=FAIL)"
case "${HARNESS_RC}" in
  0) echo "GATE 1 (token-id equality): PASS" ;;
  1) echo "GATE 1 (token-id equality): FAIL" ;;
  *) echo "GATE 1: UNKNOWN" ;;
esac
G2_PASS=$(awk -v r=${RATIO} 'BEGIN{print (r != "NA" && r+0 <= 2.0) ? "PASS" : "FAIL"}')
echo "GATE 2 (artifact <= 2x weights): ${G2_PASS}  (ratio=${RATIO})"
G3_PASS=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')
echo "GATE 3 (restore < 30s on NVMe): ${G3_PASS}  (${RESTORE} s)"

cat > "${SMOKE_OUT_DIR}/bridging-result.json" <<EOF
{
  "cuda_checkpoint_seconds": ${CP_DUR},
  "criu_dump_seconds": ${CRIU_DUMP},
  "criu_restore_seconds": ${RESTORE},
  "artifact_bytes": ${ART_BYTES},
  "artifact_gib": ${ART_GIB},
  "weights_bytes": ${WEIGHTS_BYTES},
  "weights_gib": ${WEIGHTS_GIB},
  "artifact_to_weights_ratio": ${RATIO},
  "harness_rc": "${HARNESS_RC}",
  "gate_1_token_equality": "$([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)",
  "gate_2_artifact_size": "${G2_PASS}",
  "gate_3_restore_latency": "${G3_PASS}"
}
EOF
echo
echo "result written to ${SMOKE_OUT_DIR}/bridging-result.json"
