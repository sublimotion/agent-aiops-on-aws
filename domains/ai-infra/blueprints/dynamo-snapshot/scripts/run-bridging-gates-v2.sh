#!/usr/bin/env bash
# Phase-3 gate runner v2.
# Per cuda_plugin.c: PAUSE_DEVICES + CHECKPOINT_DEVICES + RESUME_DEVICES_LATE
# hooks are invoked by `criu dump` automatically per-pid in the tree. So
# DO NOT call cuda-checkpoint manually — that conflicts with the plugin.
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-checkpoint
SMOKE_OUT_DIR=/mnt/nvme/smoke
LOG_DIR=/mnt/nvme/smoke/logs
HARNESS=/home/ec2-user/dynamo-scripts/smoke-vllm-sleep.py
VENV=/mnt/nvme/venv

mkdir -p "${CKPT_DIR}" "${SMOKE_OUT_DIR}" "${LOG_DIR}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
HLOG="${LOG_DIR}/harness.log"
: > "${HLOG}"

echo "[orch] launching workload via seccomp-wrap"
source "${VENV}/bin/activate"

setsid --fork bash -c "HF_HOME=/mnt/nvme/hf UV_USE_IO_URING=0 exec /usr/local/bin/seccomp-wrap ${VENV}/bin/python ${HARNESS}" \
  >"${HLOG}" 2>&1 &
ORCH_CHILD=$!

WPID=""
for i in $(seq 1 600); do
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  if ! pgrep -f smoke-vllm-sleep >/dev/null && [ ${i} -gt 5 ]; then
    echo "[orch] harness exited before READY"; tail -60 "${HLOG}"; exit 2
  fi
  sleep 1
done
[ -z "${WPID}" ] && { echo "[orch] timeout"; tail -80 "${HLOG}"; exit 2; }
echo "[orch] READY pid=${WPID}"

echo "[orch] process tree:"
pstree -p "${WPID}" || ps --ppid "${WPID}" -o pid,cmd
echo "[orch] GPU mem after sleep():"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "[orch] harness RSS: $(awk "/VmRSS/ {print \$2}" /proc/${WPID}/status) kB"

# Single criu dump invocation — let cuda_plugin handle cuda-checkpoint per-pid.
T2=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --link-remap --leave-running \
  --log-file dump.log -v3; then
  echo "[orch] criu dump FAILED"
  echo "[orch] === last 200 lines of dump.log ==="
  sudo tail -200 "${CKPT_DIR}/dump.log"
  exit 3
fi
T3=$(date +%s.%N)
CRIU_DUMP=$(awk -v a=${T2} -v b=${T3} 'BEGIN{print b-a}')
echo "[orch] criu dump dur=${CRIU_DUMP} s"

ART_BYTES=$(sudo du -sb "${CKPT_DIR}" | awk '{print $1}')
ART_GIB=$(awk -v b=${ART_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
echo "[orch] artifact=${ART_BYTES} bytes (${ART_GIB} GiB)"

echo "[orch] killing original tree"
sudo pkill -9 -P "${WPID}" 2>/dev/null || true
sudo kill -9 "${WPID}" 2>/dev/null || true
sleep 2

T4=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu restore \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --restore-detached \
  --log-file restore.log -v3; then
  echo "[orch] criu restore FAILED"
  sudo tail -200 "${CKPT_DIR}/restore.log"
  exit 4
fi
T5=$(date +%s.%N)
RESTORE=$(awk -v a=${T4} -v b=${T5} 'BEGIN{print b-a}')
echo "[orch] criu restore dur=${RESTORE} s"

RESTORED_PID=""
for _ in $(seq 1 30); do
  if sudo kill -0 "${WPID}" 2>/dev/null; then RESTORED_PID="${WPID}"; break; fi
  CAND=$(pgrep -f smoke-vllm-sleep.py | head -1 || true)
  [ -n "${CAND}" ] && { RESTORED_PID="${CAND}"; break; }
  sleep 1
done
echo "[orch] restored pid=${RESTORED_PID:-<none>}"
[ -z "${RESTORED_PID}" ] && { echo "[orch] no restored process"; exit 5; }

sudo kill -USR1 "${RESTORED_PID}"
for _ in $(seq 1 90); do
  [ -s "${SMOKE_OUT_DIR}/post.json" ] && break
  sleep 1
done

HARNESS_RC="?"
grep -q "GATE_1_PASS" "${HLOG}" && HARNESS_RC=0
grep -q "GATE_1_FAIL" "${HLOG}" && HARNESS_RC=1

WEIGHTS_BYTES=$(du -sb /mnt/nvme/hf/models--Qwen--Qwen3-0.6B 2>/dev/null | awk '{print $1}')
WEIGHTS_GIB=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
RATIO=$(awk -v a=${ART_BYTES} -v b=${WEIGHTS_BYTES} 'BEGIN{if (b>0) printf "%.3f", a/b; else print "NA"}')

echo
echo "=== BRIDGING GATE RESULTS ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "harness_rc=${HARNESS_RC}"
G2=$(awk -v r=${RATIO} 'BEGIN{print (r != "NA" && r+0 <= 2.0) ? "PASS" : "FAIL"}')
G3=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')
echo "GATE 1 token-id equality: $([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)"
echo "GATE 2 artifact <= 2x weights: ${G2} (ratio=${RATIO})"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"

cat > "${SMOKE_OUT_DIR}/bridging-result.json" <<EOF
{
  "criu_dump_seconds": ${CRIU_DUMP},
  "criu_restore_seconds": ${RESTORE},
  "artifact_bytes": ${ART_BYTES},
  "artifact_gib": ${ART_GIB},
  "weights_bytes": ${WEIGHTS_BYTES},
  "weights_gib": ${WEIGHTS_GIB},
  "artifact_to_weights_ratio": ${RATIO},
  "harness_rc": "${HARNESS_RC}",
  "gate_1_token_equality": "$([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)",
  "gate_2_artifact_size": "${G2}",
  "gate_3_restore_latency": "${G3}"
}
EOF
echo "result -> ${SMOKE_OUT_DIR}/bridging-result.json"
