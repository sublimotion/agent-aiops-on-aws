#!/usr/bin/env bash
# Anchor-cell gate runner: Gemma-4-26B-A4B-it on H200.
# Mirrors run-bridging-gates-v2.sh with size-tiered Gate 2 (>=3 GiB → 2x weights).
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-checkpoint
SMOKE_OUT_DIR=/mnt/nvme/smoke
LOG_DIR=/mnt/nvme/smoke/logs
HARNESS=/home/ec2-user/dynamo-scripts/smoke-vllm-gemma4-anchor.py
VENV=/mnt/nvme/venv
PATH_MODE="${PATH_MODE:-A}"

mkdir -p "${CKPT_DIR}" "${SMOKE_OUT_DIR}" "${LOG_DIR}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
HLOG="${LOG_DIR}/harness-${PATH_MODE}.log"
: > "${HLOG}"

echo "[orch] PATH_MODE=${PATH_MODE}"
echo "[orch] launching workload via seccomp-wrap"
source "${VENV}/bin/activate"

setsid --fork bash -c "HF_HOME=/mnt/nvme/hf UV_USE_IO_URING=0 PATH_MODE=${PATH_MODE} SMOKE_OUT_DIR=${SMOKE_OUT_DIR} exec /usr/local/bin/seccomp-wrap ${VENV}/bin/python ${HARNESS}" \
  >"${HLOG}" 2>&1 &

WPID=""
for i in $(seq 1 1800); do  # up to 30 min for first-load on Gemma-4 26B
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  if ! pgrep -f smoke-vllm-gemma4 >/dev/null && [ ${i} -gt 30 ]; then
    echo "[orch] harness exited before READY"; tail -120 "${HLOG}"; exit 2
  fi
  sleep 1
done
[ -z "${WPID}" ] && { echo "[orch] timeout 30min, still not READY"; tail -120 "${HLOG}"; exit 2; }
echo "[orch] READY pid=${WPID}"

echo "[orch] process tree:"
pstree -p "${WPID}" 2>/dev/null | head -20 || ps --ppid "${WPID}" -o pid,cmd
echo "[orch] GPU mem after sleep():"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "[orch] harness RSS: $(awk '/VmRSS/ {print $2}' /proc/${WPID}/status) kB"

# CRIU dump — let cuda_plugin handle cuda-checkpoint per-pid.
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
sleep 3

# Mark restore start; we want pod-create-equivalent → first-token-streamed.
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
  CAND=$(pgrep -f smoke-vllm-gemma4 | head -1 || true)
  [ -n "${CAND}" ] && { RESTORED_PID="${CAND}"; break; }
  sleep 1
done
echo "[orch] restored pid=${RESTORED_PID:-<none>}"
[ -z "${RESTORED_PID}" ] && { echo "[orch] no restored process"; exit 5; }

# Mark "container start" equivalent: criu restore exit. Now signal wake_up + measure first token.
T_WAKE_SIGNAL=$(date +%s.%N)
sudo kill -USR1 "${RESTORED_PID}"
for _ in $(seq 1 180); do
  [ -s "${SMOKE_OUT_DIR}/post.json" ] && break
  sleep 1
done
T_POST=$(date +%s.%N)
ANCHOR_TOTAL=$(awk -v a=${T4} -v b=${T_POST} 'BEGIN{print b-a}')
ANCHOR_RESTORE_PLUS_WAKE=$(awk -v a=${T4} -v b=${T_POST} 'BEGIN{print b-a}')

HARNESS_RC="?"
grep -q "GATE_1_PASS" "${HLOG}" && HARNESS_RC=0
grep -q "GATE_1_FAIL" "${HLOG}" && HARNESS_RC=1

# Gemma-4 weights size on disk
WEIGHTS_BYTES=$(du -sb /mnt/nvme/hf/hub/models--google--gemma-4-26B-A4B-it 2>/dev/null | awk '{print $1}')
WEIGHTS_GIB=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
RATIO=$(awk -v a=${ART_BYTES} -v b=${WEIGHTS_BYTES} 'BEGIN{if (b>0) printf "%.3f", a/b; else print "NA"}')

# Size-tiered Gate 2 (per spec amendment 2026-05-30):
#   weights >= 3 GiB → ratio <= 2.0
#   weights <  3 GiB → artifact <= weights + 4 GiB
WEIGHTS_FLOAT=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
GATE_2_RULE=""
if awk -v w=${WEIGHTS_FLOAT} 'BEGIN{exit !(w >= 3.0)}'; then
  GATE_2_RULE="ratio<=2.0 (weights ${WEIGHTS_GIB} GiB >= 3 GiB)"
  G2=$(awk -v r=${RATIO} 'BEGIN{print (r != "NA" && r+0 <= 2.0) ? "PASS" : "FAIL"}')
else
  GATE_2_RULE="artifact<=weights+4GiB (weights ${WEIGHTS_GIB} GiB < 3 GiB)"
  G2=$(awk -v a=${ART_GIB} -v w=${WEIGHTS_GIB} 'BEGIN{print (a+0 <= w+0+4.0) ? "PASS" : "FAIL"}')
fi
G3=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')

# Pull TTFT from harness output
TTFT_SEC=$(python3 -c "import json; d=json.load(open('${SMOKE_OUT_DIR}/ttft.json')); print(d['wake_seconds']+d['ttft_after_wake_seconds'])" 2>/dev/null || echo "NA")
WARMUP_LOAD=$(python3 -c "import json; d=json.load(open('${SMOKE_OUT_DIR}/pre.json')); print(d['load_seconds'])" 2>/dev/null || echo "NA")
WARMUP_GEN=$(python3 -c "import json; d=json.load(open('${SMOKE_OUT_DIR}/pre.json')); print(d['warm_seconds'])" 2>/dev/null || echo "NA")
COLD_BASELINE=$(awk -v a=${WARMUP_LOAD} -v b=${WARMUP_GEN} 'BEGIN{if (a=="NA"||b=="NA") print "NA"; else print a+b}')

echo
echo "=== ANCHOR GATE RESULTS (PATH=${PATH_MODE}) ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "harness_rc=${HARNESS_RC}"
echo "GATE 1 token-id equality: $([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)"
echo "GATE 2 ${GATE_2_RULE}: ${G2} (ratio=${RATIO})"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"
echo
echo "=== ANCHOR COMPARISON vs Modal 22s floor ==="
echo "cold baseline (load+warmup gen) ~= ${COLD_BASELINE} s"
echo "restore + wake + first-token = ${ANCHOR_TOTAL} s (criu restore ${RESTORE} + wake/ttft ${TTFT_SEC} s)"
echo "Modal AOT-cache HIT floor (compile+warmup) = ~22 s"
echo

cat > "${SMOKE_OUT_DIR}/anchor-result-${PATH_MODE}.json" <<EOF
{
  "path_mode": "${PATH_MODE}",
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
  "gate_2_rule_applied": "${GATE_2_RULE}",
  "gate_3_restore_latency": "${G3}",
  "anchor_restore_to_first_token_seconds": ${ANCHOR_TOTAL},
  "anchor_ttft_seconds": "${TTFT_SEC}",
  "cold_baseline_load_plus_warm_seconds": "${COLD_BASELINE}",
  "modal_floor_reference_seconds": 22.0
}
EOF
echo "result -> ${SMOKE_OUT_DIR}/anchor-result-${PATH_MODE}.json"
