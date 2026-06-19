#!/usr/bin/env bash
# Stage 0.1 — Containerized single-GPU C/R smoke.
#
# Hypothesis: vLLM running inside an nvidia-container-runtime container with
# --gpus '"device=0"' only sees /dev/nvidia0 in its mount namespace, so CRIU
# (driven from the host) does NOT trip on the 51-fd-across-all-GPUs problem
# that killed the p5e bare-EC2 cell.
#
# Layout:
#   - Workload: docker container running smoke-vllm-sleep.py (Qwen3-0.6B).
#   - Orchestrator: this script, on the host, drives criu dump against the
#     container's host PID after the workload prints READY_FOR_CHECKPOINT.
#   - cuda-checkpoint + cuda_plugin.so are bind-mounted into the container so
#     the cuda_plugin (loaded by host criu via --tree) can find them on the
#     workload's filesystem if it execs them in-container. Host criu also has
#     them locally.
#
# Gates (mirror predecessor):
#   1. Token-id equality pre/post restore (SHA256 of first 64 IDs).
#   2. Artifact size <= weights + 4 GiB (relaxed per predecessor lesson —
#      tiny model dominated by ~3 GiB process overhead).
#   3. CRIU restore wall-clock < 30 s.
#
# Plus the *new* gate this cell exists to prove:
#   4. Inside-container `ls /dev/nvidia*` shows ONLY /dev/nvidia0 + nvidiactl
#      + nvidia-uvm{,-tools} — no peer GPU nodes.
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-stage01
LOG_DIR=/mnt/nvme/stage01/logs
HARNESS_HOST_DIR=/home/ec2-user/dynamo-eks-scripts
HARNESS_NAME=smoke-vllm-sleep.py
SMOKE_OUT=/mnt/nvme/stage01

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SMOKE_OUT}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true

# vLLM image — use upstream openai-compatible image. v0.10.2 matches predecessor.
IMAGE="vllm/vllm-openai:v0.10.2"

# Pull image if missing (do separately so first-pull time is not in measurement)
if ! sudo docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[orch] pulling ${IMAGE} ..."
  sudo docker pull "${IMAGE}"
fi

# Pre-stage HF weights on host NVMe so the container can mount them
HF_HOME=/mnt/nvme/hf
mkdir -p "${HF_HOME}"
if [ ! -d "${HF_HOME}/models--Qwen--Qwen3-0.6B/snapshots" ] || \
   [ -z "$(ls ${HF_HOME}/models--Qwen--Qwen3-0.6B/snapshots/ 2>/dev/null)" ]; then
  echo "[orch] downloading Qwen3-0.6B weights to host NVMe ..."
  /mnt/nvme/venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='Qwen/Qwen3-0.6B', cache_dir='${HF_HOME}')
print('weights at:', p)
" || { python3 -m venv /mnt/nvme/venv && /mnt/nvme/venv/bin/pip install -q huggingface_hub && \
       HF_HOME=${HF_HOME} /mnt/nvme/venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B', cache_dir='${HF_HOME}'))"; }
fi

# Container name
NAME=stage01-vllm
sudo docker rm -f "${NAME}" 2>/dev/null || true

HLOG="${LOG_DIR}/harness.log"
: > "${HLOG}"

echo "[orch] launching containerized vLLM ..."
T_LAUNCH=$(date +%s.%N)

# Run container detached; stream logs to host file via docker logs --follow.
# --pid=host so criu dump from host can find the container's process tree by PID.
# (Alternative: leave container in own pid namespace and use docker top -- but
#  --pid=host keeps the host criu's --tree <pid> path simple.)
# Note: --pid=host is a deliberate compromise for Stage 0.1; production EKS
# uses pod pid namespace + criu running in a privileged sidecar that joins it.
sudo docker run -d --name "${NAME}" \
  --runtime=nvidia --gpus '"device=0"' \
  --pid=host \
  --ipc=host \
  --shm-size=4g \
  -v "${HARNESS_HOST_DIR}:/harness:ro" \
  -v "${HF_HOME}:/hf" \
  -v "${SMOKE_OUT}:/out" \
  -e HF_HOME=/hf \
  -e SMOKE_OUT_DIR=/out \
  -e UV_USE_IO_URING=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "python3 /harness/${HARNESS_NAME} 2>&1"

# Stream container logs to host file in background
sudo docker logs -f "${NAME}" >"${HLOG}" 2>&1 &
LOGS_PID=$!

# 4. Confirm ONLY /dev/nvidia0 visible inside the container
DEV_LIST=$(sudo docker exec "${NAME}" ls /dev/ 2>&1 | grep -E '^nvidia' || true)
echo "[orch] container /dev/nvidia* = ${DEV_LIST}"
EXPECTED='nvidia-modeset nvidia-uvm nvidia-uvm-tools nvidia0 nvidiactl'
ACTUAL=$(echo "${DEV_LIST}" | sort | tr '\n' ' ' | sed 's/ $//')
echo "[orch] expected: ${EXPECTED}"
echo "[orch] actual:   ${ACTUAL}"
GATE_4="FAIL"
if echo "${ACTUAL}" | grep -qv 'nvidia[1-7]'; then GATE_4="PASS"; fi
echo "[orch] GATE 4 (single-GPU device isolation): ${GATE_4}"

# Wait for READY_FOR_CHECKPOINT
WPID=""
for i in $(seq 1 600); do
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    # WPID inside container == host PID since --pid=host
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  if ! sudo docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null | grep -q true && [ ${i} -gt 5 ]; then
    echo "[orch] container exited before READY"; tail -100 "${HLOG}"; exit 2
  fi
  sleep 2
done
[ -z "${WPID}" ] && { echo "[orch] timeout"; tail -120 "${HLOG}"; exit 2; }
echo "[orch] READY pid=${WPID}"

# Inspect host's view of /proc/<pid>/fd to confirm only /dev/nvidia0 fds
echo "[orch] host /proc/${WPID}/maps grep nvidia:"
sudo grep -E '/dev/nvidia' /proc/${WPID}/maps 2>&1 | sort -u | head -20 || true
echo "[orch] host fds for ${WPID}:"
sudo ls -la /proc/${WPID}/fd/ 2>&1 | grep -E '/dev/nvidia' | sort -u | head -20 || true

T_DUMP_S=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --link-remap --leave-running \
  --log-file dump.log -v3; then
  echo "[orch] criu dump FAILED"
  echo "[orch] === last 200 lines of dump.log ==="
  sudo tail -200 "${CKPT_DIR}/dump.log"
  echo "[orch] container still running? $(sudo docker inspect -f '{{.State.Running}}' ${NAME})"
  exit 3
fi
T_DUMP_E=$(date +%s.%N)
CRIU_DUMP=$(awk -v a=${T_DUMP_S} -v b=${T_DUMP_E} 'BEGIN{print b-a}')
echo "[orch] criu dump dur=${CRIU_DUMP} s"

ART_BYTES=$(sudo du -sb "${CKPT_DIR}" | awk '{print $1}')
ART_GIB=$(awk -v b=${ART_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
echo "[orch] artifact=${ART_BYTES} bytes (${ART_GIB} GiB)"

# Kill the container (criu has --leave-running so the dump file is good)
sudo docker rm -f "${NAME}" 2>/dev/null || true
kill ${LOGS_PID} 2>/dev/null || true
sleep 2

# Restore from dump (host criu)
T_REST_S=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu restore \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --restore-detached \
  --log-file restore.log -v3; then
  echo "[orch] criu restore FAILED"
  sudo tail -200 "${CKPT_DIR}/restore.log"
  exit 4
fi
T_REST_E=$(date +%s.%N)
RESTORE=$(awk -v a=${T_REST_S} -v b=${T_REST_E} 'BEGIN{print b-a}')
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
  [ -s "${SMOKE_OUT}/post.json" ] && break
  sleep 1
done

HARNESS_RC="?"
grep -q "GATE_1_PASS" "${HLOG}" && HARNESS_RC=0
grep -q "GATE_1_FAIL" "${HLOG}" && HARNESS_RC=1

WEIGHTS_BYTES=$(du -sb /mnt/nvme/hf/models--Qwen--Qwen3-0.6B 2>/dev/null | awk '{print $1}')
WEIGHTS_GIB=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
RATIO=$(awk -v a=${ART_BYTES} -v b=${WEIGHTS_BYTES} 'BEGIN{if (b>0) printf "%.3f", a/b; else print "NA"}')

# Gate 2 v2 from predecessor: artifact <= weights + 4 GiB (absolute headroom)
THRESHOLD_BYTES=$(awk -v w=${WEIGHTS_BYTES} 'BEGIN{print w + 4*1024*1024*1024}')
G2=$(awk -v a=${ART_BYTES} -v t=${THRESHOLD_BYTES} 'BEGIN{print (a <= t) ? "PASS" : "FAIL"}')
G3=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')

echo
echo "=== STAGE 0.1 RESULTS ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "GATE 1 token-id equality: $([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)"
echo "GATE 2 artifact <= weights+4GiB: ${G2} (artifact=${ART_GIB} thr=$(awk -v t=${THRESHOLD_BYTES} 'BEGIN{printf "%.3f", t/1024/1024/1024}') GiB)"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"
echo "GATE 4 device isolation: ${GATE_4}"

cat > "${SMOKE_OUT}/stage01-result.json" <<EOF
{
  "stage": "0.1",
  "topology": "containerized single-GPU L4 g6.xlarge",
  "image": "${IMAGE}",
  "criu_dump_seconds": ${CRIU_DUMP},
  "criu_restore_seconds": ${RESTORE},
  "artifact_bytes": ${ART_BYTES},
  "artifact_gib": ${ART_GIB},
  "weights_bytes": ${WEIGHTS_BYTES},
  "weights_gib": ${WEIGHTS_GIB},
  "artifact_to_weights_ratio": ${RATIO},
  "harness_rc": "${HARNESS_RC}",
  "gate_1_token_equality": "$([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)",
  "gate_2_artifact_size_v2": "${G2}",
  "gate_3_restore_latency": "${G3}",
  "gate_4_device_isolation": "${GATE_4}",
  "container_dev_list": "${ACTUAL}"
}
EOF
echo "result -> ${SMOKE_OUT}/stage01-result.json"
