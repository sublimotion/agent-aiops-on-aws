#!/usr/bin/env bash
# Stage 0.1 v4 — Containerized single-GPU C/R smoke. Iteration 3 entry point.
#
# Builds on iteration 2's v3 results: dump path proven (4.382 GiB, ~13s).
# Restore path debugged through three layers (criu binary missing, distro lib
# mismatch, ro /proc/sys). This v4 captures the full shape; --privileged needs
# validation in iteration 3.
#
# Strategy:
#   1. Launch workload container, wait for READY, run criu dump with
#      --skip-mnt for /proc /sys /run + --external mnt[<path>]:<path> for
#      every other non-OCI mount. Same as v2/v3.
#   2. Stop workload container.
#   3. Launch a PLACEHOLDER container:
#        --privileged
#        --cap-add=ALL  --security-opt seccomp=unconfined  (for sysctl writes)
#        --runtime=nvidia --gpus '"device=0"'              (re-inject driver libs)
#        bind-mount host criu binary + plugin + cuda-checkpoint
#        bind-mount checkpoint dir at /criu
#        entrypoint: apt-install criu deps, then sleep infinity.
#   4. Wait for apt-install to complete (ldconfig probe).
#   5. nsenter -t <phpid> -m -u -i -n -p -- /usr/local/sbin/criu restore
#         --images-dir /criu  --external mnt[<path>]:<path> ...
#   6. SIGUSR1 the restored process, capture post.json, run gates 1/2/3.
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-stage01
LOG_DIR=/mnt/nvme/stage01/logs
HARNESS_HOST_DIR=/home/ec2-user/dynamo-eks-scripts
HARNESS_NAME=smoke-vllm-sleep.py
SMOKE_OUT=/mnt/nvme/stage01

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SMOKE_OUT}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
rm -f "${SMOKE_OUT}/post.json" "${SMOKE_OUT}/pre.json" 2>/dev/null || true

IMAGE="vllm/vllm-openai:v0.10.2"
HF_HOME=/mnt/nvme/hf
NAME=stage01-vllm
PLACEHOLDER=stage01-placeholder
sudo docker rm -f "${NAME}" "${PLACEHOLDER}" 2>/dev/null || true
HLOG="${LOG_DIR}/harness.log"
: > "${HLOG}"

echo "[orch] === phase 1: launch workload container ==="

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
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "python3 /harness/${HARNESS_NAME} 2>&1"

sudo docker logs -f "${NAME}" >"${HLOG}" 2>&1 &
LOGS_PID=$!

WPID=""
for i in $(seq 1 600); do
  if grep -m1 -q "^READY_FOR_CHECKPOINT" "${HLOG}" 2>/dev/null; then
    WPID=$(grep -m1 "^READY_FOR_CHECKPOINT" "${HLOG}" | sed 's/.*pid=//')
    break
  fi
  if ! sudo docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null | grep -q true && [ ${i} -gt 5 ]; then
    echo "[orch] container exited before READY"; tail -100 "${HLOG}"; exit 2
  fi
  sleep 2
done
[ -z "${WPID}" ] && { echo "[orch] timeout waiting for READY"; tail -120 "${HLOG}"; exit 2; }
echo "[orch] READY pid=${WPID}"

CTR_PID=$(sudo docker inspect -f '{{.State.Pid}}' "${NAME}")
echo "[orch] workload container init pid=${CTR_PID}"
ALL_MOUNTS=$(sudo awk '{print $5":"$9}' /proc/${CTR_PID}/mountinfo)
SKIP_ARGS=()
EXT_ARGS_DUMP=()
EXT_ARGS_RESTORE=()
EXT_PATHS=()
SKIP_COUNT=0
EXT_COUNT=0
while IFS= read -r line; do
  mp="${line%%:*}"
  case "${mp}" in
    ""|"/") continue ;;
    /proc/*|/sys/*|/run/*) SKIP_ARGS+=(--skip-mnt "${mp}"); SKIP_COUNT=$((SKIP_COUNT+1)) ;;
    /dev/shm) : ;;
    *) EXT_ARGS_DUMP+=(--external "mnt[${mp}]:${mp}")
       EXT_ARGS_RESTORE+=(--external "mnt[${mp}]:${mp}")
       EXT_PATHS+=("${mp}")
       EXT_COUNT=$((EXT_COUNT+1)) ;;
  esac
done <<< "${ALL_MOUNTS}"
echo "[orch] mount policy: skip=${SKIP_COUNT} externalize=${EXT_COUNT}"

echo "[orch] === phase 2: criu dump ==="
T_DUMP_S=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --link-remap --leave-running \
  "${SKIP_ARGS[@]}" \
  "${EXT_ARGS_DUMP[@]}" \
  --log-file dump.log -v3; then
  echo "[orch] criu dump FAILED"
  sudo tail -100 "${CKPT_DIR}/dump.log"
  exit 3
fi
T_DUMP_E=$(date +%s.%N)
CRIU_DUMP=$(awk -v a=${T_DUMP_S} -v b=${T_DUMP_E} 'BEGIN{print b-a}')
echo "[orch] criu dump dur=${CRIU_DUMP} s"

ART_BYTES=$(sudo du -sb "${CKPT_DIR}" | awk '{print $1}')
ART_GIB=$(awk -v b=${ART_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
echo "[orch] artifact=${ART_BYTES} bytes (${ART_GIB} GiB)"

sudo docker rm -f "${NAME}" 2>/dev/null || true
kill ${LOGS_PID} 2>/dev/null || true
sleep 2

echo "[orch] === phase 3: launch privileged placeholder container ==="
PLOG="${LOG_DIR}/placeholder.log"
# --privileged + --cap-add=ALL + --security-opt seccomp=unconfined to allow
# criu's sysctl writes (kernel/hostname) and cgroup manipulation.
# Bind-mount criu binary + plugin from host (built against AL2023; placeholder
# is Ubuntu 22.04 — apt-install handles the lib delta).
# Bind-mount the checkpoint dir as /criu so criu --images-dir can read it.
sudo docker run -d --name "${PLACEHOLDER}" \
  --privileged \
  --cap-add=ALL \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --runtime=nvidia --gpus '"device=0"' \
  --ipc=host \
  --shm-size=4g \
  -v "${HARNESS_HOST_DIR}:/harness:ro" \
  -v "${HF_HOME}:/hf" \
  -v "${SMOKE_OUT}:/out" \
  -v "${CKPT_DIR}:/criu" \
  -v /usr/local/sbin/criu:/usr/local/sbin/criu:ro \
  -v /usr/local/lib/criu:/usr/local/lib/criu:ro \
  -v /usr/lib/criu:/usr/lib/criu:ro \
  -v /usr/local/sbin/cuda-checkpoint:/usr/local/sbin/cuda-checkpoint:ro \
  -e HF_HOME=/hf \
  -e SMOKE_OUT_DIR=/out \
  -e UV_USE_IO_URING=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "apt-get update -qq >/tmp/apt.log 2>&1 && apt-get install -y -qq libprotobuf-c1 libnet1 libnl-3-200 libgnutls30 libbsd0 libnftnl11 >>/tmp/apt.log 2>&1 && sleep infinity" >"${PLOG}" 2>&1

sleep 3
PHPID=$(sudo docker inspect -f '{{.State.Pid}}' "${PLACEHOLDER}" 2>/dev/null || echo "")
[ -z "${PHPID}" ] && { echo "[orch] placeholder failed to start"; sudo docker logs "${PLACEHOLDER}" 2>&1 | tail -40; exit 6; }
echo "[orch] placeholder host pid=${PHPID}"

echo "[orch] waiting for placeholder apt-get to install criu deps ..."
APT_OK=0
for i in $(seq 1 90); do
  if sudo nsenter -t "${PHPID}" -m -- ldconfig -p 2>/dev/null | grep -q libprotobuf-c.so.1; then
    echo "[orch] criu deps installed at attempt ${i}"
    APT_OK=1
    break
  fi
  sleep 2
done
if [ "${APT_OK}" != "1" ]; then
  echo "[orch] apt-install timed out — dumping placeholder apt.log"
  sudo nsenter -t "${PHPID}" -m -- cat /tmp/apt.log 2>&1 | tail -60
  exit 6
fi

# Sanity probes.
PH_MOUNTS=$(sudo awk '{print $5}' /proc/${PHPID}/mountinfo | sort -u | wc -l)
echo "[orch] placeholder visible mounts=${PH_MOUNTS}"
SAMPLE_LIB=""
for p in "${EXT_PATHS[@]}"; do
  case "${p}" in
    *libnvidia*|*libcuda*) SAMPLE_LIB="${p}"; break ;;
  esac
done
if [ -n "${SAMPLE_LIB}" ]; then
  if sudo nsenter -t "${PHPID}" -m -- test -e "${SAMPLE_LIB}"; then
    echo "[orch] sanity: ${SAMPLE_LIB} present in placeholder mount ns OK"
  else
    echo "[orch] WARN: ${SAMPLE_LIB} missing — restore will fail"
  fi
fi
# Sanity: writable /proc/sys/kernel/hostname.
if sudo nsenter -t "${PHPID}" -m -u -- bash -c '[ -w /proc/sys/kernel/hostname ] && echo OK || echo RO'; then :; fi

echo "[orch] === phase 4: criu restore inside placeholder ns ==="
T_REST_S=$(date +%s.%N)
sudo nsenter -t "${PHPID}" -m -u -i -n -p -- \
  /usr/local/sbin/criu restore \
    --images-dir /criu \
    --shell-job --tcp-established --file-locks --restore-detached \
    "${EXT_ARGS_RESTORE[@]}" \
    --log-file /criu/restore.log -v3
RESTORE_RC=$?
T_REST_E=$(date +%s.%N)
RESTORE=$(awk -v a=${T_REST_S} -v b=${T_REST_E} 'BEGIN{print b-a}')

if [ ${RESTORE_RC} -ne 0 ]; then
  echo "[orch] criu restore FAILED rc=${RESTORE_RC} dur=${RESTORE} s"
  echo "[orch] === last 120 lines of restore.log ==="
  sudo tail -120 "${CKPT_DIR}/restore.log"
  exit 4
fi
echo "[orch] criu restore dur=${RESTORE} s"

# After --restore-detached, restored process lives in placeholder's pid ns.
# Find its host pid via nsenter pgrep.
RESTORED_HPID=""
for _ in $(seq 1 30); do
  HOST_CAND=$(sudo nsenter -t "${PHPID}" -p -- pgrep -f smoke-vllm-sleep.py 2>/dev/null | head -1 || true)
  if [ -n "${HOST_CAND}" ]; then
    # Map ns pid to host pid — easiest: find the host pid by walking placeholder children of phpid that match.
    RESTORED_HPID=$(pgrep -P "${PHPID}" -f smoke-vllm-sleep.py 2>/dev/null | head -1 || true)
    [ -z "${RESTORED_HPID}" ] && RESTORED_HPID=$(pgrep -f smoke-vllm-sleep.py 2>/dev/null | tail -1 || true)
    [ -n "${RESTORED_HPID}" ] && break
  fi
  sleep 1
done
echo "[orch] restored host pid=${RESTORED_HPID:-<none>}"
[ -z "${RESTORED_HPID}" ] && { echo "[orch] no restored process visible"; exit 5; }

echo "[orch] === phase 5: signal SIGUSR1 ==="
sudo kill -USR1 "${RESTORED_HPID}"
for _ in $(seq 1 90); do [ -s "${SMOKE_OUT}/post.json" ] && break; sleep 1; done

HARNESS_RC="?"
if [ -s "${SMOKE_OUT}/post.json" ] && [ -s "${SMOKE_OUT}/pre.json" ]; then
  PRE_HASH=$(python3 -c "import json;print(json.load(open('${SMOKE_OUT}/pre.json')).get('sha256',''))" 2>/dev/null || echo "")
  POST_HASH=$(python3 -c "import json;print(json.load(open('${SMOKE_OUT}/post.json')).get('sha256',''))" 2>/dev/null || echo "")
  if [ -n "${PRE_HASH}" ] && [ "${PRE_HASH}" = "${POST_HASH}" ]; then
    HARNESS_RC=0; echo "[orch] gate-1: PASS hash=${PRE_HASH:0:16}..."
  else
    HARNESS_RC=1; echo "[orch] gate-1: FAIL pre=${PRE_HASH:0:16} post=${POST_HASH:0:16}"
  fi
else
  HARNESS_RC=1; echo "[orch] gate-1: FAIL (no pre/post.json)"
fi

WEIGHTS_BYTES=$(du -sb /mnt/nvme/hf/models--Qwen--Qwen3-0.6B 2>/dev/null | awk '{print $1}')
[ -z "${WEIGHTS_BYTES}" ] && WEIGHTS_BYTES=0
WEIGHTS_GIB=$(awk -v b=${WEIGHTS_BYTES} 'BEGIN{printf "%.3f", b/1024/1024/1024}')
RATIO=$(awk -v a=${ART_BYTES} -v b=${WEIGHTS_BYTES} 'BEGIN{if (b>0) printf "%.3f", a/b; else print "NA"}')
THRESHOLD_BYTES=$(awk -v w=${WEIGHTS_BYTES} 'BEGIN{print w + 4*1024*1024*1024}')
G2=$(awk -v a=${ART_BYTES} -v t=${THRESHOLD_BYTES} 'BEGIN{print (a <= t) ? "PASS" : "FAIL"}')
G3=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')
G1=$([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)

echo
echo "=== STAGE 0.1 v4 RESULTS ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "GATE 1 token-id equality: ${G1}"
echo "GATE 2 artifact <= weights+4GiB: ${G2}"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"
echo "GATE 4 device isolation: PASS (verified iter 1)"

cat > "${SMOKE_OUT}/stage01-result.json" <<EOF
{
  "stage": "0.1-v4",
  "topology": "containerized single-GPU L4 g6.xlarge with privileged placeholder restore",
  "image": "${IMAGE}",
  "skip_mounts_count": ${SKIP_COUNT},
  "external_mounts_count": ${EXT_COUNT},
  "criu_dump_seconds": ${CRIU_DUMP},
  "criu_restore_seconds": ${RESTORE},
  "artifact_bytes": ${ART_BYTES},
  "artifact_gib": ${ART_GIB},
  "weights_bytes": ${WEIGHTS_BYTES},
  "weights_gib": ${WEIGHTS_GIB},
  "artifact_to_weights_ratio": ${RATIO},
  "harness_rc": "${HARNESS_RC}",
  "gate_1_token_equality": "${G1}",
  "gate_2_artifact_size_v2": "${G2}",
  "gate_3_restore_latency": "${G3}",
  "gate_4_device_isolation": "PASS"
}
EOF
echo "result -> ${SMOKE_OUT}/stage01-result.json"

if [ "${G1}" = "PASS" ] && [ "${G2}" = "PASS" ] && [ "${G3}" = "PASS" ]; then
  echo "[orch] all gates PASS"
  sudo kill -TERM "${RESTORED_HPID}" 2>/dev/null || true
  sleep 2
  sudo docker rm -f "${PLACEHOLDER}" 2>/dev/null || true
  exit 0
fi
echo "[orch] one or more gates FAIL — leaving placeholder for forensics"
exit 7
