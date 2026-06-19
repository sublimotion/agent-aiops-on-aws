#!/usr/bin/env bash
# Stage 0.1 v3 — Containerized single-GPU C/R smoke with placeholder-container restore.
#
# v2 proved CRIU dump succeeds with Dynamo-style mount externalization, but
# restore failed: "No mapping for 1044:(null) mountpoint". Cause: --external
# mnt[<path>]:<label> on dump records mount IDs that need a corresponding
# --external mnt[<label>]:<source> on restore. Restore must run *inside* a
# placeholder container that has the same nvidia-container-runtime driver
# bind-mounts, so the labels resolve.
#
# Strategy (mirrors upstream Dynamo internal/executor/restore.go):
#   1. Launch workload container, wait for READY, run criu dump with
#      --external mnt[<path>]:<path> per non-OCI mount (same as v2).
#   2. Stop workload container.
#   3. Launch a fresh PLACEHOLDER container from the same image with
#      --gpus '"device=0"' so nvidia-container-runtime re-injects the
#      identical driver bind-mounts. Sleep infinity so the namespace stays.
#      Bind-mount the checkpoint dir read-write into the placeholder.
#   4. nsenter -t <placeholder_pid> -m -u -i -n -p -- criu restore
#         --images-dir /criu  --external mnt[<path>]:<path> ...
#      so restore happens inside the placeholder's mount namespace where
#      every externalized path actually exists.
#   5. After restore, signal SIGUSR1 to the restored process, capture
#      post.json, and run gates 1/2/3 (gate 4 already PASS in v1).
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-stage01
LOG_DIR=/mnt/nvme/stage01/logs
HARNESS_HOST_DIR=/home/ec2-user/dynamo-eks-scripts
HARNESS_NAME=smoke-vllm-sleep.py
SMOKE_OUT=/mnt/nvme/stage01

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SMOKE_OUT}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true
rm -f "${SMOKE_OUT}/post.json" 2>/dev/null || true

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

# Wait for READY
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

# Build mount policy from container's mount namespace.
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
  fst="${line#*:}"
  case "${mp}" in
    ""|"/") continue ;;
    /proc/*|/sys/*|/run/*)
      SKIP_ARGS+=(--skip-mnt "${mp}")
      SKIP_COUNT=$((SKIP_COUNT + 1))
      ;;
    /dev/shm)
      :
      ;;
    *)
      EXT_ARGS_DUMP+=(--external "mnt[${mp}]:${mp}")
      # On restore, label must map to a real source path inside the
      # placeholder's mount namespace. Same path applies because
      # nvidia-container-runtime re-injects identical bind-mount targets.
      EXT_ARGS_RESTORE+=(--external "mnt[${mp}]:${mp}")
      EXT_PATHS+=("${mp}")
      EXT_COUNT=$((EXT_COUNT + 1))
      ;;
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

# Stop workload (we used --leave-running but the new restore creates a fresh process tree).
sudo docker rm -f "${NAME}" 2>/dev/null || true
kill ${LOGS_PID} 2>/dev/null || true
sleep 2

echo "[orch] === phase 3: launch placeholder container ==="
# Same image, same --gpus, sleep infinity so namespace persists.
# Bind-mount the checkpoint dir into the placeholder so criu --images-dir
# can read it from inside the namespace.
# Also bind-mount /out so restored process can write post.json.
PLOG="${LOG_DIR}/placeholder.log"
sudo docker run -d --name "${PLACEHOLDER}" \
  --runtime=nvidia --gpus '"device=0"' \
  --pid=host \
  --ipc=host \
  --shm-size=4g \
  -v "${HARNESS_HOST_DIR}:/harness:ro" \
  -v "${HF_HOME}:/hf" \
  -v "${SMOKE_OUT}:/out" \
  -v "${CKPT_DIR}:/criu" \
  -e HF_HOME=/hf \
  -e SMOKE_OUT_DIR=/out \
  -e UV_USE_IO_URING=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "sleep infinity" >"${PLOG}" 2>&1

sleep 3
PHPID=$(sudo docker inspect -f '{{.State.Pid}}' "${PLACEHOLDER}" 2>/dev/null || echo "")
[ -z "${PHPID}" ] && { echo "[orch] placeholder failed to start"; sudo docker logs "${PLACEHOLDER}" 2>&1 | tail -40; exit 6; }
echo "[orch] placeholder host pid=${PHPID}"

# Sanity: confirm placeholder has the same nvidia bind-mounts.
PH_MOUNTS=$(sudo awk '{print $5}' /proc/${PHPID}/mountinfo | sort -u | wc -l)
echo "[orch] placeholder visible mounts=${PH_MOUNTS}"

# Sanity: confirm a sample externalized path actually exists in the placeholder ns.
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
    echo "[orch] WARN: ${SAMPLE_LIB} missing in placeholder mount ns — restore will fail on this path"
  fi
fi

echo "[orch] === phase 4: criu restore inside placeholder ns ==="
# nsenter into placeholder's mnt+uts+ipc+net+pid namespaces, run criu.
# Note: --pid=host on placeholder + entering its pid ns is unusual; we set
# --pid=host on the container so docker's monitor doesn't reap restored
# children unexpectedly. Entering the host pid ns from the placeholder is
# effectively a no-op on pid since placeholder shares host pid ns.
# Actually we want the restored process to live in the host pid ns so
# `kill -USR1 $WPID` from this script works. So we drop -p from nsenter.
# The mount, uts, ipc, net namespaces are the placeholder container's; pid
# stays as host. This matches what cuda-checkpoint expects (host pid view).

T_REST_S=$(date +%s.%N)
sudo nsenter -t "${PHPID}" -m -u -i -n -- \
  /usr/local/sbin/criu restore \
    --images-dir /criu \
    --shell-job --tcp-established --file-locks --restore-detached \
    "${EXT_ARGS_RESTORE[@]}" \
    --log-file /criu/restore.log -v3
RESTORE_RC=$?
T_REST_E=$(date +%s.%N)
RESTORE=$(awk -v a=${T_REST_S} -v b=${T_REST_E} 'BEGIN{print b-a}')

if [ ${RESTORE_RC} -ne 0 ]; then
  echo "[orch] criu restore FAILED rc=${RESTORE_RC}"
  echo "[orch] === last 100 lines of restore.log ==="
  sudo tail -100 "${CKPT_DIR}/restore.log"
  echo "[orch] artifact ${ART_GIB} GiB; dump_s=${CRIU_DUMP}; restore failed"
  # Don't delete placeholder — leave for forensics
  exit 4
fi
echo "[orch] criu restore dur=${RESTORE} s"

# Find the restored process. With --restore-detached and host pid ns,
# the original WPID should be revived (criu re-creates with same pid if free).
RESTORED_PID=""
for _ in $(seq 1 30); do
  if sudo kill -0 "${WPID}" 2>/dev/null; then RESTORED_PID="${WPID}"; break; fi
  CAND=$(pgrep -f smoke-vllm-sleep.py | head -1 || true)
  [ -n "${CAND}" ] && { RESTORED_PID="${CAND}"; break; }
  sleep 1
done
echo "[orch] restored pid=${RESTORED_PID:-<none>}"
[ -z "${RESTORED_PID}" ] && { echo "[orch] no restored process visible after ${RESTORE} s"; exit 5; }

echo "[orch] === phase 5: signal SIGUSR1 and capture post.json ==="
sudo kill -USR1 "${RESTORED_PID}"
for _ in $(seq 1 90); do
  [ -s "${SMOKE_OUT}/post.json" ] && break
  sleep 1
done

# Tail any container logs that the restored process may write via /out.
# Read the harness log for gate-1 markers (smoke-vllm-sleep.py prints to stdout
# via /out/post.json and a side-channel file).
HARNESS_RC="?"
if [ -s "${SMOKE_OUT}/post.json" ]; then
  PRE_HASH=$(python3 -c "import json;d=json.load(open('${SMOKE_OUT}/pre.json'));print(d.get('sha256',''))" 2>/dev/null || echo "")
  POST_HASH=$(python3 -c "import json;d=json.load(open('${SMOKE_OUT}/post.json'));print(d.get('sha256',''))" 2>/dev/null || echo "")
  if [ -n "${PRE_HASH}" ] && [ "${PRE_HASH}" = "${POST_HASH}" ]; then
    HARNESS_RC=0
    echo "[orch] gate-1 token-id equality: PASS  hash=${PRE_HASH:0:16}..."
  else
    HARNESS_RC=1
    echo "[orch] gate-1 token-id equality: FAIL  pre=${PRE_HASH:0:16} post=${POST_HASH:0:16}"
  fi
else
  echo "[orch] gate-1 token-id equality: FAIL  (no post.json)"
  HARNESS_RC=1
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
echo "=== STAGE 0.1 v3 RESULTS ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "GATE 1 token-id equality: ${G1}"
echo "GATE 2 artifact <= weights+4GiB: ${G2}"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"
echo "GATE 4 device isolation: PASS (verified in v1)"

cat > "${SMOKE_OUT}/stage01-result.json" <<EOF
{
  "stage": "0.1-v3",
  "topology": "containerized single-GPU L4 g6.xlarge with placeholder restore",
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

# Cleanup placeholder if all gates passed
if [ "${G1}" = "PASS" ] && [ "${G2}" = "PASS" ] && [ "${G3}" = "PASS" ]; then
  echo "[orch] all gates PASS — cleaning placeholder"
  sudo kill -TERM "${RESTORED_PID}" 2>/dev/null || true
  sleep 2
  sudo docker rm -f "${PLACEHOLDER}" 2>/dev/null || true
  exit 0
else
  echo "[orch] one or more gates FAIL — leaving placeholder ${PLACEHOLDER} for forensics"
  exit 7
fi
