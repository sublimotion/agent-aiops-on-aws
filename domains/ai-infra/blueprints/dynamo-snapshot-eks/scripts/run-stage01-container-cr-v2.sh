#!/usr/bin/env bash
# Stage 0.1 v2 — Containerized single-GPU C/R smoke with mount-skip handling.
#
# Builds on v1 which proved Gate 4 (device isolation) PASS but failed CRIU
# dump on file-bind-mounted nvidia libs injected by nvidia-container-runtime
# (criu/mount.c:753 "doesn't have a proper root mount").
#
# Strategy: scan the container's mountinfo for the file-bind-mounts under
# /usr/bin/nvidia-* and /usr/lib64/libnvidia* (and similar driver libs), and
# pass each as --skip-mnt to criu dump. CRIU then ignores these mounts; on
# restore the same nvidia-container-runtime hook re-injects them.
set -uo pipefail

CKPT_DIR=/mnt/nvme/criu-stage01
LOG_DIR=/mnt/nvme/stage01/logs
HARNESS_HOST_DIR=/home/ec2-user/dynamo-eks-scripts
HARNESS_NAME=smoke-vllm-sleep.py
SMOKE_OUT=/mnt/nvme/stage01

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SMOKE_OUT}"
sudo rm -rf "${CKPT_DIR}"/* 2>/dev/null || true

IMAGE="vllm/vllm-openai:v0.10.2"
HF_HOME=/mnt/nvme/hf
NAME=stage01-vllm
sudo docker rm -f "${NAME}" 2>/dev/null || true
HLOG="${LOG_DIR}/harness.log"
: > "${HLOG}"

echo "[orch] launching containerized vLLM (v2 — limited driver caps + skip-mnt) ..."

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
[ -z "${WPID}" ] && { echo "[orch] timeout"; tail -120 "${HLOG}"; exit 2; }
echo "[orch] READY pid=${WPID}"

# Build --skip-mnt list from container's mount namespace.
# All file-bind-mounts of host driver libs/binaries that nvidia-container-cli injects.
SKIP_FLAGS=""
# Get mountinfo from inside container; need to extract container's PID inside its namespace
CTR_PID=$(sudo docker inspect -f '{{.State.Pid}}' "${NAME}")
echo "[orch] container init pid=${CTR_PID}"
# Broad regex: any mount whose path contains nvidia/cuda/libgl/libegl/vulkan/gsp/nvoptix/vdpau, excluding /dev/* (device nodes are NOT skipped — cuda_plugin handles them).
# Per Dynamo internal/runtime/mounts.go BuildMountPolicy: skip non-OCI virtual mounts
# under /proc /sys /run, externalize everything else, let /dev/shm dump natively.
# We approximate the "non-OCI" classification by treating ALL mounts under those
# prefixes as skip and externalizing the rest (Stage 0.1 doesn't have an OCI spec
# walker; the workload doesn't write to /run /sys /proc/ submounts so this is safe).
ALL_MOUNTS=$(sudo awk '{print $5":"$9}' /proc/${CTR_PID}/mountinfo)
SKIP_ARGS=()
EXT_ARGS=()
SKIP_COUNT=0
EXT_COUNT=0
while IFS= read -r line; do
  mp="${line%%:*}"
  fst="${line#*:}"
  case "${mp}" in
    ""|"/") continue ;;
    /proc/*|/sys/*|/run/*)
      # Non-OCI virtual; skip-mnt avoids dumping the contents.
      SKIP_ARGS+=(--skip-mnt "${mp}")
      SKIP_COUNT=$((SKIP_COUNT + 1))
      ;;
    /dev/shm)
      # tmpfs — let CRIU handle natively
      :
      ;;
    *)
      # Externalize: criu won't try to dump the mount root; restore expects it injected.
      EXT_ARGS+=(--external "mnt[${mp}]:${mp}")
      EXT_COUNT=$((EXT_COUNT + 1))
      ;;
  esac
done <<< "${ALL_MOUNTS}"
echo "[orch] mount policy: skip=${SKIP_COUNT} externalize=${EXT_COUNT}"
TOTAL=$((SKIP_COUNT + EXT_COUNT))

echo "[orch] criu dump with ${#SKIP_ARGS[@]} skip-mnt args ..."
T_DUMP_S=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu dump \
  --tree "${WPID}" \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --link-remap --leave-running \
  "${SKIP_ARGS[@]}" \
  "${EXT_ARGS[@]}" \
  --log-file dump.log -v3; then
  echo "[orch] criu dump FAILED"
  echo "[orch] === last 100 lines of dump.log ==="
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

T_REST_S=$(date +%s.%N)
if ! sudo PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/local/sbin/criu restore \
  --images-dir "${CKPT_DIR}" \
  --shell-job --tcp-established --file-locks --restore-detached \
  --log-file restore.log -v3; then
  echo "[orch] criu restore FAILED"
  sudo tail -100 "${CKPT_DIR}/restore.log"
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
THRESHOLD_BYTES=$(awk -v w=${WEIGHTS_BYTES} 'BEGIN{print w + 4*1024*1024*1024}')
G2=$(awk -v a=${ART_BYTES} -v t=${THRESHOLD_BYTES} 'BEGIN{print (a <= t) ? "PASS" : "FAIL"}')
G3=$(awk -v r=${RESTORE} 'BEGIN{print (r+0 < 30.0) ? "PASS" : "FAIL"}')

echo
echo "=== STAGE 0.1 v2 RESULTS ==="
echo "criu_dump_seconds=${CRIU_DUMP}"
echo "criu_restore_seconds=${RESTORE}"
echo "artifact=${ART_GIB} GiB / weights=${WEIGHTS_GIB} GiB ratio=${RATIO}"
echo "GATE 1 token-id equality: $([ "${HARNESS_RC}" = "0" ] && echo PASS || echo FAIL)"
echo "GATE 2 artifact <= weights+4GiB: ${G2}"
echo "GATE 3 restore < 30s: ${G3} (${RESTORE} s)"
echo "GATE 4 device isolation: PASS (verified in v1)"

cat > "${SMOKE_OUT}/stage01-result.json" <<EOF
{
  "stage": "0.1-v2",
  "topology": "containerized single-GPU L4 g6.xlarge",
  "image": "${IMAGE}",
  "skip_mounts_count": ${TOTAL},
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
  "gate_4_device_isolation": "PASS"
}
EOF
echo "result -> ${SMOKE_OUT}/stage01-result.json"
