#!/usr/bin/env bash
# =============================================================================
# MiniMax-M2 B200 — CPU-OFFLOAD RE-TEST (2026-06-30), the customer's architecture question.
# P1 proved gpu-only saturates the GPU KV pool (util->1.00, preemptions) in the distinct-prefix
# regime — exactly where a CPU offload tier SHOULD help. P1's offload arms failed only because
# gen-manifest emitted BOGUS --kv_offloading_* flags (now fixed, source-verified: OffloadingConnector
# is configured via --kv-transfer-config ONLY; CPUOffloadingSpec reads num_cpu_blocks).
#
# This run: node-up -> stage -> gpu-only boot smoke -> CPU-OFFLOAD boot gate (NEW: proves the flag fix
# on hardware before the sweep) -> scoped tiering sweep {gpu-only, cpu-offload} x {N=86,172} x {c=64,128}
# at the saturation knee -> trap scaledown of the CORRECT -maz NG. Detached, self-terminating.
# =============================================================================
set -uo pipefail
CTX="qwen3-next-bench-eks-cluster"
NODEGROUP="ai-infra-use2-b200-spot-maz"; REGION="us-east-2"
K=(kubectl --context "$CTX")
KVT_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2-kv-tiering"
M2_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2"
R="${KVT_BP}/results"; LOG="${R}/ep-recovery.log"; PHASE_STATUS="${R}/EP-STATUS"
PREFLIGHT_OK=0; mkdir -p "$R"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
pstat(){ echo "$*" > "$PHASE_STATUS"; }
scale0(){ [ "$PREFLIGHT_OK" = "1" ] || { log "scale0 SKIPPED (preflight failed -> node may not be ours)"; return; }
  for i in 1 2 3 4 5; do aws eks update-nodegroup-config --cluster-name "$CTX" --nodegroup-name "$NODEGROUP" \
    --region "$REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0 >/dev/null 2>&1 \
    && { log "scale-to-zero accepted ($i)"; return; }; sleep 10; done
  log "SCALE-TO-ZERO FAILED 5x — MANUAL ACTION"; pstat "MANUAL_ACTION_REQUIRED"; }
trap 'rc=$?; log "EXIT rc=$rc"; scale0' EXIT
trap 'log "SIGINT"; exit 130' INT
trap 'log "SIGTERM"; exit 143' TERM
( sleep $((300*60)); log "WALL CAP hit"; kill -TERM -$$ 2>/dev/null ) & WATCHDOG=$!

log "=== EP-RECOVERY RUN START (does the DP-based EP path recover 0.19's +40%?) ==="
ACTUAL=$(kubectl config view -o jsonpath="{.contexts[?(@.name=='${CTX}')].context.cluster}" 2>/dev/null)
echo "$ACTUAL" | grep -q "$CTX" || { log "PREFLIGHT FAIL: context->cluster mismatch. ABORT."; exit 1; }

log "waiting for node Ready (up to 45min for spot join)..."
for i in $(seq 1 45); do
  "${K[@]}" get nodes -l ai-infra/role=b200-spot 2>/dev/null | grep -q Ready && break; sleep 60; done
"${K[@]}" label nodes -l ai-infra/role=b200-spot blueprint=minimax-m2 nvidia.com/gpu.present=true --overwrite >/dev/null 2>&1
"${K[@]}" get nodes -l blueprint=minimax-m2 2>/dev/null | grep -q Ready \
  || { log "PREFLIGHT FAIL: no Ready blueprint=minimax-m2 node. ABORT."; exit 1; }
PREFLIGHT_OK=1; log "PREFLIGHT OK — scaledown ARMED (NG=${NODEGROUP})."

# ── NVMe SELF-HEAL: a FRESH spot node has raw, UNMOUNTED instance-store; /mnt/nvme won't exist and the
# stage pod's hostPath(type:Directory) mount fails forever. Format+mount one disk before staging. Idempotent.
log "ensuring /mnt/nvme is mounted on the node..."
NODE=$("${K[@]}" get nodes -l blueprint=minimax-m2 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
"${K[@]}" delete pod nvme-prep --ignore-not-found --wait=true >>"$LOG" 2>&1
cat <<YAML | "${K[@]}" apply -f - >>"$LOG" 2>&1
apiVersion: v1
kind: Pod
metadata: {name: nvme-prep, namespace: default}
spec:
  restartPolicy: Never
  nodeName: ${NODE}
  hostPID: true
  tolerations: [{operator: Exists}]
  containers:
  - name: p
    image: busybox
    securityContext: {privileged: true}
    command: ["nsenter","--target=1","--mount","--uts","--ipc","--net","--pid","--","sh","-c","if mountpoint -q /mnt/nvme; then echo ALREADY; else mkdir -p /mnt/nvme; blkid /dev/nvme1n1 >/dev/null 2>&1 || mkfs.xfs -f /dev/nvme1n1; mount /dev/nvme1n1 /mnt/nvme; fi; mkdir -p /mnt/nvme/models /mnt/nvme/hf-cache /mnt/nvme/kv-cache /mnt/nvme/vllm-cache /mnt/nvme/bench-results; chmod 777 /mnt/nvme; df -h /mnt/nvme"]
YAML
for i in $(seq 1 20); do
  ph=$("${K[@]}" get pod nvme-prep -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ph" = "Succeeded" ] && { log "NVMe mounted OK"; break; }
  [ "$ph" = "Failed" ] && { log "FATAL: nvme-prep failed"; "${K[@]}" logs nvme-prep 2>&1 | tail -10 | tee -a "$LOG"; exit 1; }
  sleep 10
done
"${K[@]}" delete pod nvme-prep --ignore-not-found --wait=false >>"$LOG" 2>&1

# fresh stage
"${K[@]}" delete job stage-minimax-m2 --ignore-not-found >/dev/null 2>&1; sleep 3
"${K[@]}" apply -f "${KVT_BP}/k8s/stage-model.yaml" >>"$LOG" 2>&1
log "staging weights (~25min)..."
for i in $(seq 1 50); do
  st=$("${K[@]}" get job stage-minimax-m2 -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
  case "$st" in *Complete*) log "staging done"; break;; *Failed*) log "FATAL staging failed"; exit 1;; esac; sleep 60; done
"${K[@]}" apply -f "${KVT_BP}/k8s/observability.yaml" >>"$LOG" 2>&1; log "observability applied"

# ── CPU-OFFLOAD BOOT GATE (NEW): prove the FIXED offload flags boot M2 on hardware before the sweep.
# This is the load-bearing check — if cpu-offload still won't boot, the sweep is pointless.
boot_gate(){  # $1=arm
  local shape="$1"
  local arm="${2:-gpu-only}"
  local tag="${shape}-${arm}"
  local yaml="/tmp/gate-${tag}.yaml"
  local pod ph code i
  bash "${M2_BP}/k8s/gen-serving-manifest.sh" "$shape" "$arm" > "$yaml" 2>>"$LOG" || { log "GATE gen-manifest FAILED ${tag}"; return 1; }
  "${K[@]}" delete pod -l app=vllm-minimax-m2 --ignore-not-found --wait=true >>"$LOG" 2>&1
  "${K[@]}" apply -f "$yaml" >>"$LOG" 2>&1
  pod=$("${K[@]}" get -f "$yaml" -o jsonpath='{.metadata.name}' 2>/dev/null)
  log "GATE ${tag}: pod=${pod}; waiting boot (~35min cold)..."
  for i in $(seq 1 80); do
    ph=$("${K[@]}" get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$ph" != "Running" ] && [ "$ph" != "Pending" ]; then
      log "GATE ${tag} FAILED phase=$ph — capturing WORKER trace (deep tail)."
      "${K[@]}" logs "$pod" --tail=400 2>&1 | grep -iE "EngineCore|Worker|Error|assert|raise|Exception|Traceback|not supported|incompatible|CUDA|NCCL|ValueError|RuntimeError|offload|num_cpu" | tail -50 | tee -a "$LOG"
      "${K[@]}" delete pod "$pod" --ignore-not-found >>"$LOG" 2>&1
      return 1
    fi
    code=$("${K[@]}" exec "$pod" -- curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
    log "GATE ${tag}: phase=$ph health=$code"
    [ "$code" = "200" ] && { log "GATE ${tag} PASS (health 200)."; "${K[@]}" delete pod "$pod" --ignore-not-found --wait=true >>"$LOG" 2>&1; sleep 5; return 0; }
    sleep 40
  done
  log "GATE ${tag}: never healthy"; "${K[@]}" delete pod "$pod" --ignore-not-found >>"$LOG" 2>&1; return 1
}

# QUESTION: 0.19 tp4ep4 (TP4,DP1,--enable-expert-parallel) gave +40% (2158 vs 1544 @c128 shared-prefix).
# The SAME shape REGRESSED on 0.23 (1486 vs 1573). Per vLLM EP docs, current-vLLM EP benefit comes via the
# DP path (EP_SIZE=TP*DP; --enable-expert-parallel + --data-parallel-size TOGETHER). We NEVER ran that combo
# (our DP shapes had EP="" ; our EP shape had DP=1). This run tests the docs' single-node EP recipe.
# Baseline tp4 is re-measured in the SAME run for a clean like-for-like on this exact node.

pstat "GATE tp4 (baseline)"; boot_gate tp4 gpu-only || { log "tp4 baseline gate failed (unexpected). ABORT."; exit 1; }

# Try the DP-based EP shapes in order of docs-preference. First that BOOTS wins the throughput compare.
# If BOTH fail to boot, the deep-tail WORKER trace (in boot_gate) tells us WHY DP won't start for M2 on 0.23.
EP_SHAPE=""
for cand in tp1dp4ep tp2dp2ep; do
  pstat "GATE ${cand}"
  if boot_gate "$cand" gpu-only; then EP_SHAPE="$cand"; log "=== ${cand} BOOTS on 0.23 — this is a real DP-based EP config. ==="; break
  else log "=== ${cand} FAILED to boot — worker trace captured above (this is the DP-boot blocker). ==="; fi
done

if [ -n "$EP_SHAPE" ]; then
  log "=== EP RECOVERY: comparing tp4 (baseline) vs ${EP_SHAPE} (DP-based EP) on shared-prefix, gpu-only. ==="
  pstat "SWEEP ep-recovery tp4 vs ${EP_SHAPE} @ shared-prefix"
  SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" \
    SHAPES="tp4 ${EP_SHAPE}" KV_ARMS="gpu-only" SCENARIOS="shared-prefix" CONC_STEPS="64 128 256" \
    bash "${M2_BP}/k8s/run-pareto-sweep.sh" >>"$LOG" 2>&1 \
    && log "EP-RECOVERY sweep complete — compare ${EP_SHAPE} vs tp4 (1848@c256) and 0.19 tp4ep4 (2158@c128)." \
    || log "EP-RECOVERY sweep ended rc=$? (check EP-STATUS)"
else
  log "=== NEITHER DP-based EP shape booted on 0.23 for M2. EP-via-DP is UNAVAILABLE until the DP-boot bug is"
  log "    root-caused (worker traces captured above). The +40% lever cannot be recovered on 0.23 as-is. ==="
  pstat "ABORT all-EP-DP-shapes-failed (see worker traces in ep-recovery.log)"
fi

pstat "DONE — scaling to 0"; log "=== EP-RECOVERY DONE ==="
kill "$WATCHDOG" 2>/dev/null
