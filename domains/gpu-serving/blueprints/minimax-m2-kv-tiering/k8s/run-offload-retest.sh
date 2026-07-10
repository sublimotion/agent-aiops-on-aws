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
R="${KVT_BP}/results"; LOG="${R}/offload-retest.log"; PHASE_STATUS="${R}/OFFLOAD-STATUS"
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

log "=== CPU-OFFLOAD RE-TEST START ==="
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

pstat "GATE gpu-only"; boot_gate tp4 gpu-only || { log "gpu-only gate failed (unexpected — P1 booted it). ABORT."; exit 1; }
pstat "GATE cpu-offload"
if boot_gate tp4 cpu-offload; then
  log "=== CPU-OFFLOAD BOOTS (cpu_bytes_to_use fix). Running scoped knee sweep incl nvme-tiering (fs disk tier, supported on 0.23). ==="
  pstat "SWEEP gpu-only+cpu-offload+nvme @ knee"
  SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" \
    KV_ARMS="gpu-only cpu-offload nvme-tiering" NUM_PREFIXES="86 172" ACCESS="zipfian" CONC_STEPS="64 128" \
    bash "${KVT_BP}/k8s/run-tiering-sweep.sh" >>"$LOG" 2>&1 \
    && log "OFFLOAD SWEEP complete" || log "OFFLOAD SWEEP ended rc=$? (check OFFLOAD-STATUS)"
else
  log "=== CPU-OFFLOAD STILL FAILS TO BOOT even with fixed flags — worker trace captured above. ==="
  log "Do NOT run the sweep (it would be all gpu-only). The offload boot is the blocker; diagnose from the WORKER trace."
  pstat "ABORT cpu-offload-gate-failed (see worker trace in offload-retest.log)"
fi
# ── PHASE 3: EP comparison — does vLLM 0.23 keep the +40% tp4ep4 lever that 0.19 had? ──
# 0.19 reference (shared-prefix, gpu-only): tp4=1544 vs tp4ep4=2158 tok/s @c128 (+40%). Re-measure on 0.23,
# SAME scenario/shape/conc, like-for-like. This is the one comparison that settles the 0.19-vs-0.23 question.
# Boot-gate ep4tp4 first (it uses the DP-adjacent expert-parallel path — must confirm it even boots on 0.23).
pstat "GATE tp4ep4"
if boot_gate tp4ep4 gpu-only; then
  log "=== tp4ep4 BOOTS on 0.23. Running EP-vs-TP throughput comparison (shared-prefix, gpu-only). ==="
  pstat "SWEEP ep-compare tp4 vs tp4ep4 @ shared-prefix"
  SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" \
    SHAPES="tp4 tp4ep4" KV_ARMS="gpu-only" SCENARIOS="shared-prefix" CONC_STEPS="64 128 256" \
    bash "${M2_BP}/k8s/run-pareto-sweep.sh" >>"$LOG" 2>&1 \
    && log "EP-COMPARE complete" || log "EP-COMPARE ended rc=$? (check status)"
else
  log "=== tp4ep4 FAILS TO BOOT on 0.23 — the +40% EP lever from 0.19 is GONE on this engine (worker trace above). ==="
  log "This is a material 0.23 REGRESSION vs 0.19 for throughput headroom — flag it."
  pstat "tp4ep4-gate-failed (EP lever lost on 0.23)"
fi

pstat "DONE — scaling to 0"; log "=== RE-TEST DONE ==="
kill "$WATCHDOG" 2>/dev/null
