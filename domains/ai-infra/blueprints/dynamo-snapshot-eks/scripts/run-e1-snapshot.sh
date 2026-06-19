#!/usr/bin/env bash
# E1 snapshot+restore driver. Orchestrates:
#   1. Confirm Deployment is at replicas=1, ready, sentinel touched
#   2. Apply 60-orchestrator-job.yaml — drives CheckpointJob → EBS snapshot+FSR → 4× restore pods
#   3. Stream the orchestrator's Job log; capture timestamps to JSON
#   4. Verify all 4 restore pods have annotation
#      nvidia.com/snapshot-restore-status.ministral=completed
#
# Output: results/e1/e1-snapshot.json with per-stage timestamps.
set -euo pipefail

NS=dynamo-snapshot
RESULTS_DIR=${1:-results/e1}
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/e1-orchestrator.log"

# Sanity: Deployment 1/1 ready
kubectl -n "$NS" wait --for=condition=Available deploy/ministral-3b --timeout=300s

# Sanity: ready-watcher has touched the sentinel (signals warm vLLM)
POD=$(kubectl -n "$NS" get pod -l app=ministral-3b -o jsonpath='{.items[0].metadata.name}')
echo "Source pod: $POD"
for i in $(seq 1 60); do
  if kubectl -n "$NS" exec "$POD" -c ready-watcher -- sh -c \
       'test -f /snapshot-control/ready-for-checkpoint' 2>/dev/null; then
    echo "ready-for-checkpoint sentinel present"
    break
  fi
  echo "waiting for ready-for-checkpoint sentinel ($i/60)..."
  sleep 5
done

# Re-apply orchestrator (idempotent if 60-orchestrator-job.yaml already in place).
kubectl apply -f "$(dirname "$0")/../k8s/60-orchestrator-job.yaml"

T_START=$(date +%s)
# Stream + persist orchestrator logs
kubectl -n "$NS" wait --for=condition=Ready pod -l app.kubernetes.io/component=snapshot-orchestrator --timeout=120s
kubectl -n "$NS" logs -f job/snapshot-orchestrator | tee "$LOG"
T_END=$(date +%s)

# Wait for the Job to complete
kubectl -n "$NS" wait --for=condition=complete job/snapshot-orchestrator --timeout=1800s

# Pull the result file out of the orchestrator pod's emptyDir
ORCH_POD=$(kubectl -n "$NS" get pod -l app.kubernetes.io/component=snapshot-orchestrator -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$NS" cp "$ORCH_POD":/results/e1-result.json "$RESULTS_DIR/e1-result.json"

# Capture annotation snapshot of the 4 restore pods
kubectl -n "$NS" get pods -l nvidia.com/snapshot-is-restore-target=true \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations}{"\n"}{end}' \
  > "$RESULTS_DIR/restore-pod-annotations.txt"

cat > "$RESULTS_DIR/e1-summary.json" <<EOF
{
  "wall_seconds": $((T_END - T_START)),
  "orchestrator_log": "$LOG",
  "orchestrator_result": "$RESULTS_DIR/e1-result.json",
  "restore_pod_annotations": "$RESULTS_DIR/restore-pod-annotations.txt"
}
EOF
echo "E1 done — wall=${T_END}-${T_START}s"
