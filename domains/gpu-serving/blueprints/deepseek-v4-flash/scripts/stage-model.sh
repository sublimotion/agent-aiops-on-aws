#!/usr/bin/env bash
# Stage DeepSeek-V4-Flash weights to NVMe on the B300 node.
# Pattern matches qwen3-235b-b300/k8s/download-model.yaml: K8s Job → hf_transfer → /mnt/nvme.
# No S3 round-trip — saves 2 hours and ~$5.
set -euo pipefail

NS=default
JOB=download-deepseek-v4-flash

cd "$(dirname "$0")/.."

echo "=== Verifying NVMe is mounted on B300 node ==="
NODE=$(kubectl get nodes -l model=deepseek-v4-flash -o name | head -1)
if [[ -z "$NODE" ]]; then
  echo "[error] No node with label model=deepseek-v4-flash. Run launch-nodegroup.sh first."
  exit 1
fi
echo "Target node: $NODE"

echo
echo "=== Submitting download Job ==="
kubectl apply -f k8s/download-model.yaml

echo
echo "=== Waiting for pod to start (Job timeout: 1.5h for ~150GB at hf_transfer ~30MB/s) ==="
kubectl wait --for=condition=Ready pod -l job-name="$JOB" --timeout=300s -n "$NS" || true
POD=$(kubectl get pods -l job-name="$JOB" -n "$NS" -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"

echo
echo "=== Streaming logs (Ctrl+C to detach; Job will continue) ==="
kubectl logs -f "$POD" -n "$NS" || true

echo
echo "=== Waiting for Job completion ==="
kubectl wait --for=condition=complete job/"$JOB" --timeout=5400s -n "$NS"

echo
echo "=== Done. Verifying weight files on node NVMe ==="
kubectl debug "$NODE" --image=alpine:3.20 -- sh -c "ls -la /host/mnt/nvme/models/DeepSeek-V4-Flash/ | head; du -sh /host/mnt/nvme/models/DeepSeek-V4-Flash/" 2>/dev/null || echo "[note] kubectl debug not available; inspect via SSM if needed"

echo
echo "[done] Model staged to /mnt/nvme/models/DeepSeek-V4-Flash/"
echo "Next: kubectl apply -f k8s/vllm-serve.yaml"
