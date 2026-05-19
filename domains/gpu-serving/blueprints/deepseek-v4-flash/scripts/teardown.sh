#!/usr/bin/env bash
# Tear down everything created for this benchmark.
# Order matters: pods first (release GPUs cleanly), then nodegroup.
set -euo pipefail

CLUSTER=qn-sglang-eks-cluster
REGION=us-west-2
NG_NAME=dsv4-b300-spot

echo "=== 1. Delete serving pods + jobs ==="
kubectl delete pod vllm-deepseek-v4-flash --ignore-not-found
kubectl delete pod sglang-deepseek-v4-flash --ignore-not-found
kubectl delete job download-deepseek-v4-flash --ignore-not-found

echo
echo "=== 2. Wait for pods to terminate (release GPU allocation) ==="
kubectl wait --for=delete pod -l app=vllm-deepseek-v4-flash --timeout=120s 2>/dev/null || true
kubectl wait --for=delete pod -l app=sglang-deepseek-v4-flash --timeout=120s 2>/dev/null || true

echo
echo "=== 3. Delete the B300 nodegroup (terminates the spot instance) ==="
if aws eks describe-nodegroup --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws eks delete-nodegroup --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION"
  echo "[wait] Nodegroup deletion (typically 3-5 min)..."
  aws eks wait nodegroup-deleted --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION"
else
  echo "[skip] Nodegroup $NG_NAME does not exist"
fi

echo
echo "[done] Teardown complete. Spot charges stop now."
