#!/usr/bin/env bash
# Pre-flight check before launching the B300 nodegroup.
# Verifies: AWS auth, EKS access, B300 spot capacity in usw2-az2, kubeconfig, NVMe pattern.
set -euo pipefail

REGION=us-west-2
CLUSTER=qn-sglang-eks-cluster
INSTANCE=p6-b300.48xlarge
TARGET_AZ=us-west-2b

echo "=== 1. AWS identity ==="
aws sts get-caller-identity --query 'Arn' --output text

echo
echo "=== 2. EKS cluster status ==="
aws eks describe-cluster --name "$CLUSTER" --region "$REGION" --query 'cluster.{Status:status,Version:version}' --output table

echo
echo "=== 3. B300 capacity in $TARGET_AZ ==="
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE" "Name=location,Values=$TARGET_AZ" \
  --region "$REGION" --output table

echo
echo "=== 4. Recent B300 spot price ==="
aws ec2 describe-spot-price-history \
  --instance-types "$INSTANCE" \
  --product-descriptions "Linux/UNIX" \
  --availability-zone "$TARGET_AZ" \
  --max-items 3 \
  --region "$REGION" \
  --query 'SpotPriceHistory[].{Time:Timestamp,Price:SpotPrice}' --output table

echo
echo "=== 5. kubectl context ==="
if ! kubectl config current-context >/dev/null 2>&1; then
  echo "[setup] Updating kubeconfig"
  aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"
fi
kubectl config current-context
kubectl get nodes -o wide

echo
echo "=== 6. Existing GPU nodes ==="
kubectl get nodes -l role=gpu -o custom-columns=NAME:.metadata.name,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type,STATUS:.status.conditions[-1].type 2>/dev/null || echo "[none]"

echo
echo "=== 7. Disk space check on existing GPU nodes (NVMe pattern reuse) ==="
for node in $(kubectl get nodes -l role=gpu -o name 2>/dev/null | head -1); do
  echo "Node: $node"
  kubectl describe "$node" | grep -E "Capacity|Allocatable|nvidia.com/gpu" | head -5 || true
done

echo
echo "=== Pre-flight complete ==="
echo "Next: bash launch-nodegroup.sh"
