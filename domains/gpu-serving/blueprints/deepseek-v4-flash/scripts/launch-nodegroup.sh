#!/usr/bin/env bash
# Provision the B300 spot nodegroup for DeepSeek-V4-Flash benchmark.
# Idempotent — safe to re-run; will skip if the nodegroup already exists.
set -euo pipefail

CLUSTER=qn-sglang-eks-cluster
REGION=us-west-2
NG_NAME=dsv4-b300-spot
SUBNET=subnet-001db6882dbb5ac72   # usw2-az2 — only AZ with B300 capacity
NODE_ROLE=arn:aws:iam::615299764834:role/gpu-eks-node-group-20260303162535678600000025
INSTANCE=p6-b300.48xlarge

if aws eks describe-nodegroup --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "[skip] Nodegroup $NG_NAME already exists"
  aws eks describe-nodegroup --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION" \
    --query 'nodegroup.{Status:status,Capacity:capacityType,Instance:instanceTypes,Subnets:subnets}'
  exit 0
fi

echo "[create] Nodegroup $NG_NAME on $INSTANCE (SPOT) in $SUBNET"
aws eks create-nodegroup \
  --cluster-name "$CLUSTER" \
  --region "$REGION" \
  --nodegroup-name "$NG_NAME" \
  --subnets "$SUBNET" \
  --instance-types "$INSTANCE" \
  --capacity-type SPOT \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  --disk-size 200 \
  --ami-type AL2023_x86_64_NVIDIA \
  --node-role "$NODE_ROLE" \
  --taints key=nvidia.com/gpu,value=true,effect=NO_SCHEDULE \
  --labels role=gpu,model=deepseek-v4-flash,instance=p6-b300.48xlarge \
  --tags Project=deepseek-v4-flash,SpotReclaim=true,SpecRef=domains/gpu-serving/specs/deepseek-v4-flash.md

echo "[wait] Nodegroup ACTIVE (typically 3-8 min)"
aws eks wait nodegroup-active --cluster-name "$CLUSTER" --nodegroup-name "$NG_NAME" --region "$REGION"

echo "[done] Nodegroup ready. Verify with: kubectl get nodes -l model=deepseek-v4-flash"
