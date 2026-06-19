#!/usr/bin/env bash
# Scale the existing ai-infra-b300-spot managed node group up (1) or down (0).
# B300 spot is ~$27/hr — scale to 0 the moment you hit a hard blocker or finish.
set -euo pipefail

CLUSTER="${CLUSTER:-qn-sglang-eks-cluster}"
NG="${NG:-ai-infra-b300-spot}"
REGION="${REGION:-us-west-2}"
DESIRED="${1:?usage: scale-node.sh <0|1>}"

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NG" \
  --scaling-config "minSize=0,maxSize=1,desiredSize=${DESIRED}" \
  --region "$REGION"

echo "Requested desiredSize=$DESIRED for $NG on $CLUSTER ($REGION)."
echo "Watch: aws eks describe-nodegroup --cluster-name $CLUSTER --nodegroup-name $NG --region $REGION --query 'nodegroup.status'"
echo "       kubectl get nodes -l ai-infra/role=b300-spot -w"
