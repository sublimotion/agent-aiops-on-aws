#!/usr/bin/env bash
# Chase scarce g7e on-demand capacity in us-east-2 (offered only in 2a/2b).
# EKS managed nodegroups do NOT auto-retry after CREATE_FAILED, so this loop
# recreates a single-size nodegroup whenever it fails, racing 48/24/12xl.
# Stops as soon as ANY size reaches ACTIVE (an instance launched).
set -uo pipefail

CL=qwen3-next-bench-eks-cluster
REGION=us-east-2
ROLE=arn:aws:iam::615299764834:role/ai-infra-use2-b200-node
SUBNETS="subnet-0fced510ea62b874e subnet-03d03f1fb8d62d6a5"   # us-east-2a + 2b (g7e NOT offered in 2c)
AMI=AL2023_x86_64_NVIDIA
DISK=200
MAX_ROUNDS=${MAX_ROUNDS:-60}    # ~ each round waits up to ~6 min; 60 rounds ~ 6h

declare -A NG=( [48]=fin-rag-g7e-od-48 [24]=fin-rag-g7e-od-24 [12]=fin-rag-g7e-od-12 )

create_ng() {
  local size=$1 name=${NG[$1]}
  aws eks create-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "$name" \
    --node-role "$ROLE" \
    --subnets $SUBNETS \
    --instance-types "g7e.${size}xlarge" \
    --capacity-type ON_DEMAND \
    --ami-type "$AMI" \
    --disk-size "$DISK" \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --labels fin-rag/hardware=g7e-sm120 \
    --taints key=ai-infra/g7e,value=true,effect=NO_SCHEDULE \
    --query 'nodegroup.status' --output text 2>&1
}

ng_status() {
  aws eks describe-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "${NG[$1]}" --query 'nodegroup.status' --output text 2>/dev/null \
    || echo "ABSENT"
}

delete_ng() {
  aws eks delete-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "${NG[$1]}" >/dev/null 2>&1
}

wait_gone() {
  local size=$1
  for _ in $(seq 1 30); do
    [ "$(ng_status "$size")" = "ABSENT" ] && return 0
    sleep 15
  done
}

echo "$(date +%H:%M:%S) launching initial nodegroups (48/24/12)"
for s in 48 24 12; do echo "  od-$s: $(create_ng "$s")"; done

for round in $(seq 1 "$MAX_ROUNDS"); do
  sleep 45
  line="$(date +%H:%M:%S) round=$round |"
  for s in 48 24 12; do
    st=$(ng_status "$s")
    line="$line od-$s=$st"
    if [ "$st" = "ACTIVE" ]; then
      echo "$line"
      echo "WON: g7e.${s}xlarge ACTIVE — node launched. Stopping chase."
      kubectl get nodes -l fin-rag/hardware=g7e-sm120 2>&1 | head
      exit 0
    fi
  done
  echo "$line"
  # Recreate any that failed.
  for s in 48 24 12; do
    st=$(ng_status "$s")
    if [ "$st" = "CREATE_FAILED" ] || [ "$st" = "DEGRADED" ]; then
      echo "  od-$s $st -> delete + recreate"
      delete_ng "$s"; wait_gone "$s"
      echo "  od-$s recreate: $(create_ng "$s")"
    fi
  done
done
echo "GAVE_UP after $MAX_ROUNDS rounds — g7e capacity never freed."
exit 1
