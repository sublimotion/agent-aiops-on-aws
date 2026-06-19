#!/usr/bin/env bash
# Monitor-only variant of chase-g7e-capacity.sh: does NOT create at start
# (assumes nodegroups already exist from the chase loop). Recreates any that
# hit CREATE_FAILED/DEGRADED, stops + reports the moment any reaches ACTIVE.
set -uo pipefail
CL=qwen3-next-bench-eks-cluster
REGION=us-east-2
ROLE=arn:aws:iam::615299764834:role/ai-infra-use2-b200-node
SUBNETS="subnet-0fced510ea62b874e subnet-03d03f1fb8d62d6a5"
AMI=AL2023_x86_64_NVIDIA
DISK=200
MAX_ROUNDS=${MAX_ROUNDS:-180}   # ~ up to ~3h at ~60s/round

ng_name() { case "$1" in 48) echo fin-rag-g7e-od-48;; 24) echo fin-rag-g7e-od-24;; 12) echo fin-rag-g7e-od-12;; esac; }
create_ng() {
  aws eks create-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "$(ng_name "$1")" --node-role "$ROLE" --subnets $SUBNETS \
    --instance-types "g7e.${1}xlarge" --capacity-type ON_DEMAND --ami-type "$AMI" \
    --disk-size "$DISK" --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --labels fin-rag/hardware=g7e-sm120 \
    --taints key=ai-infra/g7e,value=true,effect=NO_SCHEDULE \
    --query 'nodegroup.status' --output text 2>&1
}
ng_status() {
  aws eks describe-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "$(ng_name "$1")" --query 'nodegroup.status' --output text 2>/dev/null || echo ABSENT
}
delete_ng() { aws eks delete-nodegroup --region "$REGION" --cluster-name "$CL" --nodegroup-name "$(ng_name "$1")" >/dev/null 2>&1; }
wait_gone() { for _ in $(seq 1 40); do [ "$(ng_status "$1")" = "ABSENT" ] && return 0; sleep 15; done; }

for round in $(seq 1 "$MAX_ROUNDS"); do
  line="$(date +%H:%M:%S) round=$round |"
  for s in 48 24 12; do
    st=$(ng_status "$s"); line="$line od-$s=$st"
    if [ "$st" = "ACTIVE" ]; then
      echo "$line"; echo "WON: g7e.${s}xlarge ACTIVE — node launched. Stopping watch."
      kubectl get nodes -l fin-rag/hardware=g7e-sm120 2>&1 | head; exit 0
    fi
  done
  echo "$line"
  for s in 48 24 12; do
    st=$(ng_status "$s")
    if [ "$st" = "CREATE_FAILED" ] || [ "$st" = "DEGRADED" ] || [ "$st" = "ABSENT" ]; then
      echo "  od-$s $st -> (re)create"; [ "$st" != "ABSENT" ] && { delete_ng "$s"; wait_gone "$s"; }
      echo "  od-$s create: $(create_ng "$s")"
    fi
  done
  sleep 50
done
echo "GAVE_UP after $MAX_ROUNDS rounds — g7e capacity never freed."
exit 1
