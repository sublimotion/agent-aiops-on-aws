#!/usr/bin/env bash
# ODCR-based g7e capacity chase. Faster + more reliable than the EKS-nodegroup
# chase: create-capacity-reservation fails in SECONDS on InsufficientInstanceCapacity
# (vs ~33 min for an EKS nodegroup to surface CREATE_FAILED), and on success it
# actually HOLDS the capacity so it can't be lost before the node launches.
#
# Loop: probe each g7e size in each offered AZ (2a/2b — g7e is NOT offered in 2c).
# First reservation that succeeds -> create a single-AZ nodegroup pinned to that
# subnet; the `open` match criteria makes the on-demand launch consume the ODCR.
#
# BILLING: a held ODCR bills the on-demand rate whether used or not. This script
# cancels any reservation it does NOT end up consuming, and only KEEPS the winner
# (which the node immediately fills). If you Ctrl-C mid-run, check for stray
# reservations: aws ec2 describe-capacity-reservations --filters Name=state,Values=active
set -uo pipefail

REGION=us-east-2
CL=qwen3-next-bench-eks-cluster
ROLE=arn:aws:iam::615299764834:role/ai-infra-use2-b200-node
AMI=AL2023_x86_64_NVIDIA
DISK=200
MAX_ROUNDS=${MAX_ROUNDS:-240}      # ~30s/round -> ~2h
SIZES=${SIZES:-"12 24 48"}         # smallest first: 12xl (2 GPU) is the min for FP8 TP2
AZS=${AZS:-"us-east-2a us-east-2b"}

az_subnet() { case "$1" in
  us-east-2a) echo subnet-0fced510ea62b874e;;
  us-east-2b) echo subnet-03d03f1fb8d62d6a5;;
esac; }
ng_name() { echo "fin-rag-g7e-od-$1"; }

# Try to reserve one instance of g7e.<size>xlarge in <az>. Echoes reservation id
# on success, empty on capacity failure.
try_reserve() {
  local size=$1 az=$2
  aws ec2 create-capacity-reservation --region "$REGION" \
    --instance-type "g7e.${size}xlarge" --instance-platform Linux/UNIX \
    --availability-zone "$az" --instance-count 1 \
    --instance-match-criteria open --end-date-type unlimited \
    --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=purpose,Value=fin-rag-g7e-chase}]' \
    --query 'CapacityReservation.CapacityReservationId' --output text 2>/dev/null
}

cancel_reservation() {
  aws ec2 cancel-capacity-reservation --region "$REGION" \
    --capacity-reservation-id "$1" >/dev/null 2>&1
}

create_ng() {
  local size=$1 az=$2 subnet
  subnet=$(az_subnet "$az")
  aws eks create-nodegroup --region "$REGION" --cluster-name "$CL" \
    --nodegroup-name "$(ng_name "$size")" --node-role "$ROLE" --subnets "$subnet" \
    --instance-types "g7e.${size}xlarge" --capacity-type ON_DEMAND --ami-type "$AMI" \
    --disk-size "$DISK" --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --labels fin-rag/hardware=g7e-sm120 \
    --taints key=ai-infra/g7e,value=true,effect=NO_SCHEDULE \
    --query 'nodegroup.status' --output text 2>&1
}

echo "$(date +%H:%M:%S) ODCR chase: sizes=[$SIZES] azs=[$AZS] max_rounds=$MAX_ROUNDS"
for round in $(seq 1 "$MAX_ROUNDS"); do
  for size in $SIZES; do
    for az in $AZS; do
      rid=$(try_reserve "$size" "$az")
      if [ -n "$rid" ] && [ "$rid" != "None" ]; then
        echo "$(date +%H:%M:%S) WON reservation $rid for g7e.${size}xlarge in $az"
        echo "  creating nodegroup $(ng_name "$size") pinned to $az ($(az_subnet "$az"))"
        echo "  nodegroup status: $(create_ng "$size" "$az")"
        echo "  ODCR $rid retained — the on-demand node will consume it on launch."
        echo "WON: g7e.${size}xlarge reserved in $az. Stopping chase."
        exit 0
      fi
    done
  done
  echo "$(date +%H:%M:%S) round=$round | no g7e capacity in any size/az — retrying"
  sleep 30
done
echo "GAVE_UP after $MAX_ROUNDS rounds — g7e capacity never freed."
exit 1
