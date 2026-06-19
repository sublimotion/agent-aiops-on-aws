#!/usr/bin/env bash
# Multi-region ODCR chase for g7e. Probes every AZ that offers g7e across all
# regions where it's available (10 AZs / 4 regions). create-capacity-reservation
# fails in ~2s on InsufficientInstanceCapacity, so we sweep the whole surface in
# seconds and catch a capacity window the instant it opens anywhere.
#
# On WIN: holds the reservation and STOPS (does NOT launch). Launch path differs
# per region (EKS where a cluster exists, plain EC2 otherwise), so a human/agent
# decides next. The held ODCR guarantees the capacity won't be lost meanwhile.
#
# BILLING: a held ODCR bills the on-demand rate. The winner is intentionally
# KEPT (that's the point) — consume it promptly or cancel it:
#   aws ec2 cancel-capacity-reservation --region <r> --capacity-reservation-id <id>
# Probe reservations that briefly succeed are always cancelled.
set -uo pipefail

SIZE=${SIZE:-12}                  # 12xl = 2 GPU = FP8 TP2 minimum; smallest = easiest to land
MAX_ROUNDS=${MAX_ROUNDS:-360}     # ~30s/round -> ~3h

# region:az pairs that actually offer g7e.<SIZE>xlarge (verified 2026-06-12)
TARGETS=${TARGETS:-"\
us-east-2:us-east-2a us-east-2:us-east-2b \
us-west-2:us-west-2a us-west-2:us-west-2b us-west-2:us-west-2c us-west-2:us-west-2d \
us-east-1:us-east-1a us-east-1:us-east-1c \
ap-northeast-1:ap-northeast-1a ap-northeast-1:ap-northeast-1c"}

echo "$(date +%H:%M:%S) multi-region ODCR chase: g7e.${SIZE}xlarge across:"
echo "  $TARGETS"

for round in $(seq 1 "$MAX_ROUNDS"); do
  miss=""
  for pair in $TARGETS; do
    region=${pair%%:*}; az=${pair##*:}
    rid=$(aws ec2 create-capacity-reservation --region "$region" \
      --instance-type "g7e.${SIZE}xlarge" --instance-platform Linux/UNIX \
      --availability-zone "$az" --instance-count 1 \
      --instance-match-criteria open --end-date-type unlimited \
      --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=purpose,Value=fin-rag-g7e-chase}]' \
      --query 'CapacityReservation.CapacityReservationId' --output text 2>/dev/null)
    if [[ "$rid" == cr-* ]]; then
      echo ""
      echo "$(date +%H:%M:%S) *** WON g7e.${SIZE}xlarge in $region / $az ***"
      echo "  reservation_id=$rid  region=$region  az=$az  (HELD — do not lose)"
      echo "WON: region=$region az=$az rid=$rid"
      exit 0
    fi
    miss="$miss $region/$az"
  done
  echo "$(date +%H:%M:%S) round=$round | no capacity:$miss"
  sleep 30
done
echo "GAVE_UP after $MAX_ROUNDS rounds — no g7e capacity in any region."
exit 1
