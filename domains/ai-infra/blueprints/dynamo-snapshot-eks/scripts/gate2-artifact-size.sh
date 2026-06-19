#!/usr/bin/env bash
# Gate 2 — checkpoint artifact size cap for Ministral-3B.
#
# Cap = weights + 4 GiB + 1 GiB × TP_SIZE
#     = 6 GiB (model) + 4 GiB + 1 GiB × 1 = 11 GiB
#
# Reads the EBS snapshot size from the AWS API for the snapshot recorded
# in results/e1/e1-result.json.
set -euo pipefail
RESULT=${1:-results/e1/e1-result.json}
CAP_GIB=${CAP_GIB:-11}
SID=$(jq -r '.snapshot_id' "$RESULT")
[ -z "$SID" ] || [ "$SID" = "null" ] && { echo "snapshot_id missing in $RESULT"; exit 2; }

SIZE_GIB=$(aws ec2 describe-snapshots --snapshot-ids "$SID" --region us-west-2 \
  --query 'Snapshots[0].VolumeSize' --output text)

# VolumeSize is the SOURCE volume size (50 GiB in our case). We want the
# actual delta = used blocks. Use describe-snapshots' StorageTier or fall
# back to the source PVC's used bytes via kubectl exec on the agent.
# For E1, we approximate: read the per-checkpoint dir size from the
# agent's /checkpoints inside the snapshot via a temp-mount probe pod.
# Iter 5b: implement after first successful snapshot.
echo "snapshot $SID volume_size=${SIZE_GIB}GiB cap=${CAP_GIB}GiB"

# For now, read the orchestrator's report which includes a `dump_bytes` field
# the agent emits via event annotations (TODO: stitch in iter 5b).
python3 - <<EOF
import json, sys
r=json.load(open("$RESULT"))
size_gib=$SIZE_GIB
cap=$CAP_GIB
ok = size_gib <= cap
print(json.dumps({"snapshot_id":"$SID","volume_size_gib":size_gib,"cap_gib":cap,"pass":ok}, indent=2))
sys.exit(0 if ok else 1)
EOF
