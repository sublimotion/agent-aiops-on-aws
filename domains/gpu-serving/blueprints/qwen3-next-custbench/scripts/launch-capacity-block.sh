#!/usr/bin/env bash
# Launch p5en.48xlarge into existing qwen3-next EKS cluster via capacity block
# Reservation: cr-027183248ccb9f13e (Feb 26 11:30 UTC - Feb 27 11:30 UTC)
#
# Prerequisites:
#   - Capacity block must be in 'active' state (check: aws ec2 describe-capacity-reservations)
#   - EKS cluster qwen3-next-bench-eks-cluster must be running
#   - Run from repo root or set BLUEPRINT_DIR
#
# Usage:
#   ./scripts/launch-capacity-block.sh
#   DRY_RUN=1 ./scripts/launch-capacity-block.sh   # print command without executing
set -euo pipefail

# --- Configuration ---
# These values come from the parent qwen3-next terraform outputs.
# Override via environment variables or update defaults before running.
REGION="${REGION:-us-east-2}"
CAPACITY_RESERVATION_ID="${CAPACITY_RESERVATION_ID:-cr-027183248ccb9f13e}"  # UPDATE: set to your active capacity block
INSTANCE_TYPE="${INSTANCE_TYPE:-p5en.48xlarge}"
SUBNET_ID="${SUBNET_ID:-subnet-04be09c7bf104edb8}"                         # us-east-2c private subnet
CLUSTER_NAME="${CLUSTER_NAME:-qwen3-next-bench-eks-cluster}"
VOLUME_SIZE="${VOLUME_SIZE:-500}"

# Instance profile ARN — get from terraform: cd ../qwen3-next && terraform output gpu_instance_profile_arn
INSTANCE_PROFILE_ARN="${INSTANCE_PROFILE_ARN:-arn:aws:iam::615299764834:instance-profile/qwen3-next-bench-gpu-node-20260223212318728300000009}"

# Security groups: EKS node SG + NodePort SG + FSx SG + FSx-EFA SG
# Get from terraform: cd ../qwen3-next && terraform output -json security_group_ids
SG_EKS_NODE="${SG_EKS_NODE:-sg-0bf5ad07fc6c29df1}"
SG_GPU_NODEPORT="${SG_GPU_NODEPORT:-sg-02aeef8a922c45276}"
SG_FSX="${SG_FSX:-sg-07c6da755ffe8af2d}"
SG_FSX_EFA="${SG_FSX_EFA:-sg-073b15d42ff3f4006}"

# EFA-enabled FSx for Dynamo GDS (T5d)
FSX_EFA_DNS="${FSX_EFA_DNS:-fs-0952e4fd84eed47af.fsx.us-east-2.amazonaws.com}"
FSX_EFA_MOUNT="${FSX_EFA_MOUNT:-falszb4v}"

# AMI: Use AL2 GPU AMI (not AL2023) to match the existing managed node group.
# AL2 uses /etc/eks/bootstrap.sh; AL2023 uses nodeadm (different user data format).
# Get the latest recommended AMI from SSM parameter store.
AMI_ID="${AMI_ID:-$(aws ssm get-parameter \
  --name /aws/service/eks/optimized-ami/1.32/amazon-linux-2-gpu/recommended/image_id \
  --region "$REGION" --query 'Parameter.Value' --output text 2>/dev/null || echo 'ami-01750357dd1909542')}"

DRY_RUN="${DRY_RUN:-0}"

# --- Preflight checks ---
echo "=== Preflight Checks ==="

# 1. Check capacity block state
CB_STATE=$(aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids "$CAPACITY_RESERVATION_ID" \
  --region "$REGION" \
  --query "CapacityReservations[0].State" \
  --output text 2>&1)
echo "Capacity block state: $CB_STATE"

if [ "$CB_STATE" != "active" ]; then
  echo "ERROR: Capacity block is '$CB_STATE', not 'active'."
  echo "Start time: $(aws ec2 describe-capacity-reservations \
    --capacity-reservation-ids "$CAPACITY_RESERVATION_ID" \
    --region "$REGION" \
    --query "CapacityReservations[0].StartDate" \
    --output text 2>&1)"
  echo "Wait until the block activates before launching."
  exit 1
fi

# 2. Check EKS cluster
EKS_STATE=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
  --query "cluster.status" --output text 2>&1)
echo "EKS cluster state: $EKS_STATE"
if [ "$EKS_STATE" != "ACTIVE" ]; then
  echo "ERROR: EKS cluster is not active."
  exit 1
fi

# 3. Get user data from terraform output and append FSx-EFA mount
BLUEPRINT_DIR="${BLUEPRINT_DIR:-$(cd "$(dirname "$0")/../../../qwen3-next" && pwd)}"
echo "Getting user data from: $BLUEPRINT_DIR"
USER_DATA_RAW=$(cd "$BLUEPRINT_DIR" && terraform output -raw gpu_user_data_base64 2>/dev/null) || {
  echo "ERROR: Could not get user data from terraform output."
  echo "Make sure you're in the qwen3-next blueprint directory or set BLUEPRINT_DIR."
  exit 1
}

# Decode, append FSx-EFA mount for T5d (Dynamo KVBM), re-encode
# The parent user data mounts the original FSx at /mnt/fsx.
# We additionally mount the EFA-enabled FSx at /mnt/fsx-efa for Dynamo GDS benchmarks.
USER_DATA_DECODED=$(echo "$USER_DATA_RAW" | base64 -d)
FSX_EFA_APPEND=$(cat <<'EFAMOUNT'

# =============================================================================
# 6. FSx Lustre EFA Mount (for Dynamo KVBM T5d benchmarks)
# EFA-enabled FSx: fs-0952e4fd84eed47af (PERSISTENT_2, 4800 GiB, EFA=true)
# =============================================================================
mkdir -p /mnt/fsx-efa
EFAMOUNT
)
FSX_EFA_APPEND+=$'\n'"mount -t lustre -o noatime,flock ${FSX_EFA_DNS}@tcp:/${FSX_EFA_MOUNT} /mnt/fsx-efa || echo 'WARNING: FSx-EFA mount failed (non-fatal for T1-T4 benchmarks)'"
FSX_EFA_APPEND+=$'\n'"chmod 777 /mnt/fsx-efa 2>/dev/null || true"

USER_DATA=$(echo "${USER_DATA_DECODED}${FSX_EFA_APPEND}" | base64 | tr -d '\n')
echo "User data size: $(echo -n "$USER_DATA" | wc -c | tr -d ' ') bytes (base64)"

echo ""
echo "=== Launch Configuration ==="
echo "AMI:              $AMI_ID (AL2 GPU)"
echo "Instance type:    $INSTANCE_TYPE"
echo "Capacity block:   $CAPACITY_RESERVATION_ID"
echo "Subnet:           $SUBNET_ID (us-east-2c)"
echo "Instance profile: $INSTANCE_PROFILE_ARN"
echo "Security groups:  $SG_EKS_NODE (node), $SG_GPU_NODEPORT (nodeport), $SG_FSX (fsx), $SG_FSX_EFA (fsx-efa)"
echo "Root volume:      ${VOLUME_SIZE}GB gp3"
echo "FSx-EFA mount:    ${FSX_EFA_DNS}@tcp:/${FSX_EFA_MOUNT} → /mnt/fsx-efa"
echo ""

# --- Launch ---
LAUNCH_CMD=(
  aws ec2 run-instances
  --region "$REGION"
  --image-id "$AMI_ID"
  --instance-type "$INSTANCE_TYPE"
  --instance-market-options "MarketType=capacity-block"
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CAPACITY_RESERVATION_ID}"
  --placement "AvailabilityZone=us-east-2c"
  --iam-instance-profile "Arn=$INSTANCE_PROFILE_ARN"
  --subnet-id "$SUBNET_ID"
  --security-group-ids "$SG_EKS_NODE" "$SG_GPU_NODEPORT" "$SG_FSX" "$SG_FSX_EFA"
  --user-data "$USER_DATA"
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=qwen3-custbench-gpu-node},{Key=Project,Value=qwen3-custbench},{Key=Purpose,Value=customer-benchmark},{Key=kubernetes.io/cluster/$CLUSTER_NAME,Value=owned}]"
  --output json
)

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN — would execute:"
  echo "${LAUNCH_CMD[*]}"
  exit 0
fi

echo "Launching instance..."
RESULT=$("${LAUNCH_CMD[@]}" 2>&1)
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")

echo ""
echo "=== Instance Launched ==="
echo "Instance ID: $INSTANCE_ID"
echo ""
echo "=== Next Steps ==="
echo "1. Wait for instance to join EKS cluster:"
echo "   aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME"
echo "   kubectl get nodes -w"
echo ""
echo "2. Copy model from FSx to NVMe (SSH or SSM to node):"
echo "   # Model should already be on FSx at /mnt/fsx/models/qwen3-next/"
echo "   mkdir -p /mnt/nvme/models"
echo "   cp -r /mnt/fsx/models/qwen3-next /mnt/nvme/models/qwen3-next-fp8"
echo ""
echo "3. Build customer Docker image on the node:"
echo "   # Copy docker/Dockerfile.vllm-customer to the node"
echo "   nerdctl build -f Dockerfile.vllm-customer -t qwen3-next-custbench:latest ."
echo ""
echo "4. Run benchmarks:"
echo "   # Config A (customer baseline):"
echo "   ./configs/vllm-customer-baseline.sh"
echo "   ./scripts/run-benchmarks.sh t1"
echo ""
echo "   # Config B (optimized) — stop Config A first:"
echo "   nerdctl stop vllm-custbench-baseline"
echo "   ./configs/vllm-optimized.sh"
echo "   ./scripts/run-benchmarks.sh t2"
echo ""
echo "Monitor instance:"
echo "   aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name' --output text"
