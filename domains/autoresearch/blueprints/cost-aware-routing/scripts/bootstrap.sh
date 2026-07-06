#!/usr/bin/env bash
# Cost-aware-routing bootstrap script — provisions a fresh p5.48xlarge spot
# instance for training. Run on a control host (laptop or other EC2) with
# the AWS profile that owns the agent-aiops-research bucket.
#
# Usage:
#   ./bootstrap.sh launch       # request spot, wait for running, configure
#   ./bootstrap.sh provision    # ssh-step: install deps, clone repo, stage data
#   ./bootstrap.sh resume       # on existing instance: pull latest checkpoint, kick off train
#   ./bootstrap.sh teardown     # cancel spot request, terminate instance
#
# Idempotent: re-runs are safe (apt installs check, S3 syncs are no-op when up to date).

set -euo pipefail

# ---------- config (verified 2026-05-25) ----------
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
AZ="${AZ:-us-east-1a}"                  # cheapest spot AZ ($9.63/hr last cycle, $9.83 current)
REGION="${REGION:-us-east-1}"
SUBNET_ID="${SUBNET_ID:-subnet-04b15a3e755dfdad6}"   # default subnet in us-east-1a
AMI_ID="${AMI_ID:-ami-02bb9f913067dadb1}"   # AL2023 NVIDIA (verified for B200/H200; NCCL works)
KEY_NAME="${KEY_NAME:-g7e-bench}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/g7e-bench.pem}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-0d2ed1207a5be7e04}"   # gpu-bench-sg, SSH 22 open from 0.0.0.0/0
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-g7e-bench-profile}"  # bedrock-invoke + s3-artifacts-rw

S3_BUCKET="${S3_BUCKET:-agent-aiops-checkpoints}"
S3_PREFIX="${S3_PREFIX:-cost-aware-routing}"
TAG_PROJECT="cost-aware-routing"

# Per-run identifier (set via env when launching multiple alphas)
ALPHA="${ALPHA:-1.0}"
RUN_ID="${RUN_ID:-alpha${ALPHA}-$(date -u +%Y%m%dT%H%M%S)}"

REPO_URL="${REPO_URL:-https://github.com/your-org/agent-aiops-on-aws.git}"  # TODO set actual repo
REPO_BRANCH="${REPO_BRANCH:-main}"

# ---------- subcommands ----------

cmd_launch() {
    echo "[launch] requesting $INSTANCE_TYPE spot in $AZ ($REGION)"

    # Check current spot price first; bail if > 1.5x expected $9.63/hr
    cur_price=$(aws ec2 describe-spot-price-history \
        --instance-types "$INSTANCE_TYPE" \
        --product-descriptions "Linux/UNIX" \
        --availability-zone "$AZ" \
        --start-time "$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ)" \
        --region "$REGION" \
        --max-results 1 \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "NA")
    echo "[launch] current spot price: \$$cur_price/hr in $AZ"

    user_data=$(cat <<'EOF' | base64
#!/bin/bash
# AL2023 NVIDIA AMI already has driver/CUDA. Just expand NVMe and tag.
set -e
mkfs.xfs -f /dev/nvme1n1 || true
mkdir -p /mnt/nvme
mount /dev/nvme1n1 /mnt/nvme || true
chown ec2-user:ec2-user /mnt/nvme
echo "/dev/nvme1n1 /mnt/nvme xfs defaults,nofail 0 2" >> /etc/fstab
EOF
)

    spot_request_id=$(aws ec2 request-spot-instances \
        --instance-count 1 \
        --type one-time \
        --launch-specification "{
            \"ImageId\":\"$AMI_ID\",
            \"InstanceType\":\"$INSTANCE_TYPE\",
            \"KeyName\":\"$KEY_NAME\",
            \"SubnetId\":\"$SUBNET_ID\",
            \"SecurityGroupIds\":[\"$SECURITY_GROUP\"],
            \"IamInstanceProfile\":{\"Name\":\"$IAM_INSTANCE_PROFILE\"},
            \"BlockDeviceMappings\":[{
                \"DeviceName\":\"/dev/sda1\",
                \"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}
            }],
            \"UserData\":\"$user_data\"
        }" \
        --region "$REGION" \
        --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
        --output text)
    echo "[launch] spot request: $spot_request_id"

    # Wait for fulfillment
    echo "[launch] waiting for fulfillment..."
    aws ec2 wait spot-instance-request-fulfilled \
        --spot-instance-request-ids "$spot_request_id" \
        --region "$REGION"

    instance_id=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids "$spot_request_id" \
        --region "$REGION" \
        --query 'SpotInstanceRequests[0].InstanceId' \
        --output text)
    echo "[launch] instance: $instance_id"

    aws ec2 create-tags \
        --resources "$instance_id" "$spot_request_id" \
        --tags "Key=Project,Value=$TAG_PROJECT" "Key=RunId,Value=$RUN_ID" \
        --region "$REGION"

    aws ec2 wait instance-running --instance-ids "$instance_id" --region "$REGION"

    public_ip=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    echo "[launch] running at $public_ip"
    echo
    echo "Next: export INSTANCE_IP=$public_ip && ./bootstrap.sh provision"
    echo "Note: instance_id=$instance_id  spot_request_id=$spot_request_id"
    echo "$instance_id" > .last_instance_id
    echo "$public_ip" > .last_instance_ip
    echo "$spot_request_id" > .last_spot_request_id
}

cmd_provision() {
    : "${INSTANCE_IP:=$(cat .last_instance_ip 2>/dev/null || true)}"
    if [ -z "${INSTANCE_IP:-}" ]; then
        echo "ERROR: set INSTANCE_IP env or run launch first"; exit 1
    fi
    echo "[provision] configuring $INSTANCE_IP"

    # Wait for SSH
    for i in {1..30}; do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
            -i "$KEY_FILE" "ec2-user@$INSTANCE_IP" 'echo ok' >/dev/null 2>&1 && break
        sleep 5
    done

    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ec2-user@$INSTANCE_IP" bash -s <<'EOF'
set -euo pipefail
sudo dnf install -y git tmux jq python3.12 python3.12-devel || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /mnt/nvme/cost-aware-routing
cd /mnt/nvme/cost-aware-routing
if [ ! -d agent-aiops-on-aws ]; then
    git clone --depth 1 -b "${REPO_BRANCH:-main}" "${REPO_URL}" agent-aiops-on-aws
fi
cd agent-aiops-on-aws/domains/autoresearch/blueprints/cost-aware-routing
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt 2>/dev/null || \
    uv pip install torch transformers accelerate datasets boto3 pyyaml sympy pytest \
                   trl peft tqdm zstandard
echo
nvidia-smi
echo "[provision] ready"
EOF
    echo "[provision] done. ssh -i $KEY_FILE ec2-user@$INSTANCE_IP"
}

cmd_resume() {
    : "${INSTANCE_IP:=$(cat .last_instance_ip 2>/dev/null || true)}"
    if [ -z "${INSTANCE_IP:-}" ]; then
        echo "ERROR: set INSTANCE_IP env or launch first"; exit 1
    fi

    echo "[resume] starting trainer for alpha=$ALPHA on $INSTANCE_IP"
    ssh -i "$KEY_FILE" "ec2-user@$INSTANCE_IP" bash -s <<EOF
set -euo pipefail
cd /mnt/nvme/cost-aware-routing/agent-aiops-on-aws/domains/autoresearch/blueprints/cost-aware-routing
source .venv/bin/activate

# Pull latest checkpoint from S3 if any
LATEST=\$(aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/checkpoints/alpha${ALPHA}/ 2>/dev/null \\
    | awk '/iter-/ {print \$2}' | tr -d '/' | sort -V | tail -1 || echo "")
if [ -n "\$LATEST" ]; then
    echo "[resume] found checkpoint: \$LATEST"
else
    echo "[resume] no prior checkpoint; starting fresh"
fi

# Kick off training inside tmux so the SSH session can drop
tmux new-session -d -s train-${ALPHA} \\
    "python -m scripts.train --alpha ${ALPHA} \\
        --s3-prefix s3://${S3_BUCKET}/${S3_PREFIX} \\
        2>&1 | tee /mnt/nvme/cost-aware-routing/train-${ALPHA}.log"
echo "[resume] tmux session 'train-${ALPHA}' detached. Reattach: tmux a -t train-${ALPHA}"
EOF
}

cmd_teardown() {
    : "${INSTANCE_ID:=$(cat .last_instance_id 2>/dev/null || true)}"
    : "${SPOT_REQUEST_ID:=$(cat .last_spot_request_id 2>/dev/null || true)}"
    [ -n "${SPOT_REQUEST_ID:-}" ] && aws ec2 cancel-spot-instance-requests \
        --spot-instance-request-ids "$SPOT_REQUEST_ID" --region "$REGION" || true
    [ -n "${INSTANCE_ID:-}" ] && aws ec2 terminate-instances \
        --instance-ids "$INSTANCE_ID" --region "$REGION" || true
    echo "[teardown] cancelled $SPOT_REQUEST_ID, terminated $INSTANCE_ID"
    rm -f .last_instance_id .last_instance_ip .last_spot_request_id
}

# ---------- dispatch ----------
case "${1:-}" in
    launch) cmd_launch ;;
    provision) cmd_provision ;;
    resume) cmd_resume ;;
    teardown) cmd_teardown ;;
    *) echo "usage: $0 {launch|provision|resume|teardown}" ; exit 1 ;;
esac
