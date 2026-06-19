#!/bin/bash
# Launch a p4de.24xlarge spot instance in us-east-1 az6 for self-coding-agent-loop.
#
# Prerequisites before running this script:
#   1. AWS CLI configured with credentials that can run ec2:RunInstances.
#   2. A keypair exists (name set below; default: g7e-bench reusing existing ~/.ssh/g7e-bench.pem).
#   3. A security group allowing SSH from your IP (ID set below).
#   4. A subnet in us-east-1 az6 (subnet ID set below).
#   5. An IAM instance profile with S3 read/write access to agent-aiops-artifacts.
#
# Missing pieces (fill these in before running):
#   AWS_SUBNET_ID         — subnet in us-east-1 az6
#   AWS_SECURITY_GROUP_ID — SG that allows port 22 from your laptop's public IP
#   AWS_INSTANCE_PROFILE  — IAM instance profile ARN with S3 access
#   AWS_KEYPAIR_NAME      — EC2 keypair name (default: g7e-bench)
#
# What this script does:
#   1. Refreshes the repo snapshot at s3://agent-aiops-artifacts/self-coding-agent-loop/repo-snapshot.tar.gz
#   2. Runs `aws ec2 run-instances` with spot market options and user-data.
#   3. Prints the instance ID and waits for it to reach `running`.
#   4. Prints an SSH command for connecting.

set -euo pipefail

REGION="us-east-1"
# us-east-1 az6 = zone ID use1-az6 = zone NAME us-east-1c (verified 2026-05-10)
# p4de.24xlarge offered in us-east-1c ($13/hr median) AND us-east-1b ($21/hr median)
AZ="us-east-1c"
INSTANCE_TYPE="p4de.24xlarge"
# Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) — ami-091f07e77f51e6b42 (2026-05-05)
# This is Ubuntu, not AL2023 — user_data script is Ubuntu-flavored (apt vs dnf).
# Re-verify latest: aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning Base OSS Nvidia*Ubuntu*" --region us-east-1 --query 'sort_by(Images,&CreationDate)[-1].ImageId'
AMI_ID="${AMI_ID:-ami-091f07e77f51e6b42}"
KEY_NAME="${AWS_KEYPAIR_NAME:-g7e-bench}"

# === user must set these (will error loudly if unset) ===
: "${AWS_SUBNET_ID:?Set AWS_SUBNET_ID to a subnet in $AZ before running}"
: "${AWS_SECURITY_GROUP_ID:?Set AWS_SECURITY_GROUP_ID to an SG allowing SSH}"
: "${AWS_INSTANCE_PROFILE:?Set AWS_INSTANCE_PROFILE to an IAM instance profile ARN with S3 access}"

EXPERIMENT_BUCKET="s3://agent-aiops-artifacts/self-coding-agent-loop"
REPO_ROOT="/Users/phi/Documents/workbench/agent-aiops-on-aws"
BLUEPRINT_DIR="$REPO_ROOT/domains/autoresearch/blueprints/self-coding-agent-loop"
SNAPSHOT="/tmp/aiops-repo-snapshot-$(date +%s).tar.gz"

echo "[1/4] snapshotting repo → $SNAPSHOT"
tar -czf "$SNAPSHOT" \
    --exclude='.terraform' \
    --exclude='node_modules' \
    --exclude='.DS_Store' \
    --exclude='*.tfstate*' \
    --exclude="$BLUEPRINT_DIR/data/nebius" \
    -C "$(dirname "$REPO_ROOT")" "$(basename "$REPO_ROOT")"
echo "    size: $(ls -lh "$SNAPSHOT" | awk '{print $5}')"

echo "[2/4] uploading snapshot to S3"
aws s3 cp "$SNAPSHOT" "$EXPERIMENT_BUCKET/repo-snapshot.tar.gz" --region us-west-2
rm -f "$SNAPSHOT"

echo "[3/4] launching spot instance"
USER_DATA_B64=$(base64 -i "$BLUEPRINT_DIR/scripts/user_data.sh")

INSTANCE_JSON=$(aws ec2 run-instances \
    --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --image-id "$AMI_ID" \
    --key-name "$KEY_NAME" \
    --subnet-id "$AWS_SUBNET_ID" \
    --security-group-ids "$AWS_SECURITY_GROUP_ID" \
    --iam-instance-profile "Arn=$AWS_INSTANCE_PROFILE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "$USER_DATA_B64" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=self-coding-agent-loop},{Key=Project,Value=autoresearch},{Key=SpotSafe,Value=true}]' \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2' \
    --count 1)

INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.Instances[0].InstanceId')
echo "    instance id: $INSTANCE_ID"

echo "[4/4] waiting for instance to reach 'running'..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

========================================================================
Spot p4de launched.
  Instance ID: $INSTANCE_ID
  Public IP:   $PUBLIC_IP
  Region/AZ:   $REGION / $AZ

Connect:
  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP

Bootstrap progress:
  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP 'tail -f /var/log/user-data.log'

Wait for bootstrap to finish (~5-10 min):
  until ssh -o ConnectTimeout=5 -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP test -f /mnt/nvme/self-coding-agent-loop/.bootstrap-complete 2>/dev/null; do
      sleep 15; echo waiting...; done

Then run:
  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP \\
    'cd /mnt/nvme/self-coding-agent-loop/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop/scripts && bash run_week1.sh'

Monitor S3 artifacts (durable backup):
  watch -n 60 'aws s3 ls --recursive $EXPERIMENT_BUCKET/runs/ | tail -10'

Terminate when done:
  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
========================================================================
EOF
