#!/usr/bin/env bash
# launch-helper-ec2.sh — Launch the weight-staging helper EC2 in us-east-1.
# Uses IAM instance profile "kimi-bench-helper" with S3 write access to the model bucket.
# On first run, creates the role + instance profile automatically.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
SUBNET="${HELPER_SUBNET:-subnet-0ee4f9c19d21d5532}"    # us-east-1b, has IGW
SG="${HELPER_SG:-}"
KEY="${KEY_NAME:-g7e-bench}"
AMI="${HELPER_AMI:-ami-0d5027c9d489eb3ba}"             # AL2023 minimal x86_64
INSTANCE_TYPE="c6i.8xlarge"                            # 32 vCPU, fast net for HF→S3
ROLE_NAME="kimi-bench-helper"
PROFILE_NAME="$ROLE_NAME"
BUCKET="kimi-k2-bench-models-20260216163240701700000006"
VPC="vpc-07ff0e6bdc3cac475"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

# ---------- IAM (idempotent) ----------
ensure_role() {
  if ! aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" >/dev/null 2>&1; then
    log "Creating IAM role $ROLE_NAME"
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null

    aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name S3ModelAccess --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [
        {\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:ListBucket\",\"s3:DeleteObject\"],
         \"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}
      ]
    }"
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
      --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null
  fi

  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null
    log "Waiting for instance profile to propagate"
    sleep 10
  fi
}

# ---------- SG (idempotent) ----------
ensure_sg() {
  if [[ -z "$SG" ]]; then
    SG=$(aws ec2 describe-security-groups --region "$REGION" \
      --filters "Name=group-name,Values=kimi-bench-helper" "Name=vpc-id,Values=$VPC" \
      --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
    if [[ "$SG" == "None" || -z "$SG" ]]; then
      log "Creating SG kimi-bench-helper"
      SG=$(aws ec2 create-security-group --region "$REGION" \
        --group-name kimi-bench-helper --description "Kimi bench helper SSH" \
        --vpc-id "$VPC" --query 'GroupId' --output text)
      aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
    fi
  fi
  log "Using SG $SG"
}

# ---------- Launch ----------
launch() {
  local udata
  udata=$(cat <<'EOF'
#!/bin/bash
set -eux
dnf install -y git tar gzip xz zstd
sudo -u ec2-user bash -lc 'aws s3 cp s3://kimi-k2-bench-models-20260216163240701700000006/scripts/stage-weights.sh ~/stage-weights.sh 2>/dev/null || true'
EOF
  )

  local iid
  iid=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":1500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
    --associate-public-ip-address \
    --user-data "$udata" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kimi-bench-helper},{Key=Project,Value=kimi-k2.6-speculative}]' \
    --query 'Instances[0].InstanceId' --output text)

  log "Launched $iid"
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"
  local ip
  ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  log "Helper ready at $ip"
  log "SSH: ssh -i ~/.ssh/${KEY}.pem ec2-user@${ip}"
  printf '%s\n' "$iid" "$ip"
}

main() {
  ensure_role
  ensure_sg
  launch
}

main "$@"
