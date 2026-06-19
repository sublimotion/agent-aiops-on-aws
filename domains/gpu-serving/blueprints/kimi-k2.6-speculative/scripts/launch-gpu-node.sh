#!/usr/bin/env bash
# launch-gpu-node.sh — Request a p6-b300.48xlarge spot instance in us-east-1c (use1-az6).

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
SUBNET="${GPU_SUBNET:-subnet-079555bada92f1c9b}"  # public az6 subnet (IGW-routed)
SG="${GPU_SG:-sg-02c4dbf438ee20741}"
AMI="${GPU_AMI:-ami-027c3ae8019fc0d3a}"
KEY="${KEY_NAME:-g7e-bench}"
INSTANCE_TYPE="p6-b300.48xlarge"
ROLE_NAME="kimi-k26-gpu"
PROFILE_NAME="$ROLE_NAME"
BUCKET="kimi-k2-bench-models-20260216163240701700000006"
RESULTS_BUCKET="kimi-k2-bench-results-20260216163240701700000007"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
UDATA="$SCRIPT_DIR/bootstrap-gpu-node.sh"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

ensure_role() {
  if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    log "IAM role $ROLE_NAME exists"
  else
    log "Creating IAM role $ROLE_NAME"
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
    aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name S3ModelAndResults --policy-document "{
      \"Version\":\"2012-10-17\",
      \"Statement\":[
        {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],
         \"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]},
        {\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:ListBucket\",\"s3:AbortMultipartUpload\"],
         \"Resource\":[\"arn:aws:s3:::$RESULTS_BUCKET\",\"arn:aws:s3:::$RESULTS_BUCKET/*\"]}
      ]
    }"
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly >/dev/null
  fi

  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null
    sleep 10
  fi
}

launch() {
  ensure_role
  [[ -f "$UDATA" ]] || { log "$UDATA missing"; exit 1; }

  local iid
  iid=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-market-options '{"MarketType":"spot"}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
    --associate-public-ip-address \
    --metadata-options 'HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled' \
    --user-data "file://$UDATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kimi-k26-spec-gpu},{Key=Project,Value=kimi-k2.6-speculative},{Key=Spec,Value=domains/gpu-serving/specs/kimi-k2.6-speculative.md}]' \
    --query 'Instances[0].InstanceId' --output text)

  log "Requested spot instance $iid — waiting for running state"
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"

  local ip
  ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  local piid
  piid=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

  log "Running: $iid  public=$ip  private=$piid"
  log "SSH:     ssh -i ~/.ssh/${KEY}.pem ec2-user@${ip}"
  log "Bootstrap logs: /var/log/kimi-bootstrap.log"
  printf 'INSTANCE_ID=%s\nPUBLIC_IP=%s\nPRIVATE_IP=%s\n' "$iid" "$ip" "$piid" > "$SCRIPT_DIR/../results/.gpu-node.env"
  log "Saved endpoint info to results/.gpu-node.env"
}

launch "$@"
