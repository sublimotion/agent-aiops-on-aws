#!/usr/bin/env bash
# launch-gpu-node.sh — Request p6-b300.48xlarge spot in us-west-2b (usw2-az2)
# and join qn-sglang-eks-cluster via nodeadm (AL2023 EKS bootstrap).

set -euo pipefail

REGION="us-west-2"
SUBNET="subnet-001db6882dbb5ac72"       # EKS private subnet in usw2-az2
SG="sg-070da338e3796648d"                # eks-cluster-sg (auto-generated, has egress; existing nodes use this)
AMI="ami-0d868cc255a3e103a"              # EKS-optimized AL2023 + NVIDIA 1.32 (ships nodeadm)
KEY="g7e-bench"
INSTANCE_TYPE="p6-b300.48xlarge"
ROLE_NAME="qwen3-spec-gpu"
CLUSTER="qn-sglang-eks-cluster"

CLUSTER_ENDPOINT="https://F303CCD2F1FB751B0ED541D5C3372181.gr7.us-west-2.eks.amazonaws.com"
CLUSTER_CA="LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURCVENDQWUyZ0F3SUJBZ0lJUDcza0FSc2IvVFl3RFFZSktvWklodmNOQVFFTEJRQXdGVEVUTUJFR0ExVUUKQXhNS2EzVmlaWEp1WlhSbGN6QWVGdzB5TmpBek1ETXhOakU0TXpkYUZ3MHpOakF5TWpreE5qSXpNemRhTUJVeApFekFSQmdOVkJBTVRDbXQxWW1WeWJtVjBaWE13Z2dFaU1BMEdDU3FHU0liM0RRRUJBUVVBQTRJQkR3QXdnZ0VLCkFvSUJBUURIcUlNMnRwVVh3MmJsM3JiZlFBaGN3U3FHY0xnUG5DQzlTL3RMSTZjM3QvL0htZTRiZXhSbkFLRWcKYk5RdVFxRWwwS0s4ZFc0NFFpc1dNZW5SN0xsMElZb2xNWUg3Y3VrYUFoeWd1VGVFWjNlczFwTlhJNm5PYlEvUQpWN092VFBWbXlaSlVndjhTVzBYdGo3a05HZFhiVVBaZTRvakVreWVEalI4VklWUmpaN25EL205Ylhxb1h4UFROCndCcmJPRVdJRzRjdVRBaXgrRm5XZ1paNEl0aHdDOXpaU2dNbWhjOFlPQXRYZDQ4YXpua1UzaUFncE1hRDdVQ24KOXFvakp3d2ZtWENXK2hJcVFnOW5Lc1pCeW00cWdnaE9UZVhZYTJHeHNUOWJXNUozMGErZWlLaW9LN1pDYmlBVgpTWUN6OHJIN0tvcHNSTEE5Q2krOHdpZ1RGUjl6QWdNQkFBR2pXVEJYTUE0R0ExVWREd0VCL3dRRUF3SUNwREFQCkJnTlZIUk1CQWY4RUJUQURBUUgvTUIwR0ExVWREZ1FXQkJUUFptdXJUZ2tlZmlrRlh5NXNEQ2M1eHcwNS9EQVYKQmdOVkhSRUVEakFNZ2dwcmRXSmxjbTVsZEdWek1BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQTh5SkVRenlucwpkVnNONHBneUt3OEYwd3ZjUTBuK0xCMWxoTVdQOXZZWUplcXhFUmlMeTZveHF2VWUzbGl1VFNYUTVJaGhNMnJ6CmFrVkFMZjJWb1BBSzFJc1hpa1hTV0FERG9zNDVuSnJlNExqSjhJUVJqYU9XNWdza1g3WCtHVTcrOEdwSDFKMHoKUnJhN01WSDluZGtEMERjL3k4MEhtTS9HRXZaYnd6RndZa0hVazc2RDcrVzB4TVRKUm5QeDZpbDd4dnR1WVZkNApyaU83OTZ6OXZQTnUzQ3AxVGx1eWp0VE5JVzdHVGdta1dtU0pxRnJYZ3RhRTA3KzJoLytQSkVqSGN5MnhMQnp6CjNPemR0bVM0LzRPRytWdU5GbUZFTDBtSzBsRjRmT0R0TlZXbHRxbnBiaCtlM080aWdyVUFWUldlYVErK25NMzcKSnpNSmQrZEFiWXhaCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BOOTSTRAP="$SCRIPT_DIR/bootstrap-gpu-node.sh"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

TMP=$(mktemp -d)
cat > "$TMP/nodeconfig.yaml" <<EOF
apiVersion: node.eks.aws/v1alpha1
kind: NodeConfig
spec:
  cluster:
    name: $CLUSTER
    apiServerEndpoint: $CLUSTER_ENDPOINT
    certificateAuthority: $CLUSTER_CA
    cidr: 10.100.0.0/16
  kubelet:
    config:
      maxPods: 110
    flags:
      - "--node-labels=node.kubernetes.io/instance-type=p6-b300.48xlarge,blueprint=qwen3-235b-speculative"
      - "--register-with-taints=nvidia.com/gpu=Exists:NoSchedule"
EOF

MIME_BOUNDARY="==BOUNDARY=="
cat > "$TMP/userdata.txt" <<EOF
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="$MIME_BOUNDARY"

--$MIME_BOUNDARY
Content-Type: application/node.eks.aws

$(cat "$TMP/nodeconfig.yaml")

--$MIME_BOUNDARY
Content-Type: text/x-shellscript

$(cat "$BOOTSTRAP")

--$MIME_BOUNDARY--
EOF

log "Launching $INSTANCE_TYPE spot in $REGION / subnet=$SUBNET / sg=$SG"
iid=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY" \
  --subnet-id "$SUBNET" \
  --security-group-ids "$SG" \
  --iam-instance-profile "Name=$ROLE_NAME" \
  --instance-market-options '{"MarketType":"spot"}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
  --metadata-options 'HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled' \
  --user-data "file://$TMP/userdata.txt" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=qwen3-spec-gpu},{Key=Project,Value=qwen3-235b-speculative},{Key=Spec,Value=domains/gpu-serving/specs/qwen3-235b-speculative.md},{Key=kubernetes.io/cluster/'"$CLUSTER"',Value=owned}]' \
  --query 'Instances[0].InstanceId' --output text)

log "Requested spot $iid — waiting for running"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"
ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
log "Running: $iid  private=$ip"
log "Next: wait 3-5 min for nodeadm to join cluster. Check with: kubectl --context qn-sglang get nodes | grep p6-b300"
printf 'INSTANCE_ID=%s\nPRIVATE_IP=%s\n' "$iid" "$ip" > "$SCRIPT_DIR/../results/.gpu-node.env"

rm -rf "$TMP"
echo "$iid"
