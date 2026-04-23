#!/bin/bash
# Setup EKS managed nodegroup for Qwen3-32B benchmark on finetune-eks
# Uses g6e.2xlarge (1x L40S, 8 vCPU, 64 GiB) in us-east-1
set -euo pipefail

CLUSTER_NAME="finetune-eks"
NODEGROUP_NAME="g6e-benchmark"
REGION="us-east-1"
# Public subnets with internet access
SUBNETS="subnet-09c51a62517440cab"  # us-east-1a, public, 249 IPs
INSTANCE_TYPE="g6e.2xlarge"
DISK_SIZE=200  # GB — enough for container image (~15GB) + model (~34GB) + headroom
NODE_ROLE_ARN=""  # Set below after creation/lookup

echo "=== Qwen3-32B EKS Benchmark Setup ==="
echo "Cluster: $CLUSTER_NAME"
echo "Instance: $INSTANCE_TYPE (1x L40S 48GB, 8 vCPU, 64 GiB RAM)"
echo ""

# Step 1: Create or reuse node IAM role
ROLE_NAME="eks-g6e-benchmark-node-role"
echo "--- Step 1: IAM Role ---"
if aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" &>/dev/null; then
    echo "Role $ROLE_NAME already exists"
    NODE_ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text --region "$REGION")
else
    echo "Creating role $ROLE_NAME..."
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' \
        --region "$REGION"

    # Attach required managed policies
    for policy in \
        arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy \
        arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy \
        arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly \
        arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess; do
        aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$policy" --region "$REGION"
        echo "  Attached: $policy"
    done

    NODE_ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text --region "$REGION")
    echo "  Role ARN: $NODE_ROLE_ARN"
    echo "  Waiting 10s for IAM propagation..."
    sleep 10
fi

# Step 2: Create managed nodegroup
echo ""
echo "--- Step 2: Create Nodegroup ---"
if aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --nodegroup-name "$NODEGROUP_NAME" --region "$REGION" &>/dev/null; then
    echo "Nodegroup $NODEGROUP_NAME already exists"
    STATUS=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --nodegroup-name "$NODEGROUP_NAME" --region "$REGION" --query 'nodegroup.status' --output text)
    echo "  Status: $STATUS"
else
    echo "Creating nodegroup $NODEGROUP_NAME..."
    aws eks create-nodegroup \
        --cluster-name "$CLUSTER_NAME" \
        --nodegroup-name "$NODEGROUP_NAME" \
        --node-role "$NODE_ROLE_ARN" \
        --subnets "$SUBNETS" \
        --instance-types "$INSTANCE_TYPE" \
        --disk-size "$DISK_SIZE" \
        --scaling-config minSize=0,maxSize=1,desiredSize=1 \
        --ami-type AL2023_x86_64_NVIDIA \
        --region "$REGION"
    echo "  Nodegroup creation initiated. This takes ~5-10 minutes."
    echo "  Monitor with: aws eks describe-nodegroup --cluster-name $CLUSTER_NAME --nodegroup-name $NODEGROUP_NAME --region $REGION --query 'nodegroup.status'"
fi

# Step 3: Wait for nodegroup to be active
echo ""
echo "--- Step 3: Waiting for nodegroup... ---"
aws eks wait nodegroup-active \
    --cluster-name "$CLUSTER_NAME" \
    --nodegroup-name "$NODEGROUP_NAME" \
    --region "$REGION" 2>/dev/null || true

STATUS=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --nodegroup-name "$NODEGROUP_NAME" --region "$REGION" --query 'nodegroup.status' --output text)
echo "Nodegroup status: $STATUS"

if [ "$STATUS" = "ACTIVE" ]; then
    echo ""
    echo "=== Setup complete ==="
    echo ""
    echo "Next steps:"
    echo "  1. Update kubeconfig: aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION"
    echo "  2. Verify GPU node: kubectl get nodes -l node.kubernetes.io/instance-type=$INSTANCE_TYPE"
    echo "  3. Install NVIDIA device plugin (if not present):"
    echo "     kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml"
    echo "  4. Deploy config0: kubectl apply -f configs/config0-nocache.yaml"
    echo "  5. Wait for model download + vLLM startup (~10-15 min)"
    echo "  6. Run benchmark: kubectl exec bench-runner -- python /scripts/benchmark.py --config config0-nocache"
fi
