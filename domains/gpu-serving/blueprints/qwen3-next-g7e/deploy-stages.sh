#!/bin/bash
# Deploy qwen3-next-g7e in stages to handle provider dependencies

set -e

echo "Stage 1: Apply EKS addons and node groups..."
terraform apply \
  -target='module.eks.module.eks.aws_eks_addon["vpc-cni"]' \
  -target='module.eks.module.eks.aws_eks_addon["kube-proxy"]' \
  -target='module.eks.module.eks.aws_eks_addon["coredns"]' \
  -target='module.eks.module.eks.aws_eks_addon["aws-ebs-csi-driver"]' \
  -target='module.eks.module.eks.module.eks_managed_node_group["gpu"]' \
  -target='module.eks.module.eks.module.eks_managed_node_group["system"]' \
  -target='module.fsx_csi_irsa' \
  -target='module.vllm_irsa' \
  -target='aws_s3_bucket.results' \
  -auto-approve

echo "Waiting for node groups to be ready..."
sleep 30

echo "Stage 2: Update kubeconfig..."
aws eks update-kubeconfig --name qwen3-g7e-eks-cluster --region us-east-2

echo "Stage 3: Apply remaining resources..."
terraform apply -auto-approve

echo "Deployment complete!"