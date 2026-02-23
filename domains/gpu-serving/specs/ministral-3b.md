# ML Inference Platform Requirements

## Overview
Deploy a prototyping environment for ML inference on AWS using EKS with GPU nodes and SageMaker Studio for development.

## Components

### 1. SageMaker Studio
- **IDE**: Code Editor (VSCode-based)
- **Permissions**:
  - EKS cluster admin access (for kubectl)
  - EBS volume management (required for Code Editor workspaces)
  - S3 access for model storage
- **Network**: Deploy in private subnets with connectivity to EKS

### 2. EKS Cluster
- **Version**: 1.29+
- **Node Groups**:
  - **System nodes**: 2x m6i.large for cluster workloads
  - **GPU nodes**: g6/g6e instances for ML inference
- **GPU Instance Types** (in priority order for capacity fallback):
  - g6e.xlarge (preferred - 24GB VRAM)
  - g6.xlarge
  - g6e.2xlarge
  - g6.2xlarge
- **GPU Availability Zones**: Exclude AZs with limited GPU capacity (e.g., us-east-1b often has limited g6e)
- **Addons**: CoreDNS, kube-proxy, VPC-CNI, EBS-CSI driver, NVIDIA device plugin

### 3. vLLM Inference Server
- **Model**: `mistralai/Ministral-3-3B-Instruct-2512`
  - Note: Use `-2512` version (not `-2410`) for proper vLLM support
  - Note: Use non-GGUF version; vLLM serves safetensors natively
- **Container Image**: `vllm/vllm-openai:latest` (required for newer model architectures)
- **Required Arguments** (for Ministral/Mistral3 models):
  ```
  --tokenizer_mode mistral
  --config_format mistral
  --load_format mistral
  ```
- **Resource Settings**:
  - GPU memory utilization: 0.9
  - Max model length: 4096 (adjust based on use case)
- **Exposure**: LoadBalancer service for external access

### 4. Networking
- **VPC**: /16 CIDR with public/private subnets across 3 AZs
- **NAT Gateway**: Single (non-prod) for private subnet internet access
- **VPC Endpoints**: S3, ECR, STS, CloudWatch Logs (cost optimization)
- **Security Groups**:
  - SageMaker → EKS API server (port 443)
  - EKS nodes self-communication
  - VPC endpoints HTTPS access

### 5. Storage
- **S3 Bucket**: For model storage (KMS encrypted)
  - Note: S3 streaming via Run:ai Model Streamer is incompatible with `--load_format mistral`
  - Models load from HuggingFace Hub on pod startup instead
- **EBS**: gp3 storage class for persistent volumes

## Non-Requirements (Prototyping Scope)
- No multi-replica deployment
- No autoscaling beyond basic HPA
- No production-grade monitoring/alerting
- No multi-region redundancy

## Security Requirements
- All storage encrypted (KMS)
- Private subnets for compute
- IAM roles with least privilege (IRSA for EKS workloads)
- No public SSH access to nodes
- VPC Flow Logs enabled

## Known Limitations
1. **Ministral model format**: Requires `--load_format mistral`, incompatible with S3 streaming
2. **GPU capacity**: g6e instances have limited availability in some AZs; use multi-instance-type fallback
3. **vLLM version**: Newer models require latest vLLM image; pinned versions may not support new architectures
