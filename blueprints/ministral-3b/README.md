# Ministral-3B Blueprint

Deploy vLLM serving Mistral's Ministral-3-3B-Instruct model on EKS with SageMaker Studio for development.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          VPC                                 │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │  SageMaker      │  │           EKS Cluster           │   │
│  │  Studio         │──│  ┌─────────┐   ┌─────────────┐  │   │
│  │  (Code Editor)  │  │  │ System  │   │ GPU Nodes   │  │   │
│  └─────────────────┘  │  │ Nodes   │   │ (g6e/g6)    │  │   │
│                       │  └─────────┘   │             │  │   │
│                       │                │ ┌─────────┐ │  │   │
│                       │                │ │ vLLM    │ │  │   │
│                       │                │ │Ministral│ │  │   │
│                       │                │ └─────────┘ │  │   │
│                       │                └─────────────┘  │   │
│                       └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- HuggingFace account (for model access)

## Quick Start

```bash
# Initialize
terraform init

# Plan
terraform plan

# Deploy
terraform apply
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | us-east-1 | AWS region |
| `gpu_instance_types` | g6e.xlarge, g6.xlarge, ... | GPU instances (priority order) |
| `vllm_model_id` | mistralai/Ministral-3-3B-Instruct-2512 | Model to serve |
| `enable_sagemaker` | true | Deploy SageMaker Studio |

## Usage

After deployment:

```bash
# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name aiops-eks-cluster

# Test vLLM
curl http://<load-balancer>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Ministral-3-3B-Instruct-2512", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Known Limitations

1. **Model format**: Ministral requires `--load_format mistral`, incompatible with S3 streaming
2. **GPU capacity**: g6e instances have limited availability; uses multi-instance fallback
3. **vLLM version**: Requires latest image for Mistral3 architecture support

## Cost Estimate

| Resource | Type | ~Monthly Cost |
|----------|------|---------------|
| EKS Cluster | Control plane | $73 |
| System Nodes | 2x m6i.large | $140 |
| GPU Nodes | 1x g6e.xlarge | $380 |
| NAT Gateway | Single | $45 |
| **Total** | | **~$640/mo** |

*Costs vary by usage. GPU nodes can scale to 0 when idle.*
