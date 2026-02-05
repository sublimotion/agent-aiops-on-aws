# AI/ML Infrastructure on AWS

This Terraform configuration deploys a complete AI/ML infrastructure stack on AWS, including:

- **VPC** with public/private subnets, NAT gateway, and VPC endpoints
- **EKS Cluster** with system and GPU (g6e.xlarge) node groups
- **SageMaker Studio** with Code Editor (VSCode) and EKS connectivity
- **S3 Model Bucket** with KMS encryption for model storage
- **vLLM** deployment with Run:ai Model Streamer for efficient S3 model loading

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   VPC                                        │
│  ┌─────────────────────┐      ┌─────────────────────────────────────────┐   │
│  │   Public Subnets    │      │           Private Subnets                │   │
│  │  ┌───────────────┐  │      │  ┌─────────────────────────────────┐    │   │
│  │  │   NAT GW      │  │      │  │         EKS Cluster             │    │   │
│  │  └───────────────┘  │      │  │  ┌─────────┐    ┌────────────┐  │    │   │
│  └─────────────────────┘      │  │  │ System  │    │    GPU     │  │    │   │
│                               │  │  │ Nodes   │    │   Nodes    │  │    │   │
│                               │  │  └─────────┘    └────────────┘  │    │   │
│                               │  │                       ▲         │    │   │
│  ┌─────────────────────┐      │  │        ┌──────────────┴───────┐ │    │   │
│  │    S3 Model Bucket  │◄─────┼──┼────────│   vLLM + Run:ai      │ │    │   │
│  │   (KMS Encrypted)   │      │  │        │   Model Streamer     │ │    │   │
│  │                     │      │  │        └──────────────────────┘ │    │   │
│  │  models/            │      │  └─────────────────────────────────┘    │   │
│  │   └─ministral-3b/   │      │                                         │   │
│  └─────────────────────┘      │  ┌─────────────────────────────────┐    │   │
│          ▲                    │  │       SageMaker Studio          │    │   │
│          │ Upload model       │  │      (VSCode/Code Editor)       │    │   │
│          │                    │  │              │                  │    │   │
│          └────────────────────┼──┼──────────────┘                  │    │   │
│                               │  │         kubectl → EKS           │    │   │
│                               │  └─────────────────────────────────┘    │   │
│                               └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- kubectl
- helm

## Quick Start

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name aiops-eks-cluster
```

## Configuration

Key variables in `variables.tf`:

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | us-east-1 | AWS region |
| `project_name` | aiops | Project name prefix |
| `eks_gpu_instance_type` | g6e.xlarge | GPU instance type |
| `vllm_model_id` | mistralai/Ministral-3B-Instruct-2410 | Model ID (HuggingFace) |
| `vllm_use_s3_model` | true | Load model from S3 via Run:ai Streamer |
| `vllm_s3_model_path` | models/ministral-3b | S3 path within bucket |
| `enable_gpu_nodes` | true | Enable GPU node group |
| `enable_sagemaker_studio` | true | Enable SageMaker Studio |
| `enable_vllm_deployment` | true | Enable vLLM deployment |

## Uploading Models to S3

Models are loaded from S3 using Run:ai Model Streamer, which streams weights directly to GPU memory for faster startup.

### From SageMaker Studio (Recommended)

```bash
# 1. Download model from HuggingFace
pip install huggingface_hub
huggingface-cli download mistralai/Ministral-3B-Instruct-2410 --local-dir ./model

# 2. Upload to S3 (bucket name from terraform output)
aws s3 cp ./model s3://<model-bucket-name>/models/ministral-3b/ --recursive

# 3. Verify upload
aws s3 ls s3://<model-bucket-name>/models/ministral-3b/
```

### From Local Machine

```bash
# Get bucket name from terraform
BUCKET=$(terraform output -raw model_bucket_name)

# Download and upload
huggingface-cli download mistralai/Ministral-3B-Instruct-2410 --local-dir ./model
aws s3 cp ./model s3://$BUCKET/models/ministral-3b/ --recursive
```

### Restart vLLM After Upload

```bash
kubectl rollout restart deployment/vllm-ministral -n ml-inference
kubectl logs -f deployment/vllm-ministral -n ml-inference
```

## Connecting from SageMaker Studio

1. Open SageMaker Studio and launch Code Editor (VSCode)
2. Open a terminal
3. Configure kubectl:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name aiops-eks-cluster
   ```
4. Verify connection:
   ```bash
   kubectl get nodes
   kubectl get pods -n ml-inference
   ```

## Testing vLLM

From within the cluster (e.g., SageMaker Studio):

```bash
# Check vLLM health
curl http://vllm-ministral.ml-inference.svc.cluster.local:8000/health

# List models
curl http://vllm-ministral.ml-inference.svc.cluster.local:8000/v1/models

# Chat completion
curl http://vllm-ministral.ml-inference.svc.cluster.local:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Ministral-3B-Instruct-2410",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Security

- All storage encrypted with KMS
- VPC endpoints for AWS services
- Private subnets for workloads
- Security groups with least-privilege
- IRSA for Kubernetes service accounts

## Costs

Estimated monthly costs (varies by usage):
- EKS cluster: ~$73/month
- g6e.xlarge (1 instance): ~$650/month
- NAT Gateway: ~$45/month
- SageMaker Studio: Pay per use

## Cleanup

```bash
terraform destroy
```
