# Getting Started

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- kubectl (for EKS interaction)

## Deploying a Blueprint

1. **Choose a blueprint** from the `blueprints/` directory
2. **Review the spec** in `specs/` for requirements
3. **Deploy**:

```bash
cd blueprints/ministral-3b
terraform init
terraform plan
terraform apply
```

## Creating a New Blueprint

1. Copy an existing blueprint or start from scratch
2. Create a spec in `specs/` using `_template.md`
3. Use modules from `modules/` for reusable components
4. Document in the blueprint's `README.md`

## Module Reference

| Module | Description |
|--------|-------------|
| `networking` | VPC, subnets, NAT, VPC endpoints |
| `eks-cluster` | EKS with system/GPU nodes, addons |
| `sagemaker-studio` | SageMaker domain, user, IAM |
| `vllm` | vLLM deployment on Kubernetes |

## Troubleshooting

### GPU Capacity Issues
- Use multiple instance types in `gpu_instance_types`
- Exclude problematic AZs via `gpu_availability_zones`
- Check AWS Service Quotas for limits

### vLLM Model Compatibility
- Check model architecture support in vLLM docs
- Use `latest` image for newer models
- Some models require special args (e.g., `--load_format mistral`)

### SageMaker Code Editor
- Ensure EBS permissions are attached to execution role
- Security group must allow VPC traffic
