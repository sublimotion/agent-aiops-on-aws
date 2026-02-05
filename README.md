# Agent AIOps on AWS

ML inference platform on AWS using EKS, SageMaker, and vLLM - built with Claude Code.

## Quick Start

```bash
# Deploy Ministral-3B inference
cd blueprints/ministral-3b
terraform init
terraform plan
terraform apply
```

## Repository Structure

```
├── modules/              # Reusable Terraform modules
│   ├── networking/       # VPC, subnets, endpoints
│   ├── eks-cluster/      # EKS with GPU support
│   ├── sagemaker-studio/ # SageMaker domain + IAM
│   └── vllm/             # vLLM on Kubernetes
│
├── blueprints/           # Deployable examples
│   └── ministral-3b/     # Ministral-3B + SageMaker
│
├── specs/                # Requirements documents
│   ├── ministral-3b.md   # Current spec
│   └── _template.md      # Template for new specs
│
└── docs/                 # Documentation
    └── getting-started.md
```

## Available Blueprints

| Blueprint | Model | GPU | Description |
|-----------|-------|-----|-------------|
| [ministral-3b](blueprints/ministral-3b/) | Ministral-3-3B-Instruct | g6e.xlarge | vLLM + SageMaker Studio |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          VPC                                 │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │  SageMaker      │  │           EKS Cluster           │   │
│  │  Studio         │──│  ┌─────────┐   ┌─────────────┐  │   │
│  │  (Code Editor)  │  │  │ System  │   │ GPU Nodes   │  │   │
│  └─────────────────┘  │  │ Nodes   │   │ (vLLM)      │  │   │
│                       │  └─────────┘   └─────────────┘  │   │
│                       └─────────────────────────────────┘   │
│                                    │                        │
│                              LoadBalancer                   │
└────────────────────────────────────┼────────────────────────┘
                                     │
                              vLLM API Endpoint
```

## Modules

### networking
VPC with public/private subnets, NAT gateway, VPC endpoints.

```hcl
module "networking" {
  source             = "./modules/networking"
  project_name       = "myproject"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}
```

### eks-cluster
EKS with system and GPU node groups, NVIDIA plugin, EBS CSI driver.

```hcl
module "eks" {
  source             = "./modules/eks-cluster"
  project_name       = "myproject"
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnets
  enable_gpu_nodes   = true
  gpu_instance_types = ["g6e.xlarge", "g6.xlarge"]
}
```

### sagemaker-studio
SageMaker domain with Code Editor, EKS access, proper IAM.

```hcl
module "sagemaker" {
  source           = "./modules/sagemaker-studio"
  project_name     = "myproject"
  vpc_id           = module.networking.vpc_id
  subnet_ids       = module.networking.private_subnets
  eks_cluster_name = module.eks.cluster_name
}
```

### vllm
vLLM deployment on Kubernetes with GPU support.

```hcl
module "vllm" {
  source          = "./modules/vllm"
  project_name    = "myproject"
  model_id        = "mistralai/Ministral-3-3B-Instruct-2512"
  extra_args      = ["--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral"]
}
```

## Creating a New Blueprint

1. Create spec in `specs/<name>.md` using `_template.md`
2. Create `blueprints/<name>/` directory
3. Compose modules in `main.tf`
4. Test deployment
5. Document lessons learned

See [docs/getting-started.md](docs/getting-started.md) for details.

## Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- Sufficient GPU quota (g6e/g6 instances)

## Known Limitations

1. **Ministral models** require `--load_format mistral`, incompatible with S3 streaming
2. **GPU capacity** varies by AZ; use multi-instance fallback
3. **vLLM version** - newer models require `latest` image

## Development

This repo uses:
- **Claude Code** with steering files in `.claude/steering/`
- **RALPH loops** for iterative development
- **AWS Labs MCP servers** for Terraform best practices

## License

MIT
