# Common Terraform Patterns for AWS

Reference patterns for the terraform-automation skill. Consult when creating new infrastructure.

## File Structure Convention

```
infrastructure/
├── main.tf              # Main configuration
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── providers.tf         # Provider configuration
├── backend.tf           # State backend config
├── versions.tf          # Version constraints
├── modules/             # Local modules
│   └── vpc/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/        # Environment-specific
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── prod.tfvars
└── tests/               # Infrastructure tests
    └── validate.sh
```

## VPC with Best Practices

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = false  # One per AZ for HA
  enable_dns_hostnames   = true
  enable_flow_log        = true

  tags = {
    Environment = var.environment
    Project     = var.project
    ManagedBy   = "terraform"
  }
}
```

## S3 Bucket with Security

```hcl
resource "aws_s3_bucket" "secure" {
  bucket_prefix = "myapp-data-"
}

resource "aws_s3_bucket_versioning" "secure" {
  bucket = aws_s3_bucket.secure.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secure" {
  bucket = aws_s3_bucket.secure.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "secure" {
  bucket = aws_s3_bucket.secure.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

## EKS Cluster with GPU Support

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.project_name
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    system = {
      instance_types = ["m6i.xlarge"]
      min_size       = 1
      max_size       = 2
      desired_size   = 1
    }
  }

  # GPU nodes typically added via capacity blocks or self-managed
  # See .claude/steering/tech-stack.md for capacity block patterns
}
```

## Remote State Backend

```hcl
terraform {
  backend "s3" {
    bucket         = "terraform-state-ACCOUNT_ID"
    key            = "blueprints/BLUEPRINT_NAME/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## Infrastructure Toggle Pattern

```hcl
variable "enable_waf" {
  description = "Enable WAF protection"
  type        = bool
  default     = false
}

resource "aws_wafv2_web_acl" "main" {
  count = var.enable_waf ? 1 : 0
  # ...
}
```

## Provider Configuration

```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = "~> 1.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}
```
