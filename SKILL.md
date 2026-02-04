---
name: terraform-automation
description: Automate AWS infrastructure deployment using Terraform with security scanning, best practices, and iterative development. Use when creating, validating, or deploying Terraform configurations for AWS.
---

# Terraform Automation Skill

Comprehensive skill for AWS infrastructure automation using Terraform, integrating AWS MCP servers for best practices, security scanning, and deployment workflows.

## When to Use This Skill

- Creating new AWS infrastructure with Terraform
- Validating existing Terraform configurations
- Running security scans on infrastructure code
- Debugging failed deployments
- Following AWS Well-Architected patterns
- Implementing TDD for infrastructure

## Available MCP Tools

### Terraform MCP Server (`awslabs.terraform-mcp-server`)

| Tool | Purpose |
|------|---------|
| `terraform://workflow_guide` | Security-focused development workflow |
| `terraform://aws_best_practices` | AWS-specific Terraform guidance |
| `terraform://aws_provider_resources_listing` | AWS provider resource docs |
| `terraform://awscc_provider_resources_listing` | AWSCC provider resource docs |
| `RunTerraformCommand` | Execute terraform init/plan/validate/apply/destroy |
| `RunTerragruntCommand` | Execute terragrunt commands |
| `RunCheckovScan` | Security and compliance scanning |
| `SearchTerraformRegistry` | Find modules and providers |
| `AnalyzeTerraformModule` | Analyze module inputs/outputs |

### AWS IaC MCP Server (`awslabs.aws-iac-mcp-server`)

| Tool | Purpose |
|------|---------|
| `validate_cloudformation_template` | Validate CFN templates with cfn-lint |
| `check_cloudformation_template_compliance` | Security compliance with cfn-guard |
| `troubleshoot_cloudformation_deployment` | Debug failed deployments |
| `search_cdk_documentation` | Find CDK patterns and examples |
| `search_cdk_samples_and_constructs` | Working code examples |
| `cdk_best_practices` | Security and development guidelines |

## Development Workflow

### Phase 1: Research and Design

```
1. Define infrastructure requirements
2. Search Terraform Registry for existing modules
3. Review AWS best practices for the services needed
4. Identify security requirements and compliance rules
```

### Phase 2: Implementation (TDD Approach)

```
1. Write specification/requirements file first
2. Create initial Terraform configuration
3. Run terraform validate - expect failures initially
4. Run Checkov security scan - document baseline
5. Iterate until validation passes
6. Iterate until Checkov passes (no HIGH/CRITICAL)
```

### Phase 3: Validation and Security

```
1. terraform fmt - format code
2. terraform validate - syntax validation
3. Checkov scan - security validation
4. terraform plan - review changes
5. Manual review of plan output
```

### Phase 4: Deployment

```
1. terraform apply (with approval)
2. Verify resources created
3. Run post-deployment tests
4. Document infrastructure
```

## Best Practices

### Provider Selection

- **Prefer AWSCC provider** for consistent API behavior and better security defaults
- Use AWS provider for resources not yet available in AWSCC
- Never mix providers for the same resource type

### Resource Naming

- **Let Terraform/CDK generate unique names** - don't hardcode resource names
- Use consistent tagging strategy
- Include environment, project, and owner tags

### Security First

- Run Checkov on every change
- Fix security issues rather than suppressing
- Document justifications for any necessary exceptions
- Use least-privilege IAM policies
- Enable encryption by default (S3, RDS, EBS)

### State Management

- Use remote state (S3 + DynamoDB locking)
- Enable state encryption
- Implement state file access controls

## Example Prompts

### Create New Infrastructure

```
Create a production-ready VPC with:
- 3 availability zones
- Public and private subnets
- NAT gateways for private subnet egress
- VPC flow logs enabled
- Follow AWS Well-Architected security pillar

Use the terraform-mcp-server for best practices and run Checkov scan.
```

### Validate Existing Code

```
Validate the Terraform configuration in ./infrastructure/:
1. Run terraform validate
2. Run Checkov security scan
3. Report any HIGH or CRITICAL findings
4. Suggest fixes for each issue
```

### Debug Deployment Failure

```
The Terraform apply failed with this error: [error message]

Use the AWS IaC MCP server to:
1. Analyze the failure pattern
2. Check CloudTrail for related events
3. Suggest resolution steps
```

## RALPH Loop Integration

For autonomous iterative development, use with the ralph-loop plugin:

```bash
/ralph-loop "Create AWS infrastructure using Terraform.

Requirements:
- VPC with public/private subnets
- EKS cluster
- RDS PostgreSQL
- S3 buckets with encryption

Process each iteration:
1. Write/update Terraform code
2. Run terraform validate
3. Run Checkov security scan
4. Fix any issues found
5. Run terraform plan

Success criteria:
- terraform validate passes
- Checkov shows no HIGH/CRITICAL issues
- terraform plan shows expected resources

Output <promise>READY_FOR_APPLY</promise> when complete." --completion-promise "READY_FOR_APPLY" --max-iterations 25
```

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

## Common Terraform Patterns for AWS

### VPC with Best Practices

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

### S3 Bucket with Security

```hcl
resource "aws_s3_bucket" "secure" {
  # Let AWS generate unique name
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

## Troubleshooting

### Checkov False Positives

If Checkov reports issues that are intentional:

1. Document the reason clearly
2. Use inline skip comments sparingly:
   ```hcl
   #checkov:skip=CKV_AWS_XX:Reason for skipping
   ```
3. Prefer fixing over skipping

### State Lock Issues

```bash
# If state is locked unexpectedly
terraform force-unlock LOCK_ID
```

### Provider Version Conflicts

```hcl
terraform {
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
```
