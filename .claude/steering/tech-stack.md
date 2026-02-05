# Technology Stack

## Infrastructure

| Technology | Purpose | Preference |
|------------|---------|------------|
| **Terraform** | Infrastructure as Code | Primary |
| **AWS CDK** | Infrastructure as Code | Secondary |
| **CloudFormation** | Infrastructure as Code | Avoid (use Terraform/CDK) |

## Terraform Conventions

### Provider Priority

1. **AWSCC Provider** - Prefer for consistent API behavior
2. **AWS Provider** - Use when AWSCC doesn't support resource

### Resource Naming

```hcl
# DO: Let AWS generate unique names
resource "aws_s3_bucket" "data" {
  bucket_prefix = "myapp-data-"
}

# DON'T: Hardcode names
resource "aws_s3_bucket" "data" {
  bucket = "myapp-data-bucket"  # Avoid
}
```

### Security Defaults

- Enable encryption on all storage (S3, RDS, EBS)
- Block public access on S3 buckets
- Use least-privilege IAM policies
- Enable versioning on S3
- Enable flow logs on VPCs

## Languages

| Language | Use Case |
|----------|----------|
| **HCL** | Terraform configurations |
| **TypeScript** | CDK, Claude plugins |
| **Python** | Scripts, automation |
| **Bash** | CI/CD, simple automation |

## Tools

| Tool | Purpose |
|------|---------|
| **Checkov** | Security scanning |
| **terraform fmt** | Code formatting |
| **terraform validate** | Syntax validation |
| **tfsec** | Additional security scanning |
| **pre-commit** | Git hooks for quality gates |
| **tflint** | Terraform linting |
| **terraform-docs** | Auto-generate documentation |

## Pre-commit Hooks

Required hooks for all commits:

| Hook | Purpose |
|------|---------|
| `terraform fmt` | Enforce consistent formatting |
| `terraform validate` | Syntax validation |
| `tflint` | Terraform best practices |
| `terraform-docs` | Auto-generate module docs |
| `checkov` | Security scanning |
| `tfsec` | Additional security checks |
| `trufflehog` | Secret detection |
| `detect-aws-credentials` | Prevent credential leaks |

Setup: `pre-commit install && pre-commit run -a`

## Infrastructure Toggle Pattern

All optional features should default to `false` in variables.tf:

```hcl
# DO: Default to disabled, enable per-environment
variable "enable_waf" {
  description = "Enable WAF protection"
  type        = bool
  default     = false
}

# Override in environment tfvars
# prod.tfvars: enable_waf = true
```

Benefits:
- Explicit opt-in for features
- Clear visibility of what's enabled
- Easier cost control
- Simpler testing of base infrastructure
