# Technology Stack

> This file covers conventions for all domains. See section headers to find the right section for your domain.

## GPU Serving Conventions

### Infrastructure

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

## AgentCore Conventions

> This section grows as AgentCore Runtime blueprints accumulate lessons. Populated by `compound-learner` after each agent-runtime deployment.

### Key AWS services

| Service | Purpose |
|---------|---------|
| Bedrock AgentCore Runtime | Managed agent orchestration and session management |
| Amazon Cognito | User pool + JWT auth for WebSocket proxy |
| ECS Fargate (ARM64) | WebSocket proxy deployment (cost-efficient Graviton) |
| DynamoDB | Session state storage (agent-memory module) |
| CodeBuild | ARM64 container image builds |

### VPC requirements

AgentCore Runtime requires VPC endpoints for: `bedrock-runtime`, `bedrock-agent-runtime`, `ecr.api`, `ecr.dkr`, `s3` (gateway), `dynamodb` (gateway), `secretsmanager`.
Verify all endpoints exist before starting a capacity block — missing endpoints cause silent failures at runtime.

### Auth flow

Always enable `ALLOW_USER_PASSWORD_AUTH` and `ALLOW_REFRESH_TOKEN_AUTH` on the Cognito app client. Do not enable `ALLOW_ADMIN_USER_PASSWORD_AUTH` in production.

### Deployment sequence

Follow the agentcore-deployer 8-stage sequence: Foundation → Container Build → AgentCore Runtime → Auth Wiring → WebSocket Proxy → Integration Test → Readiness Audit → Compound.
Do not skip stages — each gate catches failures that are expensive to debug later.
