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
