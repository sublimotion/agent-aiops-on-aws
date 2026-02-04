# Terraform Automation Quick Reference

## Setup Checklist

```bash
# Prerequisites
brew install uv terraform
uv python install 3.10
pip install checkov

# Verify AWS credentials
aws sts get-caller-identity

# Install RALPH plugin (in Claude Code)
/plugin install ralph-loop@claude-plugins-official

# Restart Claude Code to load MCP servers
```

## MCP Server Tools

### Terraform Operations

| Action | MCP Tool |
|--------|----------|
| Validate syntax | `RunTerraformCommand` with `validate` |
| Security scan | `RunCheckovScan` |
| Plan changes | `RunTerraformCommand` with `plan` |
| Apply changes | `RunTerraformCommand` with `apply` |
| Search modules | `SearchTerraformRegistry` |
| Get best practices | `terraform://aws_best_practices` |

### CloudFormation/CDK

| Action | MCP Tool |
|--------|----------|
| Validate CFN | `validate_cloudformation_template` |
| Check compliance | `check_cloudformation_template_compliance` |
| Debug failures | `troubleshoot_cloudformation_deployment` |
| Search CDK docs | `search_cdk_documentation` |
| Get CDK samples | `search_cdk_samples_and_constructs` |

## Common Prompts

### Create Infrastructure

```
Create a VPC with public/private subnets in 3 AZs.
Use Terraform with AWS best practices.
Run Checkov scan and fix any HIGH/CRITICAL issues.
```

### Validate Code

```
Validate ./infrastructure/ with:
1. terraform validate
2. Checkov security scan
Report and fix any issues.
```

### RALPH Loop

```bash
/ralph-loop "Create secure S3 bucket.
Requirements: encryption, versioning, block public access.
Success: Checkov passes, terraform plan succeeds.
Output <promise>DONE</promise> when complete." --completion-promise "DONE" --max-iterations 15
```

### Debug Failure

```
CloudFormation stack "my-stack" failed in us-east-1.
Analyze with AWS IaC MCP server and suggest fixes.
```

## Validation Pipeline

```bash
# Format
terraform fmt -check -recursive

# Validate
terraform validate

# Security scan
checkov -d . --framework terraform --check HIGH,CRITICAL

# Plan
terraform plan -out=tfplan

# Apply (after review)
terraform apply tfplan
```

## Best Practices Checklist

- [ ] Use AWSCC provider when available
- [ ] Let Terraform generate resource names (`name_prefix`)
- [ ] Enable encryption by default
- [ ] Block public access on S3
- [ ] Use least-privilege IAM
- [ ] Enable versioning on S3
- [ ] Use remote state with locking
- [ ] Tag all resources consistently
- [ ] Run Checkov before every apply

## File Structure

```
infrastructure/
├── main.tf
├── variables.tf
├── outputs.tf
├── providers.tf
├── backend.tf
├── versions.tf
├── modules/
├── environments/
│   ├── dev.tfvars
│   └── prod.tfvars
└── tests/
```

## Config Locations

| File | Purpose |
|------|---------|
| `.claude/mcp.json` | MCP server configuration |
| `.claude/skills/terraform-automation/` | Custom skill |
| `.claude/settings.local.json` | Local permissions |

## Change AWS Profile

Edit `.claude/mcp.json`:

```json
"env": {
  "AWS_PROFILE": "your-profile-name",
  "FASTMCP_LOG_LEVEL": "ERROR"
}
```

Then restart Claude Code.
