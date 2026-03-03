---
name: terraform-automation
description: Automate AWS infrastructure deployment using Terraform with security scanning, best practices, and iterative development. Use when user says "create infrastructure", "deploy with Terraform", "run Checkov scan", "validate Terraform", or "debug deployment failure". Do NOT use for reading or explaining existing Terraform code, answering general Terraform questions, or CloudFormation/CDK-only workflows.
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

## Context References

For project conventions, consult these canonical steering files (do not duplicate their content):

- **Technology preferences**: `.claude/steering/tech-stack.md`
- **Project structure**: `.claude/steering/project-structure.md`
- **Quality standards**: `.claude/steering/product.md`

For HCL code patterns, see `references/terraform-patterns.md`.

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

1. Define infrastructure requirements
2. Search Terraform Registry for existing modules
3. Review AWS best practices for the services needed
4. Identify security requirements and compliance rules

### Phase 2: Implementation (TDD Approach)

1. Write specification/requirements file first
2. Create initial Terraform configuration
3. Run `terraform validate` — expect failures initially
4. Run Checkov security scan — document baseline
5. Iterate until validation passes
6. Iterate until Checkov passes (no HIGH/CRITICAL)

### Phase 3: Validation and Security

Run `scripts/validate.sh [directory]` or execute manually:

1. `terraform fmt` — format code
2. `terraform validate` — syntax validation
3. Checkov scan — security validation
4. `terraform plan` — review changes

### Phase 4: Deployment

1. `terraform apply` (with approval)
2. Verify resources created
3. Run post-deployment tests
4. Document infrastructure

## Best Practices

### Provider Selection

- **Prefer AWSCC provider** for consistent API behavior and better security defaults
- Use AWS provider for resources not yet available in AWSCC
- Never mix providers for the same resource type

### Resource Naming

- **Let Terraform/CDK generate unique names** — use `bucket_prefix`, `name_prefix`
- Use consistent tagging: environment, project, owner, ManagedBy=terraform
- Keep `var.project_name` to 12 characters or fewer to avoid IAM role name length limits (64 char max)

### Security First

- Run Checkov on every change
- Fix security issues rather than suppressing
- Document justifications for any necessary exceptions
- Use least-privilege IAM policies
- Enable encryption by default (S3, RDS, EBS)

### State Management

- Use remote state (S3 + DynamoDB locking)
- Enable state encryption
- Never run terraform in parallel background tasks — state locks cause deadlocks
- If a lock persists, check for orphaned processes with `ps aux | grep terraform`

## RALPH Loop Integration

For autonomous iterative development, use with the ralph-loop plugin:

```bash
/ralph-loop "Create AWS infrastructure using Terraform.

Requirements:
- VPC with public/private subnets
- EKS cluster
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

## Troubleshooting

### State Lock from Background Tasks

**Error**: `Error locking state: Error acquiring the state lock`
**Cause**: Multiple terraform processes or orphaned background tasks hold the state lock.
**Solution**:
1. Check for running processes: `ps aux | grep terraform`
2. Kill orphaned processes: `pkill -9 terraform`
3. If lock persists: `terraform force-unlock <LOCK_ID>`
4. Prevention: never run terraform in parallel background tasks.

### IAM Role Name Length Exceeded

**Error**: `IAM role name exceeds 64 character limit` or `InvalidParameterValue`
**Cause**: Long `var.project_name` combined with module suffixes like `-eks-cluster-node-role` exceeds the 64-character IAM limit.
**Solution**: Keep `var.project_name` to 12 characters or fewer. Example: `qwen3-next-g7e-bench` (20 chars) + `-eks-cluster-node-role` (22 chars) = 42 chars before prefix, which can push over the limit with account-specific prefixes.

### Kubernetes Provider Dependency During First Apply

**Error**: `The kubernetes/helm provider configuration is not valid` on first `terraform apply`
**Cause**: K8s and Helm providers depend on EKS cluster outputs that don't exist yet during initial apply.
**Solution**: Apply in stages using `-target`:
```bash
terraform apply -target=module.networking -target=module.eks_cluster
terraform apply  # Full apply after cluster exists
```

### GPU Resource Change Causes Scheduling Deadlock

**Error**: Pod stuck in `Pending` after changing GPU count (e.g., TP=4 to TP=8)
**Cause**: Rolling update requires new pod's GPUs before releasing old pod's allocation.
**Solution**: Scale to 0, apply, scale back:
```bash
kubectl -n <ns> scale deployment <name> --replicas=0
terraform apply -target='<deployment_resource>' -auto-approve
kubectl -n <ns> scale deployment <name> --replicas=1
```

### State Attribute Mismatch After Import

**Error**: `terraform plan` shows resource destruction for an imported resource due to attribute mismatch (e.g., `bootstrap_self_managed_addons`)
**Cause**: Imported state has different default values than Terraform config expects.
**Solution**: Edit state directly:
```bash
terraform state pull > state.json
# Fix the mismatched attribute in JSON
terraform state push state.json
terraform plan  # Verify no destructive changes
```

### Checkov False Positives

**Error**: Checkov flags intentional configurations (e.g., public ALB)
**Cause**: Security policy doesn't match deployment intent.
**Solution**: Use inline skip with documented reason:
```hcl
#checkov:skip=CKV_AWS_XX:Public ALB required for external API access
```
Prefer fixing over skipping. Document all skips in the blueprint's `lessons.md`.

### Provider Version Conflicts

**Error**: `Could not retrieve the list of available versions for provider`
**Cause**: Version constraints too tight or provider not available.
**Solution**: Use flexible constraints in `versions.tf`:
```hcl
terraform {
  required_providers {
    aws   = { source = "hashicorp/aws", version = "~> 5.0" }
    awscc = { source = "hashicorp/awscc", version = "~> 1.0" }
  }
}
```

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Skill triggers on relevant queries | 90%+ | Test with 10 prompts that should trigger (create infra, deploy, validate, scan, debug) |
| Workflow completes without user correction | 80%+ | Run same request 3x, compare outputs for structural consistency |
| Zero failed API/MCP calls per workflow | 0 errors | Monitor MCP server logs during test runs |
| Validation pipeline passes on first try | 70%+ | Track how often `scripts/validate.sh` passes without iteration |
| Consistent file structure across runs | 100% | Verify output matches `references/terraform-patterns.md` conventions |
