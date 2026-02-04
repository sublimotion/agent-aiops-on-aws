# AIOps - AWS Infrastructure Automation with Claude Code

This folder documents the setup for automating AWS infrastructure deployment using Claude Code with MCP servers, skills, and iterative development patterns.

## Overview

This setup combines:
- **AWS Labs MCP Servers** - Real-time access to Terraform/CDK best practices and security scanning
- **Terraform Automation Skill** - Custom skill for infrastructure development workflows
- **RALPH Plugin** - Autonomous iterative loops for TDD-style infrastructure development
- **Steering Files** - Kiro-inspired persistent context for consistent AI behavior

## Comparison with Kiro

This setup is inspired by [Kiro](https://kiro.dev), AWS's agentic IDE. Here's how they compare:

| Feature | Kiro | Our Claude Code Setup |
|---------|------|----------------------|
| **Environment** | Dedicated IDE (Code OSS) | CLI + any editor |
| **Spec-Driven Dev** | Built-in specs system | RALPH loops + custom skill |
| **Persistent Context** | `.kiro/steering/` files | `.claude/steering/` files |
| **Hooks** | Built-in event automation | Claude Code hooks |
| **MCP Servers** | Native support | Native support |
| **Focus** | General development | AWS infrastructure |

### Kiro Concepts Adapted

1. **Specs → RALPH Loops**: Instead of Kiro's structured specs, we use RALPH's iterative loops with explicit success criteria
2. **Steering → Steering Files**: Same concept, adapted for Claude Code in `.claude/steering/`
3. **Hooks → Claude Hooks**: Event-driven automation in `.claude/settings.json`

## Steering Files (Kiro-Inspired)

Steering files provide persistent context to Claude, similar to Kiro's steering system.

Location: `/Users/phi/Documents/workbench/.claude/steering/`

| File | Purpose |
|------|---------|
| `product.md` | Business context and quality standards |
| `tech-stack.md` | Technology preferences and conventions |
| `project-structure.md` | Directory layouts and naming patterns |

These files are automatically included when Claude needs context about the project.

## Hooks (Kiro-Inspired)

Claude Code hooks provide event-driven automation similar to Kiro's hooks.

Location: `/Users/phi/Documents/workbench/.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "terraform fmt check when .tf files are modified"
          }
        ]
      }
    ]
  }
}
```

### Available Hook Events

| Event | Trigger |
|-------|---------|
| `PreToolUse` | Before a tool executes |
| `PostToolUse` | After a tool completes |
| `Stop` | When Claude attempts to stop (used by RALPH) |

### Potential Hooks for Infrastructure

- **PostToolUse on Write/Edit**: Run `terraform fmt` on `.tf` files
- **PostToolUse on Write/Edit**: Run Checkov on infrastructure changes
- **Stop**: Validate all Terraform before allowing completion

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  RALPH Plugin   │  │ Terraform Skill │  │   MCP Servers   │  │
│  │  (Iteration)    │  │  (Guidance)     │  │  (Tools/APIs)   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
├────────────────────────────────┼────────────────────────────────┤
│                                ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    AWS Labs MCP Servers                   │   │
│  ├──────────────────────────┬───────────────────────────────┤   │
│  │  terraform-mcp-server    │    aws-iac-mcp-server         │   │
│  │  • Terraform CLI         │    • CloudFormation lint      │   │
│  │  • Checkov scanning      │    • Compliance checking      │   │
│  │  • AWS best practices    │    • CDK documentation        │   │
│  │  • Terragrunt support    │    • Deployment debugging     │   │
│  │  • Registry modules      │    • CDK samples              │   │
│  └──────────────────────────┴───────────────────────────────┘   │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │         AWS            │
                    │  (Your Infrastructure) │
                    └────────────────────────┘
```

## Prerequisites

### Required Tools

```bash
# Install uv (Python package runner for MCP servers)
brew install uv
# or: pip install uv

# Install Python 3.10+
uv python install 3.10

# Install Terraform CLI
brew install terraform

# Install Checkov for security scanning
pip install checkov

# Verify AWS CLI is configured
aws sts get-caller-identity
```

### AWS Credentials

Ensure your AWS credentials are configured:

```bash
# Option 1: Use named profile
aws configure --profile your-profile

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
```

## Configuration Files

### MCP Server Configuration

Location: `/Users/phi/Documents/workbench/.claude/mcp.json`

```json
{
  "mcpServers": {
    "awslabs.terraform-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.terraform-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "default",
        "FASTMCP_LOG_LEVEL": "ERROR"
      },
      "disabled": false,
      "autoApprove": []
    },
    "awslabs.aws-iac-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.aws-iac-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "default",
        "FASTMCP_LOG_LEVEL": "ERROR"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**To change AWS profile**, edit the `AWS_PROFILE` value in both server configurations.

### Terraform Automation Skill

Location: `/Users/phi/Documents/workbench/.claude/skills/terraform-automation/`

```
terraform-automation/
├── SKILL.md           # Skill instructions and patterns
└── scripts/
    └── validate.sh    # Validation pipeline script
```

## MCP Server Capabilities

### Terraform MCP Server (`awslabs.terraform-mcp-server`)

| Tool | Description |
|------|-------------|
| `terraform://workflow_guide` | Security-focused development workflow |
| `terraform://aws_best_practices` | AWS-specific Terraform guidance |
| `terraform://aws_provider_resources_listing` | AWS provider resource documentation |
| `terraform://awscc_provider_resources_listing` | AWSCC provider resource documentation |
| `RunTerraformCommand` | Execute `terraform init/plan/validate/apply/destroy` |
| `RunTerragruntCommand` | Execute Terragrunt commands |
| `RunCheckovScan` | Security and compliance scanning |
| `SearchTerraformRegistry` | Find modules and providers |
| `AnalyzeTerraformModule` | Analyze module inputs/outputs/README |

### AWS IaC MCP Server (`awslabs.aws-iac-mcp-server`)

| Tool | Description |
|------|-------------|
| `validate_cloudformation_template` | Validate CFN templates with cfn-lint |
| `check_cloudformation_template_compliance` | Security compliance with cfn-guard |
| `troubleshoot_cloudformation_deployment` | Debug failed deployments with CloudTrail |
| `search_cdk_documentation` | Find CDK patterns and examples |
| `search_cdk_samples_and_constructs` | Working code examples |
| `cdk_best_practices` | Security and development guidelines |
| `read_iac_documentation_page` | Read full documentation pages |

## Usage Examples

### Basic Infrastructure Creation

```
Create a production-ready VPC with:
- 3 availability zones
- Public and private subnets
- NAT gateways for private subnet egress
- VPC flow logs enabled

Use Terraform best practices and run Checkov scan.
```

### Validate Existing Terraform

```
Validate the Terraform configuration in ./infrastructure/:
1. Run terraform validate
2. Run Checkov security scan
3. Report any HIGH or CRITICAL findings
4. Suggest fixes for each issue
```

### Debug Deployment Failure

```
The CloudFormation stack "my-stack" failed in us-east-1.
Use the AWS IaC MCP server to analyze the failure and suggest fixes.
```

### RALPH Iterative Development

For autonomous iterative development until success criteria are met:

```bash
/ralph-loop "Create AWS infrastructure using Terraform.

Requirements:
- VPC with public/private subnets across 3 AZs
- S3 bucket with encryption and versioning
- IAM roles following least privilege

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

## Development Workflow

### TDD for Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure TDD Cycle                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │  RED    │────▶│  GREEN  │────▶│ REFACTOR│                  │
│   └─────────┘     └─────────┘     └─────────┘                  │
│        │               │               │                        │
│        ▼               ▼               ▼                        │
│   Write spec      Minimal code    Clean up                     │
│   Run validate    to pass         Keep tests                   │
│   Expect fail     Run validate    passing                      │
│   Run Checkov     Run Checkov                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Research and Design

1. Define infrastructure requirements
2. Search Terraform Registry for existing modules
3. Review AWS best practices via MCP servers
4. Identify security and compliance requirements

### Phase 2: Implementation

1. Write specification/requirements first
2. Create initial Terraform configuration
3. Run `terraform validate` - expect failures initially
4. Run Checkov security scan - document baseline
5. Iterate until validation passes
6. Iterate until Checkov passes (no HIGH/CRITICAL)

### Phase 3: Validation

```bash
# Use the validation script
./scripts/validate.sh ./infrastructure/

# Or manually:
terraform fmt -check -recursive
terraform validate
checkov -d . --framework terraform --check HIGH,CRITICAL
terraform plan
```

### Phase 4: Deployment

```bash
terraform apply
# Verify resources
# Run post-deployment tests
```

## Best Practices

### Provider Selection

- **Prefer AWSCC provider** for consistent API behavior and better security defaults
- Use AWS provider for resources not yet available in AWSCC
- Never mix providers for the same resource type

### Resource Naming

- **Let Terraform generate unique names** using `name_prefix` or `bucket_prefix`
- Don't hardcode resource names
- Use consistent tagging strategy

### Security

- Run Checkov on every change
- Fix security issues rather than suppressing
- Document justifications for any necessary exceptions
- Use least-privilege IAM policies
- Enable encryption by default

### State Management

- Use remote state (S3 + DynamoDB locking)
- Enable state encryption
- Implement state file access controls

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

## Common Patterns

### Secure S3 Bucket

```hcl
resource "aws_s3_bucket" "secure" {
  bucket_prefix = "myapp-data-"  # Let AWS generate unique name
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

  enable_nat_gateway   = true
  single_nat_gateway   = false  # One per AZ for HA
  enable_dns_hostnames = true
  enable_flow_log      = true

  tags = {
    Environment = var.environment
    Project     = var.project
    ManagedBy   = "terraform"
  }
}
```

## Troubleshooting

### MCP Servers Not Loading

```bash
# Restart Claude Code to reload MCP configuration
# Check server status in Claude Code

# Verify uv is installed
which uv

# Test MCP server manually
uvx awslabs.terraform-mcp-server@latest --help
```

### Checkov False Positives

If Checkov reports intentional configurations as issues:

```hcl
# Use inline skip with documented reason
#checkov:skip=CKV_AWS_XX:Public bucket required for static website hosting
resource "aws_s3_bucket" "website" {
  # ...
}
```

### AWS Credentials Issues

```bash
# Verify credentials
aws sts get-caller-identity

# Check profile
aws configure list --profile your-profile

# Update MCP config to use correct profile
# Edit .claude/mcp.json and change AWS_PROFILE
```

## Resources

### AWS Labs MCP Servers
- Repository: https://github.com/awslabs/mcp
- Documentation: https://awslabs.github.io/mcp/

### RALPH Plugin
- Enables autonomous iteration loops
- Install: `/plugin install ralph-loop@claude-plugins-official`
- Commands: `/ralph-loop`, `/cancel-ralph`

### Terraform Resources
- Terraform AWS Provider: https://registry.terraform.io/providers/hashicorp/aws/latest
- Terraform AWSCC Provider: https://registry.terraform.io/providers/hashicorp/awscc/latest
- Checkov: https://www.checkov.io/

### Claude Code Skills
- aws-skills plugin: https://github.com/zxkane/aws-skills
- TDD skill: https://github.com/obra/superpowers/tree/main/skills/test-driven-development
