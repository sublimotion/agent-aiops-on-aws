# Terraform Automation Plugin for Claude Code

Automate AWS infrastructure deployment using Terraform with security scanning, best practices, and iterative development.

## Features

- **TDD Workflow** - Test-driven infrastructure development
- **Security First** - Integrated Checkov scanning for compliance
- **AWS Best Practices** - Follows Well-Architected Framework
- **MCP Integration** - Works with AWS Labs MCP servers
- **RALPH Compatible** - Supports autonomous iterative loops

## Installation

```bash
/plugin install github:sublimotion/agent-aiops-on-aws
```

## Requirements

### Tools
```bash
# Terraform CLI
brew install terraform

# Checkov security scanner
pip install checkov

# AWS CLI (configured)
aws sts get-caller-identity
```

### MCP Servers (Optional but Recommended)
Add to your `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "awslabs.terraform-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.terraform-mcp-server@latest"],
      "env": { "AWS_PROFILE": "default" }
    },
    "awslabs.aws-iac-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.aws-iac-mcp-server@latest"],
      "env": { "AWS_PROFILE": "default" }
    }
  }
}
```

## Usage

### Basic Usage
```
/terraform-automation Create a secure S3 bucket with encryption and versioning
```

### Aliases
```
/tf Create a VPC with public and private subnets
/terraform Validate my existing infrastructure
```

### With RALPH Loop
For autonomous iterative development:

```bash
/ralph-loop "Create EKS cluster with Terraform.

Requirements:
- VPC with private subnets
- EKS cluster with managed node groups
- IAM roles with least privilege

Success criteria:
- terraform validate passes
- Checkov shows no HIGH/CRITICAL issues
- terraform plan succeeds

Output <promise>DONE</promise> when complete." --completion-promise "DONE" --max-iterations 20
```

## Workflow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ DESIGN  │────▶│  CODE   │────▶│ VALIDATE│────▶│ DEPLOY  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
  Research       Write TF        fmt/validate    apply
  Best practices Iterate         Checkov scan    Verify
  Registry search               Plan review
```

## Validation Pipeline

Run the included validation script:

```bash
./scripts/validate.sh ./infrastructure/
```

Or manually:
```bash
terraform fmt -check -recursive
terraform validate
checkov -d . --framework terraform
terraform plan
```

## Security Controls

The skill enforces these security defaults:

| Resource | Controls |
|----------|----------|
| S3 | KMS encryption, versioning, public access block |
| IAM | Least privilege, specific ARNs |
| VPC | Flow logs, private subnets |
| KMS | Key rotation enabled |

## Steering Files (Kiro-Inspired)

This plugin includes steering files that provide persistent context to Claude, similar to [Kiro's](https://kiro.dev) steering system.

Copy the `steering/` folder to your project's `.claude/` directory:

```bash
cp -r steering/ your-project/.claude/steering/
```

| File | Purpose |
|------|---------|
| `product.md` | Business context and quality standards |
| `tech-stack.md` | Technology preferences and conventions |
| `project-structure.md` | Directory layouts and naming patterns |

These files are automatically included when Claude needs context about the project.

## License

MIT
