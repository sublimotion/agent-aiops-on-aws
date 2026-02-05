# Agent AIOps on AWS

This repo explores Claude Code plugins, AWS infrastructure automation, and AI agent patterns.

## Quick Start

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run -a

# Security scan
checkov -d .

# Format Terraform
terraform fmt -recursive
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `terraform/` | Infrastructure as Code |
| `.claude/` | Claude Code configuration |
| `.claude/steering/` | Detailed context files |
| `blueprints/` | Reusable infrastructure patterns |

## Steering Files

Detailed context is in `.claude/steering/`:

- `product.md` - Business context and quality standards
- `tech-stack.md` - Technology preferences and conventions
- `project-structure.md` - Project layout and naming conventions

## Key Conventions

- **Terraform Provider**: Prefer AWSCC, fallback to AWS
- **Naming**: Use `bucket_prefix` over hardcoded names
- **Security**: Encryption enabled, public access blocked
- **Features**: Default to `false`, enable per-environment
