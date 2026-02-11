# Project Structure

## AIOps Layout

```
aiops/
├── .claude/                    # Claude Code configuration
│   ├── steering/               # Context files (this directory)
│   │   ├── product.md          # Business context
│   │   ├── tech-stack.md       # Technology preferences
│   │   └── project-structure.md # This file
│   └── ralph-loop.local.md     # RALPH loop state
├── main.tf                     # Main Terraform configuration
├── .checkov.yaml               # Security scan exceptions
├── .terraform.lock.hcl         # Provider lock file
└── README.md                   # Project documentation
```

## Terraform Project Structure

When creating new Terraform projects:

```
infrastructure/
├── main.tf              # Main configuration, root module
├── variables.tf         # Input variables with descriptions
├── outputs.tf           # Output values
├── providers.tf         # Provider configuration
├── backend.tf           # Remote state configuration
├── versions.tf          # Version constraints
├── locals.tf            # Local values (optional)
├── data.tf              # Data sources (optional)
├── modules/             # Local reusable modules
│   └── <module-name>/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/        # Environment-specific variables
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── prod.tfvars
└── tests/               # Infrastructure tests
    └── validate.sh
```

## File Naming Conventions

| Pattern | Purpose |
|---------|---------|
| `*.tf` | Terraform configuration |
| `*.tfvars` | Variable values |
| `.checkov.yaml` | Checkov skip configuration |
| `SKILL.md` | Claude skill definition |
| `CLAUDE.md` | Project context for Claude |
| `README.md` | Human documentation |

## Configuration Locations

| File | Scope | Purpose |
|------|-------|---------|
| `.claude/steering/` | Project | Persistent context for Claude |
| `.claude/mcp.json` | Project | MCP servers |
| `.claude/settings.json` | Project | Shared settings |
| `.checkov.yaml` | Project | Security scan exceptions |
