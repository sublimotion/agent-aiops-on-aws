# Project Structure

## Repository Layout

```
agent-aiops-on-aws/
├── .claude/                    # Claude Code configuration
│   ├── steering/               # Context files
│   │   ├── product.md          # Business context
│   │   ├── tech-stack.md       # Technology preferences
│   │   └── project-structure.md # This file
│   └── ralph-loop.local.md     # RALPH loop state
│
├── modules/                    # Reusable Terraform modules
│   ├── networking/             # VPC, subnets, endpoints
│   ├── eks-cluster/            # EKS with GPU support
│   ├── sagemaker-studio/       # SageMaker domain + IAM
│   └── vllm/                   # vLLM Kubernetes deployment
│
├── blueprints/                 # Complete, deployable examples
│   ├── ministral-3b/           # Ministral-3B on EKS + SageMaker
│   ├── llama-8b/               # (future) Llama model
│   └── multi-model/            # (future) Multiple models
│
├── specs/                      # Requirements/specs (input docs)
│   ├── ministral-3b.md         # Ministral-3B requirements
│   └── _template.md            # Template for new specs
│
├── docs/                       # Documentation
│   └── getting-started.md      # Quick start guide
│
├── terraform/                  # Legacy monolithic deployment
│                               # (kept for reference/migration)
│
├── CLAUDE.md                   # Root context for Claude
└── README.md                   # Project documentation
```

## Module Structure

Each module follows this pattern:

```
modules/<module-name>/
├── main.tf             # Core resources
├── variables.tf        # Inputs with descriptions
├── outputs.tf          # Outputs for consumers
└── README.md           # Auto-generated docs
```

**Variable Requirements**:
- Every variable must have a `description`
- Include `type` constraint
- Provide sensible `default` where appropriate

**Output Requirements**:
- Every output must have a `description`
- Export values needed by dependent modules

## Blueprint Structure

Each blueprint is self-contained and deployable:

```
blueprints/<name>/
├── main.tf             # Composes modules
├── variables.tf        # Blueprint-specific config
├── outputs.tf          # Useful outputs
└── README.md           # Usage + architecture diagram
```

**Naming Convention**: `<model>-<variant>` or `<purpose>-<details>`

Examples:
- `ministral-3b` - Ministral-3B inference
- `llama-70b-multi-gpu` - Llama-70B on multiple GPUs
- `multi-model-router` - Multiple models with routing

## Spec Files

Specs in `specs/` define requirements before implementation:

```
specs/
├── _template.md        # Template for new specs
├── ministral-3b.md     # Ministral-3B requirements
└── <new-blueprint>.md  # Requirements for new blueprints
```

Use specs to:
1. Define requirements before coding
2. Document lessons learned
3. Capture known limitations

## File Naming Conventions

| Pattern | Purpose |
|---------|---------|
| `*.tf` | Terraform configuration |
| `*.tfvars` | Variable values |
| `.checkov.yaml` | Checkov skip configuration |
| `CLAUDE.md` | Project context for Claude |
| `README.md` | Human documentation |

## Configuration Locations

| File | Scope | Purpose |
|------|-------|---------|
| `.claude/steering/` | Project | Persistent context |
| `.checkov.yaml` | Blueprint | Security exceptions |
| `.pre-commit-config.yaml` | Root | Pre-commit hooks |

## Adding a New Blueprint

1. Create spec in `specs/<name>.md` using template
2. Create `blueprints/<name>/` directory
3. Compose existing modules in `main.tf`
4. Add blueprint-specific `variables.tf`
5. Document in `README.md` with architecture diagram
6. Test deployment end-to-end
7. Update spec with lessons learned
