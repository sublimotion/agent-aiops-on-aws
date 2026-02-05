# Agent AIOps on AWS

A Claude Code template for autonomous AWS infrastructure deployment using spec-driven development and iterative RALPH loops.

## What This Is

This repository demonstrates patterns for using Claude Code to autonomously build and deploy infrastructure:

1. **Spec-Driven Development** - Define requirements in markdown, let Claude implement
2. **Steering Files** - Provide persistent context across sessions
3. **RALPH Loops** - Autonomous iteration until deployment succeeds
4. **Modular Terraform** - Reusable infrastructure components

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sublimotion/agent-aiops-on-aws
cd agent-aiops-on-aws
pre-commit install

# Start Claude Code
claude

# Run the RALPH loop with your spec
/ralph-loop Deploy the infrastructure in specs/ministral-3b.md
```

## Repository Structure

```
.claude/
├── steering/                 # Persistent context files
│   ├── product.md            # Business context, quality standards
│   ├── tech-stack.md         # Technology preferences
│   └── project-structure.md  # Layout and conventions
├── settings.local.json       # Permissions (gitignored)
└── skills/                   # Custom skills (optional)

specs/                        # Input specifications
├── _template.md              # Template for new specs
└── ministral-3b.md           # Example: ML inference platform

modules/                      # Reusable Terraform modules
├── networking/               # VPC, subnets, endpoints
├── eks-cluster/              # EKS with GPU support
├── sagemaker-studio/         # SageMaker domain + IAM
└── vllm/                     # vLLM on Kubernetes

blueprints/                   # Deployable compositions
└── ministral-3b/             # Example deployment
```

## Setting Up Steering Files

Steering files in `.claude/steering/` provide persistent context to Claude across all sessions.

### 1. product.md - Business Context

```markdown
# Product Context

## Primary Use Cases
- What this project does
- Who uses it

## Quality Standards
- Security requirements
- Testing expectations
- Review process
```

### 2. tech-stack.md - Technology Preferences

```markdown
# Technology Stack

## Infrastructure
| Technology | Purpose | Preference |
|------------|---------|------------|
| Terraform  | IaC     | Primary    |

## Conventions
- Provider preferences (AWSCC vs AWS)
- Naming patterns
- Security defaults
```

### 3. project-structure.md - Layout

```markdown
# Project Structure

## Directory Layout
- Where modules live
- How blueprints are organized
- Naming conventions
```

## Writing a Spec File

Specs define what you want built. Claude uses them as requirements.

### Example: specs/my-deployment.md

```markdown
# My Deployment Requirements

## Overview
Brief description of what this deployment does.

## Components

### 1. Compute
- **Platform**: EKS / Lambda / ECS
- **Instance Types**: t3.medium, with fallbacks
- **Scaling**: Min 1, Max 3

### 2. Networking
- **VPC**: 10.0.0.0/16, 3 AZs
- **Access**: Private subnets only

### 3. Storage
- **Database**: RDS PostgreSQL
- **Cache**: ElastiCache Redis

## Non-Requirements
- No multi-region (this is dev/prototype)
- No HA/DR setup

## Security Requirements
- Encryption at rest
- Private subnets for all compute
- IAM least privilege

## Known Limitations
(Updated during development with lessons learned)
```

### Spec Template

Copy `specs/_template.md` to start a new spec:

```bash
cp specs/_template.md specs/my-new-deployment.md
```

## Running with RALPH Loop

The RALPH loop enables autonomous iteration. Claude will keep working on your task, seeing its previous progress each iteration.

### Basic Usage

```bash
# Start Claude Code
claude

# Invoke the RALPH loop with your task
/ralph-loop Deploy the Terraform infrastructure in specs/my-deployment.md
```

### How It Works

1. **First iteration**: Claude reads your spec, plans the implementation
2. **Subsequent iterations**: Claude sees previous work in files, continues from where it left off
3. **Loop continues**: Until the task is complete or you cancel

### Example Session

```bash
$ claude
> /ralph-loop Deploy the infrastructure: terraform init, plan, and apply for specs/ministral-3b.md

# Iteration 1: Claude reads spec, creates modules
# Iteration 2: Claude runs terraform init, fixes provider issues
# Iteration 3: Claude runs terraform plan, adjusts for capacity
# Iteration 4: Claude runs terraform apply, deployment succeeds
# Loop exits when completion criteria met
```

### Monitoring Progress

```bash
# Check loop state
head -20 .claude/ralph-loop.local.md

# View terraform state
terraform state list

# Check deployment
kubectl get pods -A
```

### Canceling the Loop

Press `Ctrl+C` or use:
```bash
/ralph-loop:cancel-ralph
```

## Creating a New Blueprint

1. **Write the spec**
   ```bash
   cp specs/_template.md specs/my-blueprint.md
   # Edit with your requirements
   ```

2. **Create blueprint directory**
   ```bash
   mkdir blueprints/my-blueprint
   ```

3. **Run RALPH loop**
   ```bash
   claude
   > /ralph-loop Implement blueprints/my-blueprint using specs/my-blueprint.md
   ```

4. **Update spec with lessons learned**
   - Add any limitations discovered
   - Document workarounds
   - Note version requirements

## Example Blueprints

| Blueprint | Description | Spec |
|-----------|-------------|------|
| [ministral-3b](blueprints/ministral-3b/) | vLLM + EKS + SageMaker | [specs/ministral-3b.md](specs/ministral-3b.md) |

## Key Patterns

### Iterative Development

Instead of trying to get everything right upfront:
1. Write a minimal spec
2. Run RALPH loop
3. Learn from failures
4. Update spec with lessons learned
5. Repeat

### Steering File Updates

When you discover new patterns or preferences:
1. Update relevant steering file
2. Claude will use this context in future sessions
3. Keeps knowledge persistent across conversations

### Module Composition

Build complex deployments from simple modules:

```hcl
module "networking" {
  source = "../../modules/networking"
  # ...
}

module "eks" {
  source             = "../../modules/eks-cluster"
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnets
  # ...
}
```

## Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- Claude Code with ralph-loop plugin
- pre-commit (optional but recommended)

## Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run -a

# Verify tools
terraform --version
aws sts get-caller-identity
```

## License

MIT
