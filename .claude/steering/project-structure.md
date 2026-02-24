# Project Structure

## Repository Layout

```
agent-aiops-on-aws/
├── .claude/                    # Claude Code configuration
│   ├── steering/               # Context files (loaded on demand)
│   │   ├── product.md          # Business context
│   │   ├── tech-stack.md       # Technology preferences (all domains)
│   │   └── project-structure.md # This file
│   ├── agents/                 # Sub-agents
│   │   ├── blueprint-reviewer.md  # Coherence auditor
│   │   ├── spec-writer.md         # Spec authoring
│   │   ├── benchmark-analyst.md   # Results analysis
│   │   ├── infra-deployer.md      # GPU-serving deployment (8-stage)
│   │   ├── agentcore-deployer.md  # Agent Runtime deployment (8-stage)
│   │   └── compound-learner.md    # Post-deployment lesson elevation
│   ├── skills/                 # Custom skills
│   │   └── visual-explainer/   # Render dense output as interactive HTML
│   └── ralph-loop.local.md     # RALPH loop state
│
├── domains/                    # Domain-specific modules, specs, blueprints
│   ├── gpu-serving/            # GPU Serving domain
│   │   ├── modules/            # Reusable Terraform modules
│   │   │   ├── networking/     # VPC, subnets, endpoints
│   │   │   ├── eks-cluster/    # EKS with GPU support
│   │   │   ├── sagemaker-studio/ # SageMaker domain + IAM
│   │   │   ├── vllm/           # vLLM Kubernetes deployment
│   │   │   ├── fsx-lustre/     # FSx for Lustre filesystem
│   │   │   └── monitoring/     # Prometheus + Grafana
│   │   ├── blueprints/         # GPU Serving blueprints
│   │   │   ├── ministral-3b/   # Ministral-3B on EKS + SageMaker
│   │   │   ├── kimi-k2.5/     # Kimi K2.5 MoE on p5e (8x H200)
│   │   │   └── qwen3-next/    # Qwen3-Next MoE on p5en (8x H200)
│   │   └── specs/              # GPU Serving specs
│   │       ├── _template.md    # Template for new specs
│   │       ├── ministral-3b.md # Ministral-3B requirements
│   │       ├── kimi-k2.5.md   # KV cache benchmark requirements
│   │       └── qwen3-next.md  # Qwen3-Next KV cache benchmark
│   └── agent-runtime/          # Bedrock AgentCore Runtime domain
│       ├── modules/            # Reusable Terraform modules for agent runtime
│       │   ├── agentcore-runtime/  # Bedrock AgentCore Runtime resource
│       │   ├── cognito-app-auth/   # User pool + app client
│       │   ├── websocket-proxy/    # Node.js proxy on ECS Fargate
│       │   └── agent-memory/       # DynamoDB session state
│       ├── blueprints/         # Agent Runtime blueprints
│       │   └── research-agent/ # Multi-agent research system on AgentCore Runtime
│       └── specs/              # Agent Runtime specs
│           ├── _template-agent-runtime.md
│           └── research-agent.md
│
├── scripts/                    # Shared utility scripts
│   └── stage-images-ecr.sh    # Mirror images to private ECR
│
├── CLAUDE.md                   # Root context (routing layer)
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

Blueprints are self-contained: they own their infrastructure, operational artifacts, and results. Specs define what to build; blueprints own what happened.

### Core files (always present)

```
blueprints/<name>/
├── main.tf             # Composes modules
├── variables.tf        # Blueprint-specific config
├── outputs.tf          # Useful outputs
└── README.md           # Architecture + quick start
```

### Operational artifacts (added during/after deployment)

```
blueprints/<name>/
├── lessons.md          # Operational lessons (grows over time)
├── configs/            # Launch configurations per serving variant
│   ├── baseline.sh
│   └── <variant>.sh
├── scripts/            # Orchestration + validation tooling
│   ├── run-benchmarks.py
│   └── validate.sh
├── docker/             # Container images (if needed)
│   ├── Dockerfile
│   └── requirements.txt
└── results/            # Benchmark reports, architecture diagrams
    ├── <report>.md
    └── <diagram>.html
```

## Spec Structure

Specs live in `domains/<domain>/specs/` and define requirements for a blueprint.

```markdown
# <Domain> Spec: <Name>

## Overview
Brief description (1-3 sentences)

## Configuration
Table of parameters (model, hardware, etc.)

## Architecture
Components and their relationships

## Success Criteria
Concrete, testable outcomes

## Non-Requirements
What this does NOT need to do
```

## Steering Files

`.claude/steering/` contains persistent context loaded on demand by Claude Code.

| File | Purpose | When to Update |
|------|---------|----------------|
| `product.md` | Business context, quality standards | Rarely (foundational) |
| `tech-stack.md` | Technology preferences (all domains) | When adopting new tools |
| `project-structure.md` | Layout and conventions | When adding domains/blueprints |

## File Naming Conventions

- Terraform: `main.tf`, `variables.tf`, `outputs.tf`
- Modules: lowercase with hyphens (`eks-cluster/`, `agent-memory/`)
- Blueprints: lowercase with hyphens (`ministral-3b/`, `research-agent/`)
- Specs: match blueprint name (`ministral-3b.md`, `research-agent.md`)
- Scripts: lowercase with hyphens (`run-benchmarks.py`, `stage-images-ecr.sh`)
- Results: descriptive names with dates (`benchmark-report-20260221.md`)

## Adding a New Blueprint

1. Write spec in `domains/<domain>/specs/<name>.md`
2. Run `/ralph-loop:ralph-loop Deploy domains/<domain>/specs/<name>.md`
3. Claude creates `domains/<domain>/blueprints/<name>/` with terraform files
4. Deployment succeeds → capture lessons in `lessons.md`
5. Run compound-learner to elevate cross-cutting lessons to steering files
6. Update this file's repository layout tree if the blueprint introduces new patterns
