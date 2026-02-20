# Project Structure

## Repository Layout

```
agent-aiops-on-aws/
├── .claude/                    # Claude Code configuration
│   ├── steering/               # Context files (loaded on demand)
│   │   ├── product.md          # Business context
│   │   ├── tech-stack.md       # Technology preferences
│   │   └── project-structure.md # This file
│   └── ralph-loop.local.md     # RALPH loop state
│
├── modules/                    # Reusable Terraform modules
│   ├── networking/             # VPC, subnets, endpoints
│   ├── eks-cluster/            # EKS with GPU support
│   ├── sagemaker-studio/       # SageMaker domain + IAM
│   ├── vllm/                   # vLLM Kubernetes deployment
│   ├── fsx-lustre/             # FSx for Lustre filesystem
│   └── monitoring/             # Prometheus + Grafana
│
├── blueprints/                 # Self-contained deployable compositions
│   ├── ministral-3b/           # Ministral-3B on EKS + SageMaker
│   └── kimi-k2.5/             # Kimi K2.5 MoE on p5e (8x H200)
│
├── specs/                      # Requirements (input to blueprints)
│   ├── _template.md            # Template for new specs
│   ├── ministral-3b.md         # Ministral-3B requirements
│   └── vllm-kv-cache-benchmark.md # KV cache benchmark requirements
│
├── docs/                       # Project-wide documentation
│   ├── getting-started.md      # Quick start guide
│   └── 2026-gtm-architecture.md
│
├── scripts/                    # Shared utility scripts
│   └── validate.sh
│
├── terraform/                  # Legacy monolithic deployment
│                               # (kept for reference/migration)
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
│   └── validate-*.sh
├── results/            # All benchmark outputs
│   ├── benchmark-report.md    # Consolidated findings
│   ├── execution-log.md       # Commands used to run benchmarks
│   └── <run-name>/            # Per-run JSON results
├── docs/               # All written knowledge (reference + decisions)
│   ├── <topic>.md             # Technical reference
│   └── <assessment>.md        # Technology evaluations / design plans
├── docker/             # Custom container images (if needed)
└── templates/          # EC2 user data, Helm values (if needed)
```

### Directory purposes

| Directory | Contains | Examples |
|-----------|----------|---------|
| `configs/` | What to launch | Shell scripts, YAML configs per serving variant |
| `scripts/` | How to run and validate | Benchmark orchestrators, storage validators |
| `results/` | What happened | Reports, execution logs, raw JSON data |
| `docs/` | What we know | Technical reference, design plans, assessments |

### File naming conventions

| Rule | Example |
|------|---------|
| `lowercase-kebab-case` for all files | `moe-loading-best-practices.md` |
| No model prefix on filenames | `run-benchmarks.py` not `run-kimi-benchmarks.py` |
| Flat configs, one file per variant | `configs/lmcache.sh` not `configs/lmcache/run.sh` |
| Single consolidated report | `results/benchmark-report.md` |

**Blueprint naming**: `<model>-<variant>` or `<purpose>-<details>`

Examples:
- `ministral-3b` — Ministral-3B inference
- `kimi-k2.5` — Kimi K2.5 MoE benchmarking

## Specs vs Blueprints

| Concern | Lives in | Example |
|---------|----------|---------|
| Requirements | `specs/<name>.md` | What to deploy, success criteria |
| Infrastructure | `blueprints/<name>/*.tf` | Terraform code |
| Lessons learned | `blueprints/<name>/lessons.md` | Operational gotchas |
| Launch configs | `blueprints/<name>/configs/` | Per-variant shell scripts |
| Benchmark tooling | `blueprints/<name>/scripts/` | Orchestrators, validators |
| Benchmark outputs | `blueprints/<name>/results/` | Reports, raw JSON, execution logs |
| Knowledge | `blueprints/<name>/docs/` | Reference, decisions, assessments |

Specs stay at the repo root because they're authored before the blueprint exists. Everything operational belongs with the blueprint.

## Spec Files

Specs define requirements and are the input to blueprint creation:

```
specs/
├── _template.md                  # Template for new specs
├── ministral-3b.md               # Ministral-3B requirements
└── vllm-kv-cache-benchmark.md    # KV cache benchmark requirements
```

Spec lifecycle:
1. Define requirements before coding (use `_template.md`)
2. Deploy via `/ralph-loop Deploy specs/<name>.md`
3. Operational artifacts go into the blueprint, not the spec

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

1. Create spec in `specs/<name>.md` using `_template.md`
2. Create `blueprints/<name>/` directory
3. Compose existing modules in `main.tf`
4. Add `variables.tf`, `outputs.tf`, `README.md`
5. Deploy and validate end-to-end
6. Add `lessons.md` with operational findings
7. Add `results/` and `benchmarks/` as work progresses
