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
│   │   ├── carryover-auditor.md   # Prior-lesson carryover auditor
│   │   ├── spec-writer.md         # Spec authoring
│   │   ├── benchmark-analyst.md   # Results analysis
│   │   ├── infra-deployer.md      # GPU-serving deployment (8-stage)
│   │   ├── agentcore-deployer.md  # Agent Runtime deployment (8-stage)
│   │   ├── autoresearch-runner.md # Autoresearch experiment loop (8-stage)
│   │   └── compound-learner.md    # Post-deployment lesson elevation
│   ├── skills/                 # Custom skills
│   │   ├── terraform-automation/ # Terraform deployment with security scanning
│   │   ├── visual-explainer/   # Render dense output as interactive HTML
│   │   ├── deployment-orchestrator/ # Pre-flight, post-deploy, failure recovery
│   │   ├── benchmark-runner/   # LLM serving benchmark planning, execution, analysis
│   │   └── tests/              # Skill trigger + functional tests
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
│   │   │   ├── qwen3-next/    # Qwen3-Next MoE on p5en (8x H200)
│   │   │   ├── qwen3-next-custbench/ # Customer A/B benchmark + KV offloading
│   │   │   ├── qwen3-next-sglang/ # SGLang + HiCache on g7e (coding agent feasibility)
│   │   │   ├── ray-serve-ft/     # Ray Serve fault tolerance with ElastiCache GCS FT
│   │   │   └── ray-serve-video/ # Multi-framework video pipeline (PT + TF, Kafka, in-memory)
│   │   └── specs/              # GPU Serving specs
│   │       ├── _template.md    # Template for new specs
│   │       ├── ministral-3b.md # Ministral-3B requirements
│   │       ├── kimi-k2.5.md   # KV cache benchmark requirements
│   │       ├── qwen3-next.md  # Qwen3-Next KV cache benchmark
│   │       ├── qwen3-next-custbench.md # Customer A/B benchmark spec
│   │       ├── qwen3-next-sglang.md # SGLang + HiCache coding agent feasibility
│   │       ├── ray-serve-ft.md    # Ray Serve fault tolerance with ElastiCache
│   │       └── ray-serve-video.md # Multi-framework video pipeline with Kafka
│   ├── agent-runtime/          # Bedrock AgentCore Runtime domain
│   │   ├── modules/            # Reusable Terraform modules for agent runtime
│   │   │   ├── agentcore-runtime/  # Bedrock AgentCore Runtime resource
│   │   │   ├── cognito-app-auth/   # User pool + app client
│   │   │   ├── websocket-proxy/    # Node.js proxy on ECS Fargate
│   │   │   └── agent-memory/       # DynamoDB session state
│   │   ├── blueprints/         # Agent Runtime blueprints
│   │   │   └── research-agent/ # Multi-agent research system on AgentCore Runtime
│   │   └── specs/              # Agent Runtime specs
│   │       ├── _template-agent-runtime.md
│   │       └── research-agent.md
│   └── autoresearch/           # Autonomous experiment loop domain
│       ├── specs/
│       │   ├── _template.md
│       │   ├── training-recipes.md  # GPT-2 training recipe optimization
│       │   ├── agent-harness.md     # Turn degradation + multi-harness comparison
│       │   ├── finetuning-recipes.md # LoRA/QLoRA fine-tuning with Unsloth (Qwen3-0.6B)
│       │   └── finetuning-recipes-1.7b.md # LoRA/QLoRA fine-tuning with Unsloth (Qwen3-1.7B)
│       └── blueprints/
│           ├── training-recipes/    # autoresearch-colab on g7e
│           │   ├── program.md       # Agent loop instructions
│           │   ├── scripts/         # Setup and launch scripts
│           │   ├── lessons.md
│           │   └── results/
│           ├── agent-harness/       # Coding agent harness optimization
│           │   ├── program.md       # Phase 1 + Phase 2 experiment instructions
│           │   ├── README.md        # Overview and references
│           │   ├── lessons.md
│           │   ├── scripts/         # Evaluation scripts
│           │   │   ├── harness_eval.py       # Phase 1: turn degradation evaluator
│           │   │   ├── multi_harness_eval.py # Phase 2: multi-harness comparison
│           │   │   ├── setup_vllm.sh         # vLLM serving startup
│           │   │   └── adapters/             # Per-harness adapter scripts
│           │   └── results/
│           └── finetuning-recipes/  # LoRA/QLoRA fine-tuning with Unsloth
│               ├── program.md       # Agent loop instructions
│               ├── README.md        # Architecture and quick start
│               └── lessons.md
│
├── standards/                  # Cross-domain "commons": declarative spec → pure resolver → fail-closed
│   ├── benchmark-commons/      # Workload card → benchmark argv (compile_card; raises UnsupportedWorkload)
│   │   ├── runner/             # registry.py + compiler.py + platforms/ + tests/
│   │   └── workloads/          # 7 standard workload cards (catalog source of truth)
│   └── serving-commons/        # Serving config → validated config (compile_serving_config; raises InvalidServingConfig)
│       └── resolver/           # model.py + registry.py + compiler.py + corpus.py + CLI + tests/
│
├── scripts/                    # Shared utility scripts
│   └── stage-images-ecr.sh    # Mirror images to private ECR
│
├── CLAUDE.md                   # Root context (routing layer)
└── README.md                   # Project documentation
```

## Standards (the "commons" tier)

`standards/` holds the deterministic, fail-closed resolvers shared across domains.
Each follows the same shape — a **declarative input** is parsed into a typed model,
a **pure compiler** runs a **registry** of rules over it, and an **invalid input
raises** rather than silently degrading. A stdlib `unittest` conformance suite is
the contract: green = correct by construction. This is where hard-won knowledge
gets codified so an LLM cannot reinterpret it.

| Package | Input | Output | Raises on | Gate |
|---------|-------|--------|-----------|------|
| `benchmark-commons` | workload card + benchmark.yaml sidecar | compiled benchmark argv | `UnsupportedWorkload` | benchmark-runner skill |
| `serving-commons` | benchmark.yaml sidecar (+ mdc card, + lessons corpus) | `ValidationReport` (fail/warn/info) | `InvalidServingConfig` | infra-deployer **Stage 0c** |

`serving-commons/resolver/corpus.py` is the I/O boundary that harvests every
blueprint's `lessons.md` field-note frontmatter — its `CATEGORY_TO_RULE` dict is
the **single source of truth** for the `failure_categories` vocabulary (kept in
sync with `docs/card-format.md` by the conformance test). To add a serving rule,
see `standards/serving-commons/CONTRIBUTING.md`.

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
├── lessons.md          # Operational lessons (append-only, grows over time)
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
    ├── progress.md                     # Lifecycle progress tracker (live + reconstructable)
    ├── readiness-audit-<YYYYMMDD>.md   # Pre-flight checklist (Stage 7)
    ├── deployment-log-<YYYYMMDD>.md    # Timestamped deployment log
    ├── compound-<YYYYMMDD>.md          # Compound learner summary (Stage 8)
    ├── benchmark-report.md             # Benchmark analysis (if applicable)
    └── <visual>-<YYYYMMDD>.html        # Interactive HTML reports
```

**Required artifacts**: Every deployment must produce `lessons.md`, `results/readiness-audit-*.md`, `results/deployment-log-*.md`, `results/compound-*.md`, and `results/progress.md`. See `domains/gpu-serving/specs/_template-artifacts.md` for templates and `docs/progress-format.md` for the progress schema.

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

## External Dependencies

### Model Deployment Cards (mdc)

Sibling repo at `../model-deployment-card/`. Provides curated deployment recipes, upstream PR tracking, and tribal knowledge for LLM serving engines (vLLM, SGLang).

- **Before deploying**: `mdc get <model> --engine <engine>` loads the card; `mdc prs <model>` checks upstream PRs
- **After deploying**: `mdc learn <model> <engine> "<note>"` feeds lessons back to the card
- **When writing specs**: `mdc get` pre-fills recommended flags and known issues

The `infra-deployer` agent runs `mdc get` as Stage 0 before any infrastructure work. The `compound-learner` agent runs `mdc learn` to close the feedback loop.

### GPU Infrastructure (gpu-infra)

Sibling repo at `../gpu-infra-troubleshooting/`. MCP server for live diagnostics + CLI for feedback. Config in `.mcp.json`.

- **Stage 4a (proactive)**: MCP tools `discover_cluster`, `check_gpu_health`, `run_nccl_test` validate hardware before deploying the serving stack
- **Reactive**: MCP tools `explain_xid` for Xid error lookup, `get_gpu_metrics` for Prometheus/DCGM metrics
- **After deploying**: `gpu-infra learn -c <category> "<note>"` feeds hardware/platform lessons into `field-notes.md` inbox
- **Review**: `gpu-infra inbox` shows pending notes for triage into reference docs

The `infra-deployer` agent uses MCP tools at Stage 4a. The `compound-learner` agent uses `gpu-infra learn` to feed back hardware lessons.

## Adding a New Blueprint

1. Run `mdc get <model> --engine <engine>` to check for an existing deployment card
2. Write spec in `domains/<domain>/specs/<name>.md` (incorporate card recommendations)
3. Run `/ralph-loop:ralph-loop Deploy domains/<domain>/specs/<name>.md`
4. Claude creates `domains/<domain>/blueprints/<name>/` with terraform files
5. Deployment succeeds → capture lessons in `lessons.md`
6. Run compound-learner to elevate cross-cutting lessons to steering files and feed model-specific lessons back to `mdc learn`
7. Update this file's repository layout tree if the blueprint introduces new patterns
