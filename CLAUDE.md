# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in the correct domain location (see Domain Routing below)
2. Run `mdc get <model> --engine <engine>` to load the deployment card before deploying
3. Run `/ralph-loop:ralph-loop Deploy <spec-path>`
4. Claude selects the deployer agent automatically based on spec path (see Domain Routing)
5. Claude iterates until deployment succeeds
6. Capture lessons in the blueprint's `lessons.md` using the field note schema (see `docs/card-format.md` and `domains/gpu-serving/blueprints/LESSONS-TEMPLATE.md`)
7. Run compound step: invoke `compound-learner` agent — it elevates cross-cutting lessons to `.claude/steering/*.md` **and generates the YAML frontmatter** for `lessons.md` (model, engine, hardware, outcome, failure_categories, mdc_learn_commands, gpu_infra_learn_commands)
8. Run `scripts/fe.sh learn <blueprint-path>` to apply the learn commands from the frontmatter
9. Optionally run `scripts/fe.sh contribute <blueprint-path>` to open a community contribution issue

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| About to deploy a model | `mdc get <model> [--engine <engine>]` and `mdc prs <model>` for upstream deployment card + recent PRs |
| Deploying or modifying a GPU-serving blueprint | `domains/gpu-serving/specs/<matching-spec>.md` |
| Deploying an agent-runtime blueprint | `domains/agent-runtime/specs/<name>.md` |
| Running an autoresearch experiment | `domains/autoresearch/specs/<name>.md` |
| Running the compound step after deployment | `domains/<domain>/blueprints/<name>/lessons.md` + `.claude/steering/*.md`; compound-learner must generate YAML frontmatter (see `docs/card-format.md`) as its final step, then run `scripts/fe.sh learn <path>` |
| Running pre-flight or post-deploy checks | Use the `deployment-orchestrator` skill |
| Validating GPU hardware before serving (Stage 4a) | Use `gpu-infra` MCP tools: `discover_cluster`, `check_gpu_health`, `run_nccl_test` |
| Diagnosing GPU hardware issues | Use `gpu-infra` MCP tools: `explain_xid`, `get_gpu_metrics`, `check_gpu_health` |

**Spec routing** — match blueprints to specs by name:

| Blueprint | Spec |
|-----------|------|
| `domains/gpu-serving/blueprints/ministral-3b/` | `domains/gpu-serving/specs/ministral-3b.md` |
| `domains/gpu-serving/blueprints/kimi-k2.5/` | `domains/gpu-serving/specs/kimi-k2.5.md` |
| `domains/gpu-serving/blueprints/qwen3-next/` | `domains/gpu-serving/specs/qwen3-next.md` |
| `domains/gpu-serving/blueprints/qwen3-next-g7e/` | `domains/gpu-serving/specs/qwen3-next-g7e.md` |
| `domains/gpu-serving/blueprints/qwen3-next-custbench/` | `domains/gpu-serving/specs/qwen3-next-custbench.md` |
| `domains/gpu-serving/blueprints/qwen3-next-sglang/` | `domains/gpu-serving/specs/qwen3-next-sglang.md` |
| `domains/gpu-serving/blueprints/devstral-sera/` | `domains/gpu-serving/specs/devstral-sera.md` |
| `domains/gpu-serving/blueprints/glm5/` | `domains/gpu-serving/specs/glm5.md` |
| `domains/gpu-serving/blueprints/glm5-hyperpod/` | `domains/gpu-serving/specs/glm5-hyperpod.md` |
| `domains/gpu-serving/blueprints/glm5-lmcache/` | `domains/gpu-serving/specs/glm5-lmcache.md` |
| `domains/gpu-serving/blueprints/glm5-llmd/` | `domains/gpu-serving/specs/glm5-llmd.md` |
| `domains/gpu-serving/blueprints/nemotron-super/` | `domains/gpu-serving/specs/nemotron-super.md` |
| `domains/agent-runtime/blueprints/research-agent/` | `domains/agent-runtime/specs/research-agent.md` |
| `domains/autoresearch/blueprints/training-recipes/` | `domains/autoresearch/specs/training-recipes.md` |
| `domains/autoresearch/blueprints/agent-harness/` | `domains/autoresearch/specs/agent-harness.md` |
| `domains/autoresearch/blueprints/finetuning-recipes/` | `domains/autoresearch/specs/finetuning-recipes.md` |
| `domains/autoresearch/blueprints/agent-swarm/` | `domains/autoresearch/specs/agent-swarm.md` |

**Blueprint-local context** — for operational details (lessons, results, plans), look inside the blueprint directory itself rather than in specs.

## Domain Routing

The repo is organized into domains. **Infer the deployer agent automatically from the spec path** — no need to specify it explicitly in the RALPH loop command.

| Domain | Spec path prefix | Blueprint location | Deployer agent |
|--------|------------------|--------------------|----------------|
| GPU Serving | `domains/gpu-serving/specs/` | `domains/gpu-serving/blueprints/` | `infra-deployer` |
| Agent Runtime | `domains/agent-runtime/specs/` | `domains/agent-runtime/blueprints/` | `agentcore-deployer` |
| Autoresearch | `domains/autoresearch/specs/` | `domains/autoresearch/blueprints/` | `autoresearch-runner` |

**Auto-detection rule**: all specs live under `domains/<name>/specs/`. Use `domains/gpu-serving/` → `infra-deployer`, `domains/agent-runtime/` → `agentcore-deployer`, `domains/autoresearch/` → `autoresearch-runner`.

Examples:
```
/ralph-loop:ralph-loop Deploy domains/agent-runtime/specs/research-agent.md   → agentcore-deployer
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/kimi-k2.5.md          → infra-deployer
/ralph-loop:ralph-loop Run domains/autoresearch/specs/training-recipes.md     → autoresearch-runner
```

## External Tools

### fe CLI

Unified entry point for card lookup, field note lifecycle, and community contribution. Wraps `mdc` and `gpu-infra`.

```bash
# Before deploying — look up cards
fe card <model> --engine <engine>    # e.g. fe card glm-5 --engine sglang
fe card --hardware <instance>        # e.g. fe card --hardware p6-b200

# After compound step — apply learn commands from lessons.md frontmatter
fe learn domains/gpu-serving/blueprints/<name>/

# Contribute field note to community (opens GitHub Issue template)
fe contribute domains/gpu-serving/blueprints/<name>/
```

### Model Deployment Cards (mdc)

Curated deployment recipes with upstream PR tracking and tribal knowledge. Repo: `../model-deployment-card/`

```bash
# Before deploying — load the deployment card (also via: fe card)
mdc get <model> --engine <engine>   # e.g. mdc get glm-4.5 --engine sglang
mdc prs <model>                     # check recent upstream PRs

# After deploying — feed lessons back (also via: fe learn)
mdc learn <model> <engine> "<note>"
mdc learn <model> <engine> --from domains/gpu-serving/blueprints/<name>/lessons.md
```

### GPU Infrastructure (gpu-infra)

Live GPU diagnostics via MCP server + CLI for feedback. Repo: `../gpu-infra-troubleshooting/`. MCP config: `.mcp.json`

**MCP tools** (live diagnostics):

| Tool | When to use |
|------|-------------|
| `discover_cluster` | Stage 4a — enumerate GPUs, topology, driver, EFA before serving |
| `check_gpu_health` | Stage 4a — validate ECC, row remap, thermals, PCIe link |
| `run_nccl_test` | Stage 4a — verify NCCL collectives for multi-GPU TP |
| `explain_xid` | Any time — look up Xid error codes from dmesg |
| `get_gpu_metrics` | Benchmarking — pull DCGM metrics from Prometheus |

**CLI** (cards + feedback):

```bash
# Before deploying — load the GPU architecture card
gpu-infra card g7e                        # full card for instance type
gpu-infra cards --interconnect NVSwitch   # filter available cards

# After deploying — feed hardware lessons back
gpu-infra learn -c nccl "NCCL 2.25.1 broken on Blackwell PCIe (sm_120)"
gpu-infra learn -c platform --from path/to/lessons.md

# Review pending notes
gpu-infra inbox
```

## Commands

```bash
# Start RALPH loop
/ralph-loop:ralph-loop <task description>

# Cancel loop
/ralph-loop:cancel-ralph

# Validate
pre-commit run -a
checkov -d .
terraform fmt -recursive
```
