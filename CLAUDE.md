# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in the correct domain location (see Domain Routing below)
2. Run `/ralph-loop:ralph-loop Deploy <spec-path>`
3. Claude selects the deployer agent automatically based on spec path (see Domain Routing)
4. Claude iterates until deployment succeeds
5. Capture lessons in the blueprint's `lessons.md`
6. Run compound step: invoke `compound-learner` agent with the blueprint path to elevate cross-cutting lessons to `.claude/steering/*.md`

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| Deploying or modifying a GPU-serving blueprint | `domains/gpu-serving/specs/<matching-spec>.md` |
| Deploying an agent-runtime blueprint | `domains/agent-runtime/specs/<name>.md` |
| Running the compound step after deployment | `domains/<domain>/blueprints/<name>/lessons.md` + `.claude/steering/*.md` |

**Spec routing** — match blueprints to specs by name:

| Blueprint | Spec |
|-----------|------|
| `domains/gpu-serving/blueprints/ministral-3b/` | `domains/gpu-serving/specs/ministral-3b.md` |
| `domains/gpu-serving/blueprints/kimi-k2.5/` | `domains/gpu-serving/specs/kimi-k2.5.md` |
| `domains/gpu-serving/blueprints/qwen3-next/` | `domains/gpu-serving/specs/qwen3-next.md` |
| `domains/agent-runtime/blueprints/research-agent/` | `domains/agent-runtime/specs/research-agent.md` |

**Blueprint-local context** — for operational details (lessons, results, plans), look inside the blueprint directory itself rather than in specs.

## Domain Routing

The repo is organized into domains. **Infer the deployer agent automatically from the spec path** — no need to specify it explicitly in the RALPH loop command.

| Domain | Spec path prefix | Blueprint location | Deployer agent |
|--------|------------------|--------------------|----------------|
| GPU Serving | `domains/gpu-serving/specs/` | `domains/gpu-serving/blueprints/` | `infra-deployer` |
| Agent Runtime | `domains/agent-runtime/specs/` | `domains/agent-runtime/blueprints/` | `agentcore-deployer` |

**Auto-detection rule**: all specs live under `domains/<name>/specs/`. Use `domains/gpu-serving/` → `infra-deployer`, `domains/agent-runtime/` → `agentcore-deployer`.

Examples:
```
/ralph-loop:ralph-loop Deploy domains/agent-runtime/specs/research-agent.md   → agentcore-deployer
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/kimi-k2.5.md          → infra-deployer
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
