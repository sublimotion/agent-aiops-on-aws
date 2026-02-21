# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in `specs/<name>.md`
2. Run `/ralph-loop Deploy specs/<name>.md`
3. Claude iterates until deployment succeeds
4. Capture lessons in `blueprints/<name>/lessons.md`
5. Run compound step: invoke `compound-learner` agent with the blueprint name to elevate cross-cutting lessons to `.claude/steering/*.md`

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| Deploying or modifying a GPU-serving blueprint | `specs/<matching-spec>.md` |
| Deploying an agent-runtime blueprint | `domains/agent-runtime/specs/<name>.md` |
| Running the compound step after deployment | `blueprints/<name>/lessons.md` + `.claude/steering/*.md` |

**Spec routing** — match blueprints to specs by name:

| Blueprint | Spec |
|-----------|------|
| `blueprints/ministral-3b/` | `specs/ministral-3b.md` |
| `blueprints/kimi-k2.5/` | `specs/kimi-k2.5.md` |

**Blueprint-local context** — for operational details (lessons, results, plans), look inside the blueprint directory itself rather than in specs.

## Domain Routing

The repo is organized into domains. Each domain has its own specs, blueprints, and deployer agent.

| Domain | Spec location | Blueprint location | Deployer agent |
|--------|---------------|--------------------|----------------|
| GPU Serving (default) | `specs/` | `blueprints/` | `infra-deployer` |
| Agent Runtime | `domains/agent-runtime/specs/` | `domains/agent-runtime/blueprints/` | `agentcore-deployer` |

## Commands

```bash
# Start RALPH loop
/ralph-loop <task description>

# Cancel loop
/ralph-loop:cancel-ralph

# Validate
pre-commit run -a
checkov -d .
terraform fmt -recursive
```
