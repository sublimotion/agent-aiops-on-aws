# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in `specs/<name>.md`
2. Run `/ralph-loop Deploy specs/<name>.md`
3. Claude iterates until deployment succeeds
4. Update spec with lessons learned

## Context Loading

Read these files **on demand** based on what you're doing:

| When you are... | Read |
|-----------------|------|
| Writing or modifying Terraform | `.claude/steering/tech-stack.md` |
| Making architectural or product decisions | `.claude/steering/product.md` |
| Creating new files, modules, or blueprints | `.claude/steering/project-structure.md` |
| Deploying or modifying a specific blueprint | `specs/<matching-spec>.md` |

**Spec routing** — match blueprints to specs by name:

| Blueprint | Spec |
|-----------|------|
| `blueprints/ministral-3b/` | `specs/ministral-3b.md` |
| `blueprints/kv-cache-benchmark/` | `specs/vllm-kv-cache-benchmark.md` |

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
