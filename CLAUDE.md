# Agent AIOps on AWS

Template for autonomous infrastructure deployment using Claude Code, spec-driven development, and RALPH loops.

## Workflow

1. Write spec in `specs/<name>.md`
2. Run `/ralph-loop Deploy specs/<name>.md`
3. Claude iterates until deployment succeeds
4. Update spec with lessons learned

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `specs/` | Input specifications (requirements) |
| `modules/` | Reusable Terraform modules |
| `blueprints/` | Deployable compositions |
| `.claude/steering/` | Persistent context files |

## Steering Files

- `product.md` - Business context, quality standards
- `tech-stack.md` - Technology preferences, conventions
- `project-structure.md` - Layout, naming patterns

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

## Conventions

- Prefer AWSCC provider, fallback to AWS
- Use `bucket_prefix` over hardcoded names
- Enable encryption on all storage
- Default features to `false`
