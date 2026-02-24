# Product Context

This workbench is a collection of AI-powered development tools and learning resources focused on:

1. **Claude Code Integration** - Plugins, skills, and automation for Claude Code
2. **AWS Infrastructure** - Terraform and CDK patterns for AWS deployment
3. **Context Engineering** - Advanced prompt engineering and agent patterns

## Primary Use Cases

- Automating AWS infrastructure deployment with Terraform
- Building Claude Code plugins and skills
- Learning and experimenting with AI agent patterns

## Quality Standards

- Security-first approach (Checkov scans required)
- Infrastructure as Code best practices
- Test-driven development where applicable
- Pre-commit hooks must pass before merge

## AgentCore Runtime Design Principles

### AgentCore Runtime is not ECS

AgentCore Runtime has no task definition. This means: no EFS volume mounts, no native Secrets Manager injection, no `awslogs` log driver. Design agent containers assuming these ECS primitives are unavailable and substitute accordingly:
- Secrets: load from Secrets Manager in application code at startup
- Logs: instrument with OTEL SDK (`localhost:4318`)
- Persistent output: upload to S3 from `server.py` after each invocation

### Use S3 as the agent output bus, not shared filesystems

Agent invocation results (reports, artifacts) must be uploaded to S3 before the MCP response is returned. Consumers poll or subscribe to S3; do not rely on any shared filesystem between the invoker and the container.
This ensures outputs survive beyond the container's ephemeral storage and are accessible to callers regardless of AgentCore's container lifecycle management.

## Contribution Workflow

1. **Open Issue First** - Discuss significant changes before starting work
2. **Setup Hooks** - Run `pre-commit install` after cloning
3. **Validate Locally** - Run `pre-commit run -a` before commits
4. **Include Evidence** - Add test output or screenshots in PRs
5. **Security Scan** - Ensure `checkov -d .` passes
6. **Format Code** - Run `terraform fmt -recursive`
