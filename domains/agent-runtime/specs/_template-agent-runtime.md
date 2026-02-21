# Agent Runtime Spec: [Name]

> Copy this template to `domains/agent-runtime/specs/<name>.md` and fill in each section.
> Delete placeholder text and this instruction block before committing.

## Overview

Brief description of the agent: what it does, who uses it, and what AWS services it relies on.

## Agent Configuration

| Parameter | Value |
|-----------|-------|
| Foundation model | e.g., `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| AgentCore Runtime | Bedrock AgentCore Runtime (managed) |
| Idle session TTL | e.g., 600 seconds |
| Max concurrent sessions | e.g., 10 |
| AWS region | e.g., `us-east-1` |

## Memory Backend

Describe how session state is stored:
- **Session storage**: DynamoDB table (agent-memory module) or AgentCore managed memory
- **Conversation history**: Retained for N turns / N tokens
- **Persistent memory**: Yes/No — if yes, describe schema

## Auth / Identity

| Component | Configuration |
|-----------|---------------|
| User pool | Cognito user pool (cognito-app-auth module) |
| Auth flows | `USER_PASSWORD_AUTH`, `REFRESH_TOKEN_AUTH` |
| Token type | ID token (JWT), passed as `Authorization: Bearer` header |
| Multi-tenancy | Single-tenant / Multi-tenant (describe isolation model if multi-tenant) |
| Test identity | Describe test user provisioning approach |

## Tool Integrations

List each tool the agent has access to:

| Tool | AWS Service | Permission scope |
|------|-------------|-----------------|
| e.g., `get_document` | S3 | `s3:GetObject` on specific bucket ARN |
| e.g., `run_query` | Athena | `athena:StartQueryExecution` on specific workgroup |

## WebSocket Proxy

| Parameter | Value |
|-----------|-------|
| Runtime | ECS Fargate (ARM64) / AppRunner |
| Image | `domains/agent-runtime/blueprints/<name>/docker/Dockerfile` |
| Port | 8080 (default) |
| Health check path | `/health` |
| Message format | JSON: `{"message": "<text>", "sessionId": "<optional>"}` |

## Infrastructure Modules

| Module | Source | Notes |
|--------|--------|-------|
| Networking | `domains/agent-runtime/modules/agentcore-runtime` | VPC, subnets, VPC endpoints |
| AgentCore Runtime | `domains/agent-runtime/modules/agentcore-runtime` | Bedrock AgentCore resource |
| Auth | `domains/agent-runtime/modules/cognito-app-auth` | User pool + app client |
| Session storage | `domains/agent-runtime/modules/agent-memory` | DynamoDB session table |
| WebSocket proxy | `domains/agent-runtime/modules/websocket-proxy` | ECS service + task def |

## Success Criteria

Define what "deployed and working" means for this blueprint:

- [ ] AgentCore Runtime status is `PREPARED`
- [ ] Cognito user pool active, test user in `CONFIRMED` state
- [ ] WebSocket proxy ECS service at desired count
- [ ] End-to-end test: Cognito token → WebSocket → agent response in < 30s
- [ ] All readiness audit checks PASS or have documented PENDING rationale

## Non-Requirements

List things explicitly out of scope for this deployment:
- No multi-region
- No HA / failover (single AZ for prototype)
- No production Cognito email/phone verification
- (add others as appropriate)

## Known Limitations

Update this section during development as issues are discovered:
- (none yet)

## Spec history

| Date | Change |
|------|--------|
| <!-- date --> | Initial spec created |
