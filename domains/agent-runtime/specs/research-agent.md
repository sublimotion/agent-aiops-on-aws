# Agent Runtime Spec: Research Agent

## Overview

A multi-agent research system that accepts a natural language query and produces a structured
PDF report. The lead agent decomposes the query into 2–4 subtopics, spawns parallel researcher
subagents that gather quantitative data via web search, passes findings through a data analyst
for visualization, and delivers a synthesized PDF report.

Based on: `anthropics/claude-agent-sdk-demos/research-agent`

Exposed to end-users via **AgentCore Gateway as an MCP server** — Claude Desktop or Claude
Code connects to the gateway endpoint and invokes `research(<query>)` as a tool. No separate
frontend is required.

## Agent Configuration

| Parameter | Value |
|-----------|-------|
| Foundation model | `us.anthropic.claude-opus-4-6-20250514-v1:0` |
| AgentCore Runtime | Amazon Bedrock AgentCore Runtime (managed container) |
| Agent architecture | Multi-agent: Lead + Researcher × N + Data Analyst + Report Writer |
| Idle session TTL | 1800 seconds (30 min — research runs can be long) |
| Max concurrent sessions | 5 |
| AWS region | `us-east-1` |

## Agent Architecture (4 roles)

| Role | Spawned by | Tools | Output |
|------|-----------|-------|--------|
| **Lead Agent** | HTTP /invoke endpoint | Task only | Orchestrates all others |
| **Researcher** (× 2–4, parallel) | Lead via Task tool | WebSearch, Write | `research_notes/*.md` on EFS |
| **Data Analyst** (× 1) | Lead via Task tool | Glob, Read, Bash, Write | `data/data_summary.md`, `charts/*.png` on EFS |
| **Report Writer** (× 1) | Lead via Task tool | Glob, Read, Bash, Write, Skill (pdf) | `reports/<topic>_YYYYMMDD.pdf` → S3 |

## Memory Backend

| Component | Configuration |
|-----------|---------------|
| Inter-agent file coordination | EFS volume mounted at `/app/files` in ECS task |
| Session state | DynamoDB table (agent-memory module), session_id hash key, TTL 2h |
| Conversation history | Single-turn per invocation (lead agent holds context in-process) |
| Persistent memory | No (research notes persist on EFS within a session; cleared after report is uploaded to S3) |

**File layout on EFS** (mirrors original local structure):
```
/app/files/
├── research_notes/   ← Researcher outputs (markdown)
├── data/             ← Data Analyst summary
├── charts/           ← PNG visualizations (matplotlib)
└── reports/          ← Temporary PDF staging before S3 upload
```

## Auth / Identity

| Component | Configuration |
|-----------|---------------|
| Primary UI path | MCP protocol via AgentCore Gateway — no separate auth layer |
| AgentCore Gateway auth | IAM-based (gateway validates AWS SigV4 from MCP client config) |
| Direct API callers | Cognito user pool (cognito-app-auth module) for non-MCP HTTP clients |
| Auth flows | `ALLOW_USER_PASSWORD_AUTH`, `ALLOW_REFRESH_TOKEN_AUTH` |
| Token type | ID token (JWT) via `Authorization: Bearer` header for HTTP path |
| Test identity | Single test user provisioned via `aws cognito-idp admin-create-user` |

**Claude Desktop MCP config** (end-user setup, post-deploy):
```json
{
  "mcpServers": {
    "research-agent": {
      "url": "<agentcore_gateway_mcp_endpoint_url>",
      "headers": { "Authorization": "Bearer <cognito_id_token>" }
    }
  }
}
```

## Tool Integrations

| Tool | Source | AWS Service | Permission scope |
|------|--------|-------------|-----------------|
| `WebSearch` | claude-agent-sdk built-in | Secrets Manager (Brave API key) | `secretsmanager:GetSecretValue` on search key ARN |
| `Write` | claude-agent-sdk built-in | EFS (via ECS mount) | `elasticfilesystem:ClientMount`, `ClientWrite` |
| `Read` / `Glob` | claude-agent-sdk built-in | EFS (via ECS mount) | `elasticfilesystem:ClientMount`, `ClientRead` |
| `Bash` | claude-agent-sdk built-in | Container process | matplotlib, reportlab installed in image |
| `Task` | claude-agent-sdk built-in | AgentCore Runtime (sub-invoke) | `bedrock:InvokeModel` on Opus 4.6 model ARN |
| `Skill` (pdf) | `.claude/skills/pdf/` | Container process (reportlab) | None |
| S3 upload | Report Writer (Bash script) | S3 output bucket | `s3:PutObject` on `research-output-*` bucket |

## WebSocket Proxy / HTTP Server

| Parameter | Value |
|-----------|-------|
| Runtime | ECS Fargate (ARM64 / Graviton) |
| Image | `domains/agent-runtime/blueprints/research-agent/docker/Dockerfile` |
| Port | 8080 |
| Health check path | `/health` |
| Invoke path | `POST /invoke` |
| Response format | Server-Sent Events (SSE), `text/event-stream` |
| Message format (request) | `{"query": "<string>", "session_id": "<optional string>"}` |
| Message format (response) | SSE stream of `{"type": "text"|"tool_use"|"result", "content": "..."}` |

## Infrastructure Modules

| Module | Source | Notes |
|--------|--------|-------|
| Networking | `domains/agent-runtime/modules/agentcore-runtime` | VPC, private subnets, VPC endpoints |
| AgentCore Runtime | `domains/agent-runtime/modules/agentcore-runtime` | Bedrock agent resource + alias |
| AgentCore Gateway | `domains/agent-runtime/modules/agentcore-gateway` | MCP server endpoint (new module) |
| Auth | `domains/agent-runtime/modules/cognito-app-auth` | User pool + app client |
| Session storage | `domains/agent-runtime/modules/agent-memory` | DynamoDB session table |
| WebSocket proxy | `domains/agent-runtime/modules/websocket-proxy` | ECS Fargate service + ALB |

**Additional resources (blueprint-level, not in modules)**:
- `aws_efs_file_system` — inter-agent file coordination
- `aws_efs_mount_target` — one per private subnet
- `aws_s3_bucket` — research output storage, lifecycle 30d delete
- `aws_secretsmanager_secret` — Brave Search API key

## Required VPC Endpoints

| Endpoint | Type | Required for |
|----------|------|-------------|
| `com.amazonaws.us-east-1.bedrock-runtime` | Interface | AgentCore model invocation |
| `com.amazonaws.us-east-1.bedrock-agent-runtime` | Interface | AgentCore Gateway routing |
| `com.amazonaws.us-east-1.ecr.api` | Interface | Container image pull |
| `com.amazonaws.us-east-1.ecr.dkr` | Interface | Container image pull |
| `com.amazonaws.us-east-1.s3` | Gateway | Research output upload |
| `com.amazonaws.us-east-1.secretsmanager` | Interface | Brave API key retrieval |
| `com.amazonaws.us-east-1.elasticfilesystem` | Interface | EFS mount in Fargate |

## Code Changes from Source

Source: `anthropics/claude-agent-sdk-demos/research-agent`

| Change | File | What |
|--------|------|------|
| Model ID | `research_agent/agent.py` | `model="haiku"` → `model="us.anthropic.claude-opus-4-6-20250514-v1:0"` |
| Bedrock client | `research_agent/agent.py` | Set `ANTHROPIC_BEDROCK=1` + `AWS_BEDROCK_REGION=us-east-1` env vars |
| File paths | All agents | `/app/files/...` instead of relative `files/...` |
| S3 upload | `research_agent/prompts/report_writer.txt` | Add post-PDF step: `aws s3 cp /app/files/reports/<file> s3://<bucket>/` |
| HTTP wrapper | `docker/server.py` (new) | FastAPI `/invoke` endpoint + `/health` wrapping `agent.main()` |
| Remove dotenv | `research_agent/agent.py` | Delete `load_dotenv()` call |

## Success Criteria

- [ ] AgentCore Runtime status is `PREPARED` (`aws bedrock-agent get-agent` returns `agentStatus: PREPARED`)
- [ ] AgentCore Gateway MCP endpoint returns tool list containing `research` tool
- [ ] Claude Desktop shows `research-agent` in MCP server list (Settings → Developer)
- [ ] ECS service at desired count (1) with healthy ALB target group
- [ ] End-to-end test: `research("quantum computing 2025")` → PDF object in S3 within 5 minutes
- [ ] CloudWatch Logs show ≥ 3 distinct agent traces (RESEARCHER-*, DATA-ANALYST-1, REPORT-WRITER-1)
- [ ] All readiness audit checks PASS or have documented PENDING rationale

## Non-Requirements

- No multi-region deployment
- No HA / multi-AZ ECS (single AZ for prototype; EFS is multi-AZ by default)
- No production Cognito email verification flow
- No custom web frontend (Claude Desktop via MCP is the UI)
- No real-time streaming UI (SSE from ECS to Gateway; MCP delivers final result)
- No fine-grained session isolation between users (single shared EFS; acceptable for single-tenant)

## Known Limitations

- EFS file coordination is not safe for concurrent sessions writing to the same paths —
  single `desired_count = 1` on ECS mitigates this for the prototype
- `claude-agent-sdk` Bedrock support depends on `ANTHROPIC_BEDROCK=1` env var being picked up;
  verify at container build time by testing `AnthropicBedrock` client initialization
- Brave Search API has rate limits; set `desired_count = 1` to avoid parallel session conflicts
- Report PDFs are staged on EFS and uploaded to S3; if ECS task is recycled mid-run,
  partially written PDFs will not be recovered (acceptable for prototype)

## Spec history

| Date | Change |
|------|--------|
| 2026-02-21 | Initial spec created |
