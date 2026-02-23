# research-agent blueprint

Deploys the `claude-agent-sdk-demos/research-agent` to **Amazon Bedrock AgentCore Runtime**
and exposes it as an MCP server via **AgentCore Gateway**.

The user interface is Claude Desktop — no separate frontend is required.
Research queries take 10-15 minutes; the server streams NDJSON progress lines to keep
clients alive during the run (Lesson #16).

## Architecture

```
Claude Desktop
  │  tools/call { query }           MCP JSON-RPC over stdio
  │◄── notifications/progress ...   live progress while pipeline runs
  │◄── tools/call response          final result after pipeline completes
  ▼
mcp-proxy.py   (local stdio process, configured in claude_desktop_config.json)
  │  POST /invocations  (MCP JSON-RPC payload)
  │  accept: application/x-ndjson
  │◄── NDJSON stream:
  │     {"type":"progress","label":"Research pipeline starting..."}
  │     {"type":"progress","label":"Research in progress..."}     ← every 30 s
  │     {"type":"progress","label":"Uploading research outputs..."}
  │     {"jsonrpc":"2.0","id":1,"result":{...}}                   ← final line
  ▼
AgentCore Runtime  (managed container, HTTP serverProtocol)
  │  POST /invocations → StreamingResponse(application/x-ndjson)
  ▼
server.py  (FastAPI, port 8080)
  │  _stream_tools_call() async generator
  │  asyncio.create_task(_collect())   ← runs run_query in background
  │  yields progress ticks every 30 s while task runs
  ▼
research_agent/agent.py  (Lead → Researcher×N → Data Analyst → Report Writer)
  │  Researcher uses mcp__search__web_search tool
  │
  ├─► mcp_search_server.py  (MCP stdio server, started per sub-agent call)
  │     Tavily API (primary)  →  structured search results
  │     Brave API  (fallback) →  used if Tavily fails/rate-limits
  │
  └─► S3: s3://research-agent-output-.../sessions/<id>/
```

Supporting services: S3 (PDF output) · DynamoDB (sessions) ·
Cognito (auth) · Secrets Manager (Tavily + Brave API keys) · ECR (container image)

## Current deployment

| Parameter | Value |
|-----------|-------|
| Runtime ARN | `arn:aws:bedrock-agentcore:us-east-1:615299764834:runtime/research_agent-fyUZrR80VG` |
| Endpoint | `research_agent_endpoint` |
| Live version | **28** (Tavily primary search + Brave fallback via MCP server) |
| ECR image | `615299764834.dkr.ecr.us-east-1.amazonaws.com/research-agent:v28` |
| S3 output bucket | `research-agent-output-20260221145504871200000003` |

## Deploy

### 1. Prerequisites

```bash
# AWS credentials with Admin / PowerUser scope
export AWS_DEFAULT_REGION=us-east-1

# Tavily Search API key (primary — get one at tavily.com)
export TF_VAR_tavily_api_key="tvly-..."

# Brave Search API key (fallback — get one at brave.com/search/api)
export TF_VAR_brave_api_key="BSA..."
```

### 2. Build and push the container

No local container runtime? Use the S3 + SSM build approach (Lessons #2, #17):

```bash
cd domains/agent-runtime/blueprints/research-agent

# Package build context
tar czf /tmp/docker-context.tar.gz docker/

# Upload to S3 (bucket already exists after terraform apply)
S3_BUCKET=$(terraform output -raw s3_output_bucket)
aws s3 cp /tmp/docker-context.tar.gz s3://${S3_BUCKET}/build/docker-context.tar.gz

# Launch ARM64 build instance in default VPC (needs internet for base image pulls)
DEFAULT_SUBNET=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' --output text)

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-arm64" "Name=architecture,Values=arm64" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text) \
  --instance-type t4g.medium \
  --subnet-id $DEFAULT_SUBNET \
  --iam-instance-profile Name=research-agent-build-instance \
  --associate-public-ip-address \
  --user-data '#!/bin/bash
yum install -y docker && systemctl start docker && touch /tmp/ready' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=research-agent-build}]' \
  --query 'Instances[0].InstanceId' --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID
# Wait for SSM (poll until Online), then:

ECR_URL=$(terraform output -raw ecr_repository_url)
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --timeout-seconds 1200 \
  --parameters "commands=[
    \"aws s3 cp s3://${S3_BUCKET}/build/docker-context.tar.gz /tmp/ctx.tar.gz\",
    \"mkdir -p /tmp/build && tar xzf /tmp/ctx.tar.gz -C /tmp/build\",
    \"aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_URL}\",
    \"docker build --platform linux/arm64 -t ${ECR_URL}:latest /tmp/build/docker/\",
    \"docker push ${ECR_URL}:latest\"
  ]"

# Terminate when done
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

### 3. Full apply

```bash
terraform apply
```

### 4. Update existing runtime (skip terraform apply)

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/research-agent"
RUNTIME_ID="research_agent-fyUZrR80VG"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/research-agent-agentcore-exec"

# Update runtime → creates new version
NEW_VERSION=$(aws bedrock-agentcore-control update-agent-runtime \
  --agent-runtime-id $RUNTIME_ID \
  --role-arn $ROLE_ARN \
  --network-configuration '{"networkMode":"PUBLIC"}' \
  --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${ECR_URL}:latest\"}}" \
  --query 'agentRuntimeVersion' --output text)

echo "Created version $NEW_VERSION — waiting for READY..."
aws bedrock-agentcore-control wait ... # poll get-agent-runtime until status=READY

# Point endpoint at new version
aws bedrock-agentcore-control update-agent-runtime-endpoint \
  --agent-runtime-id $RUNTIME_ID \
  --endpoint-name research_agent_endpoint \
  --agent-runtime-version $NEW_VERSION
```

### 5. Add to Claude Desktop

```bash
# In mcp-proxy.py, RUNTIME_ARN and QUALIFIER are already set.
# Add to ~/.config/claude/claude_desktop_config.json:
```

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python3",
      "args": ["/path/to/domains/agent-runtime/blueprints/research-agent/mcp-proxy.py"]
    }
  }
}
```

## Verification

```bash
# Runtime + endpoint status
aws bedrock-agentcore-control get-agent-runtime \
  --agent-runtime-id research_agent-fyUZrR80VG \
  --query '{version:agentRuntimeVersion,status:status}'

aws bedrock-agentcore-control get-agent-runtime-endpoint \
  --agent-runtime-id research_agent-fyUZrR80VG \
  --endpoint-name research_agent_endpoint \
  --query '{live:liveVersion,status:status}'

# NDJSON streaming smoke test (should see 3+ progress lines then final result)
python3 - <<'EOF'
import boto3, json
from botocore.config import Config
client = boto3.client("bedrock-agentcore", region_name="us-east-1",
    config=Config(read_timeout=120, retries={"max_attempts": 1}))
resp = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:615299764834:runtime/research_agent-fyUZrR80VG",
    qualifier="research_agent_endpoint",
    payload=json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call",
        "params":{"name":"research","arguments":{"query":"What is NDJSON?"},
        "_meta":{"progressToken":"smoke-1"}}}).encode(),
    contentType="application/json", accept="application/x-ndjson")
for line in resp["response"].iter_lines():
    obj = json.loads(line)
    if obj.get("type") == "progress":
        print("PROGRESS:", obj["label"])
    elif "result" in obj:
        print("RESULT:", obj["result"]["content"][0]["text"][:200])
EOF
```

## Spec

`domains/agent-runtime/specs/research-agent.md`
