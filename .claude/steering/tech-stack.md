# Technology Stack

> This file covers conventions for all domains. See section headers to find the right section for your domain.

## GPU Serving Conventions

### Infrastructure

| Technology | Purpose | Preference |
|------------|---------|------------|
| **Terraform** | Infrastructure as Code | Primary |
| **AWS CDK** | Infrastructure as Code | Secondary |
| **CloudFormation** | Infrastructure as Code | Avoid (use Terraform/CDK) |

### Deployment Conventions

#### Single-node GPU deployments: scale to 0 before changing GPU resource requests

When changing GPU resource allocation on a single-node Kubernetes deployment (e.g., TP=4 to TP=8), scale to 0 replicas before applying Terraform changes, then scale back to 1. Rolling updates cannot work when the new pod requires more GPUs than are available after the old pod's allocation is accounted for. This prevents scheduling deadlocks where the new pod waits indefinitely for resources held by the old pod.

```bash
kubectl -n <namespace> scale deployment <name> --replicas=0
terraform apply -target='<deployment_resource>' -auto-approve
kubectl -n <namespace> scale deployment <name> --replicas=1
```

#### Air-gapped serving environments require local tokenizer paths for benchmarking

When `HF_HUB_OFFLINE=1` is set in the serving container (air-gapped, no HuggingFace Hub access), benchmark tools like `vllm bench serve` must use `--tokenizer /path/to/local/model` to point at the local model directory. The `--model` flag specifies the API-facing served model name, not the filesystem path.

#### Always document benchmark execution location before running

Record whether benchmarks run via `kubectl port-forward` (from local machine) or server-side (inside the cluster via `kubectl exec`). Port-forward benchmarks measure client → API server → pod latency; server-side benchmarks measure pod-local inference latency only. This distinction is critical for interpreting TTFT and E2E latency results.

#### FP8 quantization compatibility check for MoE models

Before reserving GPU capacity for Mixture-of-Experts models with FP8 quantization, verify that all weight dimensions (including shared experts) remain divisible by `block_k` (typically 128) at the target tensor parallelism degree. Example: if a shared expert MLP `down_proj` has `input_size=512`, TP=8 produces `input_size_per_partition=64`, which is not divisible by 128 and will cause a ValueError at model load time. Test TP compatibility on a CPU-only or smaller GPU instance before committing to a capacity block.

#### Budget for JIT compilation startup time on first-run serving stacks

Serving frameworks with JIT compilation (e.g., SGLang's DeepGEMM, TensorRT-LLM engine builds) can take 10-15 minutes for first-time startup. Subsequent restarts are faster if the JIT cache is preserved. In capacity-block benchmarking scenarios, include this warmup time in the session plan to avoid losing billable GPU hours to compilation overhead. Consider pre-compiling before the capacity block starts if the serving framework supports offline compilation.

#### For MoE models, favor tensor parallelism over data parallelism with expert parallelism at single-node scale

When serving Mixture-of-Experts models with many experts (hundreds) on a single multi-GPU node, tensor parallelism typically outperforms data parallelism with expert parallelism. Expert parallelism requires cross-GPU communication for MoE routing at every layer, which adds significant latency overhead when each GPU runs a full replica with TP=1. Tensor parallelism keeps MoE routing local to each GPU's shard and benefits from weight distribution. Data parallelism with expert parallelism may become competitive in multi-node deployments where TP cannot efficiently span nodes, but at single-node scale, prioritize TP. Benchmark both configurations if the model fits in memory with either approach.

## Terraform Conventions

### Provider Priority

1. **AWSCC Provider** - Prefer for consistent API behavior
2. **AWS Provider** - Use when AWSCC doesn't support resource

### Resource Naming

```hcl
# DO: Let AWS generate unique names
resource "aws_s3_bucket" "data" {
  bucket_prefix = "myapp-data-"
}

# DON'T: Hardcode names
resource "aws_s3_bucket" "data" {
  bucket = "myapp-data-bucket"  # Avoid
}
```

### Security Defaults

- Enable encryption on all storage (S3, RDS, EBS)
- Block public access on S3 buckets
- Use least-privilege IAM policies
- Enable versioning on S3
- Enable flow logs on VPCs

## Languages

| Language | Use Case |
|----------|----------|
| **HCL** | Terraform configurations |
| **TypeScript** | CDK, Claude plugins |
| **Python** | Scripts, automation |
| **Bash** | CI/CD, simple automation |

## Tools

| Tool | Purpose |
|------|---------|
| **Checkov** | Security scanning |
| **terraform fmt** | Code formatting |
| **terraform validate** | Syntax validation |
| **tfsec** | Additional security scanning |
| **pre-commit** | Git hooks for quality gates |
| **tflint** | Terraform linting |
| **terraform-docs** | Auto-generate documentation |

## Pre-commit Hooks

Required hooks for all commits:

| Hook | Purpose |
|------|---------|
| `terraform fmt` | Enforce consistent formatting |
| `terraform validate` | Syntax validation |
| `tflint` | Terraform best practices |
| `terraform-docs` | Auto-generate module docs |
| `checkov` | Security scanning |
| `tfsec` | Additional security checks |
| `trufflehog` | Secret detection |
| `detect-aws-credentials` | Prevent credential leaks |

Setup: `pre-commit install && pre-commit run -a`

## Infrastructure Toggle Pattern

All optional features should default to `false` in variables.tf:

```hcl
# DO: Default to disabled, enable per-environment
variable "enable_waf" {
  description = "Enable WAF protection"
  type        = bool
  default     = false
}

# Override in environment tfvars
# prod.tfvars: enable_waf = true
```

Benefits:
- Explicit opt-in for features
- Clear visibility of what's enabled
- Easier cost control
- Simpler testing of base infrastructure

## AgentCore Conventions

> This section grows as AgentCore Runtime blueprints accumulate lessons. Populated by `compound-learner` after each agent-runtime deployment.

### Key AWS services

| Service | Purpose |
|---------|---------|
| Bedrock AgentCore Runtime | Managed agent orchestration and session management |
| Amazon Cognito | User pool + JWT auth for WebSocket proxy |
| ECS Fargate (ARM64) | WebSocket proxy deployment (cost-efficient Graviton) |
| DynamoDB | Session state storage (agent-memory module) |
| CodeBuild | ARM64 container image builds |

### VPC requirements

AgentCore Runtime requires VPC endpoints for: `bedrock-runtime`, `bedrock-agent-runtime`, `ecr.api`, `ecr.dkr`, `s3` (gateway), `dynamodb` (gateway), `secretsmanager`.
Verify all endpoints exist before starting a capacity block — missing endpoints cause silent failures at runtime.

### Auth flow

Always enable `ALLOW_USER_PASSWORD_AUTH` and `ALLOW_REFRESH_TOKEN_AUTH` on the Cognito app client. Do not enable `ALLOW_ADMIN_USER_PASSWORD_AUTH` in production.

### Deployment sequence

Follow the agentcore-deployer 8-stage sequence: Foundation → Container Build → AgentCore Runtime → Auth Wiring → WebSocket Proxy → Integration Test → Readiness Audit → Compound.
Do not skip stages — each gate catches failures that are expensive to debug later.

### AgentCore HTTP protocol contract

For `serverProtocol: "HTTP"`, the container must expose `POST /invocations` (MCP JSON-RPC handler) and `GET /ping` returning `{"status": "Healthy", "time_of_last_update": int(time.time())}` on port 8080. Do not use `serverProtocol: "MCP"` unless the container implements a true MCP server on port 8000.
Missing or wrong endpoints produce 404 on every invocation.

### AgentCore Runtime endpoint version pinning

After `update-agent-runtime`, always call `update-agent-runtime-endpoint --agent-runtime-version <new_version>` and wait for endpoint status READY before testing.
The endpoint is an independent routing layer that does not auto-follow the latest runtime version.

### AgentCore Runtime has no built-in Secrets Manager injection

Load secrets from Secrets Manager in Python code at server startup (`boto3.client("secretsmanager").get_secret_value()` → `os.environ[...]`). Grant the runtime IAM role `secretsmanager:GetSecretValue`.
AgentCore Runtime has no task definition and therefore no native secrets injection unlike ECS.

### AgentCore Runtime logs require OTEL, not stdout

Add `opentelemetry-sdk opentelemetry-exporter-otlp` to `requirements.txt` and configure an `OTLPLogExporter` pointing to `http://localhost:4318` at server startup. Standard stdout/stderr is not forwarded to CloudWatch.
AgentCore Runtime routes logs through an OTEL collector sidecar; the `awslogs` driver is not available.

### AgentCore Runtime has no EFS mount support

Do not use EFS for file output from AgentCore Runtime. Write output files to ephemeral local storage during invocation, then upload to S3 (`s3://$S3_OUTPUT_BUCKET/sessions/<session_id>/`) before returning the MCP response.
AgentCore Runtime manages its own container lifecycle with no task definition, so there is no supported path to attach EFS volumes.

### boto3 retry config must be disabled for long AgentCore invocations

Set `retries={"max_attempts": 1, "mode": "standard"}` alongside `read_timeout=1200` on any boto3 client used to invoke AgentCore Runtime. Apply in both CLI test clients and proxy code.
botocore retries stack multiplied by read_timeout; 3 retries × 600 s = 1800 s of blocking before surfacing a failure.

### Claude Code 2.x Bedrock environment variable

Use `CLAUDE_CODE_USE_BEDROCK=1` and `AWS_REGION=<region>`. The old `ANTHROPIC_BEDROCK=1` var silently does nothing in Claude Code 2.x.
Fetch explicit credentials via `boto3.Session().get_credentials().get_frozen_credentials()` and inject `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` into the subprocess env — IMDS credentials are not automatically inherited by subprocesses.

### Claude binary refuses --dangerously-skip-permissions as root

Always add `USER agent` (non-root) to Dockerfiles that use claude-agent-sdk. Create the user with:
```dockerfile
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent
RUN chown -R agent:agent /app
USER agent
```
The bundled claude binary checks `process.getuid() === 0` and refuses `--dangerously-skip-permissions` as root by design.

### NDJSON streaming for long-running tool handlers

For any AgentCore Runtime agent whose `tools/call` handler takes longer than ~90 seconds:
1. Return `StreamingResponse(media_type="application/x-ndjson")` from `/invocations`
2. Yield `{"type":"progress","label":"..."}` lines every ≤30 s while the pipeline runs
3. Emit the final MCP result as the last NDJSON line

In the MCP proxy, read with `iter_lines()` on the botocore `StreamingBody` and convert progress lines into `notifications/progress` JSON-RPC notifications to stdout.
Without streaming, MCP clients (Claude Desktop, mcp-proxy) kill connections at ~2 minutes even when the backend completes correctly at 10–15 minutes.

### Container build instances must be in a public subnet with internet access

Use the default VPC (always has public subnets + IGW) for build instances — not the blueprint's private-only VPC. Transfer build context via `aws s3 cp` (local → S3) and `aws ssm send-command` (S3 → EC2). Attach the existing `<name>-build-instance` IAM instance profile with ECR + S3 permissions.
The blueprint VPC is intentionally private (all egress via VPC endpoints); this is correct for the workload but incompatible with pulling base images from Docker Hub / public.ecr.aws.

### update-agent-runtime requires --role-arn on every call

Always pass `--role-arn <existing_role_arn>` when calling `update-agent-runtime`, even if only updating the container image URI. The role ARN pattern is `arn:aws:iam::<account>:role/<name>-agentcore-exec`.
Unlike most AWS update APIs, `update-agent-runtime` treats `--role-arn` as a required field on every call, not just at creation time.
