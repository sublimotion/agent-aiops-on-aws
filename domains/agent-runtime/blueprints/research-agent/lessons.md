# Lessons Learned - Research Agent Blueprint

## Lesson #1 - Build Environment Prerequisites - 2026-02-21

**Context**: Attempted to deploy Research Agent blueprint without local container runtime

**Observation**: No Docker/nerdctl available locally; EC2 build instance in private subnet couldn't reach external services

**Rule**: Always verify container build capability before starting AgentCore deployments - either local Docker, accessible build instance, or CodeBuild project

**Why**: Container images are critical path for ECS deployments; without them, infrastructure sits idle

## Lesson #2 - Private Subnet Limitations - 2026-02-21

**Context**: Created build instance in private subnet to match security best practices

**Observation**: Instance couldn't reach GitHub, Docker Hub, or activate SSM agent without NAT gateway

**Rule**: For build instances that need external access, either use public subnet with IGW or add NAT gateway to private subnet

**Why**: Build processes often require pulling base images and dependencies from internet sources

## Lesson #3 - Non-ASCII Characters in AWS Resources - 2026-02-21

**Context**: WebSocket proxy module had em-dash (—) in security group description

**Observation**: AWS API rejected the character with "Character sets beyond ASCII are not supported"

**Rule**: Use only ASCII characters in all AWS resource names and descriptions

**Why**: AWS APIs have strict character set requirements for compatibility

## Lesson #4 - Terraform Output Sensitivity - 2026-02-21

**Context**: AgentCore Gateway module returns SSM parameters that Terraform considers sensitive

**Observation**: Terraform requires explicit `sensitive = true` flag on outputs that reference sensitive values

**Rule**: Mark any Terraform output containing URLs, tokens, or configuration data as sensitive

**Why**: Prevents accidental exposure of sensitive data in logs or state files

## Lesson #6 - CloudWatch Logs VPC Endpoint Required for Fargate - 2026-02-21

**Context**: ECS Fargate tasks in private VPC without internet access kept failing at startup

**Observation**: Task failed with `ResourceInitializationError: failed to validate logger args: cannot find CloudWatch log group ... connection issue` even though the log group existed and all other VPC endpoints were present

**Rule**: Add `com.amazonaws.<region>.logs` Interface VPC endpoint to any private VPC running ECS Fargate tasks with `awslogs` log driver. Include it in the `interface_endpoints` list alongside `ecr.api`, `ecr.dkr`, `bedrock-runtime`, etc.

**Why**: Fargate validates the CloudWatch log group via the `logs` API at task start before pulling the container. Without the endpoint, the agent cannot connect to CloudWatch, and the task is killed before the container even starts.

## Lesson #7 - AgentCore Runtime HTTP Protocol Contract - 2026-02-21

**Context**: AgentCore Runtime invocation returned 404 until correct protocol endpoints were added

**Observation**: `serverProtocol: "HTTP"` requires container to expose `POST /invocations` (payload) and `GET /ping` (health check returning `{"status": "Healthy", "time_of_last_update": <epoch>}`) on port 8080. Protocol "MCP" uses port 8000 with `POST /mcp`. Using wrong protocol or missing endpoints causes 404.

**Rule**: For AgentCore HTTP protocol: implement `POST /invocations` (MCP JSON-RPC handler) and `GET /ping` returning `{"status": "Healthy", "time_of_last_update": int(time.time())}`. Do NOT use `serverProtocol: "MCP"` unless container actually implements the MCP server protocol on port 8000.

**Why**: AgentCore Runtime strictly enforces the protocol contract; wrong endpoints mean zero invocations succeed

## Lesson #8 - Claude Code Bedrock Env Var Changed in 2.x - 2026-02-21

**Context**: Used `ANTHROPIC_BEDROCK=1` (the documented env var) but claude CLI kept returning "Not logged in · Please run /login"

**Observation**: Claude Code 2.x uses `CLAUDE_CODE_USE_BEDROCK=1` + `AWS_REGION=us-east-1`. The old env var `ANTHROPIC_BEDROCK=1` no longer enables Bedrock mode. Also requires `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` to be explicitly passed to the subprocess when credentials come from IMDS (not standard ECS task metadata URI).

**Rule**: For Claude Code ≥ 2.x with Bedrock: set `CLAUDE_CODE_USE_BEDROCK=1` and `AWS_REGION=<region>`. Fetch explicit credentials via `boto3.Session().get_credentials().get_frozen_credentials()` and inject into subprocess env via `ClaudeAgentOptions(env={..., "AWS_ACCESS_KEY_ID": ..., "AWS_SESSION_TOKEN": ...})`.

**Why**: The env var name changed between Claude Code 1.x and 2.x; `ANTHROPIC_BEDROCK` silently does nothing in 2.x

## Lesson #9 - Claude Binary Refuses --dangerously-skip-permissions as Root - 2026-02-21

**Context**: claude-agent-sdk subprocess failed with exit code 1 when container ran as root

**Observation**: The bundled claude binary explicitly checks `process.getuid() === 0` and refuses to run `--dangerously-skip-permissions` (required for bypassPermissions mode) as root user

**Rule**: Always add `USER agent` (non-root) to Dockerfile when using claude-agent-sdk. Create the user with `groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent` and `chown -R agent:agent /app`.

**Why**: Security policy in claude CLI; running as root is blocked for permission bypass mode by design

## Lesson #10 - AgentCore Runtime Endpoint Pinned to Version - 2026-02-21

**Context**: Updated runtime to new version; invoke-agent-runtime still returned old behavior

**Observation**: AgentCore Runtime has separate `liveVersion` per endpoint. Updating the runtime creates a new version but the named endpoint keeps its old `liveVersion`. Must explicitly call `update-agent-runtime-endpoint --agent-runtime-version <N>` after each runtime update.

**Rule**: After `update-agent-runtime`, always follow with `update-agent-runtime-endpoint --agent-runtime-version <new_version>` and wait for endpoint status READY before testing.

**Why**: The endpoint is an independent routing layer; it doesn't auto-update to the latest runtime version

## Lesson #11 - Secrets Manager Injection for AgentCore Runtime - 2026-02-21

**Context**: ECS task definition had BRAVE_API_KEY from Secrets Manager; AgentCore Runtime did not

**Observation**: AgentCore Runtime has no built-in Secrets Manager injection like ECS task definitions do. The `get-agent-runtime` API has no `secrets` or `environmentVariables` configuration field.

**Rule**: For AgentCore Runtime, load secrets from Secrets Manager in Python code at server startup (e.g., call `boto3.client("secretsmanager").get_secret_value()` and set `os.environ["BRAVE_API_KEY"]` before the FastAPI app initializes). The IAM role must have `secretsmanager:GetSecretValue` permission.

**Why**: AgentCore Runtime is not ECS — it has no task definition secrets injection; secrets must be fetched programmatically

## Lesson #12 - AgentCore Runtime Logs Require OTEL, Not stdout - 2026-02-21

**Context**: Container logs to stdout via Python `logging`; CloudWatch log stream `otel-rt-logs` has 0 bytes despite correct IAM permissions

**Observation**: AgentCore Runtime routes container logs through an OpenTelemetry (OTEL) collector sidecar. Standard stdout/stderr is NOT automatically forwarded to CloudWatch. The IAM role can have full `logs:PutLogEvents` permissions and still produce empty log streams. The only way to get application logs into CloudWatch from AgentCore Runtime is to write to the OTEL collector endpoint (typically `localhost:4317` gRPC or `localhost:4318` HTTP) using the `opentelemetry-sdk` Python package.

**Rule**: Add OTEL logging to AgentCore Runtime containers:
1. Add `opentelemetry-sdk opentelemetry-exporter-otlp` to `requirements.txt`
2. Configure `OTLPLogExporter` pointing to `http://localhost:4318` at server startup
3. Replace or augment `logging.basicConfig` with the OTEL log handler
Until then, use `aws --cli-read-timeout 300` and watch S3 output for completion signals.

**Why**: AgentCore Runtime is not ECS — it does not use the `awslogs` log driver. Stdout goes nowhere visible without OTEL instrumentation.

## Lesson #13 - AgentCore Runtime Has No EFS Mount Support - 2026-02-21

**Context**: Dockerfile created /app/files and assumed EFS would be mounted there; EFS filesystem showed only 6 KB (effectively empty) after research queries ran

**Observation**: AgentCore Runtime is not ECS Fargate. It has no task definition and no mechanism to mount EFS volumes. Any files written to /app/files during a run exist only in the container's ephemeral local storage and are lost after the invocation. EFS can be created and health-checked, but nothing in AgentCore Runtime wires it to the container.

**Rule**: Do not rely on EFS for file coordination in AgentCore Runtime. For persistent output, have server.py upload files to S3 using boto3 after `run_query()` completes. Sub-agents write to /app/files locally during the run (which is fine — they're all in the same container process), then server.py sweeps and uploads everything to `s3://$S3_OUTPUT_BUCKET/sessions/<session_id>/` before returning the MCP response.

**Why**: AgentCore Runtime manages its own container lifecycle without a task definition; there is no supported path to attach EFS volumes

## Lesson #14 - AgentCore Idle Session Timeout Is Configurable, Not a Hard Limit - 2026-02-22

**Context**: Research query timed out with "no response byte in last 15 mins"; assumed this was a hard AgentCore limit

**Observation**: The 15-minute timeout is `idleRuntimeSessionTimeout` in `lifecycleConfiguration`, which defaults to 900s but is configurable from 60s up to 28800s (8 hours). It is NOT a hard service limit. Update via `--lifecycle-configuration '{"idleRuntimeSessionTimeout": 3600}'` on `update-agent-runtime`. Also discovered that `--environment-variables` is a first-class field on `update-agent-runtime` — env vars can be injected via the API without baking them into the Dockerfile.

**Rule**: Set `idleRuntimeSessionTimeout` to match the expected max query duration (3600s for research agents). For boto3 callers, also set `retries={"max_attempts": 1, "mode": "standard"}` and `read_timeout=1200` to avoid retry stacking.

**Why**: Default 900s is appropriate for short tool-use agents but far too short for multi-agent pipelines doing web research. Use `--environment-variables` API field instead of Dockerfile ENV for runtime-specific config (keeps the image generic).

## Lesson #15 - Brave Search Rate Limiting Between Consecutive Invocations - 2026-02-22

**Context**: Running 3 back-to-back framework research queries (SGLang → vLLM → TensorRT-LLM); TensorRT-LLM consistently timed out at 15 minutes despite vLLM completing fine

**Observation**: Brave Search API rate-limits requests after consecutive heavy searches. Each research query spawns 3 parallel sub-agents each making multiple web searches, so 2 full queries can exhaust the rate limit window. The third query's researchers hang waiting for search results, consuming the entire 15-minute invocation window without producing output.

**Rule**: Add a 60-second cooldown between consecutive research queries that each spawn multiple researchers. For automated pipelines running multiple queries, add `time.sleep(60)` between invocations or implement exponential backoff on the Brave search tool.

**Why**: Brave's rate limits are per-API-key per time window; 3 parallel researchers × multiple searches per query = rapid consumption of the quota

## Lesson #16 - NDJSON Streaming Prevents Client Timeouts for Long-Running Agents - 2026-02-23

**Context**: MCP clients (Claude Desktop, mcp-proxy) have a ~2 minute timeout on tool calls. Research queries take 10-15 minutes to complete.

**Observation**: Without streaming, `tools/call` blocks the HTTP connection silently for the full pipeline duration. Clients kill the connection before the result arrives. The solution is to return `StreamingResponse(media_type="application/x-ndjson")` from the `/invocations` handler and yield periodic progress lines while the pipeline runs. AgentCore flushes each NDJSON line to the client as soon as it is yielded — the connection stays alive because bytes are flowing.

**Rule**: For any AgentCore Runtime agent whose `tools/call` handler takes longer than ~90 seconds: (1) use `StreamingResponse` with `media_type="application/x-ndjson"`, (2) yield `{"type":"progress","label":"..."}` lines every ≤30 s while the pipeline runs, (3) emit the final MCP result as the last NDJSON line. In the MCP proxy, read with `iter_lines()` on the botocore `StreamingBody` and emit `notifications/progress` JSON-RPC notifications to stdout for each progress line.

**Why**: The MCP 2024-11-05 protocol is synchronous — one request/one response. The only mechanism to send live updates is `notifications/progress` (one-way), which requires the proxy to convert NDJSON progress lines into JSON-RPC notifications. Without this, Claude Desktop times out at ~2 minutes even though the backend completes correctly at 10-15 minutes.

## Lesson #17 - Build Instance Must Be in Public Subnet with Internet Access - 2026-02-23

**Context**: Need to rebuild container for AgentCore Runtime; previous build instance (i-0dbe76f57475bf949) was terminated and the blueprint VPC has only private subnets without NAT.

**Observation**: Container builds require pulling base images from Docker Hub / public.ecr.aws and apt packages from debian.org. A build instance in a private-only VPC without a NAT gateway cannot reach these registries. The solution is to launch the build instance in any VPC that has a public subnet + IGW (e.g., the default VPC), not in the blueprint's private VPC. S3 is used as the build context transfer mechanism: `aws s3 cp <tarball>` from local → S3 → EC2 via SSM.

**Rule**: For container build instances: (1) use the default VPC (always has public subnets + IGW), (2) attach the existing `<name>-build-instance` IAM instance profile with ECR + S3 permissions, (3) transfer build context via `aws s3 cp` + `aws ssm send-command`. Do NOT put build instances in the blueprint's private-only VPC.

**Why**: The blueprint VPC is intentionally private (all egress via VPC endpoints) for security. This is correct for the workload but incompatible with the build step that needs internet access for base image pulls.

## Lesson #18 - AgentCore Runtime update-agent-runtime Requires --role-arn - 2026-02-23

**Context**: Running `update-agent-runtime` to deploy new container version.

**Observation**: Unlike `create-agent-runtime`, the `update-agent-runtime` CLI call requires `--role-arn` even though the role hasn't changed. Omitting it returns an error: `the following arguments are required: --role-arn`.

**Rule**: Always pass `--role-arn <existing_role_arn>` when calling `update-agent-runtime`, even if only updating the container image. The role ARN is `arn:aws:iam::<account>:role/<name>-agentcore-exec`.

**Why**: The API treats role as a required field on every update, not just at creation time. This is different from most AWS update APIs which use partial-update semantics.

## Lesson #19 - Always Add Hard Timeouts to Long-Running AgentCore Pipelines - 2026-02-23

**Context**: Research pipeline (digital twins query) ran for 43+ minutes without completing or erroring, likely due to a hung researcher sub-agent waiting on Brave Search rate limits.

**Observation**: Two compounding bugs caused infinite hangs: (1) `_stream_tools_call` in server.py had no ceiling on the `while not task.done()` loop — if `run_query` hangs forever, it yields progress bytes every 30s forever. (2) botocore's `read_timeout=1200` is a *per-read socket timeout*, not a total call timeout. Since server.py sends a byte every 30s, the socket timer resets every 30s and never fires. The proxy background thread blocked indefinitely, permanently stuck in `iter_lines()`.

**Rule**: (1) In server.py, wrap the pipeline `asyncio.create_task` with a monotonic deadline: cancel the task after 25 minutes (`MAX_PIPELINE_SECONDS = 1500`) and yield an MCP error as the final NDJSON line. (2) In mcp-proxy.py `_run_job`, record `job_start = time.monotonic()` before the `iter_lines()` loop and raise `TimeoutError` if `time.monotonic() - job_start > MAX_JOB_SECONDS` (30 minutes). Never rely solely on `read_timeout` for streaming connections where the server sends periodic keep-alive bytes.

**Why**: `read_timeout` in botocore/requests is a socket-level idle timeout between individual reads, not a wall-clock limit on the total call. Any streaming response that trickles bytes (even slowly) will keep resetting the timer. Hard wall-clock timeouts must be implemented explicitly in application code.

## Lesson #20 - Bedrock web_search Tool Not Available on Amazon Bedrock - 2026-02-23

**Context**: Brave Search API rate-limiting was causing researcher sub-agents to hang. Investigated whether Amazon Bedrock natively provides a web_search tool to replace Brave.

**Observation**: The `web_search_20250305` and `web_search_20260209` tool types that appear in Anthropic API docs are NOT available when routing through Amazon Bedrock. Attempting to use them returns a 400 error. These tools are only available on the direct Anthropic API, Azure, and Vertex AI.

**Rule**: For agents running on Amazon Bedrock AgentCore that need web search: deploy a custom MCP search server (e.g., `mcp_search_server.py`) in the container image with a project-level `.claude/settings.json` to register it. Use `mcp__<server-name>__web_search` as the tool name in agent definitions. Do NOT use the built-in `WebSearch` tool (it routes through Anthropic API, not Bedrock) or rely on the `web_search_2025*` tool types.

**Why**: Claude Code's built-in WebSearch tool silently falls back to direct Anthropic API calls when Bedrock credentials are active but the API doesn't support the tool type. The result is either a 400 error or unexpected billing against the Anthropic API instead of Bedrock.

## Lesson #21 - Use Tavily as Primary Web Search with Brave as Fallback - 2026-02-23

**Context**: Brave Search API free tier (2,000 req/month) was being exhausted by multi-researcher pipelines, causing hung agents (each full research run uses ~20-40 searches across 3 researcher sub-agents).

**Observation**: Tavily is purpose-built for AI agent pipelines, supports 1,000 free searches/month (dev tier), returns AI-ready content with automatic extraction, and has a first-class Python SDK (`tavily-python`). The `TavilyClient.search()` returns structured results with title, URL, and pre-extracted content — no HTML parsing needed. Brave is kept as fallback since it has broader crawl coverage for technical content.

**Rule**: In `mcp_search_server.py`, try Tavily first (`if os.environ.get("TAVILY_API_KEY")`), catch exceptions, then fall back to Brave. Both keys come from ECS task environment variables loaded from Secrets Manager at startup. Store as separate secrets: `<name>/tavily-api-key` and `<name>/brave-api-key`.

**Why**: Tavily's rate limits are more generous for agent workloads and its SDK is simpler. Brave remains useful as a fallback with different content coverage. Dual-provider setup eliminates single-provider rate-limit failures.

## Lesson #22 - AL2023 Build Instance Needs docker Installed Manually - 2026-02-23

**Context**: Build instance (t4g.medium, AL2023) had no container runtime despite previous SSM commands reporting "BUILD SUCCEEDED". Attempts to use nerdctl failed (`command not found`).

**Observation**: Amazon Linux 2023 AMI `ami-0f075de63c9ac63d4` does NOT include Docker or nerdctl by default. `nerdctl` was never actually installed on this instance; the "BUILD SUCCEEDED" message was a false positive from the SSM command exit code, not actual image creation. Docker can be installed with `dnf install -y docker && systemctl start docker`.

**Rule**: When building on a fresh AL2023 instance: (1) run `dnf install -y docker` first, (2) `systemctl start docker`, (3) verify with `docker --version`. Do NOT assume any container runtime is pre-installed. Add docker installation as the first step of any build SSM command sequence.

**Why**: SSM commands return "Success" status based on script exit code, not on whether the application actually ran. If a binary is not found (`exit 127`), the shell script exits 127 but SSM may still report "Success" if the error was in a subshell. Always verify with `docker --version && echo DOCKER_READY`.

## Lesson #23 - AgentCore Runtime Containers Do Not Inherit ECS Task Definition Volumes - 2026-02-23

**Context**: EFS file system was configured in the ECS task definition (websocket-proxy module) with a volume mount at `/app/files`. Research pipeline outputs (PDFs, charts, research notes) were written there. Expected `_upload_outputs_to_s3` in server.py to find files and upload them. S3 upload silently returned empty after every run.

**Observation**: The ECS service (websocket-proxy module) and the AgentCore Runtime are two completely separate compute paths. `invoke_agent_runtime` runs the container in AWS-managed Fargate infrastructure, not in the ECS cluster we define. The AgentCore Runtime container has no EFS mount; all files written to `/app/files` go to ephemeral container-local storage. Additionally, env vars like `S3_OUTPUT_BUCKET` set in the ECS task definition `environment` block are NOT inherited by AgentCore Runtime containers. The S3 upload silently skipped because `S3_OUTPUT_BUCKET` was empty in the AgentCore environment.

**Rule**: (1) Pass all required env vars (`S3_OUTPUT_BUCKET`, `AWS_REGION`, `FILES_BASE`) to the AgentCore Runtime via `--environment-variables` in `update-agent-runtime`. (2) Add an auto-discovery fallback in server.py: if `S3_OUTPUT_BUCKET` is unset, call `s3.list_buckets()` to find the bucket by name prefix. (3) Add diagnostic logging of file count in `_upload_outputs_to_s3` so empty-upload failures are visible. (4) Add `S3_OUTPUT_BUCKET` and `FILES_BASE` to the `debug/env` diagnostic endpoint.

**Why**: The two compute paths (ECS service vs AgentCore Runtime) look identical from the code perspective but have completely different environment setups. Without explicit env vars on the AgentCore path, any server-side S3 uploads will silently fail. CloudWatch logs from AgentCore Runtime containers are sparse (no container-level awslogs configured by default), making silent failures very hard to diagnose.

## Lesson #24 - Use `--environment-variables` on `update-agent-runtime` for Container Config - 2026-02-23

**Context**: Need to pass `S3_OUTPUT_BUCKET`, `AWS_REGION`, and other env vars to AgentCore Runtime containers without baking them into the Docker image.

**Observation**: `aws bedrock-agentcore-control update-agent-runtime` supports `--environment-variables` as a top-level parameter (not inside `--agent-runtime-artifact`). The format is a JSON object: `{"KEY": "VALUE", ...}`. These vars are injected into every container instance spawned by the runtime. The API accepted it with the existing `--agent-runtime-artifact` and `--role-arn` params unchanged.

**Rule**: Pass runtime-specific env vars (bucket names, region, feature flags) via `--environment-variables` rather than baking them into the Docker image. This keeps the image generic and lets different runtime deployments (dev/staging/prod) use different configs. Add these vars to the `debug/env` diagnostic endpoint to verify they're set on the next smoke test.

**Why**: Docker image rebuild + ECR push + runtime update is expensive (~10 min). Env vars can be updated independently with just `update-agent-runtime` (seconds). Keeping config separate from image also avoids leaking bucket names or account IDs into the image layer history.

## Lesson #5 - AgentCore Runtime Doesn't Require Container at Creation - 2026-02-21

**Context**: Was able to create and prepare AgentCore Runtime agent without having container image ready

**Observation**: AgentCore agent reached PREPARED state using just IAM roles and foundation model config

**Rule**: AgentCore Runtime setup can proceed independently of container builds - decouple these stages

**Why**: Allows parallel work on infrastructure and application layers