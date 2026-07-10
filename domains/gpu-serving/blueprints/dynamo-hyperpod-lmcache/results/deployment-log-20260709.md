# Deployment Log — Dynamo HyperPod LMCache

## 2026-07-09 — Launch Preparation

### Inputs

- Spec: `domains/gpu-serving/specs/dynamo-hyperpod-lmcache.md`
- Primary region: `us-east-2`
- Fallback region: `us-west-2`
- Preferred instance: `ml.g7.2xlarge`
- Approved fallback instance: `ml.g6e.xlarge`
- Model: `Qwen/Qwen3-0.6B`

### Discovery Already Completed

- AWS identity works for account `615299764834`.
- No existing SageMaker HyperPod clusters in `us-east-2` or `us-west-2`.
- Existing EKS `us-east-2` target:
  - Cluster: `qwen3-next-bench-eks-cluster`
  - ARN: `arn:aws:eks:us-east-2:615299764834:cluster/qwen3-next-bench-eks-cluster`
  - VPC: `vpc-0490a5031a96f53dd`
  - Subnets: `subnet-0fced510ea62b874e`, `subnet-04be09c7bf104edb8`, `subnet-03d03f1fb8d62d6a5`
  - Security group: `sg-05896b0f6e63a8bf2`
- Existing EKS `us-west-2` fallback:
  - Cluster: `qn-sglang-eks-cluster`
  - ARN: `arn:aws:eks:us-west-2:615299764834:cluster/qn-sglang-eks-cluster`
  - VPC: `vpc-0bd6abcecded8edf6`
  - Subnets: `subnet-00db54563893dbe55`, `subnet-001db6882dbb5ac72`, `subnet-00ffd4431ec8f1352`
  - Security group: `sg-0276a843e6e5362a8`
- SageMaker CLI model did not list `ml.g7.*` as a HyperPod create-cluster instance type.
- HyperPod `ml.g6e.xlarge for cluster usage` quota increase requests were approved in both target regions.
- `mdc` has no deployment card or PR history for `Qwen/Qwen3-0.6B` with vLLM.
- `gpu-infra` has no specific G6e card; `gpu-infra card g7` returned G7e content only.
- `.claude/steering/tech-stack.md` did not contain the expected HyperPod Inference Operator release-tracking section, so no pinned operator version was available from steering.

### Launch Commands

#### HyperPod dependencies

Installed the dependency chart into the existing EKS cluster:

```bash
aws eks update-kubeconfig --region us-east-2 --name qwen3-next-bench-eks-cluster
helm upgrade --install dependencies /Users/phi/Documents/workbench/sagemaker-hyperpod-cli/helm_chart/HyperPodHelmChart \
  --namespace kube-system \
  -f /Users/phi/Documents/workbench/sagemaker-hyperpod-cli/helm_chart/HyperPodHelmChart/regional-values/values-us-east-2.yaml \
  --set global.region=us-east-2 \
  --set nvidia-device-plugin.devicePlugin.enabled=false \
  --set aws-efa-k8s-device-plugin.devicePlugin.enabled=false \
  --set neuron-device-plugin.devicePlugin.enabled=false
```

Result: `STATUS: deployed`, revision 1.

#### Lifecycle staging

Created lifecycle bucket and uploaded `on_create.sh`:

```bash
aws s3api create-bucket \
  --bucket sagemaker-hyperpod-dynamo-615299764834-us-east-2 \
  --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2

aws s3 cp domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/scripts/on_create.sh \
  s3://sagemaker-hyperpod-dynamo-615299764834-us-east-2/lifecycle/on_create.sh \
  --region us-east-2
```

#### Create attempt 1

Config: `configs/create-cluster-us-east-2-g6e.json`

```bash
aws sagemaker create-cluster \
  --region us-east-2 \
  --cli-input-json file://domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/configs/create-cluster-us-east-2-g6e.json
```

Result:

```text
An error occurred (ValidationException) when calling the CreateCluster operation: Found RestrictedInstanceGroups in the cluster input. Please remove RestrictedInstanceGroups from the request and try again.
```

Action: create a second config without `RestrictedInstanceGroups`; use a regular CPU instance group for system-node headroom.

#### Create attempt 2

Config: `configs/create-cluster-us-east-2-g6e-no-rig.json`

```bash
aws sagemaker create-cluster \
  --region us-east-2 \
  --cli-input-json file://domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/configs/create-cluster-us-east-2-g6e-no-rig.json
```

Result:

```json
{
  "ClusterArn": "arn:aws:sagemaker:us-east-2:615299764834:cluster/p4aa28nj1kdf"
}
```

Initial `describe-cluster`:

- `ClusterStatus`: `Creating`
- `g6e-workers`: `CurrentCount=0`, `TargetCount=1`, `Status=Creating`
- `system-nodes`: `CurrentCount=0`, `TargetCount=2`, `MinCount=1`, `Status=Creating`
- `TieredStorageConfig`: requested in config as `{"Mode":"Enable","InstanceMemoryAllocationPercentage":20}` but not echoed in the initial describe output.

Follow-up `describe-cluster` confirmed:

- `ClusterStatus`: `InService`
- `TieredStorageConfig`: `Mode=Enable`, `InstanceMemoryAllocationPercentage=20`
- `g6e-workers`: `ActiveOperations.Scaling=1`, `CurrentCount=0`, `TargetCount=1`
- `system-nodes`: `ActiveOperations.Scaling=1`, `CurrentCount=0`, `TargetCount=2`

SageMaker node details:

| Instance ID | Group | Type | Status | Private IP | AZ |
|---|---|---|---|---|---|
| `i-0afcf11fc61007378` | `g6e-workers` | `ml.g6e.xlarge` | `Pending` | `10.0.2.38` | `us-east-2a` |
| `i-068f190d022dd99b4` | `system-nodes` | `ml.m5.2xlarge` | `Pending` | `10.0.45.200` | `us-east-2c` |
| `i-09b9cb851fc88e86f` | `system-nodes` | `ml.m5.2xlarge` | `Pending` | `10.0.39.217` | `us-east-2c` |

Pending observation at 08:00 EDT:

- Latest events remain `Instance lifecycle script execution ... has Started`.
- No lifecycle completion or failure event yet.
- `kubectl get nodes` still shows only the original 4 EKS nodes.
- Normal EC2 `DescribeInstances` returned `InvalidInstanceID.NotFound` for the SageMaker-reported instance IDs, so use SageMaker `describe-cluster-node` for node status.

#### Update attempt 1

Hypothesis: nodes are stuck because the create config used only the EKS cluster SG `sg-05896b0f6e63a8bf2`, not the EKS node shared SG `sg-0bf5ad07fc6c29df1`. This matches an existing repo steering lesson about GPU node networking.

Config: `configs/update-cluster-add-node-sg.json`

Result:

```text
An error occurred (ValidationException) when calling the UpdateCluster operation: Updating fields OverrideVpcConfig on an InstanceGroup not supported
```

Action: delete this failed launch and recreate with both security groups in the initial `VpcConfig`.

#### Delete attempt 1

Deleted the pending cluster so the replacement can use the corrected security group set from initial creation:

```bash
aws sagemaker delete-cluster \
  --region us-east-2 \
  --cluster-name dynamo-hp-lmcache-20260709
```

Result: delete accepted for `arn:aws:sagemaker:us-east-2:615299764834:cluster/p4aa28nj1kdf`; cluster entered `Deleting`.

Delete events later showed both instance groups scaling to zero completed successfully.

#### Recreate attempt 1

Config: `configs/create-cluster-us-east-2-g6e-node-sg.json`

Submitted while the old cluster record was still `Deleting`.

Result:

```text
An error occurred (ValidationException) when calling the CreateCluster operation: Specified cluster external resource id arn:aws:eks:us-east-2:615299764834:cluster/qwen3-next-bench-eks-cluster already in use
```

Action: wait until the original HyperPod cluster is fully deleted, then retry the same corrected config.

#### Fallback to us-west-2

While the original `us-east-2` cluster remained `Deleting`, prepared fallback region `us-west-2`:

- EKS cluster: `qn-sglang-eks-cluster`
- EKS ARN: `arn:aws:eks:us-west-2:615299764834:cluster/qn-sglang-eks-cluster`
- Cluster SG: `sg-0276a843e6e5362a8`
- Node shared SG: `sg-04d7a445823deca50`
- Subnets: `subnet-00db54563893dbe55`, `subnet-001db6882dbb5ac72`, `subnet-00ffd4431ec8f1352`

Installed HyperPod dependencies in `kube-system` with `values-us-west-2.yaml` and staged lifecycle script to `s3://sagemaker-hyperpod-dynamo-615299764834-us-west-2/lifecycle/on_create.sh`.

Config: `configs/create-cluster-us-west-2-g6e-node-sg.json`

```bash
aws sagemaker create-cluster \
  --region us-west-2 \
  --cli-input-json file://domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/configs/create-cluster-us-west-2-g6e-node-sg.json
```

Result:

```json
{
  "ClusterArn": "arn:aws:sagemaker:us-west-2:615299764834:cluster/ecew0hodovyj"
}
```

Initial west events confirmed `TieredStorageConfig.Mode=Enable` and both SGs applied:

- Cluster SG: `sg-0276a843e6e5362a8`
- Node shared SG: `sg-04d7a445823deca50`

Lifecycle failed on first west node set. CloudWatch log group:

- `/aws/sagemaker/Clusters/dynamo-hp-lmcache-west-20260709/ecew0hodovyj`

Failure evidence from `LifecycleConfig/g6e-workers/i-02f19d23ecb3f975c` and both system nodes:

```text
[start] on_create.sh
Found secondary EBS volume. Setting containerd data root to /opt/sagemaker/containerd/data-root
[SageMaker] The lifecycle scripts failed.
```

Root cause: script assumed `/etc/eks/containerd/containerd-config.toml` exists after detecting `/opt/sagemaker`. On the west HyperPod AMI, that path was absent, so `sed` failed under `set -e`.

Fix: patched `scripts/on_create.sh` to check for the file before editing and log/continue if absent. Uploaded fixed script to both regional lifecycle buckets.

Replacement result:

- `system-nodes` replacement instance `i-097810cf0d14a33cb`: lifecycle succeeded; orchestration-ready.
- `system-nodes` replacement instance `i-02c602370885ec6ff`: lifecycle succeeded; orchestration-ready.
- `system-nodes` instance group reached `CurrentCount=2`, `Status=InService`.
- EKS node count increased from 2 to 4 in `us-west-2`.
- `g6e-workers` failed instance `i-02f19d23ecb3f975c` was deleted after lifecycle failure; replacement still pending/creating as of 08:36 EDT.
- Replacement `g6e-workers` instance `i-0ebd814e4a113cdce` provisioned, lifecycle completed sufficiently for orchestration readiness, and instance group reached `CurrentCount=1`, `Status=InService`.
- Cluster reached 3/3 orchestration-ready nodes across 2 instance groups at 08:38 EDT.

## 2026-07-09 — Smoke Runtime and Telemetry

### Kubernetes and GPU Readiness

- Current region: `us-west-2`
- EKS cluster: `qn-sglang-eks-cluster`
- HyperPod cluster: `dynamo-hp-lmcache-west-20260709`
- GPU node: `hyperpod-i-0ebd814e4a113cdce`
- GPU instance group: `g6e-workers`
- Workload pod after final restart:

```text
dynamo-lmcache-worker-7d6b8cf556-449m5   1/1 Running   0 restarts   10.2.38.168   hyperpod-i-0ebd814e4a113cdce
```

The ai-toolkit daemon on the GPU node reports:

```text
[server]
address = "[::]:9200"

[cache]
capacity = "1GiB"
shared_memory_name = "ai_toolkit_cache"

[transfer]
address = "10.2.37.31:9240"
```

### Manifest Fixes

The initial smoke manifest used:

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector lmcache --host 0.0.0.0 --port 8000
```

That failed because `python3 -m dynamo.vllm` in `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` rejected `--host` and `--port`.

The working smoke path is raw vLLM OpenAI API server with `LMCacheConnectorV1`:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"sagemaker_hyperpod_shared_memory_name":"ai_toolkit_cache"}}'
```

Two additional fixes were required:

- Set `dnsPolicy: Default` because the pod could not resolve `huggingface.co` through ClusterDNS.
- Mount `LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml` with `extra_config.sagemaker_hyperpod_shared_memory_name=ai_toolkit_cache`; otherwise the adapter defaulted to `shared_memory`.

Successful connector evidence:

```text
Loading LMCache config file /etc/lmcache/config.yaml
Creating SageMaker HyperPod connector: url=http://10.2.37.31:9200, bucket=lmcache, shared_memory=ai_toolkit_cache
Shared memory opened: ai_toolkit_cache (1024.00 MB)
SageMaker HyperPod connector created successfully
Connection initialized/re-established at sagemaker-hyperpod://10.2.37.31:9200
Application startup complete.
```

### Traffic Results

Short smoke:

- `/health`: HTTP 200, 288 ms
- `/v1/models`: HTTP 200, `Qwen/Qwen3-0.6B`, 240 ms
- Chat request 1: HTTP 200, 48 prompt tokens, 402 ms
- Chat request 2: HTTP 200, 48 prompt tokens, 372 ms
- Artifact: `results/e2e-telemetry-20260709.json`

Long-prefix smoke:

- Request 1: HTTP 200, 935 prompt tokens, 3339 ms
- Request 2: HTTP 200, 935 prompt tokens, 965 ms
- `prefix_cache_hits_total` increased to 944
- `external_prefix_cache_queries_total` increased to 1022
- `external_prefix_cache_hits_total` remained `0.0`
- Artifact: `results/e2e-telemetry-long-prefix-20260709.json`

Post-vLLM-restart L2 replay:

- Restarted only `deploy/dynamo-lmcache-worker`; ai-toolkit daemon remained running.
- Replacement pod reconnected to `sagemaker-hyperpod://10.2.37.31:9200`.
- Replay request: HTTP 200, 937 prompt tokens, 3398 ms
- `external_prefix_cache_queries_total`: 937
- `external_prefix_cache_hits_total`: `0.0`
- Artifact: `results/e2e-telemetry-l2-replay-after-restart-20260709.json`

Forced L2 retrieval:

- Updated manifest with `PYTHONHASHSEED=0` and `save_unfull_chunk: true`.
- Store pass used exact prompt tag `force-l2-exact-key`.
- Store request: HTTP 200, 1118 prompt tokens, 3358 ms.
- Store log evidence:

```text
Initialized NONE_HASH=5313149338359954478
Stored 1118 out of total 1118 tokens. size: 0.1194 GB
```

- Restarted only `deploy/dynamo-lmcache-worker`; ai-toolkit daemon remained running.
- Replay used the exact same prompt and tag.
- Replay request: HTTP 200, 1118 prompt tokens, 349 ms.
- Replay metrics:

```text
vllm:external_prefix_cache_queries_total 1118
vllm:external_prefix_cache_hits_total 1117
```

- Replay log evidence:

```text
LMCache hit tokens: 1118, need to load: 1117
```

- Artifact: `results/e2e-telemetry-force-l2-store-20260709.json`
- Artifact: `results/e2e-telemetry-force-l2-replay-20260709.json`

ai-toolkit direct inspection:

- Safe read-only probes against `http://10.2.37.31:9200/`, `/health`, `/metrics`, `/stats`, `/status`, `/buckets`, `/cache`, and `/debug/vars` all returned HTTP 404.
- No daemon cache-content inspection API was found during this run.

### Outcome

Result is **PASS for raw vLLM + LMCache + HyperPod L2**, **PARTIAL for full Dynamo KVBM**:

- PASS: HyperPod managed tiered storage is enabled and ai-toolkit is running.
- PASS: vLLM/LMCache connects to HyperPod L2 using `sagemaker-hyperpod://<node-ip>:9200` and opens the correct shared memory segment.
- PASS: OpenAI-compatible traffic succeeds through the worker.
- PASS: vLLM metrics expose prefix-cache and external-prefix-cache counters.

## Dynamo Frontend L2 Validation Addendum

After the raw vLLM proof, the manifest was updated to run the real Dynamo frontend and Dynamo vLLM worker in one pod:

```text
python3 -m dynamo.frontend --discovery-backend file --request-plane tcp --event-plane zmq --http-port 8000
python3 -m dynamo.vllm --discovery-backend file --request-plane tcp --event-plane zmq --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both",...}'
```

The frontend health endpoint listed the routed backend:

```text
dyn://dynamo-hp-lmcache.backend.generate
```

The long 2093-token prompt wrote to HyperPod L2 through the frontend but missed on replay after restart:

```text
vllm:external_prefix_cache_hits_total 0.0
lmcache:num_lookup_tokens_total 2093
lmcache:num_lookup_hits_total 0.0
```

The shorter 1103-token deterministic prompt passed end to end through the Dynamo frontend:

- Store: HTTP 200, 1103 prompt tokens, 3334 ms, `Stored 1103 out of total 1103 tokens`, HyperPod `PUT success`.
- Restarted only `deploy/dynamo-lmcache-worker`; ai-toolkit daemon remained running.
- Replay: HTTP 200, 1103 prompt tokens, 328 ms, response usage reported `cached_tokens=1102`.
- Metrics:

```text
vllm:external_prefix_cache_hits_total 1102
lmcache:num_hit_tokens_total 1103
lmcache:num_lookup_hits_total 1103
lmcache:num_remote_read_requests_total 6
lmcache:num_remote_read_bytes_total 2.53001728e+08
```

- Replay log evidence:

```text
LMCache hit tokens: 1103, need to load: 1102
Retrieved 1103 out of 1103 required tokens
External prefix cache hit rate: 99.9%
```

Artifacts:

- `results/e2e-telemetry-dynamo-frontend-short-store-20260709.json`
- `results/e2e-telemetry-dynamo-frontend-short-replay-20260709.json`
- `scripts/dynamo_frontend_l2_probe.py`
- `README.md`

Updated outcome: **PASS for Dynamo frontend + Dynamo vLLM + LMCache + HyperPod L2** on the 1103-token deterministic prompt. The 2093-token miss remains a follow-up edge case.
- PASS: forced store/restart/replay produced external L2 hits (`external_prefix_cache_hits_total=1117`) and LMCache log hit evidence.
- PARTIAL: working runtime path is raw vLLM, not full Dynamo graph/KVBM.
