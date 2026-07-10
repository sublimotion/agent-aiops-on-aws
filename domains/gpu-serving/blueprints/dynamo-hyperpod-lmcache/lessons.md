# NVIDIA Dynamo HyperPod LMCache — Lessons

## Launch Notes (2026-07-09)

### Live Discovery

- AWS account: `615299764834`, caller `arn:aws:iam::615299764834:user/aiops`.
- No existing SageMaker HyperPod clusters were found in `us-east-2` or `us-west-2`.
- Existing EKS candidate in `us-east-2`: `qwen3-next-bench-eks-cluster`, Kubernetes 1.32, VPC `vpc-0490a5031a96f53dd`, subnets `subnet-0fced510ea62b874e`, `subnet-04be09c7bf104edb8`, `subnet-03d03f1fb8d62d6a5`, security group `sg-05896b0f6e63a8bf2`.
- Existing EKS candidate in `us-west-2`: `qn-sglang-eks-cluster`, Kubernetes 1.32, VPC `vpc-0bd6abcecded8edf6`, subnets `subnet-00db54563893dbe55`, `subnet-001db6882dbb5ac72`, `subnet-00ffd4431ec8f1352`, security group `sg-0276a843e6e5362a8`.
- `aws sagemaker create-cluster help` does not list `ml.g7.*` as an allowed HyperPod instance type. It does list `ml.g7e.*` and `ml.g6e.*`.
- EC2 offerings show G7 shapes exist in both target regions, but SageMaker HyperPod support is not confirmed.
- `ml.g6e.xlarge for cluster usage` quota was initially 0 in both target regions. Quota increase requests to 1 were approved in both `us-east-2` and `us-west-2`.
- `mdc get Qwen/Qwen3-0.6B --engine vllm` returned no card. `mdc prs Qwen/Qwen3-0.6B` returned no tracked PRs.
- `gpu-infra card g7` returned the `g7e` card, not a G7 card. `gpu-infra card g6e` had no card.

### Carryover Lessons

- HyperPod dependency Helm chart from `sagemaker-hyperpod-cli` must be installed before HyperPod cluster creation.
- `TieredStorageConfig.Mode=Enable` deploys the `ai-toolkit` daemon on GPU nodes; L2 validation requires the daemon and port 9200 to be present.
- Prior working LMCache L2 path used `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200` and host IPC at `/dev/shm/ai_toolkit_cache`.
- Kubernetes env ordering matters: define `NODE_IP` before `LMCACHE_REMOTE_URL`.
- HyperPod-managed FSx root can be root-owned and not writable by serving containers; create writable subdirectories before using FSx L3.
- LMCache HyperPod shared memory naming and permissions can block runtime connectivity. Verify the actual shared memory file and daemon logs rather than treating env vars as proof.
- NVIDIA Dynamo v0.8.1 LMCache integration should be launched with `python3 -m dynamo.vllm --connector lmcache`; raw vLLM `--kv-transfer-config` is diagnostic only.
- NVIDIA Dynamo LMCache integration is x86-only, so Stage 1 must verify `uname -m == x86_64`.

### Open Risks

- SageMaker HyperPod may reject the requested preferred G7 shape even though EC2 lists G7 offerings.
- A single `ml.g6e.xlarge` fallback node may be tight for both system and workload pods; use a restricted CPU system instance group if the create-cluster schema accepts it.
- Dynamo KVBM may remain separate from LMCache. A successful Dynamo worker with `LMCacheConnectorV1` is only a partial result unless Dynamo graph/KVBM telemetry also proves cache participation.

### Launch-Time Lessons

- The current SageMaker HyperPod `create-cluster` API path rejected `RestrictedInstanceGroups` with: `Found RestrictedInstanceGroups in the cluster input. Please remove RestrictedInstanceGroups from the request and try again.` For this smoke run, use regular instance groups and rely on labels/taints for system-node separation.
- `ml.g7.*` was not launched because the AWS CLI service model rejects it client-side as an invalid enum value for HyperPod `InstanceType`. The approved fallback is `ml.g6e.xlarge`.
- HyperPod `OverrideVpcConfig` security groups cannot be updated in place on an existing instance group. If nodes are created with the wrong SG set, recreate the cluster or create a new instance group with the correct SGs from the start.
- HyperPod AL2023 lifecycle scripts must not assume `/etc/eks/containerd/containerd-config.toml` exists. Guard the edit with `[[ -f ... ]]` or the script fails under `set -e`; CloudWatch only reports the last successful log line plus `[SageMaker] The lifecycle scripts failed.`

### Runtime Lessons

- `python3 -m dynamo.vllm --connector lmcache` in `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` accepts the LMCache connector flag but does not accept OpenAI serving flags `--host` and `--port`. For this smoke run, raw vLLM `python3 -m vllm.entrypoints.openai.api_server --kv-transfer-config ...` was required to expose `/v1/chat/completions`.
- HyperPod workload pods on this west cluster could not resolve `huggingface.co` through ClusterDNS. Setting `dnsPolicy: Default` made the pod use the node resolver; model resolution and download then succeeded.
- The SageMaker HyperPod LMCache adapter reads `sagemaker_hyperpod_shared_memory_name` from `LMCacheEngineConfig.extra_config`. Passing the key only in vLLM `kv_connector_extra_config` is not enough: the adapter falls back to `shared_memory` and logs `Shared memory segment 'shared_memory' not found`.
- Mount an `LMCACHE_CONFIG_FILE` with:
  ```yaml
  extra_config:
    sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache
  ```
  With that file mounted, logs show `shared_memory=ai_toolkit_cache`, `Shared memory opened: ai_toolkit_cache (1024.00 MB)`, and `Connection initialized/re-established at sagemaker-hyperpod://<node-ip>:9200`.
- Short prompts below the 256-token LMCache chunk size do not create useful L2-hit evidence. The 48-token smoke requests completed successfully but had `LMCache hit tokens: 0`.
- A 935-token long-prefix test showed local prefix-cache reuse (`prefix_cache_hits_total` increased and the second request latency dropped from ~3339 ms to ~965 ms), but `external_prefix_cache_hits_total` remained `0.0`.
- Restarting the vLLM pod while leaving ai-toolkit running still produced `external_prefix_cache_queries_total=937` and `external_prefix_cache_hits_total=0.0` on replay. This proves the connector is queried, but does not prove L2 KV reuse. Treat the current result as PARTIAL for Dynamo KVBM/LMCache compatibility.
- Forced L2 retrieval requires stable LMCache hash keys across processes. Set `PYTHONHASHSEED=0` before vLLM starts. Without it, LMCache warns about inconsistent distributed caching and post-restart L2 replay can miss.
- Setting `save_unfull_chunk: true` in `LMCACHE_CONFIG_FILE` made the forced test more deterministic. The store pass logged `Stored 1118 out of total 1118 tokens`, then a vLLM pod restart plus exact prompt replay logged `LMCache hit tokens: 1118, need to load: 1117`; metrics showed `external_prefix_cache_hits_total=1117`.
- The ai-toolkit service on port 9200 did not expose obvious read-only introspection endpoints in this run. Probes for `/`, `/health`, `/metrics`, `/stats`, `/status`, `/buckets`, `/cache`, and `/debug/vars` all returned HTTP 404. Use LMCache/vLLM logs and `external_prefix_cache_*` metrics as the validation surface unless AWS documents a daemon inspection API.
- The real Dynamo frontend path works in `vllm-runtime:1.0.1` when frontend and worker are launched together with file discovery: `python3 -m dynamo.frontend --discovery-backend file --request-plane tcp --event-plane zmq` and `python3 -m dynamo.vllm --discovery-backend file --request-plane tcp --event-plane zmq`.
- In this image, `--connector lmcache` is rejected for the vLLM backend with `ValueError: --connector is no longer supported for the vLLM backend. Use --kv-transfer-config instead.` The equivalent accepted config is `{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}` plus `kv_connector_extra_config` for the HyperPod shared memory name.
- Dynamo frontend validation through `svc/dynamo-lmcache-frontend` proved routing and writes: `/health` listed `dyn://dynamo-hp-lmcache.backend.generate`, `dynamo_component_requests_total{dynamo_endpoint="generate"} 1`, and the short store pass wrote `lmcache:num_stored_tokens_total 1103`.
- Dynamo frontend post-restart replay hit HyperPod L2 with the 1103-token deterministic prompt: request latency dropped to 328 ms, usage reported `cached_tokens=1102`, metrics showed `vllm:external_prefix_cache_hits_total 1102`, `lmcache:num_hit_tokens_total 1103`, `lmcache:num_remote_read_requests_total 6`, and logs showed `Retrieved 1103 out of 1103 required tokens`.
- The longer 2093-token Dynamo frontend replay missed L2 after restart and stored again. Keep the published smoke prompt under the single-step/chunk-boundary-sensitive regime until the long prompt miss is understood.
