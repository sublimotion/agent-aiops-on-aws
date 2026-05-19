# GLM-5 on SageMaker HyperPod EKS — Serving Benchmark Spec

## Status: DRAFT (2026-03-05)

## Overview

Deploy GLM-5 by Zhipu AI (THUDM) on **SageMaker HyperPod for EKS** to benchmark serving performance with FP8 quantization and EAGLE speculative decoding. This spec evolves the vanilla-EKS `glm5.md` spec to use HyperPod's managed infrastructure: Training Plans for capacity, the Inference Operator for model lifecycle, one-click observability (AMP/AMG), deep health checks, and KV cache offloading to FSx Lustre.

GLM-5 is a 744B MoE model (256 routed + 1 shared expert, top-8 routing, 40B active per token) with a hybrid architecture combining MoE, Multi-Latent Attention (MLA), and DeepSeek Sparse Attention (DSA).

**Why HyperPod over vanilla EKS:**

| Capability | Vanilla EKS (glm5.md) | HyperPod EKS (this spec) |
|---|---|---|
| GPU capacity | Manual capacity blocks + `run-instances` | Training Plans API — managed reservation |
| Node health | Self-managed DCGM exporter + Prometheus | Deep health checks (GPU, NVLink, EFA, NCCL) + auto-recovery |
| Model deployment | Hand-crafted K8s Deployment YAML | `InferenceEndpointConfig` CRD — managed lifecycle |
| Model staging | Manual FSx→NVMe init container script | `prefetchEnabled` — operator-managed S3/FSx staging |
| Monitoring | Self-hosted Prometheus + Grafana + DCGM | One-click AMP/AMG with pre-built GPU dashboards |
| Endpoint | NodePort + manual port-forward | SageMaker Endpoint API (`invoke_endpoint`) + ALB |
| KV cache | GPU VRAM only | L1 (CPU memory) + L2 (Redis/FSx) + intelligent routing |
| Autoscaling | Manual HPA or none | KEDA-based autoscaling on CloudWatch/Prometheus metrics |

**Why GLM-5:**
- 744B MoE with ~200K context, MIT license, FP8 variant (756 GB)
- EAGLE speculative decoding support — accelerates MoE decode latency
- Actively optimized by SGLang (FlashMLA, DeepGeMM FP8 kernels)
- SGLang implements GLM-5 DSA as `GlmMoeDsaForCausalLM` inheriting `DeepseekV2ForCausalLM` — benefits from DeepSeek V3's battle-tested MLA attention path

**Target Instances (dual-track benchmark):**

| Instance | GPUs | VRAM | KV Cache Headroom | Interconnect | Purpose |
|---|---|---|---|---|---|
| **ml.p5e.48xlarge** | 8× H200 | 1,128 GB | ~372 GB | NVLink 4 / NVSwitch | Primary — proven Hopper stack |
| **ml.p6-b200.48xlarge** | 8× B200 | 1,536 GB | ~780 GB | NVLink 5 / NVSwitch | Comparison — 2× KV cache, Blackwell perf |

The B200 comparison track validates whether Blackwell's 2× KV cache headroom meaningfully increases concurrent agent capacity (the key economics driver).

**SGLang architecture compatibility:**
- GLM-5 DSA uses **MLA (Multi-Latent Attention)** via `DeepseekV2AttentionMLA`
- On Hopper (H200): MLA defaults to **FA3** (FlashAttention 3)
- On Blackwell (B200): MLA defaults to **TRTLLM MLA** — optimized for sm_100, no special flags needed
- FP8 on Blackwell requires `--fp8-gemm-backend cutlass` (DeepGeMM crashes with non-ue8m0 scales)
- EAGLE3 speculative decoding is implemented (`set_eagle3_layers_to_capture` in `glm4_moe.py`)
- Tool-call parser: dedicated `glm47_moe_detector.py`

---

## Components

### 1. Compute — HyperPod EKS Cluster

- **Platform**: SageMaker HyperPod with EKS 1.32 orchestrator
- **System Nodes**: Managed by HyperPod (no manual m6i node group needed)
- **GPU Instance Groups** (dual-track via Training Plans):

  **Track A — Hopper (primary):**
  - `ml.p5e.48xlarge`: 8× H200 (141 GB HBM3e each), 2 TiB DDR5, NVLink 4
  - NVMe: 8× 3.84 TB SSDs (~30 TB total)
  - Training Plan quota: 256 instances per region

  **Track B — Blackwell (comparison):**
  - `ml.p6-b200.48xlarge`: 8× B200 (192 GB HBM3e each), NVLink 5
  - Training Plan quota: **8 instances** per region (limited)
  - Available in: us-east-1, us-east-2, us-west-2 only

- Deep health checks: `InstanceStress` + `InstanceConnectivity` enabled (both tracks)
- Auto node recovery: enabled
- **Region**: us-east-1 or us-west-2 (both p5e and p6-b200 available)
- **Availability Zone**: Determined by Training Plan offering (`availability_zone_id`)

#### Training Plan (Capacity Provisioning)

Training Plans replace manual capacity blocks. Reserve GPU capacity via the SageMaker API:

```python
import boto3
sm = boto3.client('sagemaker')

# 1. Search for available offerings
offerings = sm.search_training_plan_offerings(
    InstanceType='ml.p5e.48xlarge',
    InstanceCount=1,
    DurationHours=4,
    TargetResources=['hyperpod-cluster']
)

# 2. Select offering and create plan
plan = sm.create_training_plan(
    TrainingPlanName='glm5-hyperpod-bench',
    TrainingPlanOfferingId=offerings['TrainingPlanOfferings'][0]['TrainingPlanOfferingId']
)

# 3. Pass training_plan_arn to Terraform instance_groups
```

The Training Plan ARN is passed to the HyperPod cluster's instance group configuration (see Terraform Variables).

### 1a. Deep Health Checks (replaces manual GPU pre-flight)

HyperPod runs these automatically at cluster creation and node replacement:

| Check | What It Validates |
|---|---|
| GPU/NVLink count | All 8 H200 GPUs + NVLink present |
| DCGM Level 4 | Full memory diagnostics including stress tests |
| EFA bandwidth | Must meet threshold (~80 GB/s) |
| NCCL all_reduce | Cross-GPU collective communication performance |
| Hardware stress | CPU, memory, storage stress-ng validation |

Nodes that fail deep health checks are automatically replaced. No manual `nvidia-smi` / `nccl_diag.py` pre-flight needed.

### 1b. Blackwell B200 Compatibility Assessment

GLM-5's SGLang implementation (`GlmMoeDsaForCausalLM`) inherits from `DeepseekV2ForCausalLM`, giving it access to DeepSeek V3's mature MLA attention path. This makes B200 compatibility **low risk**:

| Component | B200 (sm_100) | Status | Evidence |
|---|---|---|---|
| **MLA attention** | TRTLLM MLA (default) | Supported | SGLang attention backend matrix: Blackwell MLA → `trtllm_mla` |
| **FP8 GeMM** | Cutlass backend | Supported | `--fp8-gemm-backend cutlass` (proven on sm_120) |
| **FP8 KV cache** | TRTLLM MLA supports FP8 KV | Supported | SGLang FP8 KV cache support matrix |
| **MoE routing** | FusedMoE Triton kernels | Proven | Qwen3-Next MoE on Blackwell sm_120 |
| **NCCL** | 2.26.2+ with NVLink 5 | Supported | NVLink topology avoids PCIe bug |
| **EAGLE3** | `set_eagle3_layers_to_capture` | In model code | `glm4_moe.py` has EAGLE3 support |
| **CUDA graphs** | Supported (MLA, not GDN) | Expected | `--disable-cuda-graph` only needed for hybrid GDN |
| **Tool-call parser** | `glm47_moe_detector.py` | Dedicated | Not hardware-dependent |

**Not needed on B200** (corrections from earlier analysis):
- ~~`--attention-backend triton`~~ — only for hybrid GDN models (Qwen3-Next), not MLA
- ~~`--disable-cuda-graph`~~ — only for hybrid GDN + HiCache, not MLA models
- ~~FlashMLA validation concern~~ — SGLang defaults to TRTLLM MLA on Blackwell, not FlashMLA

**Required on B200:**
- `--fp8-gemm-backend cutlass` — DeepGeMM crashes on all Blackwell architectures
- NCCL 2.26.2+ container (NGC 25.03+)
- CUDA 13.0+ toolkit

### 2. Model

- **Model ID**: `zai-org/GLM-5-FP8`
- **Architecture**: `glm_moe_dsa` — Hybrid MoE + Multi-Latent Attention + DeepSeek Sparse Attention
  - 744B total params (256 routed experts × 2.9B + 1 shared expert, top-8 routing, 40B active per token)
  - 80 transformer layers
  - Hidden size: 8,192, MLA heads: 64, MLA key-value heads: 8
  - Vocabulary: 256,000 tokens (multilingual tokenizer)
- **Context Length**: ~200K tokens native (testing up to 196K)
- **Format**: safetensors (FP8), 756 GB disk footprint
- **License**: MIT

#### Serving Engine — Custom SGLang Container

The HyperPod Inference Operator accepts custom container images via `worker.image`. Build a custom SGLang container following the GLM-4.5 pattern from `aws-samples/sagemaker-genai-hosting-examples`:

```dockerfile
FROM lmsys/sglang:v0.5.2-cu124

# SageMaker-compatible entrypoint
COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT ["/usr/bin/serve"]
```

The `serve` script converts `OPTION_*` environment variables to SGLang CLI flags:

```bash
#!/bin/bash
PREFIX="OPTION_"
ARG_PREFIX="--"
ARGS=(--port 8080)

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

echo "SGLang args: [${ARGS[@]}]"
exec python3 -m sglang.launch_server "${ARGS[@]}"
```

Push to ECR: `<account>.dkr.ecr.<region>.amazonaws.com/sglang-glm5:v0.5.2`

#### Parallelism Strategy

| Config | Instance | TP | GPUs | KV Cache Headroom | Use case |
|---|---|---|---|---|---|
| `tp8-h200` | p5e (H200) | 8 | 8 | ~372 GB | Hopper baseline |
| `tp8-b200` | p6-b200 (B200) | 8 | 8 | ~780 GB | Blackwell — 2× KV capacity |

#### Serving Configuration — H200 (Track A)

```yaml
OPTION_MODEL_PATH: /opt/ml/model
OPTION_TP_SIZE: "8"
OPTION_DTYPE: bfloat16
OPTION_CONTEXT_LENGTH: "131072"
OPTION_CHUNKED_PREFILL_SIZE: "32768"
OPTION_MAX_RUNNING_REQUESTS: "256"
OPTION_MEM_FRACTION_STATIC: "0.85"
OPTION_TOOL_CALL_PARSER: glm47
OPTION_REASONING_PARSER: glm45
OPTION_SERVED_MODEL_NAME: glm-5-fp8
```

> On Hopper, SGLang defaults to FA3 for MLA attention. No extra flags needed.

#### Serving Configuration — B200 (Track B)

```yaml
OPTION_MODEL_PATH: /opt/ml/model
OPTION_TP_SIZE: "8"
OPTION_DTYPE: bfloat16
OPTION_FP8_GEMM_BACKEND: cutlass          # Required: DeepGeMM crashes on Blackwell
OPTION_CONTEXT_LENGTH: "131072"
OPTION_CHUNKED_PREFILL_SIZE: "32768"
OPTION_MAX_RUNNING_REQUESTS: "512"         # Higher: 2× KV cache allows more concurrency
OPTION_MEM_FRACTION_STATIC: "0.85"
OPTION_TOOL_CALL_PARSER: glm47
OPTION_REASONING_PARSER: glm45
OPTION_SERVED_MODEL_NAME: glm-5-fp8
```

> On Blackwell, SGLang defaults to TRTLLM MLA for attention — optimized for sm_100. The only required flag is `--fp8-gemm-backend cutlass`. CUDA graphs are supported (unlike hybrid GDN models). `--max-running-requests` raised to 512 to leverage B200's larger KV cache.

### 3. Inference Operator — `InferenceEndpointConfig` CRD

The HyperPod Inference Operator manages the full model deployment lifecycle:

```yaml
apiVersion: inference.sagemaker.aws.amazon.com/v1
kind: InferenceEndpointConfig
metadata:
  name: glm5-fp8
  namespace: default
spec:
  endpointName: glm5-fp8
  modelName: glm5-fp8
  instanceType: ml.p5e.48xlarge
  invocationEndpoint: v1/chat/completions
  replicas: 1
  modelSourceConfig:
    modelSourceType: s3
    s3Storage:
      bucketName: glm5-model-weights
      region: us-west-2
    modelLocation: GLM-5-FP8
    prefetchEnabled: true
  kvCacheSpec:
    enableL1Cache: true
    enableL2Cache: true
    l2CacheSpec:
      l2CacheBackend: redis
      l2CacheLocalUrl: redis://redis.default.svc.cluster.local:6379
  intelligentRoutingSpec:
    enabled: true
    routingStrategy: prefixaware
  loadBalancer:
    healthCheckPath: /health
  worker:
    image: <account>.dkr.ecr.<region>.amazonaws.com/sglang-glm5:v0.5.2
    modelInvocationPort:
      containerPort: 8080
      name: http
    modelVolumeMount:
      name: model-weights
      mountPath: /opt/ml/model
    resources:
      limits:
        nvidia.com/gpu: 8
      requests:
        nvidia.com/gpu: 8
        cpu: 180000m
        memory: 1800Gi
    environmentVariables:
      - name: OPTION_TP_SIZE
        value: "8"
      - name: OPTION_DTYPE
        value: "bfloat16"
      - name: OPTION_CONTEXT_LENGTH
        value: "131072"
      - name: OPTION_CHUNKED_PREFILL_SIZE
        value: "32768"
      - name: OPTION_MAX_RUNNING_REQUESTS
        value: "256"
      - name: OPTION_MEM_FRACTION_STATIC
        value: "0.85"
      - name: OPTION_TOOL_CALL_PARSER
        value: "glm47"
      - name: OPTION_REASONING_PARSER
        value: "glm45"
      - name: OPTION_SERVED_MODEL_NAME
        value: "glm-5-fp8"
```

#### Model Invocation

Once deployed, invoke via SageMaker Runtime API:

```python
import boto3, json
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='glm5-fp8',
    ContentType='application/json',
    Body=json.dumps({
        "model": "glm-5-fp8",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512,
        "temperature": 0.7
    })
)
print(json.loads(response['Body'].read()))
```

### 4. Networking

- **VPC**: Created by HyperPod Terraform module (`modules/vpc`)
- **Private Subnets**: /16 CIDR per AZ for HyperPod nodes
- **Closed Network**: `closed_network = true` (air-gap deployment)
  - No internet gateway or NAT gateway
  - All traffic via VPC endpoints
- **VPC Endpoints**: S3 (gateway), ECR API, ECR DKR, STS, CloudWatch Logs, CloudWatch Monitoring, SSM, SSM Messages, EC2, EC2 Messages, EKS Auth
- **Security Group**: EFA-enabled, self-referencing for all ports (HyperPod module default)

### 5. Storage

#### Model Weights — S3 with Operator-Managed Staging

Model weights are staged by the Inference Operator's `prefetchEnabled` mechanism:

```
S3 bucket ──(prefetchEnabled)──▶ Pod volume mount ──▶ GPU VRAM
 s3://glm5-model-weights/        /opt/ml/model/       (engine startup)
 GLM-5-FP8/ (756 GB)            (S3 CSI driver)
```

**Pre-session setup** (from a host with internet access):
```bash
# Download model
huggingface-cli download zai-org/GLM-5-FP8 \
  --local-dir ./GLM-5-FP8/ \
  --local-dir-use-symlinks False

# Upload to S3
aws s3 sync ./GLM-5-FP8/ s3://glm5-model-weights/GLM-5-FP8/
```

#### FSx Lustre — KV Cache Offloading

FSx Lustre serves dual purposes:
1. **Alternative model staging**: If S3 prefetch is too slow for 756 GB, switch to `modelSourceType: fsx`
2. **KV cache offload**: L2 cache backend for long-context inference

```
FSx Lustre (PERSISTENT_2, 4.8 TiB, 500 MB/s/TiB)
├── /fsx/models/GLM-5-FP8/    ← Model weights (backup staging path)
└── /fsx/kv-cache/             ← KV cache offload (via SGLang --kv-cache-dir)
```

#### KV Cache Architecture

For GLM-5's 200K context window, KV cache can exceed GPU VRAM. The tiered approach varies by instance:

| Tier | Backend | H200 (p5e) | B200 (p6-b200) | Latency | Shared |
|---|---|---|---|---|---|
| GPU VRAM | HBM3e | ~372 GB | ~780 GB | ns | No |
| L1 Cache | CPU memory (HyperPod tiered storage) | ~400 GB | ~400 GB | μs | No |
| L2 Cache | Redis or FSx Lustre | 4.8+ TiB | 4.8+ TiB | ms | Yes |
| **Total** | | **~5.5 TiB** | **~5.9 TiB** | | |

> **MLA KV cache advantage**: GLM-5 uses Multi-Latent Attention (via `DeepseekV2AttentionMLA`) which compresses KV cache by projecting keys/values into a low-rank latent space. This means effective KV cache capacity per GB is significantly higher than standard MHA models. The actual concurrent context capacity will be validated in P1b/P1b'.

- L1 is automatically managed by HyperPod's tiered storage daemon (enabled at cluster creation)
- L2 via Redis enables KV cache sharing across replicas and survives pod restarts
- FSx Lustre with GDS provides durable, high-throughput L2 for long-context workloads
- B200's 2× GPU VRAM headroom means many agent workloads may never spill to L1/L2, eliminating offload latency entirely

### 6. Monitoring — Managed AMP/AMG Observability

HyperPod's one-click observability provisions **Amazon Managed Prometheus (AMP)** and **Amazon Managed Grafana (AMG)** automatically. All metrics flow through managed pipelines — no direct `/metrics` scraping needed.

#### Metric Collection Architecture

```
GPU Node (DCGM, node exporter, SGLang)
  │
  ├── HyperPod Health Monitoring Agent ──▶ AMP workspace
  ├── ADOT Collector (OpenTelemetry) ──▶ AMP workspace
  └── CloudWatch Container Insights ──▶ CloudWatch Logs
                                            │
                                     AMG workspace
                                     (pre-built dashboards)
```

#### Metrics Categories

| Category | Source | Key Metrics | AMP Query Pattern |
|---|---|---|---|
| **GPU Compute** | DCGM exporter | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` | `DCGM_FI_DEV_GPU_UTIL{instance=~".*p5e.*"}` |
| **GPU Memory** | DCGM exporter | `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_DEV_FB_USED` | Used to detect KV cache pressure |
| **NVLink** | DCGM exporter | `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | TP=8 all-reduce efficiency |
| **EFA** | EFA metrics | Bandwidth, latency per adapter | NVSwitch cross-GPU traffic |
| **FSx Lustre** | FSx CloudWatch | `DataReadBytes`, `DataWriteBytes`, `MetadataOperations` | KV cache offload I/O |
| **Node** | Node exporter | CPU, memory, disk I/O, network | CPU memory for L1 KV cache |
| **SGLang Engine** | SGLang `/metrics` via ADOT | See below | Custom PromQL via AMP |
| **Inference Operator** | CRD controller | Endpoint health, TTFB, restarts | Pre-built dashboard |

#### SGLang Metrics (collected via ADOT → AMP)

The SGLang container exposes Prometheus metrics on port 8080. The ADOT collector (deployed by HyperPod observability add-on) scrapes these and forwards to AMP.

**Serving metrics:**
- `sglang:num_running_requests` / `sglang:num_waiting_requests`
- `sglang:avg_prompt_throughput_toks_per_s` / `sglang:avg_generation_throughput_toks_per_s`
- `sglang:time_to_first_token_seconds` (histogram)
- `sglang:inter_token_latency_seconds` (histogram)

**KV cache metrics (critical for offloading analysis):**
- `sglang:kv_cache_usage_percent` — GPU VRAM KV cache fill level
- `sglang:prefix_cache_hit_rate` — RadixAttention cache hit ratio
- `sglang:prefix_cache_total_queries` / `sglang:prefix_cache_total_hits`
- `sglang:num_preemptions_total` — requests preempted due to KV cache pressure

**KV cache offloading metrics (when L1/L2 enabled):**
- `sglang:kv_cache_l1_usage_bytes` — CPU memory L1 cache usage
- `sglang:kv_cache_l2_usage_bytes` — Redis/FSx L2 cache usage
- `sglang:kv_cache_offload_ops_total` — GPU→L1 offload operations
- `sglang:kv_cache_restore_ops_total` — L1→GPU restore operations
- `sglang:kv_cache_eviction_total` — cache evictions per tier

> **Note**: KV cache offloading metrics availability depends on SGLang version and HyperPod operator integration. If SGLang doesn't expose L1/L2 metrics natively, use DCGM GPU memory metrics + node exporter CPU memory as proxies:
> - GPU KV pressure: `DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL`
> - CPU L1 fill: `node_memory_MemTotal - node_memory_MemAvailable` (delta from baseline)
> - FSx L2 I/O: CloudWatch `DataReadBytes` / `DataWriteBytes` for the FSx filesystem

#### Querying Metrics via AMP

All benchmark analysis queries go through AMP's PromQL endpoint, not direct pod scraping:

```python
import boto3

amp = boto3.client('amp')
workspace_id = '<from-terraform-output>'

# Query KV cache usage during swarm test
response = amp.query(
    workspaceId=workspace_id,
    query='avg(sglang:kv_cache_usage_percent{job="sglang"}) by (instance)',
    startTime=start_ts,
    endTime=end_ts,
    step='15s'
)
```

Or via Grafana explore UI for interactive analysis.

#### Pre-Built + Custom Dashboards

| Dashboard | Source | Purpose |
|---|---|---|
| Cluster Metrics | HyperPod built-in | GPU count, utilization, filesystem |
| Task Metrics | HyperPod built-in | Per-job resource utilization |
| Training & Inference | HyperPod built-in | TTFB, latency, restarts |
| **GLM-5 KV Cache** | Custom (import to AMG) | KV cache tiers, offload ops, evictions |
| **Agent Swarm** | Custom (import to AMG) | Per-agent TTFT, concurrent sessions, throughput |
| **Economics** | Custom (import to AMG) | tok/s, $/1M tokens, engineer capacity |

### 7. Node Access — SSM

```bash
# Get cluster ID and instance ID
aws sagemaker describe-cluster --cluster-name glm5-cluster
aws sagemaker list-cluster-nodes --cluster-name glm5-cluster

# SSM session
aws ssm start-session \
    --target sagemaker-cluster:<cluster-id>_glm5-p5e-<instance-id> \
    --region us-west-2

# SSH via SSM proxy (~/.ssh/config)
Host glm5-gpu
    HostName sagemaker-cluster:<cluster-id>_glm5-p5e-<instance-id>
    User ec2-user
    ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p
```

---

## Terraform Configuration

### Module Source

Uses the upstream HyperPod EKS Terraform module from `aws-samples/awsome-distributed-training`:

```hcl
# blueprints/glm5-hyperpod/main.tf

module "hyperpod" {
  source = "git::https://github.com/aws-samples/awsome-distributed-training.git//1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf?ref=<pinned-commit>"

  # Core
  kubernetes_version    = var.kubernetes_version
  eks_cluster_name      = var.eks_cluster_name
  hyperpod_cluster_name = var.hyperpod_cluster_name
  resource_name_prefix  = var.resource_name_prefix
  aws_region            = var.aws_region

  # GPU instance group with Training Plan
  instance_groups = var.instance_groups

  # Operators
  create_hyperpod_inference_operator_module = true
  create_hyperpod_training_operator_module  = false
  create_task_governance_module             = true

  # FSx Lustre
  create_fsx_module         = true
  create_new_fsx_filesystem = true
  fsx_storage_capacity      = var.fsx_storage_capacity
  fsx_throughput            = var.fsx_throughput

  # Observability
  create_observability_module      = true
  accelerated_compute_metric_level = "ADVANCED"
  network_metric_level             = "ADVANCED"
  node_metric_level                = "ADVANCED"
  cluster_metric_level             = "ADVANCED"
  logging_enabled                  = true

  # Closed network (air-gap)
  closed_network              = true
  eks_endpoint_public_access  = false
  eks_endpoint_private_access = true
  create_s3_endpoint          = true
  create_ec2_endpoint         = true
  create_ecr_api_endpoint     = true
  create_ecr_dkr_endpoint     = true
  create_sts_endpoint         = true
  create_logs_endpoint        = true
  create_monitoring_endpoint  = true
  create_ssm_endpoint         = true
  create_ssmmessages_endpoint = true
  create_ec2messages_endpoint = true
  create_eks_auth_endpoint    = true

  # Deep health checks
  enable_deep_health_check = true
  enable_job_auto_restart  = true
  auto_node_recovery       = true
}
```

### Terraform Variables

```hcl
# blueprints/glm5-hyperpod/variables.tf

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "resource_name_prefix" {
  type    = string
  default = "glm5"  # Keep short — IAM role names have 64-char limit
}

variable "kubernetes_version" {
  type    = string
  default = "1.32"
}

variable "eks_cluster_name" {
  type    = string
  default = "glm5-hyperpod-eks"
}

variable "hyperpod_cluster_name" {
  type    = string
  default = "glm5-cluster"
}

variable "instance_groups" {
  type = list(object({
    name                      = string
    instance_type             = string
    instance_count            = number
    ebs_volume_size_in_gb     = number
    threads_per_core          = number
    enable_stress_check       = bool
    enable_connectivity_check = bool
    lifecycle_script          = string
    availability_zone_id      = string
    training_plan_arn         = optional(string)
  }))
  default = [
    {
      name                      = "glm5-p5e"
      instance_type             = "ml.p5e.48xlarge"
      instance_count            = 1
      ebs_volume_size_in_gb     = 500
      threads_per_core          = 2
      enable_stress_check       = true
      enable_connectivity_check = true
      lifecycle_script          = "on_create.sh"
      availability_zone_id      = "usw2-az2"  # Set based on Training Plan offering
      training_plan_arn         = null          # Set after SearchTrainingPlanOfferings
    }
  ]
  # For B200 comparison track, use:
  # {
  #   name                      = "glm5-b200"
  #   instance_type             = "ml.p6-b200.48xlarge"
  #   instance_count            = 1
  #   ebs_volume_size_in_gb     = 500
  #   threads_per_core          = 2
  #   enable_stress_check       = true
  #   enable_connectivity_check = true
  #   lifecycle_script          = "on_create.sh"
  #   availability_zone_id      = "usw2-az2"
  #   training_plan_arn         = null
  # }
}

variable "fsx_storage_capacity" {
  type    = number
  default = 4800  # 4.8 TiB
}

variable "fsx_throughput" {
  type    = number
  default = 500  # MB/s per TiB
}
```

### Custom tfvars

```hcl
# blueprints/glm5-hyperpod/glm5.tfvars

aws_region            = "us-west-2"
resource_name_prefix  = "glm5"
kubernetes_version    = "1.32"
eks_cluster_name      = "glm5-hyperpod-eks"
hyperpod_cluster_name = "glm5-cluster"

instance_groups = [
  {
    name                      = "glm5-p5e"
    instance_type             = "ml.p5e.48xlarge"
    instance_count            = 1
    ebs_volume_size_in_gb     = 500
    threads_per_core          = 2
    enable_stress_check       = true
    enable_connectivity_check = true
    lifecycle_script          = "on_create.sh"
    availability_zone_id      = "usw2-az2"
    training_plan_arn         = "arn:aws:sagemaker:us-west-2:ACCOUNT:training-plan/glm5-bench"
  }
]

fsx_storage_capacity = 4800
fsx_throughput       = 500
```

---

## Air-Gap Deployment Requirements

Closed network mode eliminates all outbound internet access. All artifacts must be pre-staged.

### Container Images

Build and push the custom SGLang container to private ECR before the Training Plan starts:

| Image | Source | ECR Tag | Notes |
|---|---|---|---|
| SGLang server | `lmsys/sglang:v0.5.2-cu124` + custom `serve` | `<ecr>/sglang-glm5:v0.5.2` | GLM-5 arch deps baked in |
| Benchmark runner | `python:3.11-slim` | `<ecr>/bench-runner:latest` | custbench + dependencies |

Use the upstream module's ECR mirroring tools:
```bash
# From hyperpod-eks-tf/tools/
./copy-images-to-ecr.sh
```

### Model Weights

Upload to S3 before the Training Plan activates:

```bash
huggingface-cli download zai-org/GLM-5-FP8 \
  --local-dir ./GLM-5-FP8/ \
  --local-dir-use-symlinks False

aws s3 sync ./GLM-5-FP8/ s3://glm5-model-weights/GLM-5-FP8/
```

Size: 756 GB. Allow ~2-3 hours for download + ~1 hour for S3 upload on a fast connection.

### HyperPod Helm Chart

The Terraform module clones `sagemaker-hyperpod-cli` for Helm charts:

```bash
git clone https://github.com/aws/sagemaker-hyperpod-cli.git /tmp/helm-repo
```

This must be done from a host with internet access before `terraform apply` in a closed network. Set `helm_repo_path` to the local clone path.

---

## Benchmark Design — Coding Agent Economics

The primary goal is to answer: **How many concurrent coding engineers can a single p5e.48xlarge running GLM-5 support, and at what cost compared to using the Claude API?**

Benchmarks are organized to progressively validate tool-use capability, measure agent swarm pressure limits, and compute per-engineer economics.

### Invocation Pattern

```python
# HyperPod — SageMaker Endpoint
import boto3, json
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='glm5-fp8',
    ContentType='application/json',
    Body=json.dumps({
        "model": "glm-5-fp8",
        "messages": messages,
        "tools": tool_definitions,
        "max_tokens": 4096,
        "temperature": 0.2
    })
)
```

> SageMaker endpoint routing adds ~5-10ms overhead vs. direct pod access. For latency-critical comparisons, also measure direct pod access via `kubectl port-forward`.

### Priority Tiers

```
P0 (must-have): Smoke test + tool-call validation                   ~30 min
P1 (must-have): Agent pressure testing + swarm capacity              ~2 hrs
P2 (should-have): KV cache offloading under agent load               ~1 hr
P3 (should-have): Economics analysis — engineers per node            ~30 min (analysis)
Total budget: ~4 hrs
```

### P0: Smoke Test + Tool-Call Validation (MUST HAVE)

**Goal**: GLM-5 loads, serves inference, and can handle tool calls correctly.

| Step | Test | Context | Config |
|---|---|---|---|
| 0a | Health check | - | Model loads, `/v1/chat/completions` responds |
| 0b | Basic inference | 1K input / 512 output | QPS 0.5, no tools |
| 0c | BFCL tool-call accuracy | 200 scenarios | 5 categories (see below) |

**BFCL Tool-Call Categories:**

| Category | Scenarios | What It Tests |
|---|---|---|
| Simple function call | 40 | Single tool invocation with correct arguments |
| Multi-tool selection | 40 | Choose correct tool from 5+ definitions |
| Parallel tool calls | 40 | Call multiple tools in one response |
| Multi-turn tool use | 40 | Tool result → follow-up → second tool call |
| Structured output | 40 | JSON schema compliance in tool arguments |

**Tool definitions** (coding agent relevant):

```json
[
  {"name": "read_file", "parameters": {"path": "string"}},
  {"name": "write_file", "parameters": {"path": "string", "content": "string"}},
  {"name": "run_command", "parameters": {"command": "string", "timeout": "integer"}},
  {"name": "web_search", "parameters": {"query": "string"}},
  {"name": "create_pull_request", "parameters": {"title": "string", "body": "string", "branch": "string"}},
  {"name": "query_database", "parameters": {"sql": "string", "database": "string"}},
  {"name": "list_directory", "parameters": {"path": "string", "recursive": "boolean"}},
  {"name": "grep_codebase", "parameters": {"pattern": "string", "file_glob": "string"}}
]
```

**Gate**:
- BFCL < 70%: **STOP** — not viable for coding agents
- BFCL 70-75%: **CAUTION** — viable for batch/swarm only (no interactive)
- BFCL >= 75%: **PROCEED** to P1
- BFCL >= 80%: **STRONG** — competitive with Claude Sonnet for tool use

### P1: Agent Swarm Pressure Testing (MUST HAVE)

**Goal**: Determine maximum concurrent coding agents before SLO violation. This directly maps to "how many engineers can one node support."

#### P1a — Coding Agent Simulation

Each simulated agent follows a realistic multi-turn coding workflow:

```
Agent workflow (per session):
1. Receive task prompt (bug fix / feature / refactor)
2. Read file(s) via tool call           → 2K-8K tokens context
3. Analyze + plan (model thinking)      → 500-2K tokens output
4. Write fix via tool call              → 1K-4K tokens output
5. Run tests via tool call              → wait 5-30s (simulated execution)
6. Analyze results                      → 500-1K tokens
7. Iterate (steps 4-6) × 2-3 cycles
```

**Realistic agent task prompts** (12 scenarios):

| # | Task | Input Context | Expected Turns |
|---|---|---|---|
| 1 | Fix authentication bypass (JWT expiration) | 4K | 3 |
| 2 | Add rate limiting middleware | 3K | 4 |
| 3 | Database migration with rollback | 6K | 3 |
| 4 | Refactor monolith to service | 8K | 5 |
| 5 | Add unit tests for payment module | 5K | 3 |
| 6 | Profile and fix memory leak | 4K | 4 |
| 7 | WebSocket reconnection logic | 3K | 3 |
| 8 | CI/CD pipeline optimization | 2K | 2 |
| 9 | Add distributed tracing (OpenTelemetry) | 6K | 4 |
| 10 | CSV export with streaming for large datasets | 3K | 3 |
| 11 | Circuit breaker pattern implementation | 4K | 3 |
| 12 | RBAC permission system | 5K | 4 |

#### P1b — Concurrency Sweep (H200 Track A)

| Step | Concurrent Agents | Purpose | AMP Metrics to Capture |
|---|---|---|---|
| 1b-1 | 4 | Baseline — no contention | TTFT, ITL, GPU util, KV usage |
| 1b-2 | 8 | Light load | Same |
| 1b-3 | 16 | Moderate load | Same + waiting_requests |
| 1b-4 | 32 | Heavy load | Same + preemptions |
| 1b-5 | 48 | Stress | Same + error rate |
| 1b-6 | 64 | Overload | Same — find breaking point |
| 1b-7 | 96 | Extreme | Same — ceiling test |
| 1b-8 | 128 | Saturation | Same — absolute limit |

#### P1b' — Concurrency Sweep (B200 Track B)

Run the same sweep on `ml.p6-b200.48xlarge` to measure Blackwell's advantage:

| Step | Concurrent Agents | Purpose | Expected vs H200 |
|---|---|---|---|
| 1b'-1 | 4 | Baseline | Lower ITL (faster decode on B200) |
| 1b'-2 | 16 | Moderate | Same or better |
| 1b'-3 | 32 | Heavy | Better — more KV headroom |
| 1b'-4 | 64 | H200's breaking point | B200 should still be stable |
| 1b'-5 | 96 | B200 stress zone | Main differentiator region |
| 1b'-6 | 128 | B200 heavy | 2× KV should push ceiling higher |
| 1b'-7 | 192 | B200 extreme | Test 2× capacity hypothesis |
| 1b'-8 | 256 | B200 saturation | Absolute limit |

> **Key hypothesis**: B200's 780 GB KV cache headroom (vs H200's 372 GB) should roughly double the SLO-max concurrent agents. This is the economics inflection point — if B200 supports 128+ agents where H200 caps at 64, the cost-per-engineer becomes competitive with Claude Sonnet.

**Per concurrency level, capture from AMP:**

```yaml
# PromQL queries (executed against AMP workspace)
gpu_utilization:
  query: 'avg(DCGM_FI_DEV_GPU_UTIL{instance=~".*p5e.*"})'
  interval: 15s

gpu_memory_used_pct:
  query: 'avg(DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE))'
  interval: 15s

kv_cache_fill:
  query: 'sglang:kv_cache_usage_percent'
  interval: 15s

prefix_cache_hit_rate:
  query: 'rate(sglang:prefix_cache_total_hits[5m]) / rate(sglang:prefix_cache_total_queries[5m])'
  interval: 15s

waiting_requests:
  query: 'sglang:num_waiting_requests'
  interval: 5s

preemptions:
  query: 'rate(sglang:num_preemptions_total[5m])'
  interval: 15s

cpu_memory_used:
  query: 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes'
  interval: 15s

fsx_read_throughput:
  query: 'rate(aws_fsx_DataReadBytes_sum[5m])'
  interval: 60s

fsx_write_throughput:
  query: 'rate(aws_fsx_DataWriteBytes_sum[5m])'
  interval: 60s
```

**Gate**: Identify **SLO-max concurrent agents** — highest concurrency where:
- TTFT p99 < 2000ms (agent waiting for first response)
- ITL p99 < 100ms (streaming feels responsive)
- Error rate < 1%
- No preemptions

#### P1c — Functional Coding Evaluation Under Load

Run at SLO-max concurrency to validate GLM-5 actually produces correct code, not just fast tokens:

| Task | What | Pass Criteria |
|---|---|---|
| `parse_date_none` | Fix stub implementation | Tests pass |
| `off_by_one_pagination` | Fix logic error in pagination | Tests pass |
| `missing_auth_check` | Add JWT expiration check | Tests pass + security review |
| `race_condition_counter` | Fix thread-unsafe counter | Tests pass under concurrent access |
| `csv_encoding_error` | Fix Unicode handling in CSV export | Tests pass with UTF-8 input |

**Metrics**: Task completion rate, average turns to completion, code quality (does fix introduce new issues).

### P2: KV Cache Offloading Under Agent Load (SHOULD HAVE)

**Goal**: Validate that KV cache tiering extends concurrent agent capacity without latency degradation.

#### P2a — Baseline (GPU VRAM Only)

Run P1b's SLO-max concurrency with KV cache in GPU VRAM only. Record:
- KV cache fill level (from AMP)
- GPU memory utilization (from DCGM via AMP)
- Preemption count

#### P2b — L1 Enabled (GPU + CPU Memory)

Enable L1 KV cache offloading (HyperPod tiered storage daemon, 20% CPU memory allocation = ~400 GB):

```yaml
kvCacheSpec:
  enableL1Cache: true
  enableL2Cache: false
```

Re-run P1b concurrency sweep. **Expected**: higher SLO-max concurrency (more KV cache capacity before evictions).

**AMP metrics delta vs baseline:**

| Metric | P2a (GPU only) | P2b (GPU + L1) | Delta |
|---|---|---|---|
| SLO-max concurrent agents | TBD | TBD | +N agents |
| KV cache total capacity | ~350 GB | ~750 GB | +400 GB |
| TTFT p99 at SLO-max | TBD | TBD | ΔMs |
| Preemption rate | TBD | TBD | Should decrease |
| CPU memory used | baseline | baseline + L1 fill | L1 usage |

#### P2c — L1 + L2 Enabled (GPU + CPU + Redis/FSx)

Enable full tiered caching:

```yaml
kvCacheSpec:
  enableL1Cache: true
  enableL2Cache: true
  l2CacheSpec:
    l2CacheBackend: redis
    l2CacheLocalUrl: redis://redis.default.svc.cluster.local:6379
```

Re-run at P2b's SLO-max + 50% concurrency. Monitor FSx I/O metrics from AMP/CloudWatch:

```yaml
# CloudWatch metrics for FSx (available in AMP via CloudWatch data source)
fsx_data_read_bytes:  'aws_fsx_DataReadBytes_sum'
fsx_data_write_bytes: 'aws_fsx_DataWriteBytes_sum'
fsx_metadata_ops:     'aws_fsx_MetadataOperations_sum'
```

**Gate**: L2 cache provides additional headroom without TTFT degradation > 20% vs L1-only.

#### P2d — KV Cache Warm-Start Test

Simulate pod restart (inference operator auto-recovery) and measure:
1. Time to first request after restart (cold KV cache)
2. KV cache hit rate after 100 requests (warm-up)
3. Compare L2-backed (Redis persists cache) vs no-L2 (cache lost)

### P3: Economics Analysis — Engineers per Node (SHOULD HAVE)

**Goal**: Build the business case for self-hosted GLM-5 vs. Claude API.

#### Agent Profile (realistic coding engineer usage)

Based on industry coding agent usage patterns:

| Parameter | Value | Source |
|---|---|---|
| Requests per engineer per hour | 30-60 | Claude Code / Cursor / Copilot telemetry estimates |
| Avg input tokens per request | 4,000 | Code context + system prompt + conversation history |
| Avg output tokens per request | 1,500 | Code generation + explanation |
| Tool calls per request | 1.5 | Read file, write file, run test |
| Active coding hours per day | 6 | Standard engineering workday |
| Peak concurrency factor | 0.4 | Not all engineers request simultaneously |

#### Cost Model

**Self-hosted GLM-5 on HyperPod:**

```
Monthly cost = Training Plan hourly rate × hours/month
             = ~$60-98/hr × 730 hrs/month (24/7)
             = ~$43,800 - $71,540/month

Engineers supported = SLO-max concurrent agents / peak concurrency factor
                    = SLO-max / 0.4

Cost per engineer per month = monthly cost / engineers supported
```

**Claude API (Sonnet 4.6 pricing as reference):**

```
Per engineer per month:
  Input:  45 req/hr × 4,000 tok × 6 hrs × 22 days = 23.76M input tokens
  Output: 45 req/hr × 1,500 tok × 6 hrs × 22 days = 8.91M output tokens

  Cost = (23.76M × $3/1M) + (8.91M × $15/1M)
       = $71.28 + $133.65
       = ~$205/engineer/month

  Claude Opus 4.6:
  Cost = (23.76M × $15/1M) + (8.91M × $75/1M)
       = $356.40 + $668.25
       = ~$1,025/engineer/month
```

#### Break-Even Analysis

```
Break-even engineers = GLM-5 monthly cost / Claude API cost per engineer

Example (Sonnet):  $60K/month ÷ $205/eng/month = ~293 engineers
Example (Opus):    $60K/month ÷ $1,025/eng/month = ~59 engineers
```

**The benchmark must determine**: How many engineers GLM-5 actually supports at SLO-compliant quality, and whether the cost per engineer beats Claude API at the observed capacity.

#### Output Table

| Metric | GLM-5 (measured) | Claude Sonnet 4.6 | Claude Opus 4.6 |
|---|---|---|---|
| BFCL tool-call accuracy | TBD (P0) | ~88% (reference) | ~92% (reference) |
| Coding task completion rate | TBD (P1c) | ~85% (reference) | ~90% (reference) |
| SLO-max concurrent agents | TBD (P1b) | Unlimited (API) | Unlimited (API) |
| TTFT p99 at SLO-max | TBD (P1b) | ~500ms (API) | ~1000ms (API) |
| Engineers supported (24/7) | TBD | Unlimited | Unlimited |
| Monthly cost (infrastructure) | ~$60K | $0 | $0 |
| Cost per engineer per month | TBD | ~$205 | ~$1,025 |
| Break-even engineers | TBD | TBD | TBD |
| Quality-adjusted break-even | TBD | TBD | TBD |

> **Quality adjustment**: If GLM-5 task completion rate is 70% vs Claude's 85%, the effective cost per engineer increases by 85/70 = 1.21x because engineers need more iterations. Factor this into the break-even calculation.

### HyperPod-Specific Tests

Run alongside P0-P3 as infrastructure validation:

| Step | What | Purpose | Metrics Source |
|---|---|---|---|
| H0 | GPU / NVLink / EFA validation | Hardware connectivity baseline | SSM + MCP tools |
| H1 | S3 prefetch timing | Measure 756 GB S3→pod staging time | CloudWatch / ADOT |
| H2 | KV cache tier effectiveness | L1/L2 hit rates under agent load | AMP PromQL |
| H3 | Intelligent routing | `prefixaware` vs `roundrobin` on shared-prefix | AMP PromQL |
| H4 | Node auto-recovery | Kill GPU process, measure recovery | HyperPod events + AMP |
| H5 | AMG dashboard validation | Verify all custom dashboards render | Manual |

#### H0: GPU / NVLink / EFA Connectivity Validation

Run immediately after HyperPod cluster creation, before model deployment. Use both HyperPod deep health check results and manual validation via SSM:

**GPU Inventory & Health:**
```bash
# Via SSM session to GPU node
nvidia-smi                          # 8× H200, driver version, CUDA version, ECC on
nvidia-smi topo -m                  # NVLink topology (all 8 GPUs connected via NVSwitch)
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
```

**NVLink Bandwidth:**
```bash
# NCCL all_reduce bandwidth test (should see >450 GB/s bus BW on NVSwitch)
# Use gpu-infra MCP tool:
mcp__gpu-infra__run_nccl_test(hosts=["<gpu-node-ip>"], gpus_per_node=8)
```

**EFA Adapters:**
```bash
# Verify all 32 EFA adapters are present and active
fi_info -p efa                      # List EFA providers
ibv_devinfo                         # InfiniBand verbs device info
ls /sys/class/infiniband/           # Should show 32 EFA devices
```

**Cluster Profile (via MCP):**
```bash
mcp__gpu-infra__discover_cluster(host="<gpu-node-ip>")
mcp__gpu-infra__check_gpu_health(host="<gpu-node-ip>", checks=["all"])
```

**Expected Results:**

| Check | Expected | Fail Action |
|---|---|---|
| GPU count | 8× H200 | HyperPod auto-replace should trigger |
| NVLink topology | All GPUs connected via NVSwitch | Do not proceed — TP=8 requires NVLink |
| NCCL all_reduce bus BW | > 450 GB/s | Investigate — may indicate faulty NVLink |
| EFA adapters | 32 active | Check `enable_connectivity_check` results |
| ECC errors (uncorrected) | 0 | HyperPod auto-replace should trigger |
| GPU temperature | < 80°C at idle | Check cooling/airflow |
| Xid errors | None in `dmesg` | Use `mcp__gpu-infra__explain_xid` to diagnose |

> **Note**: HyperPod deep health checks (`InstanceStress` + `InstanceConnectivity`) cover most of these automatically at node creation. H0 is a manual verification + recording of baseline metrics for comparison during the benchmark session. If deep health checks pass, H0 is a quick 5-minute sanity check rather than a full diagnostic.

---

## Success Criteria

### Coding Agent Viability

| Metric | Target | Phase | Gate |
|---|---|---|---|
| BFCL tool-call accuracy | >= 75% | P0 | Proceed to P1 |
| Coding task completion rate | >= 70% | P1c | Viable for swarm |
| H200 SLO-max concurrent agents | >= 16 | P1b | Economically interesting |
| B200 SLO-max concurrent agents | >= 2× H200 | P1b' | Blackwell value validated |
| Break-even vs Claude Opus | <= 100 engineers | P3 | Primary business case |
| Break-even vs Claude Sonnet | <= 300 engineers | P3 | Stretch business case |

### Latency SLOs (Agent Workload)

| Metric | Target | Condition |
|---|---|---|
| TTFT p99 | < 2000ms | At SLO-max concurrency |
| ITL p99 | < 100ms | Streaming decode |
| E2E agent turn | < 30s | Single tool-call round-trip |
| Error rate | < 1% | At SLO-max concurrency |
| Preemptions | 0 | At SLO-max concurrency |

### KV Cache Offloading

| Metric | Target | Phase |
|---|---|---|
| L1 increases SLO-max concurrency | >= 25% improvement over GPU-only | P2b |
| L2 provides additional headroom | >= 10% over L1-only | P2c |
| L2 warm-start recovery | < 60s to 80% hit rate | P2d |
| FSx I/O throughput under load | > 500 MB/s sustained | P2c (CloudWatch) |

### HyperPod Infrastructure

| Metric | Target | Phase |
|---|---|---|
| Deep health checks (GPU, NVLink, EFA, NCCL) | All pass | H0 |
| NVLink bandwidth (all-reduce) | > 450 GB/s bus bandwidth | H0 |
| EFA connectivity | All 32 adapters active | H0 |
| S3 prefetch (756 GB) | < 30 min | H1 |
| Auto-recovery time | < 5 min | H4 |
| Custom SGLang container | Serves via SageMaker endpoint | P0 |
| Endpoint latency overhead | < 20ms vs direct pod | P0 |

---

## Non-Requirements

- Multi-node distributed inference (single ml.p5e.48xlarge only)
- BF16 inference (FP8 only — 1.51 TB BF16 does not fit single node)
- vLLM support (SGLang is the reference engine for GLM-5)
- Slurm orchestration (EKS only)
- Production autoscaling beyond KEDA evaluation
- Multi-region deployment
- JumpStart model catalog (GLM-5 is a custom HuggingFace model)

---

## Security Requirements

- All storage encrypted (S3 SSE, FSx KMS, EBS KMS)
- Closed network — no outbound internet
- VPC endpoints for all AWS service access
- IAM roles with least privilege (IRSA via OIDC provider)
- SSM-only node access (no public SSH)
- TLS certificates for inference endpoints (stored in S3)
- GuardDuty VPC endpoint cleanup on destroy

---

## Cost Considerations

### Benchmark Session Cost

| Resource | Estimated Cost | Notes |
|---|---|---|
| Training Plan (ml.p5e.48xlarge, 4 hrs) | ~$236-392 | Equivalent to capacity block pricing |
| FSx Lustre 4.8 TiB | ~$0.145/GB/month | Destroy between sessions |
| EKS control plane | $0.10/hr | Persistent |
| AMP workspace | $0.03/10K samples | Minimal for 4-hour session |
| AMG workspace | $9/active editor/month | Pro-rated |
| S3 model storage (756 GB) | ~$17/month | Persistent between sessions |
| **Total benchmark session** | ~$250-410 | Training Plan dominates |

### Production Economics (24/7 operation)

| Scenario | Monthly Cost | Engineers (est.) | Cost/Engineer/Month |
|---|---|---|---|
| **GLM-5 on H200** (1× p5e) | ~$43,800-71,540 | TBD from P1b | TBD |
| **GLM-5 on B200** (1× p6-b200) | ~$55,000-90,000 (est.) | TBD from P1b' | TBD |
| **Claude Sonnet 4.6 API** | $205 × N engineers | Unlimited | ~$205 |
| **Claude Opus 4.6 API** | $1,025 × N engineers | Unlimited | ~$1,025 |

> B200 pricing is estimated at 1.25-1.5× H200 based on historical GPU generation pricing. Actual Training Plan pricing will be visible after account allowlisting.

**Break-even calculation** (filled after benchmarks):

```
GLM-5 viable if:
  (monthly_infra_cost / SLO_max_engineers) < Claude_API_cost_per_engineer

With quality adjustment:
  effective_cost = infra_cost / (SLO_max_engineers × quality_ratio)
  where quality_ratio = GLM5_task_completion_rate / Claude_task_completion_rate
```

**H200 scenarios** (pre-benchmark estimates):

| SLO-max agents | Quality ratio | Cost/eng (H200) | vs Sonnet | vs Opus |
|---|---|---|---|---|
| 16 | 0.82 | ~$3,340 | 16x worse | 3.3x worse |
| 32 | 0.82 | ~$1,670 | 8x worse | 1.6x worse |
| 64 | 0.82 | ~$835 | 4x worse | Competitive |
| 128 | 0.82 | ~$418 | 2x worse | 2.5x better |

**B200 scenarios** (pre-benchmark estimates, ~$72K/month est.):

| SLO-max agents | Quality ratio | Cost/eng (B200) | vs Sonnet | vs Opus |
|---|---|---|---|---|
| 64 | 0.82 | ~$1,098 | 5x worse | Competitive |
| 128 | 0.82 | ~$549 | 2.7x worse | 1.9x better |
| 192 | 0.82 | ~$366 | 1.8x worse | 2.8x better |
| 256 | 0.82 | ~$274 | 1.3x worse | 3.7x better |

> **Key insight**: The economics hinge on agent density. H200 needs 200+ agents to beat Claude Sonnet — unlikely on a single node. B200's 2× KV headroom could push to 128-256 agents, making it competitive with Opus and approaching Sonnet parity. The benchmark must validate this hypothesis.
>
> **Multi-node scaling**: If a single node can't reach break-even density, multiple p5e nodes behind intelligent routing could aggregate capacity. However, this adds latency and eliminates KV cache sharing benefits. The B200 single-node path is preferable if the capacity hypothesis holds.

---

## Known Limitations

1. **Custom SGLang container unvalidated**: The Inference Operator examples use DJL/TGI/vLLM. SGLang container compatibility with the `worker.image` field needs validation — specifically the `serve` script entrypoint and `/v1/chat/completions` endpoint path
2. **S3 prefetch for 756 GB**: `prefetchEnabled` staging time for a 756 GB model is unknown. If too slow (>30 min), fall back to `modelSourceType: fsx` with pre-staged weights on FSx Lustre
3. **Training Plan allowlisting**: Account must be allowlisted for Training Plans (contact AWS account team or file support case). IAM permissions alone are not sufficient
4. **Training Plan availability**: p5e offerings are dynamic; p6-b200 quota is limited to **8 instances** per region. Must search and reserve in advance
5. **KV cache L2 with SGLang**: The Inference Operator's `kvCacheSpec` (enableL1Cache/enableL2Cache) drives an **LMCache-based** tiered KV store via the operator's daemon — it has no integration point with SGLang, which uses its own in-process HiCache. Flipping `kvCacheSpec` on an SGLang image just mounts an unused `lmcache-config` volume. For GLM-5 specifically, LMCache's SGLang adapter is also blocked on MLA attention (PR #2629 OPEN, issue #3192). **Use `--enable-hierarchical-cache` (SGLang HiCache) and leave `kvCacheSpec` unset** for SGLang tracks. Switching to vLLM+LMCache would engage the operator's cache layer, but GLM-5's multi-group MLA is separately blocked pending LMCache PR #2951 (OPEN, targets `dev`). SGLang's native `--kv-cache-dir` for disk offload to FSx still needs separate validation.
6. **Intelligent routing + SGLang**: `prefixaware` routing relies on the operator intercepting requests to track prefix trees. Must validate with SGLang's RadixAttention which does its own prefix caching internally
7. **B200 FP8 requires cutlass backend**: DeepGeMM crashes on Blackwell with non-ue8m0 scale formats. Must use `--fp8-gemm-backend cutlass`. This flag is available in SGLang nightly builds and v0.5.2+
8. **B200 NCCL version**: Requires NCCL 2.26.2+ (NGC 25.03+ containers). Earlier NCCL versions have Blackwell-specific bugs. NVLink 5 topology avoids the PCIe-specific shared memory bug but version requirement remains
9. **HyperPod Terraform module maturity**: The `hyperpod-eks-tf` module is from `aws-samples` (not official Terraform Registry). Pin to a specific commit to avoid breaking changes
10. **IAM Identity Center required**: AMG authentication requires IAM Identity Center. Must be enabled in the account before deploying observability
11. **Closed network Helm chart dependency**: `terraform apply` requires a local clone of `sagemaker-hyperpod-cli` for Helm charts — must be done from a host with internet access
12. **Endpoint latency overhead**: SageMaker endpoint routing adds ~5-10ms vs direct pod access. Benchmark both paths for accurate latency characterization
13. **B200 pricing unknown**: Training Plan pricing for p6-b200 is not yet visible (account not allowlisted). Economics analysis uses 1.25-1.5× H200 estimate pending actual pricing

---

## Deployment Sequence

```
1. Pre-session (internet access required)
   ├── Download GLM-5-FP8 weights → upload to S3
   ├── Build SGLang container → push to ECR
   ├── Clone sagemaker-hyperpod-cli for Helm charts
   └── SearchTrainingPlanOfferings → CreateTrainingPlan

2. Infrastructure (terraform apply)
   ├── VPC + private subnets + VPC endpoints
   ├── EKS cluster
   ├── Security group (EFA-enabled)
   ├── S3 bucket (lifecycle scripts + TLS certs)
   ├── IAM roles (SageMaker, inference operator, S3 CSI, ALB, KEDA)
   ├── Helm charts (NVIDIA device plugin, deep health check, cert-manager)
   ├── HyperPod cluster (Training Plan → GPU instance group)
   ├── FSx Lustre (CSI driver + optional filesystem)
   ├── Inference Operator (EKS add-on)
   └── Observability (AMP + AMG)

3. Model deployment
   ├── kubectl apply -f glm5-inference-endpoint.yaml
   ├── Wait for prefetch (S3 → pod) to complete
   └── Verify: aws sagemaker-runtime invoke-endpoint --endpoint-name glm5-fp8

4. Hardware validation
   ├── H0: GPU / NVLink / EFA connectivity (SSM + MCP tools)
   └── Deep health check results review

5. Benchmark execution — Track A (H200)
   ├── P0: Smoke test + BFCL tool-call validation          (~30 min)
   │   └── GATE: BFCL >= 75% → proceed
   ├── P1: Agent swarm pressure testing                     (~2 hrs)
   │   ├── P1a: Coding agent simulation (realistic workflow)
   │   ├── P1b: Concurrency sweep (4→128 agents)
   │   └── P1c: Functional coding eval at SLO-max
   ├── P2: KV cache offloading under agent load             (~1 hr)
   │   ├── P2a: GPU-only baseline
   │   ├── P2b: L1 (CPU memory) enabled
   │   ├── P2c: L1 + L2 (Redis/FSx) enabled
   │   └── P2d: Warm-start recovery test
   └── H1-H5: HyperPod infrastructure tests

6. Benchmark execution — Track B (B200, separate Training Plan)
   ├── P0': Smoke test (validate cutlass FP8 + TRTLLM MLA) (~15 min)
   ├── P1b': Concurrency sweep (4→256 agents)               (~2 hrs)
   │   └── Head-to-head vs H200 SLO-max
   ├── P2': KV cache comparison (GPU-only may suffice)       (~30 min)
   └── H0': GPU / NVLink 5 / EFA validation

7. Economics analysis (P3)                                    (~30 min)
   ├── H200 vs B200 agent capacity comparison
   ├── Cost per engineer per month (both instances)
   └── Break-even vs Claude Sonnet / Opus

5. Teardown
   ├── kubectl delete -f glm5-inference-endpoint.yaml
   ├── terraform destroy -var-file=glm5.tfvars
   └── (Keep S3 model weights for next session)
```

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
> See `blueprints/glm5-hyperpod/lessons.md`, `blueprints/glm5-hyperpod/results/`, etc.
