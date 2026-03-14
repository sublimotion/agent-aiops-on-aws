# Technology Stack

> This file covers conventions for all domains. See section headers to find the right section for your domain.

## GPU Serving Conventions

### Infrastructure

| Technology | Purpose | Preference |
|------------|---------|------------|
| **Terraform** | Infrastructure as Code | Primary |
| **AWS CDK** | Infrastructure as Code | Secondary |
| **CloudFormation** | Infrastructure as Code | Avoid (use Terraform/CDK) |

#### B200 NVL5+ requires AL2023 AMI — AL2 kernel 5.10 lacks ib_umad module for NVIDIA Fabric Manager

For p6-b200 instances (B200 NVSwitch topology), use Amazon Linux 2023 AMIs (e.g., `amazon-eks-node-al2023-x86_64-nvidia-1.32-v20260304` with kernel 6.1). Amazon Linux 2 (kernel 5.10) does not compile the `ib_umad` kernel module (`CONFIG_INFINIBAND_USER_MAD=m`), which is required by NVIDIA Fabric Manager on NVL5+ systems. Without Fabric Manager, CUDA returns error 802 (`cudaErrorSystemNotReady`) — nvidia-smi shows GPUs on the host but containers cannot access them. This is a platform constraint, not a model or framework limitation.

#### AL2023 EKS uses nodeadm with MIME multipart user data, not bootstrap.sh

EKS nodes on Amazon Linux 2023 use `nodeadm` for cluster joining, not `/etc/eks/bootstrap.sh`. User data must be MIME multipart format with `application/node.eks.aws` content type for the NodeConfig YAML, and `text/x-shellscript` for post-boot scripts (NVMe RAID0, FSx mount, etc.). This applies to all AL2023-based EKS node groups. AL2 continues to use `bootstrap.sh`. Do not mix the two formats — check the AMI family before writing user data templates.

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

#### DeepGEMM JIT compilation on Blackwell requires ~15 min on first startup — set readiness probe initialDelaySeconds ≥900s

SGLang's DeepGEMM JIT compilation on Blackwell sm_120 GPUs compiles 9 kernel configurations with 65536 iterations each, taking approximately 15 minutes on first startup. Subsequent restarts reuse the cached kernels (stored on NVMe if available), reducing warmup to ~5 minutes. Set Kubernetes readiness probe `initialDelaySeconds` to at least 900 seconds (15 minutes) to prevent the pod from being marked unhealthy during JIT compilation. This applies to all Blackwell FP8 serving deployments using DeepGEMM, not just specific models. Consider pre-compiling kernels with `sglang.compile_deep_gemm` in a custom image for production deployments to eliminate this overhead.

#### vLLM DeepGEMM JIT compilation on B200 takes ~16 min on first startup — cache at /root/.cache/vllm/

vLLM's DeepGEMM JIT compilation on B200 sm_100f GPUs takes approximately 16 minutes on first startup: 77s model load, 200s JIT (117 kernels), 200s warmup (2259 kernels), 509s torch.compile, 245s CUDA graph capture (51 graphs). Kernels are cached at `/root/.cache/vllm/deep_gemm/cache/` and AOT functions at `/root/.cache/vllm/torch_aot_compile/`. Subsequent restarts with a warm cache reduce startup to under 5 minutes. Set Kubernetes readiness probe `initialDelaySeconds` to at least 900 seconds (15 minutes) for cold starts. Consider pre-compiling kernels or mounting a persistent cache volume for production deployments to eliminate this overhead. This applies to all B200 FP8 serving deployments using vLLM DeepGEMM.

#### LMCache v0.3.15 incompatible with SGLang NSA/MLA attention (as of 2026-03-07) — blocks KV offloading for GLM-5, DeepSeek V3, and similar MLA models

LMCache's SGLang adapter (`lmc_radix_cache.py` line 96) expects separate `k_buffer` and `v_buffer` attributes in the KV pool. Models using NSA (Native Sparse Attention) or MLA (Multi-Head Latent Attention), such as GLM-5 (`glm_moe_dsa`) and DeepSeek V3, use `NSATokenToKVPool` which inherits from `MLATokenToKVPool` and uses a fused `kv_buffer` instead. LMCache crashes with `AttributeError: 'NSATokenToKVPool' object has no attribute 'k_buffer'` when `--enable-lmcache` is set. LMCache PR #2629 (MLA layerwise support) is open but NOT merged as of 2026-03-07. Both SGLang-side and LMCache-side changes are needed. Do not reserve GPU capacity for LMCache KV offloading (CPU, GDS, POSIX) with MLA models until PR #2629 merges. SGLang's built-in RadixAttention prefix caching works fine as a baseline. Verify PR merge status before planning capacity blocks for MLA models with external KV cache offloading.

#### p6-b200.48xlarge termination takes ~10 min before capacity block slot becomes available — plan for 10-min gaps when replacing instances

Terminating a p6-b200.48xlarge instance takes approximately 10 minutes before the capacity block slot becomes available for a new launch. This is slower than smaller instance types (e.g., p5en.48xlarge typically terminates in 2-3 minutes). When replacing instances during capacity blocks, plan for 10-minute gaps in availability. Do not poll capacity block availability aggressively — check every 30 seconds to avoid API throttling. This delay is an AWS service constraint affecting all large instance types, not specific to a particular workload or model.

#### PYTHONPATH NVMe trick for persistent pip installs on EKS nodes without buildkitd

EKS nodes on AL2023 lack `buildctl`/`buildkitd` for `nerdctl build`, and Kaniko fails on large (14+ GB) Docker Hub images. To install Python packages without rebuilding the image: (1) `pip install` in the running container, (2) copy installed packages to NVMe hostPath: `cp -a /usr/local/lib/python3.12/dist-packages/{pkg,pkg.dist-info} /mnt/nvme/<package-dir>/`, (3) set `PYTHONPATH=/mnt/nvme/<package-dir>` in the deployment env. **Critical**: copy both the package directory and its `.dist-info` directory — `importlib.metadata` needs `.dist-info` for version resolution. Packages persist across pod restarts via the hostPath volume. This pattern applies to any EKS blueprint that needs to add Python packages during rapid iteration without node-side image builds.

#### For MoE models, favor tensor parallelism over data parallelism with expert parallelism at single-node scale

When serving Mixture-of-Experts models with many experts (hundreds) on a single multi-GPU node, tensor parallelism typically outperforms data parallelism with expert parallelism. Expert parallelism requires cross-GPU communication for MoE routing at every layer, which adds significant latency overhead when each GPU runs a full replica with TP=1. Tensor parallelism keeps MoE routing local to each GPU's shard and benefits from weight distribution. Data parallelism with expert parallelism may become competitive in multi-node deployments where TP cannot efficiently span nodes, but at single-node scale, prioritize TP. Benchmark both configurations if the model fits in memory with either approach.

#### CPU weight offloading is unnecessary on high-VRAM GPUs and may be unsupported

Serving frameworks like vLLM use `--cpu-offload-gb` to offload model weights to CPU RAM, not KV cache. On high-VRAM GPUs (e.g., H200 with 141 GB HBM per GPU), weight offloading is typically unnecessary and may be unsupported (vLLM 0.16+ V1 engine blocks it entirely). Before considering CPU offloading, calculate available KV cache capacity: for FP8 models on H200, a typical TP=4 config leaves 100+ GB per GPU for KV cache, providing 30-40x concurrency at 262K context. The bottleneck at extreme context lengths is prefill computation time (O(n^2) attention layers), not VRAM capacity.

#### Batching effectiveness scales with context length

For long-context workloads (64K+ tokens), batching at moderate-to-high QPS (2.0+) can reduce TTFT by 4-6x compared to low QPS (0.5) due to GPU amortization of prefill computation across concurrent requests. This effect is stronger at long context than at short context because prefill cost dominates. When designing load balancers or capacity planning for long-context models, target steady moderate concurrency rather than bursty low-QPS patterns. Run QPS sweeps during benchmarking to identify the batching sweet spot for each context length tier.

#### Prefix caching is the key enabler for long-context serving

For workloads with shared context (RAG with document retrieval, multi-turn conversations with long system prompts), prefix caching can reduce TTFT by 50-60% and extend the viable context range by 2-4x. Always enable prefix caching (`--enable-prefix-caching` in vLLM, on by default in SGLang via RadixAttention) for production deployments. During benchmarking, test both random context and shared-prefix patterns to capture the prefix cache effect.

#### For scarce GPU instances, shotgun launch across multiple regions before reserving capacity

When targeting newly-launched or scarce GPU instance types (e.g., g7e Blackwell), EC2 dry-run validates permissions and quotas but not physical capacity. Do not trust dry-run success as a capacity signal. Instead, shotgun `aws ec2 run-instances` across multiple regions and AZs simultaneously to find available capacity. Capacity blocks are not supported for all instance types (e.g., g7e); on-demand or spot are the only options. If benchmarking urgency is high, consider bare EC2 in the first region with capacity rather than waiting for EKS node group capacity in a preferred region.

#### Bare EC2 with EKS-optimized AMI requires manual containerd startup

EKS-optimized AL2023 AMIs use nerdctl/containerd, not Docker. When launching bare EC2 with these AMIs (outside an EKS cluster), the containerd service is not running by default. Run `sudo systemctl start containerd` before any nerdctl commands. Use `--gpus <count>` (e.g., `--gpus 4`) instead of Docker's `--gpus '"device=0,1,2,3"'` syntax. Do not combine `--rm` with `-d` (detached mode) — nerdctl does not support this combination.

#### MTP speculative decoding degrades throughput on PCIe-interconnected GPUs

Speculative decoding with MTP (e.g., Qwen3-Next's `qwen3_next_mtp` method) adds inter-GPU communication overhead for speculative head computation and verification. On PCIe-interconnected GPUs (e.g., g7e.24xlarge, g7e.48xlarge), this overhead exceeds the speculative decoding benefit, causing throughput degradation of 2-41% across QPS levels. MTP is designed for NVLink-interconnected GPUs (H200, A100) where inter-GPU bandwidth is 10-20x higher. Always test MTP on the target hardware before enabling in production; default to baseline (no MTP) for PCIe platforms.

#### vLLM MTP speculative decode with FlashMLASparse uses PIECEWISE CUDA graph mode and forces KV cache block size to 64

vLLM's Multi-Token Prediction speculative decoding (`--speculative-config.method mtp --speculative-config.num_speculative_tokens N`) with FlashMLASparse attention uses PIECEWISE CUDA graph mode instead of FULL_AND_PIECEWISE. FULL_AND_PIECEWISE mode is not supported with speculative decoding for models using `DeepseekV32IndexerBackend` (e.g., GLM-5, DeepSeek V3). Additionally, FlashMLASparse forces the KV cache block size to 64 regardless of the `--block-size` flag. This applies to all MLA models using vLLM MTP speculative decode. Expect different CUDA graph capture and memory behavior compared to standard attention backends.

#### Mamba hybrid architectures have different caching and speculative decoding constraints

Models using hybrid attention+mamba architectures (e.g., Qwen3-Next with `Qwen3NextForCausalLM`) trigger mamba cache mode in vLLM. Prefix caching works but enables experimental mamba 'align' mode. MTP speculative decoding conflicts with mamba 'align' mode in vLLM 0.15.0, requiring `--no-enable-prefix-caching` to work at all, which further degrades performance. Verify model architecture (`transformers.AutoConfig.from_pretrained(...)` → check `architectures` field) before assuming transformer-only optimizations apply.

#### Keep project names short to avoid IAM role name length limits

AWS IAM role names have a 64-character limit. Terraform modules often compose role names from `var.project_name` + module suffixes (e.g., `-eks-cluster-node-role`). Long project names (e.g., `qwen3-next-g7e-bench`) can push generated names over the limit, causing Terraform apply failures. Keep `var.project_name` to 12 characters or fewer to provide headroom for module composition.

#### EC2 dry-run does not validate capacity — it only validates permissions and quotas

`aws ec2 run-instances --dry-run` returns "would succeed" if your IAM permissions and service quotas allow the launch. It does not check whether AWS has physical hardware available in the target AZ. A successful dry-run followed by a failed real launch (InsufficientInstanceCapacity) is expected behavior, not a bug. Do not use dry-run as a capacity check; use it only to validate IAM/quota configuration.

#### Monitor terraform background tasks to avoid state lock conflicts

Terraform state locks are stored in a local `.terraform.tfstate.lock.info` file (when using local state) or DynamoDB (when using S3 backend). Running terraform commands in parallel background tasks will cause the second command to wait indefinitely for the lock or fail with "state locked" errors. Before starting a new terraform operation, check for existing processes with `ps aux | grep terraform` and kill any orphaned background tasks. If a lock persists after killing all terraform processes, manually remove the lock file or DynamoDB entry.

#### Terraform state attribute mismatches on import can be fixed surgically

When importing existing AWS resources into Terraform, attribute mismatches (e.g., `bootstrap_self_managed_addons=true` in state vs. `false` in config) force resource destruction on the next apply. Before destroying and recreating infrastructure, use `terraform state pull > state.json`, edit the JSON directly to align the attribute values, and `terraform state push state.json` to restore the corrected state. This avoids expensive infrastructure churn. Validate with `terraform plan` before and after to confirm the mismatch is resolved.

#### SGLang on Blackwell requires cutlass FP8 backend for non-ue8m0 scale formats

DeepGemm FP8 backend crashes with "Unknown recipe" on Blackwell GPUs (sm_120) when loading models with non-ue8m0 scale formats. Use `--fp8-gemm-backend cutlass` (available in SGLang nightly, not v0.5.9 stable). This flag is required for models like Qwen3-Next FP8 on g7e instances. vLLM users should set `VLLM_USE_DEEP_GEMM=0` for equivalent behavior.

#### Hybrid DeltaNet+GQA models require triton attention backend on Blackwell

Models with hybrid attention architectures (Mamba + DeltaNet + GQA, e.g., Qwen3-Next) require `--attention-backend triton` on Blackwell GPUs. FlashInfer will fail with "triton or trtllm_mha backend are the only supported backends on Blackwell GPUs for hybrid GDN models". This is a framework constraint, not a model limitation.

#### Hybrid attention + HiCache requires CUDA graph disabled

When serving hybrid attention models (e.g., Qwen3-Next with DeltaNet+GQA) with HiCache KV offloading, use `--disable-cuda-graph`. CUDA graph compilation conflicts with HiCache's dynamic memory management for hybrid models. This constraint applies only to hybrid architectures; standard transformer models can use CUDA graphs with HiCache.

#### vLLM Mistral tool-call parser generates non-compliant tool_call IDs

The Mistral parser in vLLM (through v0.15.0) generates `call_0`, `call_1`, etc. as tool_call IDs instead of the OpenAI-spec 9-character alphanumeric format required by BFCL and most downstream tools. This causes multi-turn tool-use failures where the second turn's tool_result is rejected due to ID format validation. Workaround: patch the eval script to accept short IDs or use `--tool-call-parser hermes` if the model supports it. Track vLLM issue #23180 for upstream fix.

#### SGLang qwen3_coder parser outputs tool calls as XML in content field

SGLang's `qwen3_coder` parser correctly sets `finish_reason: "tool_calls"` but places tool calls in the `content` field as `<tool_call>{"name": ..., "arguments": ...}</tool_call>` XML tags instead of the standard OpenAI `tool_calls` array. Downstream applications must parse both formats. Add XML fallback parsing to any tool-use evaluation or agent framework that consumes SGLang responses.

#### Force-deleted Kubernetes pods leak GPU memory requiring manual cleanup

`kubectl delete pod --force --grace-period=0` does not cleanly terminate GPU processes. GPU memory remains allocated by orphaned PIDs. Before redeploying, SSH to the node and kill PIDs manually via `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`. Without this cleanup, the new pod will fail to allocate GPU memory even though `nvidia-smi` shows GPUs as idle.

#### Verify CUDA image tags before deployment — cu131 vs cu130 for Blackwell

For Blackwell GPUs (sm_120), use CUDA 13.0 (`cu130`), not CUDA 13.1 (`cu131`). Not all image registries publish `cu131` tags; attempting to pull a non-existent tag wastes deployment time. For SGLang on g7e, use `lmsysorg/sglang:nightly-dev-cu13-<date>` (which is cu130) or `v0.5.9-cu130`. Do not assume higher CUDA minor versions exist without checking the registry first.

#### MoE and FP8 kernels are not tuned for new GPU architectures at launch

Serving frameworks ship default kernel configurations that may be sub-optimal for newly-launched GPUs (e.g., RTX PRO 6000 Blackwell). Generate device-specific tuning configs using the framework's kernel benchmarking tools (e.g., `sglang/benchmark/kernels/fused_moe_triton` for MoE, vLLM's `benchmark_kernels.py` for FP8 GEMMs) before committing to a multi-hour capacity block. Tuned configs can improve throughput by 20-40% on new architectures.

#### g7e instances support EFA (kernel-bypass networking) but not GDS or NVLink

All g7e sizes support EFA: g7e.12xlarge (1 interface), g7e.24xlarge (2 interfaces), g7e.48xlarge (4 interfaces). EFA provides kernel-bypass networking (AWS SRD protocol) for inter-node communication — this is independent of GPU interconnect (PCIe vs NVLink). EFA enables NIXL LIBFABRIC disaggregated prefill/decode between nodes via Dynamo's `NixlConnector`. However, EFA is **not true RDMA** — the KV transfer path is GPU VRAM → cudaMemcpy → CPU buffer → EFA SRD (kernel-bypass) → CPU buffer → cudaMemcpy → GPU VRAM. The CPU bounce is required on both sides because EC2 EFA does not support GPUDirect RDMA (direct NIC↔GPU DMA without CPU involvement). True GPUDirect RDMA requires InfiniBand + `nvidia-peermem`, available only on p5/p5e/p5en with NVSwitch.

g7e does NOT support GPUDirect Storage (GDS) — `gdsio` compat mode shows zero benefit over standard POSIX I/O on EC2 NVMe controllers. HiCache L3/L4 KV offloading on g7e uses standard file I/O to local NVMe. For GDS-backed KV offloading via FSx Lustre, use p5en.48xlarge. Always copy models to NVMe RAID0 (`/mnt/nvme`) for best I/O throughput during model loading.

For disaggregated serving on g7e without EFA-capable instances, NIXL LIBFABRIC falls back to TCP. Add `"kv_buffer_device":"cpu"` to the `kv-transfer-config` (default `cuda` requires RDMA). See Dynamo PR #7369 for EKS Auto Mode examples with both EFA and TCP fallback configs.

#### Multi-replica architecture trades latency for reliability in tool-use workloads

Isolated single-GPU replicas behind a round-robin proxy (e.g., 4x vLLM TP=1 on g7e.24xlarge) have zero failure rates under high concurrency but 4-5x higher TTFT p50 compared to a single shared-KV-cache multi-GPU deployment (e.g., 1x SGLang TP=4). Shared KV cache benefits from batched attention and prefix caching across all requests, but saturates at high concurrency leading to failures. For latency-critical interactive agents, favor shared KV cache (SGLang TP=N). For high-reliability swarm agents, favor isolated replicas (vLLM TP=1 per GPU).

#### HiCache hybrid model support is a moving target — verify PR merge status before capacity blocks

HiCache L2 offloading for hybrid attention models (PR #19663) was not merged into SGLang nightly builds as of 2026-03-03. Always check the target nightly build's commit log against the feature PR before reserving GPU capacity for HiCache testing. If the feature is not merged, S3-level benchmarks (HiCache L2) will fail with "HiRadixCache only supports MHA and MLA" errors. Baseline KV cache configs work without HiCache for initial feasibility testing.

#### SGLang HiCache works with NSA/MLA attention where LMCache fails — use --enable-hierarchical-cache for MLA models

For models using NSA (Native Sparse Attention) or MLA (Multi-Head Latent Attention) architectures such as GLM-5 (`glm_moe_dsa`) and DeepSeek V3, use SGLang's built-in HiCache (`--enable-hierarchical-cache`) instead of LMCache for KV cache offloading. HiCache has native `NSATokenToKVPoolHost` support that understands the fused `kv_buffer` layout used by MLA models, while LMCache expects separate `k_buffer`/`v_buffer` attributes and crashes on MLA. HiCache is integrated into SGLang and evolves with the attention backend, eliminating external compatibility issues. This applies to all MLA/NSA models on SGLang until LMCache PR #2629 merges.

#### HiCache --hicache-size must exceed device KV pool size to pass initialization assertion

SGLang HiCache asserts `host_memory > device_memory` during initialization. Set `--hicache-size` to at least the device KV pool size plus margin. For example, if the device KV pool is approximately 82 GB per TP rank, use `--hicache-size 100` (100 GB per rank). Do not rely on the default `--hicache-ratio 2.0` which calculates 2x device pool per rank — this can exceed available system RAM on memory-constrained instances and cause OOM. Calculate total host memory requirement as `num_tp_ranks × hicache_size` and verify it fits within available system RAM before launching. This is a framework requirement, not a model-specific constraint.

#### For memory-constrained models, CPU KV cache offloading fundamentally changes the concurrency ceiling

When model weights consume most GPU VRAM (e.g., GLM-5 FP8 using 175 GB / 183 GB per GPU), the device KV cache becomes the primary throughput bottleneck. CPU KV cache offloading (via HiCache or similar) can deliver 2-3x throughput improvement at high concurrency by extending effective KV cache capacity beyond GPU VRAM. Superlinear scaling at high concurrency (e.g., baseline plateaus at 64 concurrent while HiCache continues scaling to 128+ concurrent) confirms that KV cache eviction was limiting throughput, not compute capacity. Always benchmark both baseline (device-only KV cache) and CPU offload configurations for large models to identify whether KV cache or compute is the true bottleneck. This pattern applies across models and frameworks, not just specific architectures.

#### Redis can run on GPU nodes with taint toleration when system nodes lack capacity

System nodes (e.g., m5.xlarge) often lack sufficient CPU or memory for auxiliary services like Redis. Adding `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` to Redis (or other non-GPU workloads) allows them to schedule on GPU nodes, which typically have abundant CPU and RAM beyond what serving workloads use. For example, on p6-b200.48xlarge nodes running vLLM, 80% of CPU and 90% of memory remain free. This pattern applies to any Kubernetes cluster where auxiliary services need more resources than the system node pool provides. Use resource requests/limits to ensure the auxiliary service does not starve the GPU workload.

### llm-d Infrastructure

llm-d is the reference inference scheduler architecture: InferencePool (GA v1 API) + EPP (Endpoint Policy Picker) + Envoy Gateway. This section captures deployment patterns for llm-d components.

#### InferencePool v1 GA API has different schema than v1alpha2 experimental API

InferencePool v1 (`inference.networking.k8s.io/v1`) uses different field names compared to the experimental v1alpha2 API (`x-k8s.io/v1alpha2`):
- v1: `endpointPickerRef` (not `extensionRef`)
- v1: `targetPorts: [{number: 8000}]` (not `targetPortNumber: 8000`)
- v1: `selector.matchLabels` (not flat `selector: {app: ...}`)

EPP v1.3.1 watches the GA group `inference.networking.k8s.io`, not the experimental group. Always use v1 InferencePool manifests with EPP v1.3.1+. Do not mix v1alpha2 manifests with GA-aware controllers — the CRDs are incompatible and will cause silent routing failures.

#### EPP v1.3.1 requires explicit configuration and uses kebab-case flags

EPP (Endpoint Policy Picker) v1.3.1 has no implicit defaults. Always provide `--config-file` or `--config-text` when deploying EPP. Flags use kebab-case: `--pool-name` (not `--poolName`), `--grpc-port` (not `--grpcPort`), `--secure-serving` (not `--secureServing`).

RBAC requirements: EPP's service account needs `list` and `watch` on:
- `pods` (core API)
- `inferencepools.inference.networking.k8s.io` (GA group)
- `inferencemodelrewrites.x-k8s.io`, `inferenceobjectives.x-k8s.io`, `inferencepoolimports.x-k8s.io` (experimental group)

Use the official image: `registry.k8s.io/gateway-api-inference-extension/epp:v1.3.1`. This applies to all llm-d deployments using EPP v1.3.1+.

#### Envoy Gateway with --skip-crds requires manual GatewayClass creation

When installing Envoy Gateway with `--skip-crds` (to avoid CRD version conflicts), the GatewayClass is not created automatically. Manually apply:
```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: eg
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
```

Additionally, set `allowedRoutes.namespaces.from: All` on the Gateway resource to enable cross-namespace HTTPRoutes. Without this, HTTPRoutes in different namespaces will show `ResolvedRefs=False` and fail to route traffic. This applies to all Envoy Gateway deployments where the Gateway and HTTPRoutes are in different namespaces.

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
