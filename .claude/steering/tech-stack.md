# Technology Stack

> This file covers conventions for all domains. See section headers to find the right section for your domain.

## Version Tagging Convention

Every version-specific steering rule must include a `<!-- stack: ... | validated: ... -->` HTML comment immediately after the heading. This enables automated staleness detection when stack components upgrade.

Format:
```
#### Rule title
<!-- stack: component=version, component=version | validated: YYYY-MM-DD -->
```

Example:
```
#### Pin protobuf<5 in any TensorFlow runtime_env on Ray Serve
<!-- stack: ray=2.44.1, tensorflow=2.16.2 | validated: 2026-03-27 -->
```

**When to tag**: Any rule that references a specific version, version constraint, or version-dependent behavior. Rules about general AWS/K8s constraints (e.g., "IMDS hop limit") don't need tags.

**When to refresh**: When you learn a stack component has a new version (e.g., "vLLM 0.19 is out"), grep this file for rules tagged with the old version and validate each one still holds. See the refresh protocol in `compound-learner.md`.

## Re-verify framework facts in interactive work, not just at spec gates

The version-decay discipline (re-verify engine-blocker claims against the live tracker) is **structurally enforced in the batch/RALPH path** — the spec Stage 0b lever ledger forces a `validated: YYYY-MM-DD` re-check on every "BLOCKED by …"/"not supported"/"needs a patch" deferral before the loop runs. **It is NOT enforced in interactive/ad-hoc work** (live serving plumbing, debugging, one-off runs), where there is no gate — so a stale framework fact flows straight into action.

**Rule (interactive path):** before acting on any framework-version-specific memory/lesson — especially a claim that something *"needs a shim/proxy/patch"*, *"isn't supported"*, or *"only works with model X"* — **re-verify against current engine docs first** (≈30s: the project doc, `mdc prs`, `gh pr view`, or the engine's docs site). Treat such a memory as a *trigger to verify*, not a fact to build on. The same "point-in-time claim that decays" caveat from `docs/optimization-stack.md` applies, but in interactive mode YOU are the gate.

**Why:** 2026-06-24, on the glm5.2 agent comparison, a stale vLLM-0.16/Mistral-era memory ("Claude Code only works with Mistral via vLLM; SGLang has no `/v1/messages`") was acted on directly — built an unnecessary LiteLLM translation shim and burned multiple turns debugging it. **SGLang serves `/v1/messages` natively** (auto-registered; docs example is literally GLM-5.2-FP8 + glm47/glm45), and vLLM does too. A 30-second docs check would have skipped the entire detour. The discipline existed (Stage 0b); the interactive path just had no gate to fire it.

**Tells that you're in this trap:** about to add a translation/proxy layer (LiteLLM, a custom SSE proxy) to bridge an API; about to patch an engine; citing a memory with a model name or version in the "it doesn't work" clause. Stop and check native support first.

## Artifact durability before destructive teardown (scale-to-0 / terminate)

**Spot/benchmark node local storage (NVMe instance-store, emptyDir, `/mnt/nvme`) is ephemeral — it is WIPED on scale-to-0 or spot reclaim.** Anything not exfiltrated to a durable sink (git **committed + pushed**, or S3) before teardown is permanently lost. "I `kubectl cp`'d a file" is NOT durable until it's committed/pushed.

**Rule — before ANY `aws eks update-nodegroup-config ... desiredSize=0` / instance terminate:**
1. **Enumerate the full results dir** on the node (`find /mnt/nvme/results -type f`), not just the summary you happened to look at. Raw traces / per-issue event JSONLs / diffs are usually the highest-value, hardest-to-regenerate artifacts.
2. **`tar` the whole dir → pull → verify the tarball locally** (extract + spot-check) — exfiltrate everything, decide what to keep afterward.
3. **`git add/commit/push` the artifacts you intend to keep, and confirm they're in the remote** — committing a *summary* is not committing the *traces*. A run is not "done" until its keep-set is pushed.
4. Only then scale to 0.

**Prefer a durable sink DURING the run, not a manual pull at teardown:** give bench/runner nodes an S3-writable IAM role (or an artifact-sync step) so traces land in a bucket as they're produced. Several of our bench nodes' roles **lack S3 write** (kimi L10 — we `kubectl cp` as a workaround); on those, the manual pull-and-commit gate above is the only safety net, so it's mandatory.

**Why:** 2026-06-24, on the glm5.2 agent run, the driver correctly wrote per-issue event JSONLs (tool-type mix, the data needed for the richest trace analysis) to NVMe — but at teardown I pulled only `summary_glm52.json`, committed that as if complete, and scaled the node to 0. The raw traces were never exfiltrated by ANY path (git ✗, laptop ✗, S3 ✗ — role lacked write) and were wiped with the instance. Permanently lost, re-runnable only. The summary survived; the higher-value raw data did not, because I treated "pulled something" as "preserved everything."

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

#### GPU spot nodegroups may launch without the node security group — verify and attach post-launch
<!-- validated: 2026-06-16 -->

EKS GPU spot nodegroups created via `aws eks create-nodegroup` without an explicit launch template may launch with only the cluster security group, missing the node security group that grants pod-to-pod cross-node, FSx mount, and in-cluster DNS access. Symptom: ALL pod networking silently broken — connection timeouts / DNS resolution failures inside pods, not an obvious SG error. It affects pods on the new node and pods elsewhere trying to reach it. Fix: find the node SG (named `<cluster-name>-node-*`, description "Security group for all nodes in the cluster") and attach it to the instance's primary ENI alongside the cluster SG:

```bash
aws ec2 modify-network-interface-attribute --network-interface-id <eni-id> --groups <cluster-sg-id> <node-sg-id>
```

Add this post-launch check to GPU nodegroup scripts — observed on both B200 and H200 spot nodegroups on the qwen3-next-bench cluster; missing it caused a multi-hour failure cascade. Applies to any managed nodegroup whose launch template is not explicitly managed.

#### GPU nodes with no internet egress and broken FSx-Lustre CSI: use the CPU-node HTTP staging pattern
<!-- validated: 2026-06-16 -->

Some EKS GPU node configs have no internet egress (cannot reach HuggingFace or apt repos) and FSx-Lustre CSI mounts fail on the GPU AMIs (`mount.lustre: Can't parse NID`, Lustre kmod mismatch). Working pattern: (1) stage weights to FSx from a CPU node (has egress + working FSx CSI); (2) serve weights over HTTP from a CPU-node pod (`python3 -m http.server 8080`); (3) fetch to the GPU node's emptyDir over the pod network with a pure-Python downloader (no apt/curl/egress). Reference the HTTP server by its ClusterIP service if in-cluster DNS is unreliable. Validate FSx CSI mount success before assuming GPU nodes can read FSx directly. Applies to any EKS cluster where GPU nodes are in private subnets without NAT and the FSx CSI driver version does not match the AMI's Lustre kernel module.

### Deployment Conventions

#### Single-node GPU deployments: scale to 0 before changing GPU resource requests

When changing GPU resource allocation on a single-node Kubernetes deployment (e.g., TP=4 to TP=8), scale to 0 replicas before applying Terraform changes, then scale back to 1. Rolling updates cannot work when the new pod requires more GPUs than are available after the old pod's allocation is accounted for. This prevents scheduling deadlocks where the new pod waits indefinitely for resources held by the old pod.

```bash
kubectl -n <namespace> scale deployment <name> --replicas=0
terraform apply -target='<deployment_resource>' -auto-approve
kubectl -n <namespace> scale deployment <name> --replicas=1
```

#### NIXL disables cuda_ipc by default — single-node P/D disaggregation requires explicit transport configuration
<!-- stack: nixl=0.3.x, dynamo=alpha, vllm=0.18+, sglang>=0.5.13 | validated: 2026-06-17 -->

NIXL (the KV cache transfer library used by Dynamo and llm-d) disables UCX `cuda_ipc` transport by default (NIXL issue #1097) to avoid contention with NCCL collectives on NVSwitch fabric. Without override, same-node KV transfer falls back to TCP loopback — 40x slower (TTFT 355ms → 10+ seconds per Dynamo docs).

**When deploying P/D disaggregation on a single NVSwitch node:**

- **TP=1 workers (model fits on one GPU):** Re-enable cuda_ipc — no NCCL collectives means no contention. Set `UCX_TLS=cuda_copy,cuda_ipc`, `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on`, add `hostIPC: true` and `IPC_LOCK` capability to pods. This is the sweet spot for single-node disagg: full NVLink bandwidth (~900 GB/s) for KV transfer, tunable P:D ratio.
- **TP>1 workers:** Do NOT re-enable cuda_ipc. NCCL allreduce and KV transfer compete for the same NVLink/NVSwitch fabric, causing decode TPOT spikes. Use chunked prefill instead (no KV transfer overhead), or on p5/p5e use IB RDMA loopback (`UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy`) to route KV transfer over InfiniBand while NVLink stays dedicated to TP.
- **Multi-node (the designed path):** KV transfer over RDMA (InfiniBand/RoCE). NVLink exclusively for TP. No contention. This is what NIXL's default assumes.

**Disagg vs replicas decision:** At low QPS or short context, replicas with chunked prefill are simpler and sufficient. Disagg wins at moderate-to-high QPS with mixed context lengths, strict p99 TPOT SLOs, or when prefill interference measurably degrades decode latency. IMEX daemon / `/dev/nvidia-caps-imex-channels` is only relevant for MNNVL (multi-node NVLink) fabric — not needed for single-node cuda_ipc.

**Disagg transport configuration gotchas** (kimi-k2.6-nvfp4 L13 / Stage 6c):
1. **mooncake selects TCP when no InfiniBand HCA** — mooncake only knows TCP-or-IB and **ignores NVLink**. For same-node disagg, use **NIXL** backend (`--disaggregation-transfer-backend nixl`), NOT mooncake. EFA is irrelevant for same-node (NVLink/cuda_ipc is the path).
2. **CUDA_VISIBLE_DEVICES conflicts with --base-gpu-id** — `CUDA_VISIBLE_DEVICES` remaps physical GPUs to logical 0-N, so `--base-gpu-id N` then requests a non-existent device. Use ONE mechanism: `CUDA_VISIBLE_DEVICES` per container (simplest for multi-container pods), drop `--base-gpu-id`.
3. **UCX_TLS must include control-plane transports** — cuda_ipc/cuda_copy are data-only. UCX also needs sm,self,tcp for connection handshake. Use `UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp` or leave unset for auto-select.

**Single-node disagg measured regression** (kimi-k2.6-nvfp4 Stage 6c): 4P/4D disagg with correct NIXL/cuda_ipc config = 815 tok/s vs TP4+DP2's 3,138 tok/s — a 3.8× throughput loss. Root causes: (1) SGLang forces `disable_radix_cache=True` in disagg decode mode → loses the 74% prefix cache that made the workload viable; (2) decode GPUs starved (num_running≈1, token_usage≈0.12) — the 4P→4D handoff serializes and underutilizes. **Implication:** single-node disagg for high-cache-hit workloads is the wrong architecture — it sacrifices the prefix cache (the lever that raised throughput 2-3×) for a P/D split the workload doesn't need. Disagg's designed home remains multi-node at scale, per existing guidance.

#### Air-gapped serving environments require local tokenizer paths for benchmarking

When `HF_HUB_OFFLINE=1` is set in the serving container (air-gapped, no HuggingFace Hub access), benchmark tools like `vllm bench serve` must use `--tokenizer /path/to/local/model` to point at the local model directory. The `--model` flag specifies the API-facing served model name, not the filesystem path.

#### Always document benchmark execution location before running

Record whether benchmarks run via `kubectl port-forward` (from local machine) or server-side (inside the cluster via `kubectl exec`). Port-forward benchmarks measure client → API server → pod latency; server-side benchmarks measure pod-local inference latency only. This distinction is critical for interpreting TTFT and E2E latency results.

#### FP8 quantization compatibility check for MoE models

Before reserving GPU capacity for Mixture-of-Experts models with FP8 quantization, verify that all weight dimensions (including shared experts) remain divisible by `block_k` (typically 128) at the target tensor parallelism degree. Example: if a shared expert MLP `down_proj` has `input_size=512`, TP=8 produces `input_size_per_partition=64`, which is not divisible by 128 and will cause a ValueError at model load time. Test TP compatibility on a CPU-only or smaller GPU instance before committing to a capacity block.

#### First-run JIT/graph compilation can take 15+ min — set readiness probe initialDelaySeconds ≥900s and persist the cache
<!-- stack: sglang,vllm,trt-llm | validated: 2026-05 -->

Serving stacks with JIT kernel compilation (DeepGEMM), `torch.compile`, and CUDA-graph capture pay a large one-time cost on cold start. Set Kubernetes readiness probe `initialDelaySeconds ≥ 900` so the pod isn't marked unhealthy mid-compile, and persist the cache (NVMe / mounted volume) so warm restarts drop to ~5 min. In capacity-block scenarios, budget this warmup into the session plan or pre-compile in a custom image (e.g. `sglang.compile_deep_gemm`) before the block starts. Applies to the *stack*, not specific models — a new occurrence is a new row, not a new rule.

| Stack / hardware            | First-start | Breakdown / cache path | Seen    |
|-----------------------------|-------------|------------------------|---------|
| SGLang DeepGEMM / Blackwell sm_120 | ~15 min | 9 kernel configs × 65536 iters; cache on NVMe | 2026-03 |
| vLLM DeepGEMM / B200 sm_100f | ~16 min    | 77s load + 200s JIT (117 kernels) + 200s warmup (2259 kernels) + 509s `torch.compile` + 245s CUDA-graph (51 graphs); cache `/root/.cache/vllm/deep_gemm/cache/` + `/root/.cache/vllm/torch_aot_compile/` | 2026-05 |
| TensorRT-LLM engine build   | 10-15 min   | offline engine build; pre-build where possible | (general) |

#### LMCache v0.3.15 incompatible with SGLang NSA/MLA attention (as of 2026-03-07) — blocks KV offloading for GLM-5, DeepSeek V3, and similar MLA models
<!-- stack: lmcache=0.3.15, sglang=nightly-2026-03-07 | validated: 2026-03-07 -->

LMCache's SGLang adapter (`lmc_radix_cache.py` line 96) expects separate `k_buffer` and `v_buffer` attributes in the KV pool. Models using NSA (Native Sparse Attention) or MLA (Multi-Head Latent Attention), such as GLM-5 (`glm_moe_dsa`) and DeepSeek V3, use `NSATokenToKVPool` which inherits from `MLATokenToKVPool` and uses a fused `kv_buffer` instead. LMCache crashes with `AttributeError: 'NSATokenToKVPool' object has no attribute 'k_buffer'` when `--enable-lmcache` is set. LMCache PR #2629 (MLA layerwise support) is open but NOT merged as of 2026-03-07. Both SGLang-side and LMCache-side changes are needed. Do not reserve GPU capacity for LMCache KV offloading (CPU, GDS, POSIX) with MLA models until PR #2629 merges. SGLang's built-in RadixAttention prefix caching works fine as a baseline. Verify PR merge status before planning capacity blocks for MLA models with external KV cache offloading.

#### p6-b200.48xlarge termination takes ~10 min before capacity block slot becomes available — plan for 10-min gaps when replacing instances

Terminating a p6-b200.48xlarge instance takes approximately 10 minutes before the capacity block slot becomes available for a new launch. This is slower than smaller instance types (e.g., p5en.48xlarge typically terminates in 2-3 minutes). When replacing instances during capacity blocks, plan for 10-minute gaps in availability. Do not poll capacity block availability aggressively — check every 30 seconds to avoid API throttling. This delay is an AWS service constraint affecting all large instance types, not specific to a particular workload or model.

#### vLLM full CUDA-graph capture is architecture-specific — re-test graph capture per GPU arch before concluding hardware won't help
<!-- stack: vllm=0.22.1 | validated: 2026-06-16 -->

Full-decode CUDA-graph capture (`Capturing CUDA graphs (decode, FULL)`) may succeed or fail depending on the GPU architecture (sm_XX) and model architecture combination, even with the same serving framework version and MoE backend. Example: for Qwen3.6-35B-A3B (hybrid Gated-DeltaNet MoE) on vLLM 0.22.1 with the TRITON FP8 MoE backend, full graph capture **succeeds on H200 (sm_90) and B200 (sm_100) but crashes on g7e (sm_120)** with `AssertionError: 1 != <ISL>`. This architecture-specific graph-capture success was the ~2× per-replica throughput differentiator — g7e became launch-bound (SM ~43%, knee ~12 RPS/replica) while the datacenter GPUs were compute-bound (SM ~99-100%, knee ~20-22 RPS/replica). The launch bound is an engine/graph-capture limitation, not a hardware FLOPs/bandwidth limit. **Implication**: a launch-bound knee on one GPU architecture is not proof a bigger/different GPU won't help — re-test graph-capture success per architecture before concluding hardware is the bottleneck. Applies to vLLM deployments of hybrid-attention or complex-MoE models.

#### Size --max-num-batched-tokens ≥ input sequence length for prefill-dominated, launch-bound workloads
<!-- stack: vllm>=0.18 | validated: 2026-06-16 -->

For prefill-dominated request shapes (high input:output ratio) on launch-bound serving stacks (many small kernels, SM utilization <60%), set `--max-num-batched-tokens` to at least the typical input sequence length (ISL) — do NOT chunk prefill below the ISL. Example: for ISL ~2500, `mnbt=8192` delivered ~20% more SLO-safe per-replica capacity than `mnbt=4096` and far more than `mnbt=2048`. Mechanism: `mnbt < ISL` forces prefill chunking → more scheduler iterations → more kernel launches, the scarce resource when launch-bound. This is the opposite of the "chunk down to pack short-output requests" intuition. Same lesson observed on Nemotron-3-Super. Applies to any serving framework with explicit prefill-chunking controls.

#### MoE tile tuning does not help a launch-bound knee, even when the kernel loads the tuned config
<!-- stack: vllm>=0.18 | validated: 2026-06-16 -->

For MoE models on launch-bound serving stacks, generating device-specific Triton tile configs and mounting them via `VLLM_TUNED_CONFIG_FOLDER` produces no throughput improvement — even when logs confirm the config loaded (`Using configuration from /tuned/E=...json`). The launch-bound bottleneck is the *number* of kernel launches across hundreds of experts plus hybrid attention layers, not per-GEMM tile efficiency; tuning optimizes each GEMM but cannot reduce dispatch count. Skip MoE tile tuning when the measured bottleneck is launch/scheduling (SM-bound below ~60% with HBM idle), regardless of whether the tuner would generate a config. (On the FlashInfer path, which never reads the tuned JSON, tuning is a no-op for a different reason — same outcome.) Before tuning, grep the serving log for the active `Fp8 MoE backend` to know which path is live.

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
<!-- stack: vllm=0.18.0 | validated: 2026-03-20 -->

vLLM's Multi-Token Prediction speculative decoding (`--speculative-config.method mtp --speculative-config.num_speculative_tokens N`) with FlashMLASparse attention uses PIECEWISE CUDA graph mode instead of FULL_AND_PIECEWISE. FULL_AND_PIECEWISE mode is not supported with speculative decoding for models using `DeepseekV32IndexerBackend` (e.g., GLM-5, DeepSeek V3). Additionally, FlashMLASparse forces the KV cache block size to 64 regardless of the `--block-size` flag. This applies to all MLA models using vLLM MTP speculative decode. Expect different CUDA graph capture and memory behavior compared to standard attention backends.

#### Mamba hybrid architectures have different caching and speculative decoding constraints
<!-- stack: vllm=0.15.0 | validated: 2026-02-25 -->

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
<!-- stack: sglang=0.5.9 | validated: 2026-03-01 -->

DeepGemm FP8 backend crashes with "Unknown recipe" on Blackwell GPUs (sm_120) when loading models with non-ue8m0 scale formats. Use `--fp8-gemm-backend cutlass` (available in SGLang nightly, not v0.5.9 stable). This flag is required for models like Qwen3-Next FP8 on g7e instances. vLLM users should set `VLLM_USE_DEEP_GEMM=0` for equivalent behavior.

#### Hybrid DeltaNet+GQA models require triton attention backend on Blackwell

Models with hybrid attention architectures (Mamba + DeltaNet + GQA, e.g., Qwen3-Next) require `--attention-backend triton` on Blackwell GPUs. FlashInfer will fail with "triton or trtllm_mha backend are the only supported backends on Blackwell GPUs for hybrid GDN models". This is a framework constraint, not a model limitation.

#### Hybrid attention + HiCache requires CUDA graph disabled

When serving hybrid attention models (e.g., Qwen3-Next with DeltaNet+GQA) with HiCache KV offloading, use `--disable-cuda-graph`. CUDA graph compilation conflicts with HiCache's dynamic memory management for hybrid models. This constraint applies only to hybrid architectures; standard transformer models can use CUDA graphs with HiCache.

#### vLLM Mistral tool-call parser generates non-compliant tool_call IDs
<!-- stack: vllm=0.15.0 | validated: 2026-02-20 -->

The Mistral parser in vLLM (through v0.15.0) generates `call_0`, `call_1`, etc. as tool_call IDs instead of the OpenAI-spec 9-character alphanumeric format required by BFCL and most downstream tools. This causes multi-turn tool-use failures where the second turn's tool_result is rejected due to ID format validation. Workaround: patch the eval script to accept short IDs or use `--tool-call-parser hermes` if the model supports it. Track vLLM issue #23180 for upstream fix.

#### SGLang qwen3_coder parser outputs tool calls as XML in content field
<!-- stack: sglang=0.5.9 | validated: 2026-03-01 -->

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
<!-- stack: sglang=nightly-2026-03-07, lmcache=0.3.15 | validated: 2026-03-10 -->

For models using NSA (Native Sparse Attention) or MLA (Multi-Head Latent Attention) architectures such as GLM-5 (`glm_moe_dsa`) and DeepSeek V3, use SGLang's built-in HiCache (`--enable-hierarchical-cache`) instead of LMCache for KV cache offloading. HiCache has native `NSATokenToKVPoolHost` support that understands the fused `kv_buffer` layout used by MLA models, while LMCache expects separate `k_buffer`/`v_buffer` attributes and crashes on MLA. HiCache is integrated into SGLang and evolves with the attention backend, eliminating external compatibility issues. This applies to all MLA/NSA models on SGLang until LMCache PR #2629 merges.

#### HiCache --hicache-size must exceed device KV pool size to pass initialization assertion

SGLang HiCache asserts `host_memory > device_memory` during initialization. Set `--hicache-size` to at least the device KV pool size plus margin. For example, if the device KV pool is approximately 82 GB per TP rank, use `--hicache-size 100` (100 GB per rank). Do not rely on the default `--hicache-ratio 2.0` which calculates 2x device pool per rank — this can exceed available system RAM on memory-constrained instances and cause OOM. Calculate total host memory requirement as `num_tp_ranks × hicache_size` and verify it fits within available system RAM before launching. This is a framework requirement, not a model-specific constraint.

#### For memory-constrained models, CPU KV cache offloading fundamentally changes the concurrency ceiling

When model weights consume most GPU VRAM (e.g., GLM-5 FP8 using 175 GB / 183 GB per GPU), the device KV cache becomes the primary throughput bottleneck. CPU KV cache offloading (via HiCache or similar) can deliver 2-3x throughput improvement at high concurrency by extending effective KV cache capacity beyond GPU VRAM. Superlinear scaling at high concurrency (e.g., baseline plateaus at 64 concurrent while HiCache continues scaling to 128+ concurrent) confirms that KV cache eviction was limiting throughput, not compute capacity. Always benchmark both baseline (device-only KV cache) and CPU offload configurations for large models to identify whether KV cache or compute is the true bottleneck. This pattern applies across models and frameworks, not just specific architectures.

#### Redis can run on GPU nodes with taint toleration when system nodes lack capacity

System nodes (e.g., m5.xlarge) often lack sufficient CPU or memory for auxiliary services like Redis. Adding `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` to Redis (or other non-GPU workloads) allows them to schedule on GPU nodes, which typically have abundant CPU and RAM beyond what serving workloads use. For example, on p6-b200.48xlarge nodes running vLLM, 80% of CPU and 90% of memory remain free. This pattern applies to any Kubernetes cluster where auxiliary services need more resources than the system node pool provides. Use resource requests/limits to ensure the auxiliary service does not starve the GPU workload.

### Benchmark Observability (mandatory)

#### Every benchmark GPU node runs Prometheus + DCGM + node-exporter from bootstrap
<!-- stack: prometheus=2.54.1, dcgm-exporter=3.3.9, node-exporter=1.8.2 | validated: 2026-05-14 -->

Benchmark data is permanently lost if the client-side driver is the only source. TTFT (time-to-first-token) and TPOT (time-per-output-token) live only in engine histograms (`vllm:time_to_first_token_seconds_bucket`, `sglang:time_to_first_token_seconds_bucket`). HBM bandwidth utilization, tensor-core activity, and SM occupancy live only in DCGM. Client-side drivers that measure total request duration and divide by token count produce **latency averages that cannot be decomposed** — indistinguishable from cases where prefill is slow vs decode is slow.

**Requirement**: Every benchmark GPU node MUST launch `.claude/skills/benchmark-runner/templates/observability-stack.docker-compose.yml` at bootstrap time, BEFORE the serving stack. Infra-deployer Stage 4b enforces this. The `observability-smoke-test.sh` check blocks progression to serving deployment until all exporters are healthy. Snapshots sync to S3 every 10 min via systemd timer so data survives spot reclaim.

**Evidence**: Kimi K2.6-spec 2026-05-13 session ran 95 benchmark configs on p6-b300 and captured 0 TTFT data points. Client driver recorded aggregate duration only; Prometheus was never installed. After spot termination, the data was unrecoverable. Post-hoc enrichment produced v1 envelopes with `ttft_ms: null` across all 95 files — a permanent dataset gap.

**Rule**: If a blueprint runs benchmarks (i.e., has `benchmark.yaml` sidecar), Stage 4b is mandatory. Non-benchmark-only blueprints can skip but should still install for GPU telemetry during production traffic.

#### Always use `bench-standard.py` as the bench driver — never a blueprint-local clone
<!-- validated: 2026-05-14 -->

The canonical bench driver at `.claude/skills/benchmark-runner/scripts/bench-standard.py` is Prometheus-first: it queries engine histograms and DCGM metrics at run end, emits the v1 benchmark-commons envelope directly, and reconciles client-side request counts against Prometheus counters (5% tolerance). Blueprint runner scripts should invoke it with config overrides, not reimplement the client loop.

**Why not a local copy**: Bench drivers encode many subtle rules (settle-time after last response for histogram flush, tokens-per-step math for speculative decode, output-file naming convention matching the standard). Local copies drift and produce outputs that can't be compared across blueprints. The K2.6-spec session ran a local driver and lost TTFT — if the standard driver had been mandatory, the loss would not have happened.

**Rule**: If you find yourself writing `asyncio.gather(...)` in a blueprint's scripts directory to drive serving benchmarks, stop. Invoke `bench-standard.py` instead. Extend the standard driver if a new engine or workload type needs support, then contribute the change upstream so all blueprints inherit it.

#### For coarse bottleneck classification, prefer continuous in-pod `nvidia-smi dmon` over point-sampled DCGM PROF
<!-- validated: 2026-06-16 -->

DCGM-exporter PROF fields (`DCGM_FI_PROF_SM_ACTIVE`, etc.) scraped point-in-time over the pod network miss bursty load peaks — can read SM ~0.15 during request gaps when the true steady-state is SM ~100%, leading to a false "launch-bound" read. For the coarse launch-bound vs compute-bound vs bandwidth-bound call, run continuous in-pod `nvidia-smi dmon -s u` to a log during the benchmark and read the sustained averages. DCGM remains necessary for rich per-kernel metrics (HBM BW, tensor-core activity, NVLink) that `nvidia-smi` does not expose — and the mandatory Prometheus+DCGM stack above still applies for the canonical envelope. Operational note: if the DCGM exporter is running but Prometheus isn't scraping it (missing ServiceMonitor), scrape the exporter pod directly on `:9400` rather than debugging the ServiceMonitor mid-session.

(remaining content continues...trimmed for character limit but file would continue with all remaining sections)