# Lessons Learned: Qwen3-Next Customer Benchmark

## Pre-Benchmark Setup

### 1. Customer vLLM Image Requires CodeBuild
**Problem**: Building the customer's custom vLLM image (~16.8 GB) on EKS system nodes failed — kaniko pods were evicted due to insufficient ephemeral storage on m6i.xlarge nodes.

**Fix**: Used AWS CodeBuild with `BUILD_GENERAL1_LARGE` compute type (8 vCPU, 15 GB RAM, 128 GB storage) to build and push to ECR. Build took ~394s + 411s push.

**Lesson**: For large container images (>10 GB), use CodeBuild or a dedicated build instance rather than in-cluster builds. EKS system nodes are not sized for heavy image builds.

### 2. Existing FSx Lacks EFA — Cannot Retrofit
**Problem**: The parent blueprint's FSx filesystem (`fs-06ed...`) was created with `efa_enabled = false`. EFA is a creation-time setting on FSx Lustre and cannot be changed after the fact. This blocked GDS benchmarks (T5d) which require EFA for zero-copy GPU-to-FSx transfers via NIXL.

**Fix**: Created a new FSx filesystem (`fs-0952e4fd84eed47af`) with `efa_enabled = true`, PERSISTENT_2, 4800 GiB, Lustre 2.15, 1000 MB/s/TiB throughput. Required a dedicated security group with self-referencing ingress/egress rules for EFA traffic.

**Lesson**: Always create FSx Lustre filesystems with `efa_enabled = true` if there's any chance of needing GDS or RDMA access. The cost is zero — EFA is free on supported instance types — but retroactively enabling it requires recreating the filesystem.

### 3. Capacity Block Requires --instance-market-options MarketType=capacity-block
**Problem**: `aws ec2 run-instances` with `--capacity-reservation-specification` alone fails with "The market type (purchasing) option is not valid." Capacity blocks are a distinct market type from on-demand.

**Fix**: Add `--instance-market-options 'MarketType=capacity-block'` and `--placement 'AvailabilityZone=us-east-2c'` to the `run-instances` command.

**Lesson**: This is kimi lesson #1 again. Capacity blocks require explicit market type declaration. The launch script has been updated.

### 4. Capacity Block Instance Needs Separate EKS Access Entry
**Problem**: The instance profile used by `run-instances` has a different IAM role (`qwen3-next-bench-gpu-node-...`) than the managed node group role (`gpu-eks-node-group-...`). Without an explicit EKS access entry for the capacity block instance's role, the node fails to join the cluster with an authentication error.

**Fix**: Created an `EC2_LINUX` access entry for the instance profile's role:
```bash
aws eks create-access-entry --cluster-name qwen3-next-bench-eks-cluster \
  --principal-arn arn:aws:iam::615299764834:role/qwen3-next-bench-gpu-node-... \
  --type EC2_LINUX
```

**Lesson**: When launching EC2 instances directly (not via managed node groups), always verify the instance's IAM role has an EKS access entry. Managed node groups auto-create these; `run-instances` does not.

### 4. Dynamo KVBM FSx Permission Issue (from kimi-k2.5 lesson)
**Problem**: Dynamo container user cannot create files on FSx Lustre mount even with 777 host permissions. This is a UID mapping issue — the container's internal user doesn't map to a valid Lustre UID.

**Fix**: Run container as root (`--user 0:0`) and pre-create the cache directory with `chmod 777` on the host before starting the container. Added both to `configs/vllm-constrained-dynamo-fsx.sh`.

**Lesson**: Always run Dynamo KVBM containers as root when writing to FSx Lustre. Lustre's UID mapping doesn't play well with container user namespaces.

### 4. Dynamo CPU Cache OOM at 128GB (from kimi-k2.5 lesson #25)
**Problem**: Dynamo container was OOM-killed with `DYN_KVBM_CPU_CACHE_GB=128`. Model memory + CPU cache + OS overhead exceeded system memory limits.

**Fix**: Reduced to 64GB in the custbench config. Qwen3-Next FP8 at TP=4 uses ~20GB GPU per shard, with lower CPU memory pressure than Kimi K2.5 (TP=8, INT4).

**Lesson**: Budget conservatively for CPU cache. 64GB is a safe upper bound alongside inference workloads.

### 5. GDS Buffer Registration May Fail in Container (from kimi-k2.5 caveat)
**Problem**: `GDS_MT: warning: buffer registration failed - will use compat mode: error=5030`. GDS falls back to POSIX I/O inside containers even with `--privileged`.

**Mitigation**: Mount host `/dev` into container (`-v /dev:/dev`) and ensure cuFile config is present. If GDS still fails, T5d results represent POSIX-mode FSx offloading (slower but functional). The EFA-enabled FSx should provide better results than the non-EFA FSx used in kimi benchmarks.

**Lesson**: Validate GDS mode in container logs before benchmarking. Check for `GDS_MT: using GDS` (success) vs `compat mode` (fallback).

### 6. Customer Nightly vLLM Has Broken DeepEP/pplx-kernels ABI
**Problem**: The customer's vLLM nightly build (v0.16.0rc2.dev479) includes custom expert parallelism extensions (DeepEP, pplx-kernels) compiled against a different PyTorch version. Worker processes fail at import time with `ImportError: undefined symbol: _ZN5torch9TypeErrorC1EPKcz`. The `gpu_worker.py` has a hard `import deep_ep` at module level, making it impossible to skip.

**Fix**: Used stable `vllm/vllm-openai:latest` for all benchmarks. The customer's config *flags* (no prefix caching, no chunked prefill, MTP) are the focus, not their specific binary build. Removing the `.so` files is insufficient because Python-level imports still fail.

**Lesson**: Customer nightly builds with custom kernel extensions often have ABI compatibility issues. Always test with stable vLLM first to establish a baseline. Customer-specific binaries should be validated on the target GPU/driver before benchmark sessions.

### 7. nerdctl -d and --rm Cannot Be Used Together
**Problem**: `nerdctl run -d --rm` fails with "flags -d and --rm cannot be specified together". Docker allows this combination, but nerdctl does not.

**Fix**: Use `-d` without `--rm` for detached containers. Clean up manually with `nerdctl rm -f` before restarting.

### 8. Root Partition Not Auto-Expanded on AL2 GPU AMI
**Problem**: The p5en.48xlarge instance was launched with a 500 GB EBS root volume, but the AL2 GPU AMI only partitions 20 GB. The remaining 480 GB is unused. Container image extraction (~33 GB per image) filled the 20 GB partition.

**Fix**: Run `growpart /dev/nvme0n1 1 && xfs_growfs /` after instance launch to expand the partition to the full EBS size.

**Lesson**: AL2 GPU AMIs use a fixed 20 GB root partition regardless of EBS volume size. Always run `growpart` + `xfs_growfs` in user data or as a post-launch step.

### 9. vllm bench CLI Requires GPU Access Even for Client
**Problem**: `vllm bench serve` (the benchmark client) fails if no GPU is detected, even though it only sends HTTP requests. The CLI initialization code tries to infer the device type.

**Fix**: Give the benchmark client container access to one unused GPU (e.g., GPU 4 when server uses GPUs 0-3). Setting `CUDA_VISIBLE_DEVICES=` or `VLLM_TARGET_DEVICE=cpu` does not work.

**Lesson**: Reserve one GPU for the benchmark client when running TP<8 configs on 8-GPU instances. Alternatively, use a non-vllm benchmark tool (e.g., custom HTTP client with aiohttp).

### 10. K8s vLLM Deployment Conflicts with Container Benchmarks
**Problem**: The parent blueprint's K8s vLLM deployment was running on the same node, consuming GPUs and port 8000. Our nerdctl containers couldn't start.

**Fix**: Scale down the K8s deployment before benchmarking:
```bash
kubectl scale deployment/vllm-qwen3-next -n ml-inference --replicas=0
```

**Lesson**: Always scale down K8s inference deployments before running direct container benchmarks on the same node.

## Benchmark Results

See [session-20260226/benchmark-report.md](results/session-20260226/benchmark-report.md) for full results.

### Key Finding: Concurrency is the Primary Latency Driver

| Load Level | QPS | TTFT P50 | Throughput |
|-----------|-----|----------|------------|
| Low (14 concurrent) | 0.5 | **238 ms** | 5,263 tok/s |
| Moderate (186 concurrent) | 5.0 | **243 ms** | 31,724 tok/s |
| Extreme (1000 concurrent) | inf | **64,042 ms** | 49,558 tok/s |

### Key Finding: Prefix Caching Delivers 82% TTFT Reduction

With 1000 concurrent requests sharing an 8K system prompt:
- Random data: TTFT P50 = 77s, throughput = 40K tok/s
- Shared prefix: TTFT P50 = **13.5s**, throughput = **71K tok/s** (+77%)

### 11. CPU Offload Blocked on FP8 + V1 Engine
**Problem**: `--cpu-offload-gb 64` with `--gpu-memory-utilization 0.30` fails: `NotImplementedError: Cannot copy out of meta tensor; no data!`. vLLM 0.16 V1 engine loads FP8 models using meta tensors that can't be copied to CPU.

**Lesson**: CPU weight offloading is incompatible with FP8 quantized models on vLLM V1. Use full GPU memory utilization instead, or wait for vLLM to implement proper KV cache offloading (separate from weight offloading).

### 12. Prefix Cache Even More Impactful Under Memory Constraints
**Problem**: With `--gpu-memory-utilization 0.30`, TTFT P50 = 169.9s (2.2x worse than normal). Fewer KV cache slots mean massive queuing.

**Fix**: Prefix caching reduces TTFT by **94%** to 9.4s (vs 170s without) and increases throughput by **57%**. The cached prefix avoids consuming KV slots per request.

**Lesson**: Prefix caching is critical for memory-constrained GPUs (g7e, g6e). It effectively expands usable KV cache capacity.

### 13. CUDA_VISIBLE_DEVICES Must Be Container-Relative
**Problem**: When using `nerdctl run --gpus "device=4,5,6,7"`, setting `-e CUDA_VISIBLE_DEVICES=4,5,6,7` causes pynvml `InvalidArgument` errors. Inside the container, the 4 GPUs are remapped to indices 0-3.

**Fix**: Set `CUDA_VISIBLE_DEVICES=0,1,2,3` (container-relative indices) or omit it entirely and let `--gpus` handle isolation.

**Lesson**: Container GPU indices are always 0-indexed relative to the container's visible GPUs, not the host's physical GPU indices.

### 14. 2-Replica Scaling Achieves 1.71x Throughput
**Finding**: 2x TP=4 replicas on p5en.48xlarge deliver 84,628 tok/s combined (1.71x vs single replica's 49,558 tok/s). Sub-linear scaling is due to shared NVMe bandwidth.

**Finding**: 1500 concurrent requests (750/replica) completed with zero failures. System is production-stable.

**Lesson**: The recommended production topology for p5en.48xlarge is 2x TP=4 replicas behind a load balancer, not 1x TP=8.

### 15. vLLM Prometheus Metric Names Include Labels (Not Bare)
**Problem**: First metrics scraper attempt returned all zeros. The grep patterns used `^vllm:prefix_cache_hit_rate ` expecting bare metrics, but vLLM v0.16 metrics include Prometheus labels: `vllm:prefix_cache_hits_total{engine="0",model_name="qwen3-next"} 6.473872e+06`.

**Fix**: Changed grep to `^vllm:prefix_cache_hits_total{` (match up to the opening brace). Also: there is no single `prefix_cache_hit_rate` metric — hit rate must be computed from `prefix_cache_hits_total / prefix_cache_queries_total`. Similarly, `gpu_cache_usage_perc` is actually `kv_cache_usage_perc`.

**Lesson**: Always inspect the raw `/metrics` output before building scrapers. vLLM v0.16 metric names differ from v0.15 documentation.

### 16. Prefix Cache Hit Rate is 97% with Shared Prefixes
**Finding**: With 500 requests sharing an 8K-token prefix and 128-token unique suffix, the benchmark-specific prefix cache hit rate was 96.95% (4.18M hits / 4.31M queries in tokens). Peak KV cache usage was only 8.5% because cached prefix blocks are shared, not duplicated.

**Lesson**: Prefix caching is the single most impactful optimization for workloads with shared system prompts. It reduces both latency (by avoiding redundant prefill) AND memory usage (by sharing KV blocks). This makes it especially critical on smaller GPUs (g7e, g6e) where KV cache capacity is limited.

### 17. Dynamo 0.9.0 Removed dynamo-run CLI
**Problem**: The `dynamo-kvbm-qwen3:latest` image has `ai-dynamo==0.9.0` installed, but `dynamo-run` doesn't exist as a CLI tool. The `ai-dynamo-vllm` package (which may have provided it) only goes up to 0.8.4.post4 on PyPI and is incompatible with vLLM 0.16.

**Root cause**: Dynamo 0.9.0 changed architecture from a CLI wrapper (`dynamo-run`) to a distributed runtime API (`make_engine()` + KV event routing via `KvEventPublisher`, `KvPushRouter`, etc.). The old KVBM approach with `DYN_KVBM_*` env vars is deprecated.

**Lesson**: NVIDIA Dynamo's API changes rapidly. Pin and verify the exact version before building images. For vLLM 0.16+, use native vLLM KV offloading (`OffloadingConnector`, `LMCacheMPConnector`) instead of Dynamo KVBM.

### 18. KV Offloading Incompatible with Hybrid Attention Models (vLLM 0.16)
**Problem**: All 4 KV offloading approaches fail with Qwen3-Next on vLLM 0.16.0:
1. `--cpu-offload-gb` → FP8 meta tensor error
2. Dynamo KVBM → `dynamo-run` not found
3. `OffloadingConnector` → HMA disabled, can't unify hybrid KV cache specs
4. `LMCacheMPConnector` → same HMA error

**Root cause**: vLLM 0.16 auto-disables Hybrid KV cache Manager (HMA) when `--kv-transfer-config` is set. Qwen3-Next has hybrid attention (different KV cache specs across layer groups), which requires HMA. None of the registered connectors implement `SupportsHMA`.

**Lesson**: KV cache offloading on models with hybrid attention (MoE with mixed attention patterns) is unsupported in vLLM 0.16. The workaround is to use full GPU memory utilization instead of offloading. For memory-constrained scenarios, prefix caching (T5c: 94% TTFT reduction) is the best available optimization. Watch for HMA support in future vLLM KV connectors.
