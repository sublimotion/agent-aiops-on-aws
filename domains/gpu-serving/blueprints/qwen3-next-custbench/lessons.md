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

## Benchmark Results

*(To be populated after benchmark session)*
