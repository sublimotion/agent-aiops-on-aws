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

## Benchmark Results

*(To be populated after benchmark session)*
