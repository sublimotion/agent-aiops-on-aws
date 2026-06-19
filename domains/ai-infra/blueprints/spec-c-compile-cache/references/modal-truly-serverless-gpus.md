# Reference: Modal "Truly Serverless GPUs" (2026-05)

**Source**: https://modal.com/blog/truly-serverless-gpus
**Date captured**: 2026-05-24
**Why this matters**: Modal documents a 40× cold-start speedup (2000s → 50s) via four compounded techniques. This re-bases Spec C's hypothesis from "AOT compile cache HIT = 22s floor" to "GPU memory snapshot can boot vLLM in 13.8s for small models."

## Headline numbers

- **40× cold-start speedup**: ~2,000 s → ~50 s end-to-end
- **vLLM Qwen 3 0.6B (1 GiB bf16)**: 95.7 s → **13.8 s** with GPU memory snapshot
- **SGLang Qwen 3 0.6B**: 83.7 s → **17.5 s** with GPU memory snapshot
- **Reducto customer result**: 70s → 12s, ~6× via GPU snapshotting

## Four-technique stack

### 1. Cloud Buffers (Instance pre-warming)
- Idle GPU pool shared across applications
- Linear program (GLOP) optimizes warm pool size against scraped prices and observed supply
- Removes "tens of minutes" of instance allocation from hot path
- Application layer can request its own `buffer_containers`
- GPU health checks: `dcgmi diag` weekly (intensive); lighter checks more frequent

### 2. Custom Filesystem (ImageFS)
- libfuse-based, content-addressed, multi-tier cache
- Lazy loading: metadata only (~few MB, <100ms) at start
- **Skips gzip decompression** — DEFLATE's ~100 MB/s single-threaded ceiling is a hidden bottleneck
- `read_ahead_kb` tuned from 128 → 32×1024 (32 MB)
- Cuts ~1 minute from container start
- **Cache hierarchy with concrete numbers**:
  | Tier | Latency | Throughput |
  |---|---|---|
  | Page cache | 0.001-0.1 µs | 10-40 GiB/s |
  | SSD | 100 µs | 4 GiB/s |
  | AZ Cache Server | 1,000 µs | 10 GiB/s |
  | Regional CDN | 100,000 µs | 3-10 GiB/s |
  | Blob storage | 200,000 µs | 3-10 GiB/s |

### 3. CPU Memory Snapshotting (host-side C/R)
- Uses **gVisor `runsc checkpoint`/`restore`** — NOT Linux CRIU
- Works because runsc is a Go state machine with cooperative preemption at await points
- **Uncompressed checkpoints** to avoid gzip bottleneck (`pages.img` 100 MB to many GB)
- ~10× reduction in host-side load time
- Exposed as `@modal.enter(snap=True)` decorator
- Caveat: snapshots tied to host CPU features (`pclmulqdq` etc.) — must match instance type

### 4. GPU Memory Snapshotting (CUDA context)
- Uses **Nvidia driver feature** to checkpoint device memory → host RAM → disk
- Builds on host-side C/R + custom filesystem
- 4-10× speedup typical
- Caveats:
  - **Multi-GPU snapshotting deadlocks via NCCL** (open problem; relevant for our TP=8 Kimi work)
  - vLLM/SGLang need weight offloading before snapshot
  - KV cache better recreated than restored

## Scale evidence

- CPU snapshots: ~35M replicas restored (Feb-Apr 2026), >5M execution hours, ~1M distinct snapshots
- CPU+GPU snapshots: ~15M replicas restored, >2M execution hours, ~700K distinct snapshots

## What this means for our lab

### Spec C is partially obsolete

Our Spec C (compile cache strategies) was framed against Modal's **22s AOT compile cache HIT floor** for Gemma-4 26B. Modal's "Truly Serverless" post shows the actual production floor is **~13.8s for Qwen3 0.6B via GPU memory snapshot**. The compile cache is just one component of what gets snapshotted.

Spec C should be split:
- **Spec C-AOT** (existing): AOT compile cache HIT — what we can do without snapshot tooling, ~22s floor for 26B class
- **Spec C-Snapshot** (new): GPU memory snapshot via Nvidia driver feature — needs investigation of whether the API is publicly available outside Modal's stack

### The cudagraph trim finding still stands

Modal's snapshot includes the captured CUDA graphs. Our trim measurement (162s saved on Kimi K2.6 warm path, 28% of stage) is the without-snapshot equivalent — it's what you get when you can't snapshot.

### Multi-GPU NCCL deadlock is the killer

Modal explicitly says **multi-GPU snapshotting deadlocks via NCCL**. Our Kimi K2.6 runs are TP=8. So the GPU snapshot technique that gives Modal 4-10× speedup on small single-GPU models may not apply to our frontier-MoE workload at all. This is the most important caveat in the post for our work.

### The image-pull stage finding

Modal's `read_ahead_kb=32×1024` matches what we read in their earlier "Fast lazy container loading" writeup. Confirms the FUSE tunable as a portable optimization. Spec E (FUSE tuning) variant should adopt this.

### Open question for our lab

Can we replicate any subset of the snapshot stack on EKS without Modal's gVisor/ImageFS infrastructure?

- **CPU memory snapshots via Linux CRIU**: possible in principle, never validated on a vLLM/Python process tree. Modal explicitly avoided CRIU.
- **GPU memory snapshots**: needs Nvidia driver feature access; investigate whether NVIDIA's publicly-documented APIs (cuMemMap, IPC handles, MPS) allow this without Modal's infrastructure.
- **Realistic for our budget**: probably no. Spec C-Snapshot is out of scope for this lab cycle.

## Updated Spec C target floor

For frontier MoE on EKS without Modal-class infrastructure:
- Today's Kimi K2.6 warm-NVMe (B200 baseline): **573 s** main-to-ready
- After cudagraph trim (measured today): **411 s** (162 s saved)
- After AOT compile cache HIT (projected, Spec C-AOT): **~150-200 s** (compile portion ~22s floor + remaining warmup + capture not snapshot-eliminated)
- **Modal-equivalent if we had GPU snapshot**: ~50 s (40× ceiling on full stack)
- **Current best achievable on EKS without Modal**: ~150-200 s
