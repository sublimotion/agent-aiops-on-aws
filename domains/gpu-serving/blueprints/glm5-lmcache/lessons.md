# GLM-5 LMCache Deployment Lessons

## Lesson #1: B200 NVL5+ requires AL2023 AMI (not AL2)

**Problem**: NVIDIA Fabric Manager failed to start on Amazon Linux 2 (kernel 5.10) because the `ib_umad` kernel module is not compiled into the AL2 kernel. The Fabric Manager start script requires `ib_umad` for NVL5+ systems.

**Error**: `Kernel module "ib_umad" has not been loaded, fabric manager cannot be started`

**Root cause**: AL2 kernel 5.10 doesn't include `CONFIG_INFINIBAND_USER_MAD=m`. The `rdma-core` and `libibumad` userspace packages are installed, but the kernel module is missing.

**Fix**: Use `amazon-eks-node-al2023-x86_64-nvidia-1.32-v20260304` (ami-02bb9f913067dadb1) which uses kernel 6.1 with full IB stack including `ib_umad`.

**Impact**: Without Fabric Manager, CUDA returns error 802 (`cudaErrorSystemNotReady`) — no GPU access from containers despite `nvidia-smi` showing GPUs from the host.

## Lesson #2: AL2023 EKS uses nodeadm, not bootstrap.sh

**Problem**: User data format for EKS nodes differs between AL2 and AL2023. AL2 uses `/etc/eks/bootstrap.sh`, AL2023 uses `nodeadm` with YAML config in MIME multipart format.

**Fix**: Use multipart MIME user data with `application/node.eks.aws` content type for the NodeConfig, and `text/x-shellscript` for post-boot scripts (NVMe RAID0, FSx mount, etc.).

## Lesson #3: GLM-5 (glm_moe_dsa) requires specialized SGLang image

**Problem**: The `lmsysorg/sglang:latest` image doesn't recognize the `glm_moe_dsa` model type. Standard `transformers` library doesn't include this architecture.

**Fix**: Use `lmsysorg/sglang:glm5-blackwell` for Blackwell GPUs (or `glm5-hopper` for Hopper). These images include the model architecture code and DeepGEMM optimizations.

## Lesson #4: SGLang defaults to host 127.0.0.1

**Problem**: SGLang's Uvicorn server defaults to listening on `127.0.0.1:30000`, making it unreachable from Kubernetes services.

**Fix**: Add `--host 0.0.0.0` to the server launch arguments.

## Lesson #5: DeepGEMM JIT takes ~15 min on first startup

**Problem**: First startup of GLM-5 on Blackwell requires DeepGEMM JIT compilation — 9+ kernel configurations with 65536 iterations each. This takes about 15 minutes.

**Mitigation**: Set readiness probe `initialDelaySeconds` to at least 900s (15 min). Consider pre-compiling with `sglang.compile_deep_gemm` in a custom image.

## Lesson #6: B200 termination is slow (~10 min)

**Problem**: Terminating p6-b200.48xlarge instances takes ~10 minutes before the capacity block slot becomes available for a new launch. This is slower than smaller instance types.

**Mitigation**: Plan for 10-minute gaps when replacing instances. Do not poll aggressively — check capacity block availability every 30 seconds.

## Lesson #7: hf_xet deadlocks on macOS, works on Linux

**Problem**: HuggingFace's XET storage library deadlocks on macOS (`_dispatch_semaphore_wait_slow`) when downloading large model files without authentication. Setting `HF_HUB_DISABLE_XET=1` did not fix it.

**Fix**: Download model files directly on the EC2 instance using `huggingface_hub.snapshot_download()`. EC2 instances don't have the macOS deadlock and have better bandwidth to HF CDN.

## Lesson #8: GLM-5-FP8 uses 175 GB / 183 GB per GPU

**Problem/Observation**: After loading GLM-5-FP8 with TP8, each B200 GPU uses ~175 GB of 183 GB HBM, leaving only ~8 GB for KV cache. With `mem_fraction_static=0.90`, ~9 GB is reserved for KV cache per GPU.

**Implication**: KV cache memory is tight. LMCache offloading (CPU/GDS/POSIX) is important for this model to handle long contexts or high concurrency. The 1.43 TB total HBM across 8 GPUs is necessary for this 744B MoE model.

## Lesson #9: Throughput scales well with batching on MoE

**Observation**: GLM-5 throughput scales from 90 tok/s (1 concurrent) to 1,530 tok/s (32 concurrent) — a 17x improvement. This is because only ~40B parameters are active per token (top-8 out of 256 experts), so batching amortizes the routing overhead effectively.

## Lesson #10: LMCache incompatible with SGLang NSA/MLA attention (as of v0.3.15)

**Problem**: LMCache `--enable-lmcache` crashes on GLM-5 with `AttributeError: 'NSATokenToKVPool' object has no attribute 'k_buffer'. Did you mean: 'kv_buffer'?`

**Root cause**: GLM-5 uses NSA (Native Sparse Attention) with DSA (DeepSeek Sparse Attention). `NSATokenToKVPool` inherits from `MLATokenToKVPool` which uses a fused `kv_buffer` instead of separate `k_buffer`/`v_buffer`. LMCache's SGLang adapter (`lmc_radix_cache.py` line 96) expects the separate buffers.

**Status**: LMCache PR #2629 (MLA layerwise support) is open but NOT merged. Both SGLang-side and LMCache-side changes are needed. Tracked in LMCache issue #2636.

**Impact**: LMCache KV cache offloading (CPU, GDS, POSIX) cannot be used with GLM-5 or any MLA/NSA model on SGLang until PR #2629 merges. SGLang's built-in RadixAttention prefix caching works fine as a baseline.

**Workaround**: Use SGLang RadixAttention only (no `--enable-lmcache`). The baseline benchmark shows prefix caching is effective with consistent latency across rounds.

## Lesson #11: PYTHONPATH NVMe trick for persistent pip installs on EKS

**Problem**: Installing Python packages in a running pod is lost on restart. EKS nodes with AL2023 lack `buildctl`/`buildkitd` for `nerdctl build`, and Kaniko fails on 14 GB Docker Hub images.

**Workaround**:
1. `pip install` in running container
2. Copy installed packages to NVMe hostPath: `cp -a /usr/local/lib/python3.12/dist-packages/{pkg,pkg.dist-info} /mnt/nvme/lmcache-packages/`
3. Set `PYTHONPATH=/mnt/nvme/lmcache-packages` in deployment env
4. **Must copy `.dist-info` directories** — `importlib.metadata` needs them for version resolution
5. Packages persist across pod restarts via hostPath volume

## Lesson #12: LMCache `dev` branch overrides transformers version

**Problem**: LMCache's dependency chain pulls `transformers 5.2.0.dev0`, overriding SGLang's pinned `transformers==4.57.1`. This causes RoPE parameter warnings but doesn't break model loading.

**Impact**: `WARNING: Transformers version 5.2.0.dev0 is used for model type glm_moe_dsa`. The glm5-blackwell image's custom model code still functions with the newer transformers, but this is fragile.

## Lesson #13: SGLang HiCache works with NSA/MLA attention — use instead of LMCache

**Problem**: LMCache is blocked on NSA/MLA (lesson #10), but SGLang's built-in HiCache (`--enable-hierarchical-cache`) has native `NSATokenToKVPoolHost` support.

**Key finding**: HiCache's `hiradix_cache.py` has explicit code paths for `NSATokenToKVPool`, creating a `NSATokenToKVPoolHost` instance that understands the fused `kv_buffer` layout. No external library needed.

**Performance**: HiCache CPU offload delivered 2,602 tok/s at 128 concurrent (vs baseline 909 tok/s at 64 concurrent — 2.86x improvement). Single-request throughput unchanged (48 tok/s).

**Recommendation**: For MLA/NSA models on SGLang, use `--enable-hierarchical-cache` instead of `--enable-lmcache`. HiCache is built into SGLang and evolves with the attention backend.

## Lesson #14: HiCache `--hicache-size` must exceed device KV pool

**Problem**: HiCache asserts `host_memory > device_memory` during initialization. With `--hicache-size 50` (50 GB per rank) but device KV pool ~82 GB per rank, the assertion fails: `AssertionError: The host memory should be larger than the device memory with the current protocol`

**Fix**: Set `--hicache-size` to at least the device KV pool size + margin. For GLM-5 on B200 with `mem_fraction_static=0.90`, use `--hicache-size 100` (100 GB per rank). Total: 8 x 100 = 800 GB, fits in p6-b200.48xlarge's 2 TB RAM.

**Avoid**: Default `--hicache-ratio 2.0` calculates 2x device pool per rank (165 GB x 8 = 1,325 GB) which can OOM on memory-constrained systems.

## Lesson #15: HiCache superlinear scaling at high concurrency indicates KV cache was the bottleneck

**Observation**: Baseline throughput hit a ceiling at 64 concurrent (909 tok/s) while HiCache continued scaling to 128 concurrent (2,602 tok/s). The 71% improvement at 64 concurrent (909→1,556 tok/s) confirms that KV cache eviction was the primary throughput bottleneck, not compute.

**Implication**: For models that use most of GPU memory for weights (GLM-5 uses 175/183 GB per GPU), CPU KV cache offloading is not just a "nice to have" — it fundamentally changes the concurrency ceiling. HiCache's write-through policy ensures evicted entries are immediately available for reload without recompute.
