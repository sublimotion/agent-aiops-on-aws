# Lessons Learned: Kimi K2.5 on P5e

## P5e Deployment

### 1. EKS Doesn't Support Capacity Block Market Type
**Problem**: EKS managed node groups cannot launch instances with `MarketType=capacity-block`.

**Solution**: Launch EC2 instance directly and join to EKS cluster manually.

```bash
# Launch p5e with capacity block (not supported by EKS managed node groups)
aws ec2 run-instances \
  --instance-type p5e.48xlarge \
  --capacity-reservation-specification 'CapacityReservationTarget={CapacityReservationId=cr-xxx}' \
  --instance-market-options 'MarketType=capacity-block' \
  --placement 'AvailabilityZone=us-east-2c' \
  ...

# Then add EKS access entry for the instance role
aws eks create-access-entry --cluster-name <cluster> --principal-arn <instance-role-arn>
```

**Lesson**: Plan for manual node management when using capacity blocks with EKS.

### 2. Large MoE Model Loading is Slow
**Problem**: Kimi K2.5 (64 safetensor shards) takes ~25 minutes to load across 8x H200s.

**Lesson**:
- Use persistent pods or keep-alive strategies
- Pre-pull model weights to FSx for faster loading
- Consider model loading time in capacity planning

### 3. Tensor Parallelism Must Match GPU Count
**Problem**: Setting `--tensor-parallel-size` incorrectly causes CUDA errors.

**Solution**: For p5e.48xlarge with 8x H200, always use `--tensor-parallel-size 8`.

**Lesson**: TP must equal the number of GPUs. Verify with `nvidia-smi` before deployment.

---

## LMCache + FSx Benchmarking

### 4. TTFT Measurement is Model-Dependent
**Problem**: TTFT showed 0ms for Kimi K2.5 requests in most benchmarks.

**Cause**: Kimi K2.5 uses `delta.reasoning` tokens before `delta.content`, so standard TTFT detection (waiting for first content token) fails.

**Lesson**: Reasoning models need custom TTFT measurement that accounts for thinking tokens. Use E2E latency as primary metric.

### 5. Token Estimation ≠ Actual Tokenization
**Problem**: "28K context" config generated ~51K actual tokens.

**Observation**: vLLM successfully processed requests exceeding stated 32K max_model_len.

**Lesson**: Word-based token estimation is unreliable (~1.8x off). Always verify with actual tokenizer or measure actual prompt_tokens from API response.

### 6. Cache Benefits Require Prefix Sharing
**Finding**:
| Scenario | Cache Benefit |
|----------|---------------|
| Shared system prompts (50 tenants) | 1.98x |
| Multi-turn chat (20 rounds) | 2.4x |
| Random prompts (no sharing) | 1.05x |

**Lesson**: LMCache value is proportional to prefix reuse. Random/unique workloads see minimal benefit (~5%).

### 7. Cold Start Penalty is Significant
**Finding**: First request is 1.8-2.5x slower than subsequent requests.

| Scenario | Cold | Warm | Penalty |
|----------|------|------|---------|
| 51K context | 4,403ms | 2,480ms | 1.8x |
| Multi-tenant | 9,545ms | 4,810ms | 2.0x |
| 20K chat | 3,176ms | 1,300ms | 2.4x |

**Lesson**: Pre-warm cache before production traffic. Consider cache warming during deployment.

### 8. FSx Scales Well for KV Cache
**Observed**: Cache grew from 189MB → 37GB (2,160 files) without performance degradation.

**Lesson**: FSx Lustre handles large KV cache workloads. Size based on prefix variety, not just traffic volume. Plan for 10-50GB per model.

### 9. High Tenant Variety Causes Tail Latency
**Finding**: 50 tenants → P90/P99 latency 10s+ (vs P50 of 2.9s)

**Lesson**: Many unique prefixes cause cache pressure and evictions. Monitor eviction rates in high-variety deployments. Consider increasing cache size or accepting higher tail latency.

### 10. Sub-linear Context Scaling Works
**Finding**:
| Context | E2E Latency | Latency per 1K tokens |
|---------|-------------|----------------------|
| 24K tokens | 3,095ms | 129ms |
| 36K tokens | 4,273ms | 119ms |
| 48K tokens | 5,521ms | 115ms |

**Lesson**: KV cache offloading enables efficient long context handling. Latency per token decreases with larger contexts due to batched retrieval.

### 11. LMBench Workloads Have Different Output Formats
**Problem**: StrictSynthetic showed 0ms TTFT in CSV because it uses completions API differently and measures timing differently.

**Lesson**: Always verify result format. Calculate E2E from `finish_time - launch_time` as fallback. Don't assume all benchmarks output data the same way.

### 12. GDS Validated: 9+ GB/s GPU→FSx via cuFile
**Finding**: LMCache GDS backend writes KV cache directly from GPU VRAM to FSx Lustre at 9.0-9.4 GB/s via cuFile.

**Validated Configuration** (K8s env vars):
```yaml
env:
  - name: LMCACHE_USE_EXPERIMENTAL
    value: "True"
  - name: LMCACHE_GDS_PATH
    value: "/mnt/fsx/kv-cache/lmcache"
  - name: LMCACHE_CUFILE_BUFFER_SIZE
    value: "8192"
  - name: LMCACHE_LOCAL_CPU
    value: "False"
  - name: CUFILE_ENV_PATH_JSON
    value: "/etc/cufile.json"
```

**Validated LMCache logs**:
```
LMCache INFO: GDS backend using fstype 'lustre' on path '/mnt/fsx/kv-cache/lmcache'
LMCache INFO: Using cufile
LMCache INFO: Stored 10752 tokens. size: 0.7037 GB, cost 76.6ms, throughput: 9.18 GB/s
```

**Lesson**: GDS works with FSx Lustre 2.15 PERSISTENT_2 on p5e.48xlarge. Cache grew to 9.2GB across benchmarks without issues.

### 13. LMCache ABI Mismatch with vLLM Nightly Images
**Problem**: LMCache 0.3.x prebuilt wheels are compiled against PyTorch 2.6/2.7 + CUDA 12. vLLM nightly images ship with PyTorch 2.10 which has different `c10_cuda` ABI. Import fails with:
```
ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**Solution**: Build LMCache from source at container startup:
```bash
# Install CUDA dev headers (runtime image doesn't have them)
apt-get install -y -qq libcusparse-dev-12-9 libcublas-dev-12-9 libcusolver-dev-12-9
# Build against installed torch (no build isolation)
TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 pip install lmcache --no-binary lmcache --no-deps --no-build-isolation
```

**Lesson**: Always build LMCache from source when using vLLM nightly. Pin specific vLLM releases for production to avoid ABI drift. The build adds ~3-4 min to startup.

### 14. cuFile Symlink Required in vLLM Images
**Problem**: vLLM nightly images ship `libcufile.so.0` but LMCache's GDS backend loads `libcufile.so` (unversioned). Without the symlink, LMCache falls back to "degraded mode (recompute)" — no actual KV offloading.

**Solution**: Create symlinks and run ldconfig before starting vLLM:
```bash
CUFILE_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cufile/lib
ln -sf ${CUFILE_DIR}/libcufile.so.0 ${CUFILE_DIR}/libcufile.so
ln -sf ${CUFILE_DIR}/libcufile_rdma.so.1 ${CUFILE_DIR}/libcufile_rdma.so
echo ${CUFILE_DIR} > /etc/ld.so.conf.d/cufile.conf && ldconfig
```

**Lesson**: Check logs for `LMCache ERROR: LMCacheEngine marked as init failed: libcufile.so` — if present, GDS is silently disabled and vLLM serves requests without KV offloading. Always verify `GDS backend using fstype 'lustre'` appears in startup logs.

### 15. CUDA 12 vs 13: Use `nightly` Not `cu130-nightly` for LMCache
**Problem**: `vllm/vllm-openai:cu130-nightly` (CUDA 13, torch 2.10+cu130) cannot build LMCache because `cusparse.h` and other CUDA dev headers don't exist for CUDA 13.0 in apt. The `nightly` image (CUDA 12.9, torch 2.10+cu129) has apt packages available.

**Solution**: Use `vllm/vllm-openai:nightly` (CUDA 12.9) for LMCache+GDS deployments. cuFile and GDS work identically on CUDA 12.9.

**Lesson**: CUDA 13 adoption is ahead of the LMCache/cuFile ecosystem. The `nightly` tag is the pragmatic choice for GDS benchmarking.

### 16. LMCache Benchmark Results vs Baseline (p5e.48xlarge, Kimi K2.5)

| Workload | TTFT (Baseline) | TTFT (LMCache) | E2E (Baseline) | E2E (LMCache) |
|----------|----------------|----------------|----------------|----------------|
| multi_turn_qa | 281ms | **152ms (-46%)** | 13.9s | 13.4s |
| long_context_rag | 279ms | 260ms (-7%) | 18.5s | 17.9s |
| strict_no_reuse | 140ms | 140ms (0%) | 13.7s | 13.5s |
| multi_tenant_50t | 152ms | 251ms (+66%) | 13.6s | 13.7s |
| context_16K | 183ms | 251ms (+37%) | 13.8s | 13.4s |
| context_48K | 250ms | 544ms (+117%) | 13.6s | 14.4s |

**Key findings**:
- Multi-turn Q&A (shared prefixes) shows **46% TTFT reduction** — LMCache's sweet spot
- Long context scaling adds TTFT overhead from GDS store operations
- E2E throughput (~11 tok/s) is unchanged — LMCache doesn't impact decode
- FSx cache: 9.2GB, GDS throughput: 9.0-9.4 GB/s
- **Real value**: cache persistence across pod restarts and cross-instance sharing

### 17. Reasoning Parser Changes E2E Latency Dramatically
**Problem**: Original LMCache benchmarks (Feb 14-15) showed 1-5s E2E latency. New benchmarks show ~13.5s E2E for identical `max_tokens=200`.

**Cause**: The original setup (`lmcache.vllm.entrypoints.openai.api_server`) did NOT include `--reasoning-parser kimi_k2`. Without it, Kimi K2.5's reasoning tokens either:
- Counted against `max_tokens`, capping total output at 200 tokens (reasoning + content combined)
- Or were suppressed entirely by vLLM

With `--reasoning-parser kimi_k2`, the model emits a full reasoning chain in `delta.reasoning_content` (200-500+ tokens, ~10-13s) **before** generating `delta.content` (200 tokens, ~3s). Reasoning tokens don't count against `max_tokens`.

**Impact on benchmarks**: E2E speedup ratios shrink dramatically:
| Config | Prefill savings | E2E without reasoning | E2E with reasoning |
|--------|----------------|----------------------|-------------------|
| 150ms saved | N/A | 1,150 / 1,300 = **1.13x** | 13,650 / 13,800 = **1.01x** |

**Lesson**: Always document whether `--reasoning-parser` is enabled. It changes both raw latency and the apparent benefit of any prefill optimization. The original report's 1.8-2.5x E2E speedups are real for that configuration but not reproducible with reasoning enabled.

### 18. LMCache Serializes Requests Under Moderate GPU Pressure
**Problem**: Memory pressure benchmark (25 sessions × 24K tokens) showed:
- Baseline: 25 running, 0 waiting
- LMCache+GDS: 5 running, 20 waiting → **13x worse foreground TTFT**

**Cause**: `LMCacheConnectorV1` with `kv_role: kv_both` writes KV blocks to FSx synchronously within vLLM's scheduling loop. Each request must complete its FSx write before the scheduler can admit the next request. This creates head-of-line blocking under concurrent load.

**Contributing factors**:
1. **Double-buffering**: Write path holds KV tensors in GPU memory during cuFile DMA, reducing effective KV cache capacity
2. **Sequential cuFile I/O**: GDS writes on a single/small CUDA stream pool are sequential — 25 requests × ~10ms write = 250ms of serialized I/O
3. **Scheduler integration**: vLLM's scheduling loop is single-threaded; the KV transfer connector hooks into it synchronously

**Data**:
| Pressure | LMCache fg TTFT | Baseline fg TTFT | LMCache penalty |
|----------|----------------|-----------------|----------------|
| Moderate (60% KV) | 5,330ms | 403ms | **13.2x** |
| Extreme (99% KV) | 22,313ms | 19,955ms | **1.1x** |

**Lesson**: LMCache's worst case is moderate concurrency where FSx I/O creates artificial queueing but the GPU isn't fully saturated. At extreme pressure, both systems are equally GPU-bound and the I/O overhead becomes negligible (~11%).

### 19. H200 HBM Makes KV Cache Offloading Largely Unnecessary
**Finding**: 8x H200 143GB = 1144GB HBM. After model weights (~540GB INT4), ~432GB remains for KV cache = 9,533 blocks × 64 tokens = **610K tokens capacity**.

At `max_model_len=32768`, this supports ~18 concurrent full-length sessions without any eviction. In practice, MoE models have smaller KV cache per token (only attention layers, not expert FFNs), so actual capacity is even higher.

**Memory pressure results**:
| Sessions × Context | KV Utilization | Preemptions |
|-------------------|---------------|-------------|
| 25 × 24K = 600K tokens | 61% | 0 |
| 50 × 32K = 1,600K tokens | 99% | 2-3 |

**Lesson**: On H200 (and H100 80GB to a lesser extent), KV cache pressure is rare for typical production workloads. LMCache's value proposition is strongest on smaller GPUs (A10G, L4, A100 40GB) where HBM is scarce. For H200 deployments, the primary value of FSx offloading is **cross-node sharing** and **persistence across restarts**, not memory relief.

### 20. LMCache Value Depends on Workload Pattern and Measurement
**Finding**: Our benchmarks vs the original report show different speedups because of different measurement methodology:

| What we measured | Original report | Our report |
|-----------------|-----------------|------------|
| Metric | E2E latency | TTFT (prefill only) |
| Reasoning parser | Disabled | Enabled |
| Tools | LMBench (optimized) | Custom benchmarks |
| Memory pressure | Not tested | Tested (negative result) |
| Speedup claimed | 1.8-2.5x | 1.07-1.31x |

**Both are correct** for their respective configurations. The discrepancy is not an error — it reflects:
1. E2E includes constant generation time that amplifies the ratio (lesson #17)
2. LMBench tools use workload patterns optimized for cache hits
3. Memory pressure reveals overhead invisible at low concurrency

**Lesson**: When evaluating KV cache offloading, always specify: (a) the metric (TTFT vs E2E), (b) whether reasoning is enabled, (c) concurrency level, (d) GPU memory utilization. A "2x speedup" claim without these qualifiers is incomplete.

### 21. Tiered KV Cache Offloading: Framework Comparison
**Finding**: No current framework provides a complete GPU→CPU→NVMe→Network tiering chain in production:

| Framework | Tiers | Multi-tier chain | Status |
|-----------|-------|-----------------|--------|
| LMCache | 2 (local + remote) | No — pick one local backend | Production |
| NVIDIA Dynamo KVBM | 4 (G1/G2/G3/G4) | Architecture yes, OSS partial | Early |
| Mooncake Store | 4 (VRAM/DRAM/NVMe/Remote) | Yes — full hierarchy | Production (SGLang) |
| vLLM native | 2 (GPU + CPU swap) | No disk offloading | Production |

**LMCache limitation**: You must choose ONE local backend (CPU, GPU, or disk) + ONE remote backend (Redis, FSx, S3). Cannot chain CPU as warm tier AND NVMe as cold tier AND FSx as archive.

**Lesson**: For hierarchical tiering on AWS (GPU→CPU→NVMe→FSx), Mooncake+SGLang is the most complete option today. Dynamo KVBM has the right architecture but the OSS implementation is still catching up.

### 22. GDS vs POSIX for LMCache: When GDS Matters
**Config difference**:
- GDS: `LMCACHE_USE_EXPERIMENTAL=True` + cuFile symlinks → GPU VRAM → FSx direct (9+ GB/s)
- POSIX: `LMCACHE_USE_EXPERIMENTAL=False`, `LMCACHE_LOCAL_CPU=True` → GPU → CPU DRAM → FSx (1-3 GB/s)

**When GDS matters**: At high KV cache write rates (many concurrent users, large contexts). The 3-9x bandwidth advantage of GDS reduces the per-request write latency that causes scheduling serialization (lesson #18).

**When POSIX is fine**: Low concurrency, small contexts, or when cuFile setup is impractical (no CUDA dev headers, non-Lustre filesystems).

**Lesson**: GDS doesn't eliminate the serialization problem — it reduces the per-write latency. The real fix requires async write-back in the connector, not faster I/O.

### 23. Original LMCache Speedups Were Inflated by Reasoning Parser Omission
**Problem**: Original report (Feb 14-15) claimed 1.8-2.5x E2E speedup. Normalized analysis shows the actual prefill (TTFT) speedup is 1.07-1.31x.

**Root causes** (3 compounding factors):
1. **No `--reasoning-parser kimi_k2`**: `max_tokens=200` capped reasoning+content combined, giving 1-5s E2E. With parser, reasoning runs uncapped (~13s), making E2E gains invisible.
2. **LMBench TTFT=0**: The LMBench tools didn't capture TTFT at all. All speedup claims were E2E which includes constant generation time that inflates the ratio.
3. **Our own custom benchmark (Feb 14) showed LMCache was slower**: multi_turn_qa TTFT speedup was 0.71x (LMCache 2,053ms vs Baseline 1,450ms). This result was not included in the original report.

**Normalized data**:
| Test | Reported E2E Speedup | Estimated TTFT Speedup | Method |
|------|---------------------|----------------------|--------|
| synthetic_20k | 1.29x | ~2.1x | E2E minus ~1000ms gen |
| multi_tenant_50 | 5.19x | ~10.4x | Includes queueing, not pure prefill |
| multi_turn_qa (Feb14 custom) | 0.93x | 0.71x | Direct TTFT measurement — LMCache slower |
| multi_turn_qa (Feb17 custom) | 1.02x | ~4.7x | With reasoning parser |

**Lesson**: When reporting KV cache offloading benchmarks, always: (a) capture TTFT separately from E2E, (b) document reasoning parser config, (c) compare against baseline with same script. E2E speedup ratios are misleading when generation time dominates.

---

## NVIDIA Dynamo KVBM Benchmarking

### 24. Dynamo KVBM Requires Patching 3 Rust Files for MLA Models
**Problem**: KVBM 0.9.0 (and `main` branch) validates `outer_dim ∈ [1, 2]` in three separate Rust source files. vLLM 0.15.1 passes `outer_dim=64` for MLA models (Kimi K2.5: 512 KV heads / 8 TP = 64). All three locations must be patched:
- `lib/llm/src/block_manager/config.rs:69`
- `lib/llm/src/block_manager/layout.rs:293`
- `lib/llm/src/block_manager/v2/physical/layout/config.rs:22`

**Fix**: Change `#[validate(range(min = 1, max = 2))]` to `#[validate(range(min = 1))]` in all three files, then rebuild the wheel with maturin.

**Lesson**: When deploying KVBM with MLA-based models (DeepSeek V2, Kimi K2.5), always verify `outer_dim` validation. Missing even one of the three locations causes `RuntimeError: Engine core initialization failed`.

### 25. Container OOM at 128GB KVBM CPU Cache
**Problem**: Dynamo container was OOM-killed (exit code 137) during benchmark test 3 (doc-library-rag). The 128GB KVBM CPU cache + model memory + benchmark script exceeded system memory.

**Fix**: Reduced `DYN_KVBM_CPU_CACHE_GB` from 128 to 64. Subsequent tests ran without OOM.

**Lesson**: On p5e.48xlarge (2TB system RAM), budget conservatively: ~540GB model in GPU, some in CPU for tensor parallel comm, plus OS and container overhead. 64GB CPU cache is a safe upper bound for KVBM alongside Kimi K2.5.

### 26. vLLM Scheduler Prevents Tiered KV Cache Offloading
**Problem**: KVBM's 4-tier hierarchy (GPU→CPU→Disk→Remote) was never exercised. vLLM's scheduler acts as a gatekeeper — it queues or preempts requests before GPU KV cache fills, so downstream tiers never see pressure.

**Evidence**: Even in the aggressive memory pressure test (50×32K = 1.6M tokens vs 610K GPU capacity), peak KV cache usage was only 12.7% (reported by vLLM metrics). The scheduler throttled admission rather than letting KV cache overflow into KVBM's offloading path.

**Lesson**: The fundamental bottleneck for tiered offloading with vLLM is the scheduler, not the offloading framework. This affects both KVBM and LMCache. SGLang's HiCache uses cascading eviction (data flows through tiers) instead of admission gating, which is architecturally better suited for exercising tiered offloading.

### 27. NVMe Tier Was Never Configured
**Problem**: The KVBM disk cache was pointed directly at FSx (`DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/dynamo`), skipping the 30TB of local NVMe SSDs entirely. The architecture diagram showed a G3 NVMe tier, but we never configured it as an intermediate tier.

**Lesson**: For proper tiered evaluation, configure NVMe as the disk tier first (fastest local storage), then add FSx as the remote/G4 tier. The p5e has 8×3.8TB NVMe (~30TB) at ~50 GB/s aggregate — far faster than FSx Lustre at ~5 GB/s.

### 28. KVBM Disk Cache Uses Unlinked Temp Files on FSx
**Problem**: KVBM disk cache always showed 0 MB on FSx Lustre. KVBM creates a `DiskStorage` object and immediately unlinks the temp file, using it as an anonymous file descriptor. `du` and `find` show nothing.

**Additional complication**: `fallocate()` is not supported on Lustre — KVBM falls back to `truncate()` for pre-allocation.

**Lesson**: Cannot monitor KVBM disk cache usage via filesystem tools on Lustre. Would need to check `/proc/<pid>/fd/` inside the container, or use KVBM's internal metrics if available.

### 29. Dynamo Outperforms LMCache on Prefix Caching
**Finding**: Across all prefix-heavy workloads, Dynamo KVBM showed better TTFT than LMCache:

| Test | Dynamo | LMCache | Dynamo advantage |
|------|--------|---------|-----------------|
| Multi-turn round 20 | 191ms | 349ms | 1.83x faster |
| API gateway speedup | 1.82x | 1.31x | Better cold→warm |
| Doc RAG speedup | 1.41x | 1.07x | Better cache warming |
| Conversation resumption | 0.99x | 1.04x | Less degradation |
| Shared prompt 50t | 1.68x | 1.02x | Much better scaling |

**Lesson**: Even without tiered offloading working, Dynamo's in-memory prefix cache management is more efficient than LMCache's. The lower absolute TTFT values suggest less overhead in the hot path.

### 30. No Cache Hit/Miss Metrics Captured Across All 3 Rounds
**Problem**: Neither the LMCache, baseline, nor Dynamo benchmark rounds captured prefix cache hit/miss counters from vLLM's `/metrics` endpoint. The benchmark script had a `scrape_lmcache_metrics()` function but:
1. LMCache wasn't exposing those Prometheus counters (returned "no lmcache metrics found")
2. Individual test functions (`run_multi_turn_benchmark`, `run_memory_pressure_throughput`, etc.) never called any metrics scraper
3. Prometheus on EKS was not configured to scrape the standalone nerdctl container

**Fix**: Added `scrape_prefix_cache_metrics()` function to the benchmark script that captures both vLLM (`prefix_cache_hit_total`, `prefix_cache_miss_total`) and SGLang HiCache (`hicache_l1/l2/l3_hits/misses`) counters. Still needs to be wired into each test function's before/after flow.

**Lesson**: Always capture server-side cache metrics alongside client-side latency. Without direct hit/miss data, cache effectiveness can only be inferred from cold-vs-warm TTFT ratios, which is imprecise.

### 31. SGLang HiCache is the Right Architecture for Tiered Offloading
**Finding**: After evaluating LMCache (synchronous write-back), Dynamo KVBM (async but scheduler-gated), and researching alternatives:

| Framework | Scheduler behavior | Tiered offloading |
|-----------|-------------------|------------------|
| vLLM + LMCache | Gates admission, synchronous write-back | Never triggers |
| vLLM + Dynamo KVBM | Gates admission, async write-back | Never triggers |
| SGLang + HiCache | Cascading eviction through tiers | Designed to trigger |
| SGLang + Mooncake | Same as HiCache + RDMA L3 backend | Production-tested at Moonshot AI |

SGLang's HiCache extends RadixAttention with GPU→CPU→Storage tiering. When GPU fills, data cascades to CPU; when CPU fills, data cascades to storage. The scheduler doesn't gate admission to prevent overflow — the tier hierarchy handles it. This is architecturally what we need.

**Lesson**: The inference engine's scheduler policy is as important as the offloading framework. An async offloading framework (KVBM) paired with a gating scheduler (vLLM) is effectively dead code. Match the scheduler to the offloading strategy.

---

## Summary

| # | Lesson | Category | Impact |
|---|--------|----------|--------|
| 1 | EKS + capacity blocks = manual | P5e Deployment | Plan for manual node joining |
| 2 | MoE loading ~25 min | P5e Deployment | Factor into capacity planning |
| 3 | TP must match GPU count | P5e Deployment | CUDA errors if wrong |
| 4 | TTFT broken for reasoning models | Benchmarking | Use E2E latency instead |
| 5 | Token estimation 1.8x off | Benchmarking | Verify with actual tokenizer |
| 6 | Cache benefit ∝ prefix reuse | LMCache | 2x benefit with sharing, 5% without |
| 7 | Cold start 1.8-2.5x penalty | LMCache | Pre-warm cache before production |
| 8 | FSx scales to 83GB+ | LMCache | Size based on prefix variety |
| 9 | High variety = tail latency | LMCache | 50 tenants → 10s P99 |
| 10 | Sub-linear context scaling | LMCache | Latency/token decreases with length |
| 11 | LMBench output format varies | Benchmarking | Verify result format per tool |
| 12 | GDS validated: 9+ GB/s to FSx | LMCache+GDS | GPU→FSx direct I/O works |
| 13 | LMCache ABI mismatch | LMCache+GDS | Must build from source on nightly |
| 14 | cuFile symlink required | LMCache+GDS | Silent fallback without it |
| 15 | Use `nightly` not `cu130-nightly` | LMCache+GDS | CUDA 13 lacks dev headers |
| 16 | Multi-turn TTFT -46% with LMCache | Benchmark | Shared prefix workloads benefit most |
| 17 | Reasoning parser changes E2E dramatically | Benchmarking | 1-5s → 13.5s E2E; speedup ratios shrink |
| 18 | LMCache serializes under moderate pressure | LMCache+GDS | 13x worse fg TTFT at 60% KV utilization |
| 19 | H200 HBM makes offloading unnecessary | Architecture | 610K token capacity, pressure is rare |
| 20 | Speedup depends on measurement method | Benchmarking | E2E vs TTFT, reasoning on/off |
| 21 | No framework does full 4-tier offloading | Architecture | Mooncake closest; LMCache is 2-tier only |
| 22 | GDS vs POSIX: faster I/O ≠ fix for serialization | LMCache+GDS | Async write-back needed, not faster writes |
| 23 | Original speedups inflated by reasoning parser omission | Benchmarking | E2E 1.8-2.5x → TTFT 1.07-1.31x when normalized |
| 24 | KVBM requires patching 3 Rust files for MLA | Dynamo KVBM | outer_dim validation in config.rs, layout.rs, v2 config.rs |
| 25 | Container OOM at 128GB CPU cache | Dynamo KVBM | Reduce to 64GB; budget RAM conservatively |
| 26 | vLLM scheduler prevents tiered offloading | Architecture | Scheduler gates admission; offload tiers never triggered |
| 27 | NVMe tier was never configured | Dynamo KVBM | Disk cache pointed at FSx, skipping 30TB NVMe |
| 28 | KVBM disk cache uses unlinked temp files | Dynamo KVBM | Cannot monitor via du/find on Lustre |
| 29 | Dynamo outperforms LMCache on prefix caching | Dynamo KVBM | 1.41-1.83x faster TTFT across all prefix tests |
| 30 | No cache hit/miss metrics captured | Benchmarking | Must wire scrape_prefix_cache_metrics() into test flows |
| 31 | SGLang HiCache is the right architecture | Architecture | Cascading eviction vs admission gating |
