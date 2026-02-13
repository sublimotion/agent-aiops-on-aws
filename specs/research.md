# Model Change Impact Research: Nemotron 8B → Ministral 3B

## Summary

The spec `vllm-kv-cache-benchmark.md` references `mistralai/Ministral-3-3B-Reasoning-2512`, but the blueprint `blueprints/vllm-kv-benchmark/` still contains configurations sized for the larger `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` model.

## Model Comparison

| Attribute | Nemotron 8B UltraLong | Ministral 3B |
|-----------|----------------------|--------------|
| Parameters | ~8B | ~3B |
| Context Length | 1M tokens | ~32K tokens |
| VRAM Required | ~16-20GB | ~6-8GB |
| Model Size on Disk | ~16GB | ~6GB |

## Items to Clean Up in `blueprints/vllm-kv-benchmark/`

### 1. Model ID (CRITICAL)

**File:** `variables.tf:110-111`

```hcl
# Current (wrong)
default = "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct"

# Should be
default = "mistralai/Ministral-3-3B-Reasoning-2512"
```

### 2. Context Length (CRITICAL)

**File:** `variables.tf:125-128`

```hcl
# Current (oversized)
variable "vllm_max_model_len" {
  description = "Maximum model context length (1M for UltraLong model)"
  default     = 131072 # 128K - designed for Nemotron UltraLong
}

# Recommended for Ministral 3B
default = 8192  # or 32768 for max context benchmarks
```

### 3. Memory Resources (OVERSIZED)

**File:** `main.tf:450-451`

```hcl
# Current
cpu    = "4"
memory = "32Gi" # Comment says "L40S needs more host memory"

# Recommended (based on ministral-3b blueprint)
cpu    = "2"
memory = "12Gi"
```

### 4. PVC Storage Size (OVERSIZED)

**File:** `main.tf:538-540`

```hcl
# Current
storage = "200Gi" # Comment says "Large for 8B model"

# Recommended
storage = "50Gi"  # Sufficient for 3B model (~6GB)
```

### 5. Probe Timeouts (CONSERVATIVE)

**File:** `main.tf:473-491`

```hcl
# Current liveness probe
initial_delay_seconds = 300 # Comment: "Larger model needs more time"

# Recommended for 3B model
initial_delay_seconds = 120  # Smaller model loads faster
```

### 6. Shared Memory Size (OVERSIZED)

**File:** `main.tf:504-508`

```hcl
# Current
size_limit = "32Gi"

# Recommended
size_limit = "16Gi"  # Sufficient for 3B model
```

### 7. Missing Mistral-specific vLLM Args

**File:** `main.tf` - `local.vllm_extra_args` needs Mistral tokenizer config

The ministral-3b blueprint uses these args that may be needed:
```hcl
extra_args = [
  "--tokenizer_mode", "mistral",
  "--config_format", "mistral",
  "--load_format", "mistral"
]
```
### 8. Outdated Comments

| Location | Current Comment | Action |
|----------|----------------|--------|
| `main.tf:106` | "g7e.xlarge for L40S" | Update to reflect actual instance |
| `main.tf:451` | "L40S needs more host memory" | Remove/update |
| `main.tf:478` | "Larger model needs more time" | Update |
| `main.tf:539` | "Large for 8B model" | Update |
| `variables.tf:58-59` | "g6e family with L40S GPUs (48GB VRAM)" | Update description |
| `variables.tf:126` | "(1M for UltraLong model)" | Update |

## Instance Type Consideration

The spec mentions `g7e` instances but the terraform uses `g6e`. Both work:

| Instance | GPU | VRAM | Suitable for Ministral 3B |
|----------|-----|------|--------------------------|
| g6e.xlarge | L40S | 48GB | Yes (overkill) |
| g6.xlarge | L4 | 24GB | Yes |
| g5.xlarge | A10G | 24GB | Yes |

For a 3B model, even `g5.xlarge` or `g6.xlarge` would suffice. The L40S (g6e/g7e) is oversized but provides headroom for KV cache offloading experiments.
<feedback>let's use g7e or g6e instances only. If the instances are bigger, we can increase the concurrency in the benchmarks run</feedback>

## Model Variant Clarification

The spec uses:
```
mistralai/Ministral-3-3B-Reasoning-2512
```

The existing ministral-3b blueprint uses:
```
mistralai/Ministral-3-3B-Instruct-2512
```

**Verify the exact model name** - "Reasoning" vs "Instruct" may be different model variants. Check HuggingFace for the correct model ID.
<feedback>Let's try the reasoning since we have the RAG and agent use cases</feedback>

## Recommended Changes Summary

| Priority | File | Change |
|----------|------|--------|
| P0 | `variables.tf` | Update `vllm_model_id` default |
| P0 | `variables.tf` | Reduce `vllm_max_model_len` to 8192-32768 |
| P1 | `main.tf` | Add Mistral tokenizer args |
| P1 | `main.tf` | Reduce memory request to 12Gi |
| P1 | `main.tf` | Reduce PVC size to 50Gi |
| P2 | `main.tf` | Reduce probe delays |
| P2 | `main.tf` | Reduce shm size to 16Gi |
| P3 | Various | Update outdated comments |

## Terraform Module Impact

No changes needed to shared modules (`modules/`). All changes are localized to the blueprint.

## Next Steps

1. Update `variables.tf` with correct model ID and context length
2. Add Mistral-specific vLLM args to deployment
3. Right-size resource requests
4. Run `terraform plan` to verify changes
5. Consider whether FSx Lustre benchmarking is still relevant for this smaller model

---

# Part 2: KV Cache Offloading Architecture Research

## Problem Statement

LMCache + FSx performed **84x worse** than baseline for multi-tenant workloads (50 tenants × 4K token prompts):

| Metric | Baseline | LMCache+FSx |
|--------|----------|-------------|
| TTFT (avg) | 482.9ms | 40,613ms |
| Achieved QPS | 2.94 | 1.60 |

## Root Cause Analysis

### Is PCIe the Bottleneck?

**No.** The bottleneck is **network I/O latency**, not PCIe bandwidth.

| Component | Bandwidth | Latency |
|-----------|-----------|---------|
| PCIe 4.0 x16 | ~32 GB/s | ~microseconds |
| FSx Lustre SCRATCH_2 (1.2 TiB) | ~240 MB/s | ~milliseconds |

PCIe is 100x+ faster. The LMCache config `LMCACHE_LOCAL_DISK=file:///fsx/...` treats network storage as "local disk" - but every cache operation incurs network round-trip latency.

### Why Thrashing Occurred

With 50 unique system prompts competing for 100GB FSx cache limit:

1. **Many small network round-trips** - Each 256-token chunk = 1 FSx I/O
2. **Cache eviction pressure** - 50 prefixes × 4K tokens each exceeds working set
3. **Write amplification** - Every new prefix writes to FSx before serving
4. **No local tier** - Every cache miss hits network immediately

## FSx Lustre GDS Support

AWS FSx Lustre supports **GPUDirect Storage (GDS)** which bypasses CPU:

```
Traditional:     GPU → PCIe → CPU RAM → Network → FSx
With GDS:        GPU → PCIe → Network → FSx (bypasses CPU)
```

Reference: https://aws.amazon.com/blogs/aws/amazon-fsx-for-lustre-unlocks-full-network-bandwidth-and-gpu-performance/

**However**, GDS reduces CPU overhead but doesn't eliminate network latency. Tiered caching is still preferred.

## Tiered Storage Architecture

Optimal KV cache offloading requires multiple tiers:

```
┌─────────────────────────────────────────────────────────┐
│  Tier 0: GPU VRAM     │  ~ns latency   │  ~2 TB/s      │
├───────────────────────┼────────────────┼───────────────┤
│  Tier 1: Host DRAM    │  ~100ns        │  ~100 GB/s    │
├───────────────────────┼────────────────┼───────────────┤
│  Tier 2: Local NVMe   │  ~10-100μs     │  ~7 GB/s      │
├───────────────────────┼────────────────┼───────────────┤
│  Tier 3: FSx (GDS)    │  ~1-10ms       │  ~240 MB/s    │
└───────────────────────┴────────────────┴───────────────┘
```

Hot prefixes stay in VRAM/DRAM, cold prefixes spill to NVMe, FSx provides cross-node sharing.

## LMCache vs Mooncake Comparison

| Capability | LMCache | Mooncake |
|------------|---------|----------|
| **Tiered caching** | Single backend only | VRAM → DRAM → NVMe → Remote |
| **GDS support** | No | Yes (optional build flag) |
| **Transport protocols** | File I/O | RDMA, TCP, NVMe-oF, CXL |
| **Hotspot handling** | No | Multi-replica support |
| **PD disaggregation** | No | Native support |
| **vLLM integration** | Native plugin | Via Transfer Engine |

### Mooncake Architecture

Mooncake implements hierarchical KV caching with the Mooncake Store:

```
┌─────────────────────────────────────────────────────┐
│                   Mooncake Store                     │
├─────────────────────────────────────────────────────┤
│  GPU VRAM → Host DRAM → Local NVMe → Remote Store   │
└─────────────────────────────────────────────────────┘
         Transfer Engine (RDMA, TCP, NVMe-of, CXL)
```

Key advantages:
- **Tiered eviction** - Hot data stays local, cold data spills to remote
- **Multi-replica** - Handles access hotspots by replicating popular prefixes
- **P2P transfer** - Direct GPU-to-GPU transfers via RDMA

Reference: https://github.com/kvcache-ai/Mooncake

## Recommended Architecture for AWS

For multi-tenant workloads with many unique system prompts:

```
┌────────────────────────────────────────────────────────────┐
│                    vLLM + Mooncake                          │
├────────────────────────────────────────────────────────────┤
│  Tier 0: GPU VRAM (native prefix cache)                    │
│  Tier 1: Host DRAM (overflow buffer)                       │
│  Tier 2: Local NVMe + GDS (g6e instance storage)           │
│  Tier 3: FSx Lustre + GDS (cross-node shared cache)        │
└────────────────────────────────────────────────────────────┘
```

### Instance Selection for Tiered Storage

| Instance | GPU | Local NVMe | FSx Access |
|----------|-----|------------|------------|
| g6e.xlarge | L40S 48GB | None | Network |
| g6e.4xlarge | L40S 48GB | 950GB NVMe | Network |
| p5.48xlarge | 8x H100 | 8x 3.84TB NVMe | Network |

For tiered caching, need instances with local NVMe (g6e.4xlarge+).

## Conclusions

1. **LMCache single-backend design** is unsuitable for high-variety prefix workloads
2. **Network latency** (not PCIe) is the bottleneck when using FSx as primary cache
3. **Tiered architecture** (Mooncake) is required for production multi-tenant deployments
4. **FSx GDS** helps but doesn't solve the fundamental latency gap
5. **For single-node**: Native vLLM prefix caching is optimal; skip LMCache
6. **For multi-node**: Consider Mooncake with tiered storage or wait for LMCache tiering support

## Future Work

- [ ] Benchmark Mooncake with tiered storage on g6e.4xlarge (has local NVMe)
- [ ] Test FSx GDS integration with Mooncake Transfer Engine
- [ ] Evaluate kvaware routing with multiple vLLM replicas
- [ ] Compare RDMA vs TCP transport for cross-node KV transfer
