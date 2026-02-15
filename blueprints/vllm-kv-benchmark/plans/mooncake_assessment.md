# Mooncake KV Cache Assessment Plan

**Date**: February 14, 2026
**Objective**: Evaluate the added value of Mooncake vs LMCache for KV cache offloading on AWS

---

## Executive Summary

Mooncake is a KVCache-centric disaggregated architecture developed by Moonshot AI (FAST 2025 Best Paper). Unlike LMCache's single-tier approach, Mooncake implements **hierarchical tiered caching**: GPU VRAM → Host DRAM → Local NVMe → Remote Store.

### Key Differentiators

| Capability | LMCache (Current) | Mooncake |
|------------|-------------------|----------|
| **Tiered caching** | Single backend only | VRAM → DRAM → NVMe → Remote |
| **Transport protocols** | File I/O, GDS | RDMA, TCP, NVMe-oF, CXL |
| **Hotspot handling** | No | Multi-replica support |
| **PD disaggregation** | No | Native support |
| **Integration** | Native vLLM plugin | Via Transfer Engine |

---

## Assessment Phases

### Phase 1: Installation & Baseline (Day 1)

**Objective**: Deploy Mooncake on existing p5e infrastructure and validate functionality.

#### 1.1 Install Mooncake Transfer Engine

```bash
# On p5e.48xlarge node
pip install mooncake-transfer-engine

# Verify installation
python -c "from mooncake.transfer_engine import TransferEngine; print('OK')"
```

#### 1.2 Configure Mooncake Store

```yaml
# mooncake-config.yaml
store:
  tiers:
    - type: vram
      capacity: 640GB  # 8x H100 total VRAM
    - type: dram
      capacity: 2TB    # Host memory
    - type: fsx
      path: /mnt/fsx/mooncake
      capacity: 100GB
  eviction_policy: lru
  transfer_protocol: tcp  # Start with TCP, upgrade to RDMA if available
```

#### 1.3 Baseline Test

Run identical workload to LMCache benchmark for direct comparison:

```bash
# Multi-tenant benchmark (50 tenants)
python multi_tenant_benchmark.py \
  --num-tenants 50 \
  --users-per-tenant 3 \
  --base-url http://localhost:8000 \
  --model moonshotai/Kimi-K2.5
```

**Success Criteria**:
- [ ] Mooncake starts without errors
- [ ] KV cache stored to FSx
- [ ] Basic request/response working

---

### Phase 2: Head-to-Head Comparison (Day 2)

**Objective**: Compare Mooncake vs LMCache across all benchmark workloads.

#### 2.1 Test Matrix

| Benchmark | Config | Mooncake | LMCache | Delta |
|-----------|--------|----------|---------|-------|
| Long Context (51K) | Single user | TBD | 2,479ms | TBD |
| Multi-Tenant (50) | 4K prompt each | TBD | 4,810ms | TBD |
| 20K Chat History | QPS 1.0 | TBD | 1,389ms | TBD |
| Random (No sharing) | 50 users | TBD | 1,150ms | TBD |
| Agentic | 5 agents | TBD | 1,097ms | TBD |

#### 2.2 Metrics to Collect

1. **Latency**
   - E2E latency (p50, p95, p99)
   - TTFT (Time To First Token)
   - Cold vs warm cache penalty

2. **Throughput**
   - Requests per second
   - Tokens per second (input/output)
   - Cache hit rate

3. **Resource Utilization**
   - GPU memory usage per tier
   - Host DRAM usage
   - FSx I/O bandwidth
   - Network utilization

4. **Cache Behavior**
   - Cache size growth over time
   - Eviction rate per tier
   - Tier hit rates (VRAM vs DRAM vs FSx)

#### 2.3 Benchmark Commands

```bash
# Long Context Benchmark
python benchmark_long_context.py --mode all --requests 15 --backend mooncake

# Multi-Tenant Benchmark
python multi_tenant_benchmark.py --num-tenants 50 --users-per-tenant 3 --backend mooncake

# LMBench Workloads
cd LMBench/3-workloads
python synthetic/multi-round-qa.py --num-users 8 --user-history-prompt 20000 --qps 1.0
python random/random-qa.py --num-users 50 --prompt-len 200 --qps 1.0
python agentic/agentic-qa.py --num-agents 5 --user-request-interval 2.0
```

---

### Phase 3: Tiered Caching Validation (Day 3)

**Objective**: Validate Mooncake's unique hierarchical caching benefits.

#### 3.1 Test Tiered Eviction

Scenario: Generate workload that exceeds VRAM capacity to force tiering.

```bash
# Generate 800GB of KV cache (exceeds 640GB VRAM)
python stress_test.py \
  --total-kv-size 800GB \
  --unique-prefixes 100 \
  --measure-tier-distribution
```

**Expected Outcome**:
- Hot prefixes remain in VRAM
- Warm prefixes spill to DRAM
- Cold prefixes evict to FSx
- Access latency varies by tier

#### 3.2 Test Hotspot Replication

Scenario: Create access hotspot and verify multi-replica handling.

```bash
# 80% requests hit same prefix, 20% random
python hotspot_benchmark.py \
  --hot-prefix-ratio 0.8 \
  --concurrent-users 100 \
  --measure-replica-count
```

**Expected Outcome**:
- Mooncake creates replicas of hot prefix
- Reduced contention vs LMCache
- Higher effective throughput

#### 3.3 Compare Tier Performance

| Test | VRAM-only | +DRAM tier | +FSx tier | Mooncake | LMCache |
|------|-----------|------------|-----------|----------|---------|
| 50K context | TBD | TBD | TBD | TBD | 2,479ms |
| Cache miss | TBD | TBD | TBD | TBD | 4,403ms |
| Warm hit | TBD | TBD | TBD | TBD | 2,480ms |

---

### Phase 4: Production Trace Replay (Day 4)

**Objective**: Validate with real production traffic patterns.

#### 4.1 Replay Mooncake Trace

Use the Mooncake production trace (87K token requests):

```bash
python LMBench/3-workloads/trace-replayer/trace-replayer-qa.py \
  --trace-file traces/mooncake_trace.jsonl \
  --preserve-timing \
  --backend mooncake
```

**Trace Characteristics** (from mooncake_trace.jsonl):
- 50+ requests with timestamps at 0, 3000, 5999, 9000, 12000, 15000ms
- Context sizes: 915 - 87,169 tokens
- Shared prefix (hash_id 0) across all requests
- High variance in output lengths (1 - 929 tokens)

#### 4.2 Replay GMI Trace (for comparison)

```bash
python LMBench/3-workloads/trace-replayer/trace-replayer-qa.py \
  --trace-file traces/gmi_trace.jsonl \
  --preserve-timing \
  --backend mooncake
```

Compare results against LMCache GMI trace benchmark (44 requests, 100% success).

---

### Phase 5: Transfer Protocol Comparison (Day 5)

**Objective**: Evaluate RDMA vs TCP transport performance.

#### 5.1 TCP Baseline (Default)

```yaml
transfer_protocol: tcp
```

#### 5.2 RDMA Configuration (Requires EFA)

```yaml
transfer_protocol: rdma
rdma_device: mlx5_0  # or EFA device
```

**P5e EFA Configuration**:
- 3200 Gbps network bandwidth
- EFA-enabled networking
- RDMA-capable NICs

#### 5.3 Expected Results

| Protocol | Throughput | Latency | CPU Overhead |
|----------|------------|---------|--------------|
| TCP | ~10 GB/s | ~100μs | High |
| RDMA | ~50+ GB/s | ~10μs | Low |

---

## Evaluation Criteria

### Primary Metrics

| Metric | LMCache Baseline | Mooncake Target | Weight |
|--------|------------------|-----------------|--------|
| E2E Latency (p50) | 1,150ms | <1,000ms | 30% |
| Cold Start Penalty | 1.8-2.5x | <1.5x | 20% |
| 50-tenant TTFT | 4,810ms | <3,000ms | 25% |
| Cache Hit Rate | 76-80% | >85% | 15% |
| Resource Efficiency | Baseline | TBD | 10% |

### Secondary Metrics

- Tier utilization distribution
- Eviction rate under pressure
- Multi-replica efficiency
- Protocol selection impact

---

## Success Criteria

### Mooncake is RECOMMENDED if:

1. **E2E latency** ≤ LMCache for standard workloads
2. **Cold start penalty** < 1.5x (vs LMCache 1.8-2.5x)
3. **Multi-tenant (50+)** handles without thrashing
4. **Tiered caching** shows measurable benefit under memory pressure
5. **RDMA transport** provides >2x throughput vs TCP

### Mooncake is NOT RECOMMENDED if:

1. Additional complexity doesn't justify performance gains
2. Integration with vLLM is unstable
3. FSx tier performance matches LMCache
4. RDMA not available/practical on AWS

---

## Deliverables

1. **Benchmark Report**: `results/MOONCAKE_BENCHMARK_REPORT.md`
2. **Comparison Matrix**: Side-by-side LMCache vs Mooncake across all workloads
3. **Recommendation**: Deploy/Not Deploy decision with justification
4. **Architecture Guide**: If recommended, deployment guide for production

---

## Resource Requirements

| Resource | Specification | Cost |
|----------|---------------|------|
| p5e.48xlarge | 8x H100, EFA | ~$98/hr |
| FSx Lustre | 1.2 TiB | $0.14/hr |
| Assessment Duration | ~5 days | ~$11,700 |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Mooncake vLLM integration issues | Medium | High | Test on separate cluster first |
| RDMA driver incompatibility | Low | Medium | Fallback to TCP |
| FSx performance variance | Low | Low | Multiple test runs |
| Model loading conflicts | Medium | Medium | Scale down before testing |

---

## Schedule

| Day | Phase | Tasks |
|-----|-------|-------|
| 1 | Installation | Install Mooncake, configure FSx, baseline test |
| 2 | Comparison | Run all LMBench workloads with Mooncake |
| 3 | Tiered Caching | Stress test tier eviction, hotspot replication |
| 4 | Production Traces | Replay mooncake_trace.jsonl and gmi_trace.jsonl |
| 5 | Protocol Testing | RDMA vs TCP comparison, final analysis |

---

## References

- [Mooncake Paper (FAST 2025)](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [LMCache Benchmark Report](results/BENCHMARK_REPORT.md)
- [LMCache Lessons Learned](lessons.md)
- [Research Notes](../../specs/research.md)
