# LMCache Use Cases and Benchmark Scenarios

## When LMCache Provides Value

LMCache excels at **prefix reuse**, not memory expansion. It caches computed KV tensors for retrieval instead of recomputation.

| Use Case | Benefit | Not Useful For |
|----------|---------|----------------|
| Shared system prompts | Compute once, reuse across users | Unique contexts per request |
| Multi-node deployments | Share cache via FSx across instances | Single-node deployments |
| RAG with common chunks | Cache frequently retrieved documents | Random document access |
| Cold start recovery | Persist cache across restarts | Ephemeral workloads |

## Benchmark Scenarios

### Scenario 1: Shared System Prompt

**Description**: Many users share the same long system prompt. Each user sends short messages with short outputs.

**Configuration**:
```yaml
num_users: 100
shared_system_prompt: 4000  # tokens
user_message: 100           # tokens
answer_len: 200             # tokens
qps: 5-10
```

**Why LMCache Helps**: Without LMCache, each of 100 requests recomputes the 4K token system prompt. With LMCache, it's computed once and retrieved 99 times.

**Expected Improvement**: 3-5x TTFT reduction after cache warm-up.

**Command**:
```bash
python LMBench/3-workloads/synthetic/multi-round-qa.py \
  --num-users 100 \
  --shared-system-prompt 4000 \
  --user-history-prompt 100 \
  --answer-len 200 \
  --num-rounds 5 \
  --qps 5.0
```

---

### Scenario 2: Multi-Node with Shared FSx Cache

**Description**: Multiple vLLM instances behind a load balancer share KV cache via FSx Lustre.

**Architecture**:
```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Load Balancer│
                    └──────┬──────┘
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐┌─────▼─────┐┌─────▼─────┐
        │  vLLM-1   ││  vLLM-2   ││  vLLM-3   │
        │ +LMCache  ││ +LMCache  ││ +LMCache  │
        └─────┬─────┘└─────┬─────┘└─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                    ┌──────▼──────┐
                    │ FSx Lustre  │
                    │ (Shared KV) │
                    └─────────────┘
```

**Why LMCache Helps**: Instance 1 computes prefix and writes to FSx. Instances 2 and 3 retrieve from FSx instead of recomputing.

**Expected Improvement**: Near-linear throughput scaling with instance count.

**Configuration**:
```yaml
# LMCache environment for each instance
LMCACHE_LOCAL_DISK: "file:///fsx/shared-kv-cache/"
LMCACHE_MAX_LOCAL_DISK_SIZE: 100.0  # GB per instance
```

---

### Scenario 3: RAG with Common Document Chunks

**Description**: Retrieval-augmented generation where certain documents are frequently retrieved across queries.

**Configuration**:
```yaml
num_documents: 20
document_size: 2000         # tokens per document
queries_per_document: 10    # average reuse
query_size: 100             # tokens
answer_len: 200             # tokens
qps: 3.0
```

**Why LMCache Helps**: Popular documents (e.g., FAQ, policy docs) are cached after first retrieval. Subsequent queries with same documents skip prefill.

**Expected Improvement**:
- First query: Full prefill (baseline)
- Subsequent queries: Cache hit, 2-4x TTFT reduction

**Workload Pattern**:
```
Query 1: [Doc A, Doc B] + question → compute Doc A, Doc B
Query 2: [Doc A, Doc C] + question → retrieve Doc A, compute Doc C
Query 3: [Doc B, Doc C] + question → retrieve Doc B, Doc C (both cached)
```

---

### Scenario 4: Cold Start Recovery

**Description**: Measure first request latency after vLLM restart with pre-populated FSx cache.

**Test Protocol**:
```bash
# 1. Warm up cache
python benchmark.py --warmup-requests 100 --shared-system-prompt 4000

# 2. Verify cache populated
kubectl exec deployment/vllm -- du -sh /fsx/kv-cache/

# 3. Restart vLLM
kubectl rollout restart deployment/vllm

# 4. Measure first request (should hit FSx cache)
time curl -X POST http://localhost:30080/v1/chat/completions ...
```

**Why LMCache Helps**: Baseline must recompute all KV from scratch. LMCache retrieves from persistent FSx storage.

**Expected Improvement**: 10x+ reduction in first request latency after restart.

---

### Scenario 5: Multi-Turn Conversation with User Affinity

**Description**: Users have long conversations where each turn builds on previous context.

**Configuration**:
```yaml
num_users: 50
turns_per_user: 20
system_prompt: 500          # tokens
history_growth: 200         # tokens per turn
answer_len: 200             # tokens
time_between_turns: 10      # seconds
```

**Why LMCache Helps**: Each turn reuses KV from previous turns. Without caching, turn 20 must recompute all 19 previous turns.

**Expected Improvement**:
- Turn 1: Full compute
- Turn 10: ~90% cache hit (only new user message computed)
- Turn 20: ~95% cache hit

---

## Benchmark Parameters Summary

| Scenario | Context Size | Prefix Sharing | Instances | QPS | Key Metric |
|----------|--------------|----------------|-----------|-----|------------|
| Shared System Prompt | 4-8K | 80-90% | 1 | 5-10 | TTFT reduction |
| Multi-Node | 4-8K | 80-90% | 2-4 | 10-20 | Throughput scaling |
| RAG Common Chunks | 4-8K | 50-70% | 1 | 3-5 | Cache hit rate |
| Cold Start | 4-8K | 100% | 1 | N/A | First request latency |
| Multi-Turn | 2-8K | 70-90% | 1 | 1-2 | Per-turn TTFT |

## Key Metrics to Track

### LMCache Metrics
```promql
# Cache hit rate (target: >80% for shared prefix workloads)
lmcache_cache_hit_rate

# Cache size on disk
lmcache_disk_usage_bytes

# Cache retrieval latency
lmcache_retrieval_latency_seconds
```

### vLLM Metrics
```promql
# Time to first token (should decrease with cache hits)
vllm:time_to_first_token_seconds

# Prefix cache effectiveness
vllm:prefix_cache_hit_rate

# Preemptions (should decrease with offloading)
vllm:num_preemptions_total
```

## When NOT to Use LMCache

| Scenario | Why LMCache Doesn't Help |
|----------|--------------------------|
| Unique contexts per request | No prefix reuse, 0% cache hits |
| Single short request | No opportunity for reuse |
| GPU memory exhaustion | LMCache caches for reuse, doesn't expand capacity |
| Latency-critical single requests | Disk I/O adds overhead |

## Recommended Test Matrix

```
LMCache Value = Prefix Reuse × Instance Count × Request Volume
```

| Test | Prefix Reuse | Instances | Requests | Expected Value |
|------|--------------|-----------|----------|----------------|
| Baseline comparison | 80% | 1 | 1000 | Medium |
| Multi-node scaling | 80% | 4 | 4000 | High |
| Low reuse workload | 20% | 1 | 1000 | Low |
| High volume shared | 90% | 2 | 10000 | Very High |
