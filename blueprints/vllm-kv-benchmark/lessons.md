# Lessons Learned: vLLM KV Cache Benchmark

## Infrastructure Deployment

### 1. NAT Gateway EIP Quota
**Problem**: AWS account had EIP quota of 5, which was exhausted.

**Solution**: Used VPC endpoints for ECR, S3, and FSx API calls instead of NAT Gateway. This also reduces data transfer costs.

```hcl
# VPC endpoints instead of NAT Gateway
resource "aws_vpc_endpoint" "ecr_api" {
  service_name = "com.amazonaws.${var.region}.ecr.api"
  vpc_endpoint_type = "Interface"
}
```

### 2. FSx CSI Driver Requirements
**Problem**: FSx CSI driver couldn't make API calls from private subnets.

**Solution**: Required FSx VPC endpoint for the FSx API.

```hcl
resource "aws_vpc_endpoint" "fsx" {
  service_name = "com.amazonaws.${var.region}.fsx"
  vpc_endpoint_type = "Interface"
}
```

### 3. Prometheus Image Deprecation
**Problem**: `jimmidyson/configmap-reload` image is deprecated and unavailable.

**Solution**: Use `prometheus-config-reloader` from prometheus-operator instead.

```yaml
configmapReload:
  prometheus:
    image:
      repository: quay.io/prometheus-operator/prometheus-config-reloader
      tag: v0.75.0
```

### 4. Prometheus Scrape Configuration
**Problem**: Setting `scrape_interval: 1s` without matching `scrape_timeout` causes config validation error.

**Solution**: Always set `scrape_timeout` equal to or less than `scrape_interval`.

```yaml
scrape_configs:
  - job_name: vllm
    scrape_interval: 1s
    scrape_timeout: 1s  # Must be <= scrape_interval
```

### 5. GPU Node Scheduling
**Problem**: New pods couldn't schedule when GPU was occupied by existing pod during rolling updates.

**Solution**: Scale to 0 first, then scale to 1 for GPU workloads with single replica.

```bash
kubectl scale deployment/vllm --replicas=0
sleep 5
kubectl scale deployment/vllm --replicas=1
```

## vLLM Configuration

### 6. CPU Offload Performance Penalty
**Problem**: Assumed `--cpu-offload-gb` would help with long contexts.

**Reality**: CPU offload causes **50x slowdown** (114 → 2.2 tok/req/s). It's designed for fitting larger models, not for KV cache expansion.

**Lesson**: Only use CPU offload when model doesn't fit in GPU memory, never for performance optimization.

### 7. Swap Space is Transparent
**Problem**: Expected `--swap-space` to have overhead.

**Reality**: Swap space has zero overhead when not triggered. It only activates during preemptions.

**Lesson**: Safe to enable swap space as a safety net without performance penalty.

### 8. Prefix Caching Effectiveness
**Finding**: Native vLLM prefix caching achieves 76-80% hit rate for multi-turn conversations without any additional configuration.

**Lesson**: For single-node deployments with shared prefixes, `--enable-prefix-caching` is sufficient. No need for LMCache.

## Long Context Limits

### 9. GPU Memory is the Real Bottleneck
**Problem**: Expected LMCache + FSx to enable 30K+ context handling.

**Reality**:
- ≤16K context: No preemptions, acceptable latency
- 24K context: 39 preemptions, 16s TTFT
- 30K context: Complete stall, requests never complete

**Lesson**: L40S 48GB cannot handle 15+ concurrent 30K contexts regardless of caching strategy. Need A100/H100 80GB for long-context workloads.

### 10. LMCache Caches for Reuse, Not Capacity
**Problem**: Assumed LMCache disk offloading would increase effective GPU memory.

**Reality**: LMCache reduces preemptions by 31% but increases TTFT by 56%. It caches computed KV for future reuse, but doesn't expand how many concurrent contexts fit in GPU memory.

**Lesson**: LMCache is best for:
- Multi-node deployments (shared KV cache)
- High prefix reuse scenarios
- Cost optimization (avoid recomputing expensive prefills)

NOT for increasing single-GPU memory capacity.

## Benchmarking

### 11. Use LMBench for Standardized Testing
**Finding**: LMBench provides well-designed workload generators (multi-round-qa.py) that simulate realistic usage patterns.

**Lesson**: Don't reinvent benchmark tooling. Use established frameworks for reproducible results.

### 12. Test at Breaking Points
**Finding**: Testing only at comfortable workloads (1-4K context) missed the 24K breaking point.

**Lesson**: Always stress test to find limits:
- Increase context length until failure
- Increase concurrency until preemptions
- Measure at multiple QPS levels

### 13. Monitor Preemptions, Not Just Latency
**Finding**: TTFT can be "acceptable" while preemptions are destroying throughput.

**Lesson**: Always monitor `vllm:num_preemptions_total`. Preemptions indicate memory pressure even before latency degrades visibly.

## Cost Optimization

### 14. Right-Size for Workload
| Context | GPU | Cost/hr | Recommendation |
|---------|-----|---------|----------------|
| ≤16K | g6e.xlarge (L40S 48GB) | $1.19 | ✅ Sufficient |
| 24K+ | p4d.24xlarge (A100 80GB) | $32.77 | Required |

**Lesson**: Don't over-provision for short contexts, but don't under-provision for long contexts. Test actual workload patterns.

### 15. FSx Lustre for Shared Caching
**Finding**: FSx Lustre at $0.14/hr provides 1.2 TiB of high-performance shared storage.

**Lesson**: FSx is cost-effective for LMCache disk backend when:
- Multiple vLLM instances share cache
- Cache persistence across restarts matters
- High I/O throughput needed

Not necessary for single-node with native prefix caching.

## Summary

| Lesson | Category | Impact |
|--------|----------|--------|
| Use VPC endpoints | Infrastructure | Avoids EIP quota issues |
| CPU offload = 50x slower | Configuration | Critical performance trap |
| Swap space is free | Configuration | Safe to enable |
| 76-80% prefix cache hit | Performance | Native caching is effective |
| L40S limit ~16K context | Capacity | Hard GPU memory constraint |
| LMCache ≠ capacity expansion | Architecture | Common misconception |
| Monitor preemptions | Operations | Early warning metric |
