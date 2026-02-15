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

---

## P5e Deployment (8x H100)

### 16. EKS Doesn't Support Capacity Block Market Type
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

### 17. Large MoE Model Loading is Slow
**Problem**: Kimi K2.5 (64 safetensor shards) takes ~25 minutes to load across 8x H100s.

**Lesson**:
- Use persistent pods or keep-alive strategies
- Pre-pull model weights to FSx for faster loading
- Consider model loading time in capacity planning

### 18. Tensor Parallelism Must Match GPU Count
**Problem**: Setting `--tensor-parallel-size` incorrectly causes CUDA errors.

**Solution**: For p5e.48xlarge with 8x H100, always use `--tensor-parallel-size 8`.

**Lesson**: TP must equal the number of GPUs. Verify with `nvidia-smi` before deployment.

---

## LMCache + FSx Benchmarking

### 19. TTFT Measurement is Model-Dependent
**Problem**: TTFT showed 0ms for Kimi K2.5 requests in most benchmarks.

**Cause**: Kimi K2.5 uses `delta.reasoning` tokens before `delta.content`, so standard TTFT detection (waiting for first content token) fails.

**Lesson**: Reasoning models need custom TTFT measurement that accounts for thinking tokens. Use E2E latency as primary metric.

### 20. Token Estimation ≠ Actual Tokenization
**Problem**: "28K context" config generated ~51K actual tokens.

**Observation**: vLLM successfully processed requests exceeding stated 32K max_model_len.

**Lesson**: Word-based token estimation is unreliable (~1.8x off). Always verify with actual tokenizer or measure actual prompt_tokens from API response.

### 21. Cache Benefits Require Prefix Sharing
**Finding**:
| Scenario | Cache Benefit |
|----------|---------------|
| Shared system prompts (50 tenants) | 1.98x |
| Multi-turn chat (20 rounds) | 2.4x |
| Random prompts (no sharing) | 1.05x |

**Lesson**: LMCache value is proportional to prefix reuse. Random/unique workloads see minimal benefit (~5%).

### 22. Cold Start Penalty is Significant
**Finding**: First request is 1.8-2.5x slower than subsequent requests.

| Scenario | Cold | Warm | Penalty |
|----------|------|------|---------|
| 51K context | 4,403ms | 2,480ms | 1.8x |
| Multi-tenant | 9,545ms | 4,810ms | 2.0x |
| 20K chat | 3,176ms | 1,300ms | 2.4x |

**Lesson**: Pre-warm cache before production traffic. Consider cache warming during deployment.

### 23. FSx Scales Well for KV Cache
**Observed**: Cache grew from 189MB → 37GB (2,160 files) without performance degradation.

**Lesson**: FSx Lustre handles large KV cache workloads. Size based on prefix variety, not just traffic volume. Plan for 10-50GB per model.

### 24. High Tenant Variety Causes Tail Latency
**Finding**: 50 tenants → P90/P99 latency 10s+ (vs P50 of 2.9s)

**Lesson**: Many unique prefixes cause cache pressure and evictions. Monitor eviction rates in high-variety deployments. Consider increasing cache size or accepting higher tail latency.

### 25. Sub-linear Context Scaling Works
**Finding**:
| Context | E2E Latency | Latency per 1K tokens |
|---------|-------------|----------------------|
| 24K tokens | 3,095ms | 129ms |
| 36K tokens | 4,273ms | 119ms |
| 48K tokens | 5,521ms | 115ms |

**Lesson**: KV cache offloading enables efficient long context handling. Latency per token decreases with larger contexts due to batched retrieval.

### 26. LMBench Workloads Have Different Output Formats
**Problem**: StrictSynthetic showed 0ms TTFT in CSV because it uses completions API differently and measures timing differently.

**Lesson**: Always verify result format. Calculate E2E from `finish_time - launch_time` as fallback. Don't assume all benchmarks output data the same way.

### 27. GDS Support Exists but Requires Specific Setup
**Finding**: LMCache supports GDS (GPU Direct Storage) for direct GPU-to-FSx transfers.

**Configuration**:
```yaml
env:
  - name: LMCACHE_USE_GDS
    value: "true"
```

**Lesson**: GDS can provide ~150 GB/s bandwidth on P5 instances with EFA, but requires proper FSx and driver configuration.

---

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
| EKS + capacity blocks = manual | P5e Deployment | Plan for manual node joining |
| MoE loading ~25 min | P5e Deployment | Factor into capacity planning |
| TTFT broken for reasoning models | Benchmarking | Use E2E latency instead |
| Token estimation 1.8x off | Benchmarking | Verify with actual tokenizer |
| Cache benefit ∝ prefix reuse | LMCache | 2x benefit with sharing, 5% without |
| Cold start 1.8-2.5x penalty | LMCache | Pre-warm cache before production |
| FSx scales to 37GB+ | LMCache | Size based on prefix variety |
| High variety = tail latency | LMCache | 50 tenants → 10s P99 |
