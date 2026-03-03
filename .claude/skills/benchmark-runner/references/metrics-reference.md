<!-- Reference: loaded on-demand by SKILL.md. Metrics collection details and PromQL queries. -->

# Metrics Reference

## Client-Side Metrics

From `vllm bench serve` JSON output:

| Metric | JSON Key | Unit | Description |
|--------|----------|------|-------------|
| TTFT | `ttft_ms_p50`, `ttft_ms_p90`, `ttft_ms_p99` | ms | Time from request submission to first token |
| ITL | `itl_ms_p50`, `itl_ms_p90`, `itl_ms_p99` | ms | Average inter-token latency during decode |
| TPOT | `tpot_ms_p50`, `tpot_ms_p90`, `tpot_ms_p99` | ms | Total decode time / output tokens |
| E2EL | `e2el_ms_p50`, `e2el_ms_p90`, `e2el_ms_p99` | ms | Full request duration |
| Output tok/s | `output_throughput` | tokens/s | Output tokens / wallclock time |
| Total tok/s | `total_throughput` | tokens/s | (Input + output tokens) / wallclock time |
| Request throughput | `request_throughput` | req/s | Completed requests / wallclock time |
| Error rate | computed | % | failed / total |

### Kimi K2.5 Additional Metrics (custom Python runner)

| Metric | Description |
|--------|-------------|
| `ttft_content_ms` | Time to first **content** token (after reasoning tokens) |
| `reasoning_tokens` | Count of reasoning/thinking tokens before content |
| `has_reasoning` | Whether response included reasoning section |

## Server-Side Metrics (Prometheus)

### vLLM Metrics

Scrape endpoint: `curl -sf "$VLLM_URL/metrics"`

#### KV Cache

```promql
# Cache utilization (0-1)
vllm:kv_cache_usage_perc

# GPU vs CPU cache split
vllm:gpu_cache_usage_perc
vllm:cpu_cache_usage_perc

# Prefix cache effectiveness
rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])

# Absolute hit/miss counts (for delta calculation)
vllm:prefix_cache_hits
vllm:prefix_cache_queries
```

#### Request Queue

```promql
# Currently processing
vllm:num_requests_running

# Waiting in queue
vllm:num_requests_waiting

# Preemptions (should be 0 — indicates memory pressure)
rate(vllm:num_preemptions_total[5m])
```

#### Latency Histograms

```promql
# TTFT p99
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# ITL p99
histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]))
```

### NVIDIA Dynamo KVBM Metrics

Enable with: `DYN_KVBM_METRICS=true`

| Metric | Description |
|--------|-------------|
| `kvbm_matched_tokens` | Token matching activity |
| `kvbm_onboard_blocks_host_to_device` | Blocks CPU → GPU |
| `kvbm_onboard_blocks_disk_to_device` | Blocks disk → GPU |
| `kvbm_host_cache_hit_rate` | CPU tier hit rate (0.0–1.0) |
| `kvbm_disk_cache_hit_rate` | Disk tier hit rate (0.0–1.0) |

### FSx Cache Metrics (disk tier)

```bash
# Cache directory size and file count
du -sh /mnt/fsx/kv-cache/
find /mnt/fsx/kv-cache/ -type f | wc -l

# Via kubectl (remote node)
kubectl exec -n <ns> <pod> -- du -sh /mnt/fsx/kv-cache/
```

## Metric Capture Workflow

### Before Each Benchmark

```bash
capture_metrics "$VLLM_URL" "${RESULT_DIR}/pre_${LABEL}_metrics.txt"
capture_kv_metrics "pre_${LABEL}" "$VLLM_URL"
```

### After Each Benchmark

```bash
capture_metrics "$VLLM_URL" "${RESULT_DIR}/post_${LABEL}_metrics.txt"
capture_kv_metrics "post_${LABEL}" "$VLLM_URL"
```

### Computing Deltas

```bash
# Extract prefix cache hits before/after
PRE_HITS=$(grep 'vllm:prefix_cache_hits' "${RESULT_DIR}/pre_${LABEL}_metrics.txt" | awk '{print $2}')
POST_HITS=$(grep 'vllm:prefix_cache_hits' "${RESULT_DIR}/post_${LABEL}_metrics.txt" | awk '{print $2}')
DELTA_HITS=$((POST_HITS - PRE_HITS))

PRE_QUERIES=$(grep 'vllm:prefix_cache_queries' "${RESULT_DIR}/pre_${LABEL}_metrics.txt" | awk '{print $2}')
POST_QUERIES=$(grep 'vllm:prefix_cache_queries' "${RESULT_DIR}/post_${LABEL}_metrics.txt" | awk '{print $2}')
DELTA_QUERIES=$((POST_QUERIES - PRE_QUERIES))

# Hit rate for this benchmark run
if [ "$DELTA_QUERIES" -gt 0 ]; then
  HIT_RATE=$(echo "scale=4; $DELTA_HITS / $DELTA_QUERIES" | bc)
  log "Prefix cache hit rate: $HIT_RATE ($DELTA_HITS / $DELTA_QUERIES)"
fi
```

## Reporting Standards

When reporting benchmark results (in markdown or to the benchmark-analyst agent):

1. **Always report absolute numbers with units** — "TTFT p50 = 42ms", not "TTFT improved"
2. **When comparing, report both absolute difference and ratio** — "1.31x faster (45ms vs 59ms)"
3. **Flag high variance** — If p99/p50 > 3x, mark result as "high variance, potentially unreliable"
4. **Note hardware config at top** — Instance type, GPU count, TP, model, max-model-len
5. **Distinguish TTFT vs throughput improvements** — TTFT improvements are prefill-bound; throughput improvements are decode-bound. Different optimizations affect each differently.
6. **Report output tok/s as primary throughput metric** — This matches how customers report and compare.
