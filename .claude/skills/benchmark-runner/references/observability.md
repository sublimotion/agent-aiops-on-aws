# Observability on benchmark GPU nodes

Mandatory stack for every benchmark run. Deployed by `bootstrap-observability.sh`, validated by `observability-smoke-test.sh`, persisted by `sync-prometheus-to-s3.sh`.

## Stack components

| Service | Port | Purpose | Retention |
|---|---|---|---|
| Prometheus | 9090 | Scrapes engine + GPU + node; serves PromQL API | 7 days local |
| DCGM exporter | 9400 | Per-GPU util, HBM BW, tensor core, power, temp, XID | (live, scraped by Prom) |
| node-exporter | 9100 | CPU, RAM, disk, network | (live, scraped by Prom) |
| S3 sync timer | — | Every 10 min: TSDB snapshot → `s3://<bucket>/prometheus/<blueprint>/<session>/` | indefinite |

## Why this matters — Kimi K2.6-spec lesson

2026-05-13: a benchmark session on p6-b300 measured 95 data points across 18 configurations (Phases 0, 1b, 4, 5a/b/c/d). The custom bench driver only recorded aggregate duration and computed `per_request_tok_per_s = total_tokens / duration`. **TTFT was never captured.**

When the session wrapped and the spot instance was terminated:
- SGLang `/metrics` was live during the run but never scraped
- TSDB was never snapshotted (no Prometheus was even installed)
- 95 JSON files ended up with `ttft_ms: null` across the board
- Data was **permanently unrecoverable**

This skill's mandate: every future run has Prometheus + DCGM running from bootstrap, smoke-tested before first request, snapshotted every 10 min to S3. `bench-standard.py` enforces this by *reading from Prometheus* — so if you're emitting the v1 envelope at all, you've already captured the data.

## Bootstrap flow (infra-deployer Stage 4b)

```
┌───────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ Stage 4: GPU node │──▶│ Stage 4a: GPU    │──▶│ Stage 4b:          │
│   provisioned     │   │   health check   │   │   observability    │
└───────────────────┘   └──────────────────┘   │   bootstrap + smoke│
                                                └─────────┬──────────┘
                                                          │ passes
                                                          ▼
                                                ┌───────────────────┐
                                                │ Stage 5: serving  │
                                                │   stack deploy    │
                                                └───────────────────┘
```

Must pass before proceeding to Stage 5.

## Smoke test checks

Script: `scripts/observability-smoke-test.sh`

1. Prometheus `/health` returns 200
2. DCGM exporter reports N GPUs (matches `nvidia-smi --query-gpu=count`)
3. node-exporter serves `node_cpu_seconds_total`
4. Prometheus targets all up (`query=up`, no zeros)
5. Engine `/metrics` reachable (vLLM :8000 or SGLang :30000, whichever is running)
6. Engine histograms present: `*:time_to_first_token_seconds_bucket`, `*:time_per_output_token_seconds_bucket`, `*:e2e_request_latency_seconds_bucket`

Any failure exits non-zero — infra-deployer must block on this.

## Key PromQL queries

### Histogram percentiles over a run window

Given a run from t0 to t1 (scaled to a `[window]` like `90s`):

```promql
# TTFT p99
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[90s])) by (le))

# TPOT (inter-token latency) p50
histogram_quantile(0.50, sum(rate(vllm:time_per_output_token_seconds_bucket[90s])) by (le))

# E2E latency mean
sum(rate(vllm:e2e_request_latency_seconds_sum[90s])) / sum(rate(vllm:e2e_request_latency_seconds_count[90s]))
```

### Engine counters (reconciliation vs client)

```promql
# Successful requests over the run window
increase(vllm:request_success_total[90s])
increase(sglang:num_requests_success_total[90s])

# Preemptions (should be 0 for most configs)
increase(vllm:num_preemptions_total[90s])
```

### GPU / DCGM during the run

```promql
# HBM bandwidth utilization (0-1, multiply by 100 for %) — critical for roofline validation
avg(avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE[90s])) * 100

# Tensor core active — proxy for compute-bound vs BW-bound
avg(avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE[90s])) * 100

# SM occupancy
avg(avg_over_time(DCGM_FI_PROF_SM_ACTIVE[90s])) * 100

# Power draw (W)
avg(avg_over_time(DCGM_FI_DEV_POWER_USAGE[90s]))

# XID errors during run — any non-zero is a bad run
sum(increase(DCGM_FI_DEV_XID_ERRORS[90s]))
```

## S3 layout for archived snapshots

```
s3://<results-bucket>/prometheus/<blueprint>/<session-id>/
├── 20260513T192839Z-4f8a.../     ← each snapshot (decomposed TSDB)
├── 20260513T193839Z-7b21.../
├── ...
└── wal-latest/                    ← continuously-synced WAL (for crash recovery)
```

To query archived data post-session, download a snapshot and mount it in a local Prometheus:

```bash
aws s3 sync s3://<bucket>/prometheus/kimi-k2.6-speculative/20260513T192839Z-4f8a/ ./snap/
docker run --rm -p 9090:9090 -v $PWD/snap:/prometheus prom/prometheus:v2.54.1 \
    --storage.tsdb.path=/prometheus \
    --web.enable-admin-api
```

## Troubleshooting

### DCGM profiling metrics (HBM BW, SM active) return empty

**Cause**: Profiling requires `--profiler=on`, which the default DCGM config enables via the included `dcp-metrics-included.csv`. Check container args in docker-compose.
**Check**: `curl localhost:9400/metrics | grep DCGM_FI_PROF_` — should show non-empty `DRAM_ACTIVE`, `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`.
**Fix**: Ensure docker-compose pulls the GPU Operator DCGM image, not the barebones one.

### Engine /metrics shows no `*_bucket` fields

**Cause**: Histogram metrics only appear after the first request completes. Cold engines expose counters but not histograms.
**Fix**: Send one warmup request before running smoke test at the serving stage.

### TSDB snapshot API returns 405

**Cause**: Prometheus launched without `--web.enable-admin-api`.
**Fix**: Check the docker-compose command args include this flag.

### Histogram percentile is NaN

**Cause**: `histogram_quantile` over a window with zero rate (no requests fell into any bucket). Happens for warmup runs or when the window is before the first request.
**Fix**: Widen the window, or fall back to `*_sum / *_count` for the mean.
