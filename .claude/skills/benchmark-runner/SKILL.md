---
name: benchmark-runner
description: Plan, execute, and analyze LLM serving benchmarks across vLLM and SGLang configurations. Use when user says "run benchmarks", "benchmark config", "QPS sweep", "compare serving configs", "custbench", "benchmark analysis", or "generate benchmark report". Do NOT use for writing Terraform (use terraform-automation), deployment validation (use deployment-orchestrator), or GPU hardware diagnostics (use gpu-infra-troubleshooting).
---

# Benchmark Runner Skill

Standardized patterns and helpers for LLM serving benchmarks on GPU infrastructure. Covers benchmark planning, execution, metrics collection, and analysis handoff.

## Phase Structure

### Standard Phases (P0–P2) — Engine Selection + Optimization

Use for new model deployments to select the best engine and configuration.

| Phase | Purpose | Typical Configs | Dataset |
|-------|---------|-----------------|---------|
| **P0** | Engine comparison (vLLM vs SGLang) | 2 configs, same TP | `random` 1024/512 @ QPS 0.5 |
| **P1a** | MTP comparison (speculative decoding) | Winner ± MTP | `random` 1024/512 @ QPS 2.0 |
| **P1b** | Context scaling | Winner at 4K/32K/64K/128K | `random` + `generated-shared-prefix` |
| **P1c** | QPS sweep (find SLO-max) | Winner at QPS 0.5/1/2/4/8 | `random` 2048/512 |
| **P1d** | Parallelism comparison | TP vs DP+EP | `random` at SLO-max QPS |
| **P2a** | KV offloading (standard ctx) | cpu-offload, LMCache, Dynamo | `generated-shared-prefix` |
| **P2b** | Extended context (126K–252K) | Winner + cpu-offload | `prefix_repetition` |

### Customer Phases (T1–T7) — Custbench Pattern

Use for customer engagements to quantify optimization impact vs customer's baseline.

| Phase | Purpose | What Changes |
|-------|---------|-------------|
| **T1** | Customer baseline reproduction | Customer's exact config (Config A) |
| **T2** | Optimized head-to-head | Our best config (Config B) vs Config A |
| **T2b** | Prefix sharing isolation | Same configs, shared-prefix dataset |
| **T3** | MTP isolation | Config B ± MTP |
| **T4** | Load scaling | Both configs at QPS 0.5/5.0/inf |
| **T5** | Memory-constrained | Simulated smaller GPU (e.g., `--gpu-memory-utilization 0.30`) |
| **T6** | Multi-replica | 2x replicas, round-robin |
| **T7** | Stress test | 1000+ concurrent requests |

## Always Standard Format — the mandate

**Every benchmark run MUST emit the v1 benchmark-commons envelope directly. No exceptions, no post-processing.**

Use `scripts/bench-standard.py` as the source of truth. It:
- Drives the engine via OpenAI-compatible or native API
- Queries Prometheus for TTFT/TPOT/E2E histograms (client-side non-streaming loses TTFT)
- Queries DCGM for GPU util, HBM BW, tensor-core activity, power, temp, XID errors
- Emits v1 envelope with filename `<model>_<substrate>_<hw>_<engine-tag>_<workload>_c<N>.json`
- **Reconciles** client request count vs Prometheus success counter (5% tolerance)

**Motivating failure (Kimi K2.6-spec, 2026-05-13)**: custom bench driver only recorded total request duration. TTFT never captured. Prometheus was running but never scraped. After spot instance terminated, data was unrecoverable. 95 JSON files needed post-hoc enrichment with missing TTFT fields — a permanent gap in the dataset. DON'T REPEAT.

**Prerequisites on every benchmark GPU node** (enforced by infra-deployer Stage 4b):
- Prometheus + DCGM exporter + node-exporter running (see `templates/observability-stack.docker-compose.yml`)
- `scripts/observability-smoke-test.sh` passes
- Systemd timer `prom-sync.timer` syncing snapshots to S3

If you find yourself writing a one-off bench script that parses client timing only, STOP. Invoke `bench-standard.py` instead.

## Post-run validity gate

After an artifact is produced, run `standards/benchmark-commons/runner/validate-run.py`
with the workload card. The gate writes `artifact.validity.classification`:
`valid_controlled_baseline`, `valid_production_representative`,
`valid_smoke_only`, or `invalid_or_incomplete`.

- Synthetic/random cards may be valid controlled baselines, but **must not** be reported as production-representative.
- Production claims require `production-mix` / `sharegpt-production-mix` / trace-replay style workloads plus the card's required metrics.
- Use `--claim production --strict-production` when publishing customer- or production-facing numbers so missing required metrics fail closed.

## Card → command is compiled, not interpreted

**You do not decide what flags a workload runs with. `compile_card` does.** The
mapping from a workload card to concrete benchmark argv lives in
`standards/benchmark-commons/runner/registry.py` + `compiler.py` and is a pure,
tested function. Your job is to pick the right **card** and **sidecar** — not to
hand-write `--dataset-name` / `--request-rate` / `--goodput` flags.

- To run a workload: `runner/run-benchmark.sh --workload <card> --sidecar <yaml> --tool <vllm|sglang> [--dry-run]`. Dry-run prints the exact compiled argv.
- If compilation **raises `UnsupportedWorkload`**: the card or sidecar is wrong, or the `(dataset.type, load.type)` pair needs a registry handler. **Fix the declaration or add a handler — do NOT improvise a one-off driver.**
- If a card resolves to an **orchestrated** executor (agentic, multi-turn, cohost, burn-in, cold-start, audio, video): it cannot run via the vendor bench tools. Use the named executor in `runner/orchestrators.py`.
- Adding a new card, dataset type, or load type? Follow `runner/CONTRIBUTING.md` and make `python3 -m unittest discover -s runner/tests` pass. That test is the gate; green = correct by construction.
- The same sidecar is **also** the input to the serving-config resolver (`standards/serving-commons/`), which gates the *deployment* (FP8/TP divisibility, AMI, spec-decode, etc.) at Stage 0c — distinct from this skill, which gates the *benchmark*. A sidecar that fails `validate-serving-config.py` describes a config that should not be deployed, let alone benchmarked.

This is why the framework is deterministic: the SKILL guides *which* benchmark to run; the compiler owns *exactly what* executes. No interpretation in between.

## Benchmark Execution Rules

These are hard-won operational lessons. Follow them exactly.

1. **Same tool for both engines**: Use `vllm bench serve` for both vLLM and SGLang (via OpenAI-compatible API). Using different tools makes results incomparable.

2. **Group phases by framework**: Minimize server restarts. Model loads take 5–30 minutes depending on size and repacking.

3. **Benchmark at realistic QPS**: Use 0.5–5 QPS for config comparison. Use `inf` QPS only for stress testing. At 1000+ concurrent, ANY config shows high TTFT due to prefill queue — that's a concurrency problem, not a config problem.

4. **Always capture Prometheus metrics**: Scrape `/metrics` before and after every benchmark. Capture prefix cache hit rate, KV cache usage, and preemption count. Without these, you can't explain WHY a config is faster.

5. **Warm up before measuring**: Run 30 warmup requests before each benchmark to populate caches and JIT compile kernels.

6. **Run 3–5 repetitions**: Report p50 of the p50s. Flag results where p99/p50 > 3x as potentially unreliable.

7. **Record benchmark execution location**: Port-forward adds network latency that inflates TTFT. Document whether benchmarks ran via port-forward or directly on the node.

8. **Cooldown between configs**: Sleep 60s between benchmark runs to let GPU memory settle and caches stabilize.

## Metrics Checklist

### Client-Side (from benchmark tool output)

| Metric | Unit | How to Get |
|--------|------|------------|
| TTFT p50/p90/p99 | ms | `vllm bench serve` JSON output |
| ITL p50/p90/p99 | ms | `vllm bench serve` JSON output |
| TPOT p50/p90/p99 | ms | `vllm bench serve` JSON output |
| Output tok/s | tokens/s | output_tokens / wallclock_time |
| Total tok/s | tokens/s | (input + output) / wallclock_time |
| Error rate | % | failed_requests / total_requests |

### Server-Side (from Prometheus `/metrics`)

| Metric | PromQL | What It Tells You |
|--------|--------|-------------------|
| KV cache usage | `vllm:kv_cache_usage_perc` | Memory pressure |
| Prefix cache hit rate | `rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])` | Cache effectiveness |
| Preemption rate | `rate(vllm:num_preemptions_total[5m])` | Memory pressure (should be 0) |
| Running requests | `vllm:num_requests_running` | Concurrency |
| Waiting requests | `vllm:num_requests_waiting` | Queue depth |

### Capture Helpers

Source `scripts/benchmark-helpers.sh` in your blueprint's runner script:

```bash
source "$(dirname "$0")/../../.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh"

# Before benchmark
capture_metrics "$VLLM_URL" "${RESULT_DIR}/pre_${LABEL}_metrics.txt"
capture_kv_metrics "pre_${LABEL}" "$VLLM_URL"

# Run benchmark
run_bench "my_test" "$VLLM_URL" random 2048 512 2.0

# After benchmark
capture_metrics "$VLLM_URL" "${RESULT_DIR}/post_${LABEL}_metrics.txt"
capture_kv_metrics "post_${LABEL}" "$VLLM_URL"
```

## Dataset Selection

| Dataset | Flag | When to Use | Cache Effect |
|---------|------|-------------|-------------|
| `random` | `--random-input-len N --random-output-len M` | Baseline throughput, no cache benefit | None (cold) |
| `generated-shared-prefix` | `--gsp-system-prompt-len N --gsp-question-len M --gsp-output-len K` | Prefix caching effectiveness | High (shared prefix) |
| `prefix_repetition` | `--prefix-repetition-prefix-len N --prefix-repetition-suffix-len M` | Extreme prefix sharing (252K context) | Maximum |

## Post-Processing Pipeline

```
Benchmark execution (this skill)
    │
    ├── JSON results in results/session-YYYYMMDD/
    │
    ▼
benchmark-analyst agent
    │
    ├── Reads JSON, computes comparisons
    ├── Writes results/benchmark-report.md
    │
    ▼
visual-explainer skill (/benchmark-visual)
    │
    ├── Reads markdown report
    ├── Populates templates/benchmark-comparison.html
    ├── Writes results/benchmark-visual-YYYYMMDD.html
    └── Opens in browser
```

## Scaffolding a New Blueprint's Benchmarks

1. Copy the template: `cp .claude/skills/benchmark-runner/templates/run-benchmarks.sh.tmpl blueprints/<name>/scripts/run-benchmarks.sh`
2. Edit the template — fill in `MODEL_NAME`, `MODEL_PATH`, phase definitions
3. Create configs: one `configs/*.sh` per serving configuration
4. Run: `bash scripts/run-benchmarks.sh [phase]`

## Reference Loading

| Situation | Reference |
|-----------|-----------|
| Full phase definitions with exact commands | `references/benchmark-patterns.md` |
| Metrics collection details + PromQL | `references/metrics-reference.md` |
| Common failures during benchmarks | `references/troubleshooting.md` |
| Observability stack setup + smoke test | `references/observability.md` |

## Observability scripts (all runs, always)

| Script | Purpose |
|--------|---------|
| `scripts/bootstrap-observability.sh <s3-bucket> <blueprint>` | Install Prometheus + DCGM + node-exporter on the GPU node and enable S3 sync timer. Runs in infra-deployer Stage 4b. |
| `scripts/observability-smoke-test.sh` | Validate all exporters healthy + engine histograms present. Must pass before Stage 5 serving deployment. |
| `scripts/sync-prometheus-to-s3.sh` | Snapshot Prometheus TSDB and push to S3 every 10 min (installed as systemd timer). |
| `scripts/bench-standard.py` | **The standard bench driver.** Prometheus-first, emits v1 envelope. Always use this. |

## Troubleshooting

### vllm bench serve requires GPU even for client-only mode

**Error**: `RuntimeError: No CUDA GPUs are available`
**Cause**: vLLM benchmark CLI imports torch.cuda even in client mode.
**Solution**: Reserve one unused GPU for the benchmark client, or run the client on a CPU-only machine using the `--tokenizer` flag pointing to local model path.

### Model load time dominates benchmark session

**Error**: 30+ minutes spent on server restarts between configs.
**Cause**: MoE models require weight repacking (Marlin format) on every load.
**Solution**: Group phases by framework. Run all vLLM phases first, then all SGLang phases. Within a framework, order phases to minimize config changes that require restarts.

### Prefix cache metrics not captured

**Error**: Cache hit rate shows 0% despite shared-prefix dataset.
**Cause**: Prometheus scrape missed, or `--enable-prefix-caching` not set.
**Solution**: Always scrape metrics before AND after each test. Verify the server was started with `--enable-prefix-caching`. Check `vllm:prefix_cache_queries` counter is incrementing.

### Results not comparable across engines

**Error**: SGLang shows 2x better TTFT than vLLM.
**Cause**: Different benchmark tools (`sglang.bench_serving` vs `vllm bench serve`) have different client implementations.
**Solution**: Use `vllm bench serve` for both engines via OpenAI-compatible API endpoint.

### High TTFT at extreme concurrency

**Error**: TTFT p50 > 500ms at 1000 concurrent requests.
**Cause**: Prefill queue saturated — this is a concurrency problem, not a config problem.
**Solution**: Benchmark at realistic QPS (0.5–5) to measure config impact. Report extreme concurrency results separately as stress test data, not config comparison data.

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Triggers on benchmark requests | 90%+ | Test with "run benchmarks", "QPS sweep", "compare configs" |
| Does NOT trigger on infra/deploy requests | 0% | Test with "deploy", "terraform", "GPU health check" |
| Helper functions sourced without errors | 100% | `bash -n scripts/benchmark-helpers.sh` |
| Template scaffold produces valid script | 100% | Copy template, fill placeholders, `bash -n` passes |
| All benchmark phases documented | 100% | P0-P2 and T1-T7 fully defined with exact commands |
