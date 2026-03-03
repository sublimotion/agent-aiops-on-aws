<!-- Reference: loaded on-demand by SKILL.md. Common benchmark failures and fixes. -->

# Benchmark Troubleshooting

## Execution Issues

### vllm bench serve requires GPU even in client mode

**Error**: `RuntimeError: No CUDA GPUs are available`
**Cause**: vLLM benchmark CLI imports torch.cuda during initialization.
**Solution**:
1. Reserve one unused GPU for the benchmark client
2. Or run from a separate CPU-only machine using `--tokenizer /path/to/model`
3. Or use `CUDA_VISIBLE_DEVICES=""` and pass `--tokenizer` explicitly

### Server health check fails / hangs

**Error**: `wait_healthy` times out after 600s
**Cause**: Model loading takes longer than expected (MoE repacking, weight conversion).
**Solution**:
- H200 + Qwen3-Next TP=4: ~390s typical (Marlin MoE repacking)
- H200 + Kimi K2.5: ~2 hours from FSx (compute-bound, not I/O)
- Increase `wait_healthy` timeout: `wait_healthy "$URL" 3600`
- Check server logs for progress: `kubectl logs -f <pod>`

### Model fails to load with FP8 + TP mismatch

**Error**: `ValueError: block_k=128 but partition_size=64`
**Cause**: FP8 quantization uses `block_k=128`; with TP=8 on MoE shared experts, partition size becomes `input_size/8 = 64` which isn't divisible by 128.
**Solution**: Use TP=4 for FP8 MoE models. Test TP compatibility on CPU-only instance before reserving GPU capacity.

### MTP warmup assertion error

**Error**: `AssertionError: num_tokens <= self.scheduler_config.max_num_batched_tokens`
**Cause**: vLLM 0.16 V1 engine `_dummy_run` warmup exceeds `max_num_batched_tokens` when MTP is enabled.
**Solution**: Blocked until vLLM 0.17+. Workaround: use vLLM 0.15 or disable MTP.

### MTP degrades throughput on PCIe GPUs

**Error**: MTP shows 2–41% throughput degradation instead of improvement.
**Cause**: MTP requires NVLink for efficient draft token verification. PCIe bandwidth bottlenecks the speculative decoding round-trip.
**Solution**: Only use MTP on NVSwitch instances (P5/P5e/P5en). On PCIe instances (g7e), disable MTP.

## Metrics Issues

### Prefix cache hit rate is 0% with shared-prefix dataset

**Error**: `vllm:prefix_cache_hits` counter not incrementing despite shared-prefix workload.
**Cause**:
1. Server started without `--enable-prefix-caching`
2. Prometheus scrape captured metrics before any requests completed
3. `prefix_cache_queries` counter also 0 = feature not enabled
**Solution**:
- Verify server flag: `ps aux | grep enable-prefix-caching`
- Scrape metrics AFTER benchmark completes, not before
- Check `vllm:prefix_cache_queries` is incrementing (proves feature is enabled)

### KV cache metrics missing for Dynamo KVBM

**Error**: No `kvbm_*` metrics in Prometheus scrape.
**Cause**: KVBM metrics not enabled.
**Solution**: Start server with `DYN_KVBM_METRICS=true` environment variable.

### FSx cache directory empty after LMCache benchmark

**Error**: `du -sh /mnt/fsx/kv-cache/` shows 0.
**Cause**: LMCache FSx write permissions issue — container UID doesn't match FSx file ownership.
**Solution**: Check FSx mount permissions. LMCache writes as the container's UID; ensure FSx allows writes from that UID (often requires `--no-root-squash` on FSx).

### nvidia-fs (GDS) not available for GPU Direct Storage

**Error**: LMCache falls back to POSIX mode instead of GDS.
**Cause**: `nvidia-fs` kernel module not included in AL2023 EKS AMIs.
**Solution**: Use POSIX mode (functional, tested). For GDS, build custom AMI with `nvidia-fs` module.

## Results Issues

### Results not comparable across engines

**Error**: SGLang shows dramatically different numbers than vLLM for same workload.
**Cause**: Different benchmark tools (`sglang.bench_serving` vs `vllm bench serve`) use different client implementations — warmup logic, metric calculation, and request scheduling differ.
**Solution**: Always use `vllm bench serve --base-url <endpoint>` for both engines. Both vLLM and SGLang expose OpenAI-compatible APIs.

### High TTFT at extreme concurrency masks config differences

**Error**: Both Config A and Config B show ~940ms TTFT p50 at 1000 concurrent.
**Cause**: At extreme concurrency (1000+ concurrent × 10K input), the prefill queue dominates latency. ANY config will show high TTFT — this is a queuing theory problem, not a config problem.
**Solution**:
- Use QPS 0.5–5 for config comparison (measures optimization impact)
- Use `inf` QPS only for stress testing (measures burst capacity)
- Report stress test results separately, clearly labeled as "stress test at 1000 concurrent"
- At realistic load (5 QPS, ~100 concurrent), TTFT p50 drops to ~243ms

### Benchmark JSON files missing or empty

**Error**: `results/session-*/` directory is empty after benchmark.
**Cause**: `--save-result` flag not passed, or `--result-dir` not set.
**Solution**: Always include in `run_bench`:
```bash
--save-result --result-dir "$RESULT_DIR" --result-filename "${LABEL}.json" --save-detailed
```

### Port-forward inflates TTFT

**Error**: TTFT p50 is 2–3x higher than expected.
**Cause**: `kubectl port-forward` adds network hop and TCP overhead.
**Solution**:
1. Document execution location in benchmark report header
2. Run benchmarks directly on the GPU node (SSH or kubectl exec)
3. If port-forward is unavoidable, use it consistently for all configs so relative comparisons are valid
