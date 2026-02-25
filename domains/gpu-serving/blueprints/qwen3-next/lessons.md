# Lessons Learned: Qwen3-Next on P5en

## P5en Deployment

### 1. FP8 block_k=128 Incompatible with TP=8
**Problem**: vLLM FP8 quantization uses `block_k=128` for weight quantization. With TP=8, the shared expert MLP `down_proj` gets partitioned to `input_size_per_partition=64`, which is not divisible by `block_k=128`.

**Error**:
```
ValueError: Weight input_size_per_partition = 64 is not divisible by weight quantization block_k = 128
```

**Impact**: Neither vLLM nor SGLang can serve Qwen3-Next FP8 at TP=8. Both engines use block-quantized FP8 with block_size=128. Verified in P0b: SGLang v0.5.9-cu130 produces `ValueError: The output_size of gate's and up's weight = 64 is not divisible by weight quantization block_n = 128`. All configs must use TP=4 as baseline.

**Root cause**: Qwen3-Next uses a Mixture-of-Experts architecture with shared experts. The shared expert `down_proj` has a specific dimension that, when divided by 8 (TP=8), produces 64 — smaller than the FP8 quantization block size of 128. With TP=4, the partition size is 128, which is exactly divisible.

**Lesson**: When using FP8 quantization with MoE models, verify that all weight dimensions (including shared experts) remain divisible by `block_k` at the target TP degree. Test TP compatibility before reserving GPU capacity.

### 2. Air-Gapped Benchmark Requires --tokenizer Flag
**Problem**: `vllm bench serve` failed with `LocalEntryNotFoundError` because the container runs with `HF_HUB_OFFLINE=1` (air-gapped, no HuggingFace Hub access). The bench tool tried to download the tokenizer from HF Hub.

**Error**:
```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder
```

**Solution**: Pass `--tokenizer /mnt/nvme/models/qwen3-next-fp8` to point the bench tool at the local model directory for tokenizer files.

**Additionally**: The `--model` flag must use the served model name (`qwen3-next`), not the model path, because the server uses `--served-model-name qwen3-next`.

**Lesson**: In air-gapped deployments, always specify `--tokenizer` pointing to the local model path. The `--model` flag is the API-facing name, not the filesystem path.

### 3. K8s GPU Scheduling Deadlock on Config Changes
**Problem**: Changing vLLM from TP=4 (4 GPUs) to TP=8 (8 GPUs) via Terraform caused a scheduling deadlock. The new pod requested 8 GPUs but the old pod still held 4, leaving insufficient resources for the new pod to schedule.

**Solution**: Scale to 0 first, apply Terraform changes, then scale back to 1:
```bash
kubectl -n ml-inference scale deployment vllm-qwen3-next --replicas=0
terraform apply -target='kubernetes_deployment.vllm_qwen3[0]' -auto-approve
kubectl -n ml-inference scale deployment vllm-qwen3-next --replicas=1
```

**Lesson**: When changing GPU resource requests on a single-node deployment, always scale to 0 before applying. Rolling updates cannot work when the new pod requires more GPUs than are available after the old pod is subtracted.

### 4. vLLM 0.16 V1 Engine MTP Warmup Bug
**Problem**: vLLM 0.16.0rc2 V1 engine crashes with `AssertionError` when MTP speculative decoding is enabled. The `_dummy_run` warmup generates more tokens than `max_num_batched_tokens` due to speculative tokens.

**Error**:
```
AssertionError: assert num_tokens <= self.scheduler_config.max_num_batched_tokens
```

**Attempts**: Tried `num_speculative_tokens=2` (32K batch), `num_speculative_tokens=2` (64K batch), `num_speculative_tokens=1` + `VLLM_USE_V1=0` (env var not recognized in 0.16). All fail.

**Impact**: MTP benchmarks blocked. Need vLLM 0.17+ or a patched build.

**Lesson**: Test speculative decoding on a smaller instance before reserving a capacity block. vLLM V1 engine warmup doesn't account for spec tokens.

### 5. SGLang DeepGEMM JIT + CUDA Graph First-Run Penalty
**Problem**: SGLang v0.5.9 takes ~640s for first-time startup on this model due to DeepGEMM JIT compilation and CUDA graph capture. Subsequent starts are faster if the JIT cache is preserved.

**Lesson**: Pre-compile DeepGEMM using `sglang.compile_deep_gemm` before production deployment. Budget 10-15 minutes for first SGLang startup.

### 6. nerdctl -d and --rm Incompatible
**Problem**: `nerdctl run --rm -d` fails with `flags -d and --rm cannot be specified together`. Docker allows this but nerdctl doesn't.

**Solution**: Use `-d` without `--rm` and clean up manually with `nerdctl stop && nerdctl rm`.

### 7. DP+EP Underperforms TP=4 for MoE with FP8
**Problem**: Data parallelism (DP=8) with expert parallelism on all 8 GPUs produces worse latency and lower throughput than TP=4 on 4 GPUs for Qwen3-Next FP8.

**Results**: At QPS=4, DP+EP achieves 1397 tok/s vs TP=4's 1528 tok/s. TTFT p50 is 5.5x worse (237ms vs 43ms), ITL p99 exceeds the 30ms SLO (96.61ms vs 20.41ms).

**Root cause**: Expert parallelism requires cross-GPU communication for MoE routing at every layer. With 512 experts and 10 active per token, the routing overhead at TP=1 (no tensor-parallel weight sharing) dominates. TP=4 keeps MoE routing local to each GPU's shard and benefits from tensor-parallel weight distribution.

**Lesson**: For MoE models with many experts, tensor parallelism typically outperforms data parallelism with expert parallelism at the single-node scale. DP+EP may be more suitable for multi-node deployments where TP cannot span nodes efficiently.

### 8. --cpu-offload-gb Blocked on vLLM 0.16 V1 Engine
**Problem**: `--cpu-offload-gb 64` causes `AssertionError` in the V1 engine's `may_reinitialize_input_batch` — it asserts `cpu_offload_gb == 0`. The `VLLM_USE_V1=0` env var is not recognized in vLLM 0.16 (V0 engine was removed).

**Error**:
```
AssertionError: Cannot re-initialize the input batch when CPU weight offloading is enabled.
See https://github.com/vllm-project/vllm/pull/18298 for more details.
```

**Note**: `--cpu-offload-gb` offloads **model weights** to CPU RAM, not KV cache. On H200, this is unnecessary — 104.5 GiB KV cache is available per GPU even without weight offloading, providing 34.6x concurrency at 262K context.

**Lesson**: CPU weight offloading is broken in vLLM 0.16 V1 engine and unnecessary on H200. Don't waste capacity block time debugging it. If KV cache pressure becomes an issue at extreme concurrency, wait for vLLM to implement proper KV cache offloading (separate from weight offloading).

### 9. Extended Context (262K) Works Without CPU Offload on H200
**Problem**: P1b showed 126K context was unusable (TTFT p50 = 5.8s, ITL p99 = 604ms). The hypothesis was that CPU offload could help by freeing GPU VRAM for KV cache.

**Finding**: Simply setting `--max-model-len 262144` (without cpu-offload) enables full 262K native context. H200's 141 GB HBM per GPU provides ample KV cache capacity. The server reported 2.28M KV cache tokens and 34.6x concurrency at 262K.

**Results**: 252K context serves successfully — TTFT p50 = 2.2s (batch/async viable), and with prefix caching (200K shared prefix), TTFT p50 = 860ms and ITL p99 = 20ms (within SLOs).

**Lesson**: On H200, don't assume long-context needs KV offloading. The bottleneck at extreme context is prefill computation (O(n²) attention layers), not VRAM capacity. Prefix caching is the key enabler for long-context workloads — it eliminates redundant prefill.

### 10. Batching Transforms Long-Context Viability
**Problem**: 126K context at QPS 0.5 produced TTFT p50 = 1.2s (P2b-1) — marginal for interactive use.

**Finding**: At QPS 2.0 (P2b-3), batching reduces TTFT p50 from 1201ms to **261ms** (4.6x improvement) and ITL p99 from 422ms to **57.7ms** (7.3x improvement). The GPU amortizes prefill computation across concurrent requests.

**Lesson**: Long-context workloads should target moderate-to-high concurrency (QPS 2.0+) rather than low QPS. The batching effect that helps at short context (P1c) is even more impactful at long context, where prefill dominates. Design load balancers to maintain steady request flow rather than bursting.

### 11. `generated-shared-prefix` Dataset Not in vLLM 0.16
**Problem**: The benchmark script used `--dataset-name generated-shared-prefix` for prefix caching tests, but vLLM 0.16 bench doesn't support this dataset. Available datasets: `sharegpt, burstgpt, sonnet, random, random-mm, random-rerank, hf, custom, custom_mm, prefix_repetition, spec_bench`.

**Solution**: Use `--dataset-name prefix_repetition` with `--prefix-repetition-prefix-len`, `--prefix-repetition-suffix-len`, `--prefix-repetition-num-prefixes`, and `--prefix-repetition-output-len` flags.

**Lesson**: Verify benchmark dataset names against the installed vLLM version before running. The dataset API changes between versions.

---

## Summary

| # | Lesson | Category | Impact |
|---|--------|----------|--------|
| 1 | FP8 block_k=128 incompatible with TP=8 on ALL engines | Serving Config | Both vLLM and SGLang blocked; use TP=4 |
| 2 | Air-gapped bench needs --tokenizer | Benchmarking | HF_HUB_OFFLINE blocks tokenizer download |
| 3 | Scale to 0 before GPU config changes | K8s Deployment | Rolling update deadlocks on GPU contention |
| 4 | vLLM 0.16 V1 engine MTP warmup bug | Serving Config | MTP blocked; need vLLM 0.17+ |
| 5 | SGLang DeepGEMM first-run 640s startup | Serving Config | Budget 10-15min for first SGLang start |
| 6 | nerdctl -d and --rm incompatible | Container Runtime | Use -d only, clean up manually |
| 7 | DP+EP underperforms TP=4 for MoE FP8 | Serving Config | TP=4 is strictly superior; DP+EP exceeds SLOs |
| 8 | --cpu-offload-gb blocked on vLLM 0.16 V1 engine | Serving Config | Weight offload broken; unnecessary on H200 |
| 9 | Extended 262K context works without offload on H200 | Serving Config | 104.5 GiB KV cache/GPU; prefix cache enables 252K |
| 10 | Batching transforms long-context viability | Operations | 126K TTFT drops 4.6x at QPS 2.0 vs QPS 0.5 |
| 11 | generated-shared-prefix dataset not in vLLM 0.16 | Benchmarking | Use prefix_repetition instead |
