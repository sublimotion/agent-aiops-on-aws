<!-- Reference: loaded on-demand by SKILL.md. Full phase definitions with exact commands. -->

# Benchmark Patterns Reference

## Standard Phases (P0–P2)

### P0: Engine Comparison

Compare vLLM vs SGLang at low QPS to select the winning engine.

```bash
# P0: vLLM baseline
run_bench "p0_vllm_tp4_32k_qps0.5" "$VLLM_URL" random 1024 512 0.5

# P0: SGLang baseline (use vllm bench serve against SGLang's OpenAI-compatible API)
run_bench "p0_sglang_tp4_32k_qps0.5" "$SGLANG_URL" random 1024 512 0.5
```

**Decision**: The engine with lower TTFT p50 AND higher output tok/s wins. If results are within 10%, prefer vLLM for ecosystem maturity.

### P1a: MTP Comparison

Test speculative decoding (Multi-Token Prediction) on the winning engine.

```bash
# Without MTP
run_bench "p1a_winner_tp4_noMTP_32k_qps2" "$WINNER_URL" random 1024 512 2.0

# With MTP (requires server restart with --num-speculative-tokens N)
prompt_restart "winner_tp4_MTP"
run_bench "p1a_winner_tp4_MTP_32k_qps2" "$WINNER_URL" random 1024 512 2.0
```

**Known issue**: MTP degrades throughput 2–41% on PCIe GPUs (no NVLink). Only use MTP on NVSwitch instances (P5/P5e/P5en).

### P1b: Context Scaling

Test how performance changes with input length.

```bash
for CTX in 1024 4096 32768 65536 131072; do
  run_bench "p1b_winner_ctx${CTX}_qps1" "$WINNER_URL" random "$CTX" 512 1.0
done

# With shared prefix (measures prefix cache benefit)
for CTX in 4096 32768 65536; do
  run_bench "p1b_winner_gsp_ctx${CTX}_qps1" "$WINNER_URL" \
    generated-shared-prefix "$CTX" 512 1.0 \
    --gsp-system-prompt-len "$((CTX - 256))" --gsp-question-len 256 --gsp-output-len 512
done
```

### P1c: QPS Sweep

Find the maximum QPS that meets SLO targets.

```bash
for QPS in 0.5 1.0 2.0 4.0 8.0; do
  run_bench "p1c_winner_qps${QPS}" "$WINNER_URL" random 2048 512 "$QPS"
done
```

**SLO targets** (from spec):
- TTFT p99 < 300ms at 32K context
- TTFT p99 < 1s at 128K context
- ITL p50 < 30ms

### P1d: Parallelism Comparison

Compare TP-only vs DP+EP (if model supports expert parallelism).

```bash
# TP=8
run_bench "p1d_tp8_qps2" "$WINNER_URL" random 2048 512 2.0

# DP=2 + EP=4 (requires different server config)
prompt_restart "dp2_ep4"
run_bench "p1d_dp2ep4_qps2" "$WINNER_URL" random 2048 512 2.0
```

### P2a: KV Cache Offloading

Test KV cache offloading at standard context length.

```bash
# CPU offload
run_bench "p2a_cpuoffload_32k_qps2" "$WINNER_URL" \
  generated-shared-prefix 32768 512 2.0 \
  --gsp-system-prompt-len 32000 --gsp-question-len 256 --gsp-output-len 512

# Dynamo KVBM (if available)
run_bench "p2a_dynamo_32k_qps2" "$WINNER_URL" \
  generated-shared-prefix 32768 512 2.0 \
  --gsp-system-prompt-len 32000 --gsp-question-len 256 --gsp-output-len 512
```

**Known blocker**: KV offloading incompatible with hybrid attention models (e.g., Qwen3-Next) in vLLM 0.16. All connectors disable Hybrid KV cache Manager (HMA).

### P2b: Extended Context

Test extreme context lengths with prefix sharing.

```bash
for PREFIX in 126000 189000 252000; do
  SUFFIX=1024
  run_bench "p2b_prefix${PREFIX}k" "$WINNER_URL" \
    prefix_repetition 0 512 1.0 \
    --prefix-repetition-prefix-len "$PREFIX" --prefix-repetition-suffix-len "$SUFFIX"
done
```

---

## Customer Phases (T1–T7) — Custbench Pattern

### T1: Customer Baseline

Reproduce the customer's exact configuration.

```bash
# Use customer's serving config (Config A)
run_bench "t1_configA_customer" 1000 10000 1000 inf
capture_kv_metrics "post_t1" "$VLLM_URL"
```

**Critical**: Match the customer's vLLM version, flags, model, quantization, and workload exactly. Any deviation invalidates the comparison.

### T2: Optimized Head-to-Head

Compare our best config against the customer baseline.

```bash
# Config A (customer)
run_bench "t2_configA" 100 10000 1000 inf

# Config B (optimized — requires server restart)
prompt_restart "configB_optimized"
run_bench "t2_configB" 100 10000 1000 inf
```

### T2b: Prefix Sharing Isolation

Test with shared-prefix dataset to measure prefix caching benefit.

```bash
# Config A — shared prefix
run_bench "t2b_configA_shared" 100 10000 1000 inf generated-shared-prefix \
  --gsp-system-prompt-len 8000 --gsp-question-len 128 --gsp-output-len 1000

# Config B — shared prefix
run_bench "t2b_configB_shared" 100 10000 1000 inf generated-shared-prefix \
  --gsp-system-prompt-len 8000 --gsp-question-len 128 --gsp-output-len 1000
```

### T3: MTP Isolation

Quantify MTP impact alone.

```bash
# Config B without MTP
run_bench "t3_noMTP" 100 10000 1000 5.0

# Config B with MTP
prompt_restart "configB_MTP"
run_bench "t3_MTP" 100 10000 1000 5.0
```

### T4: Load Scaling

Test both configs across QPS levels.

```bash
for QPS in 0.5 5.0 inf; do
  run_bench "t4_configA_qps${QPS}" 100 10000 1000 "$QPS"
  run_bench "t4_configB_qps${QPS}" 100 10000 1000 "$QPS"
done
```

### T5: Memory-Constrained

Simulate smaller GPU by reducing memory utilization.

```bash
# Restart server with --gpu-memory-utilization 0.30
prompt_restart "configB_constrained"
run_bench "t5_constrained_base" 100 10000 1000 inf
```

### T6: Multi-Replica

Round-robin across 2 replicas.

```bash
# Start 2 servers on ports 8000 and 8001
run_bench "t6_2replica" 1000 10000 1000 inf
# Client alternates between http://localhost:8000 and http://localhost:8001
```

### T7: Stress Test

Maximum concurrency.

```bash
run_bench "t7_stress" 1500 10000 1000 inf
capture_kv_metrics "post_t7" "$VLLM_URL"
```

---

## vllm bench serve Command Reference

```bash
vllm bench serve \
  --model "$MODEL_NAME" \
  --base-url "$VLLM_URL" \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 100 \
  --request-rate 2.0 \
  --warmup 30 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "${LABEL}.json" \
  --save-detailed \
  --tokenizer "$MODEL_PATH" \
  --sampling-params '{"temperature": 0.0, "top_p": 1.0}'
```

**Key flags**:
- `--request-rate inf` = send all at once (stress test)
- `--request-rate N` = N requests/second (steady state)
- `--tokenizer` = required in air-gapped environments or when model isn't on HuggingFace
- `--save-detailed` = per-request latency breakdown (needed for p99 analysis)
