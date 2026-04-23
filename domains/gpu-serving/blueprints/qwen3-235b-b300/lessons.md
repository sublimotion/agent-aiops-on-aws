---
model: Qwen3-235B-A22B-FP8
engine: vllm
engine_version: v0.19.1
hardware: p6-b300.48xlarge
gpu: B300-SXM6-AC
gpu_count: 8
driver: "580.126.09"
cuda: "13.0"
nccl: "2.28.9"
spot_price_hr: 16.47
outcome: success
peak_throughput_tps: 13877
single_stream_tps: 110
ttft_p50_ms: 42
ttft_p99_ms: 1996
slo_max_concurrent: 512
cost_per_million_tokens: 0.39
date: 2026-04-22
failure_categories:
  - fp8_block_size_mismatch
  - huggingface_cli_deprecation
  - max_position_embeddings_mismatch
  - tool_call_parser_incompatibility
mdc_learn_commands:
  - 'mdc learn qwen3-235b vllm "FP8 block_n=128: moe_intermediate_size=1536 / TP8 = 192, not divisible. TP4 (384) or TP2 (768) work. TP1/TP3 also valid."'
  - 'mdc learn qwen3-235b vllm "max_position_embeddings=40960, NOT 131072. FP8 variant does not include YaRN config. Do not set --max-model-len > 40960."'
  - 'mdc learn qwen3-235b vllm "Use --tool-call-parser hermes (not qwen3_xml). qwen3_xml does not parse <tool_call> tags in vLLM v0.19.1."'
  - 'mdc learn qwen3-235b vllm "Use --reasoning-parser deepseek_r1 per official Qwen3 docs. Thinking mode on by default; disable with /no_think in system prompt."'
  - 'mdc learn qwen3-235b vllm "TP4 FP8 on B300: 55 GiB/GPU weights, 210 GiB KV headroom. Peak 11,820 tok/s @ c=512."'
  - 'mdc learn qwen3-235b vllm "TP2+DP4+EP: 17% more peak throughput (13,877 tok/s) but 4.4x worse single-stream and 36x worse cold-start TTFT. Use TP4 unless sustained >256 concurrent."'
  - 'mdc learn qwen3-235b vllm "FLASHINFER_TRTLLM MoE backend selected automatically for FP8. DeepGEMM E8M0 enabled on B300 sm_103."'
  - 'mdc learn qwen3-235b vllm "$0.39/M output tokens at peak (c=512). Break-even vs Sonnet at just 54 engineers. Most cost-efficient self-hosted coding model tested."'
gpu_infra_learn_commands: []
---

# Qwen3-235B-A22B-FP8 on B300 — Lessons Learned

## Session: 2026-04-22

### L1: FP8 Block Size Incompatible with TP8 for Qwen3 MoE

**Severity**: BLOCKING
**Category**: fp8_block_size_mismatch

Qwen3-235B has `moe_intermediate_size=1536`. vLLM FP8 quantization uses `block_n=128` for weight blocks. When sharding with TP, the per-partition size must be divisible by 128:

| TP | moe_intermediate / TP | % 128 | Status |
|---|---|---|---|
| 1 | 1536 | 0 | OK |
| 2 | 768 | 0 | OK |
| 3 | 512 | 0 | OK |
| 4 | 384 | 0 | OK |
| **8** | **192** | **64** | **FAIL** |

Error: `ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128`

**Fix**: Use TP4 (or TP2, TP3, TP1). TP4 is optimal for B300 — 55 GiB weights/GPU leaves 210 GiB KV headroom.

**Generalization**: For any FP8 MoE model, check `moe_intermediate_size / TP_SIZE % 128 == 0` before deployment. This applies to all models with fine-grained FP8 (block_size=128).

### L2: max_position_embeddings = 40960, Not 131072

**Severity**: BLOCKING
**Category**: max_position_embeddings_mismatch

The FP8 variant's `config.json` has `max_position_embeddings: 40960`. The spec assumed 131072 (YaRN-extended) based on the BF16 model card. vLLM refuses `--max-model-len 131072` without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

**Fix**: Set `--max-model-len 40960`. Do not blindly trust model card context length — always check the actual `config.json` in the downloaded weights.

### L3: qwen3_xml Parser Does Not Work in vLLM v0.19.1

**Severity**: HIGH
**Category**: tool_call_parser_incompatibility

The Qwen3 model outputs tool calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>` — this is the Hermes format, not a Qwen-specific XML format. The `--tool-call-parser qwen3_xml` flag exists but does not parse these tags. Tool calls appear as raw text in the `content` field.

**Fix**: Use `--tool-call-parser hermes`. The Hermes parser correctly extracts `<tool_call>` blocks into the `tool_calls` response field with `finish_reason: tool_calls`.

### L4: huggingface-cli Renamed to hf in huggingface_hub v1.11+

**Severity**: MEDIUM
**Category**: huggingface_cli_deprecation

`huggingface-cli download` no longer works. The CLI has been renamed to `hf download`. Also, `--exclude` flag semantics changed — passing `--exclude "*.md"` alongside no explicit filenames causes a `File not found` error because the CLI treats excluded patterns as explicit filenames.

**Fix**: Use `hf download <repo> --local-dir <path>` without `--exclude`. Download completes in ~3 min for 235GB over the EKS VPC.

### L5: TP4 Outperforms TP2+DP4+EP at All Practical Concurrency Levels

**Severity**: INFO (architecture decision)
**Category**: configuration_optimization

Tested both configurations on 8x B300:

| Config | Single-stream | Peak @ c=512 | TTFT p50 @ c=1 |
|---|---|---|---|
| TP4 (4 GPUs) | **110 tok/s** | 11,820 tok/s | **168ms** |
| TP2+DP4+EP (8 GPUs) | 23 tok/s | **13,877 tok/s** | 6,007ms |

TP2+DP4 only wins at c≥256 (+17% peak), but has 4.4x worse single-stream and catastrophic cold-start TTFT (6s) due to 4 independent warmup cycles. The expert parallelism communication overhead dominates at low batch sizes.

**Recommendation**: TP4 for general workloads. Consider TP2+DP4 only for sustained high-concurrency batch inference (>256 concurrent).

### L6: Thinking Mode Always On by Default

**Severity**: LOW
**Category**: model_behavior

Qwen3-235B wraps all output in `<think>...</think>` blocks by default. The `deepseek_r1` reasoning parser routes these to the `reasoning` field, leaving `content: null` for thinking-only responses.

To disable thinking: include `/no_think` in the system prompt. This produces clean `content` output with empty `reasoning`.

For benchmarking throughput, always use `/no_think` — thinking tokens inflate output counts and are discarded by the parser.

### L7: VLLM_TORCH_COMPILE_CACHE Not a Valid Env Var

**Severity**: LOW
**Category**: configuration_warning

vLLM v0.19.1 logs `Unknown vLLM environment variable detected: VLLM_TORCH_COMPILE_CACHE`. The env var name may have changed or been removed. The CUDA graph and compilation caches still work via `TRITON_CACHE_DIR` and the internal vLLM cache mechanisms.

### L8: 275 GB Per GPU, Not 268 GB

**Severity**: INFO
**Category**: hardware_discovery

B300 SXM6 AC reports 275,040 MiB (275 GB) per GPU, not the 268 GB commonly cited. Total cluster VRAM: 2,200 GB (not 2,150 GB). This gives slightly more KV headroom than spec estimated.

### L9: Kubernetes Service Environment Variables Leak into vLLM

**Severity**: LOW
**Category**: kubernetes_env_leak

When a previous deployment (e.g., `sglang-kimi-k26`) created a Kubernetes Service, the env vars for that service (`VLLM_KIMI_K26_SERVICE_HOST`, `VLLM_KIMI_K26_PORT`, etc.) leak into subsequent pods via K8s service discovery. vLLM warns about these as unknown env vars because they start with `VLLM_`.

Not harmful but noisy. Clean up old Services before deploying new pods, or use `enableServiceLinks: false` in the pod spec.
