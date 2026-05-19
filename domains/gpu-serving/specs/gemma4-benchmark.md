# Gemma 4 Serving Benchmark Spec

## Status: DRAFT (2026-05-19)

## Overview

Standalone serving benchmark for **Gemma 4 31B** (dense) and optionally **Gemma 4 26B-A4B** (MoE) on H100 NVSwitch. Produces config-comparable results against other gpu-serving blueprints (Qwen3-Next, GLM-5, Kimi K2.6) using the standard `benchmark-serving.py` workload sweep + `vllm bench serve` phases.

This is a **benchmark addendum** to `gemma4-hyperpod.md`. The parent spec covers HyperPod cluster setup, GPU pre-flight, and tool-calling validation. This spec focuses purely on quantifying serving performance across vLLM configurations and isolating the impact of Gemma 4-specific features (heterogeneous head_dim, hybrid sliding-window attention, prefix caching with shared prefixes).

### Optimization Objective

```
Primary:   Characterize Gemma 4 31B throughput-latency Pareto frontier on 1x p5.48xlarge (H100 TP2)
Secondary: Quantify impact of TRITON_ATTN (forced by head_dim=512) vs FlashAttention on TTFT
Metric:    TTFT p50/p99, ITL p50/p99, output tok/s across context lengths and QPS levels
```

### Why a separate benchmark spec

- `gemma4-hyperpod.md` mixes infra setup, smoke tests, tool calling, and benchmarks under a 6-hour shared FTP window — too compressed for clean A/B sweeps
- Repo benchmark report (`reports/benchmark-results.md`) needs config-parity numbers; the hyperpod spec currently lists targets as `TBD`
- Gemma 4 has unique architecture quirks (head_dim=512, hybrid attention) that warrant isolated characterization vs other models' results

---

## Components

### 1. Compute

- **Platform**: SageMaker HyperPod EKS or standalone EKS
- **Primary GPU Instance**: **ml.p5e.48xlarge** (8x H200 141GB HBM3e, NVSwitch)
- **Primary Parallelism**: **TP=1** — 31B BF16 (~63 GB) fits on a single H200 with ~78 GB headroom for KV cache. Eliminates TP collective overhead and gives the cleanest baseline.
- **Fallback**: ml.p5.48xlarge (8x H100 80GB) at TP=2 if H200 capacity unavailable. Forces a 2-GPU minimum for 31B BF16 and adds NCCL collective cost to every layer.
- **Region**: us-east-2 (or wherever p5e capacity is reservable)

**Why H200 / TP=1 is the recommended target:**

| Factor | H200 TP=1 | H100 TP=2 |
|--------|-----------|-----------|
| 31B BF16 fit | ✅ 63 GB on 141 GB GPU | Requires TP=2 (32 GB/GPU on 80 GB) |
| KV cache headroom at 32K | ~78 GB free | ~48 GB/GPU free (sharded) |
| HBM bandwidth | 4.8 TB/s | 3.35 TB/s — ~40% lower |
| TP collective overhead | None | NCCL all-reduce per layer (TRITON_ATTN already prefill-heavy — extra cost stings) |
| FlashAttention4 (head_dim=512) | ✅ SM 9.0 | ✅ SM 9.0 |
| Cross-spec comparability | Matches Qwen3-Next/GLM-5 H200 baselines | Off-axis vs main report |

The HBM-bandwidth and TP-collective wins matter most for decode latency (ITL p50/p99) and long-context prefill — both target metrics in this spec. Once the model fits on one GPU, capacity is no longer the constraint; bandwidth and TP overhead are.

**Out of scope hardware** (validated via card + upstream issues):
- **g7e (RTX PRO 6000 Blackwell)**: model freezes during loading (#38926, OPEN, no workaround)
- **A100 / Ampere / Ada / Turing**: head_dim=512 falls back to Triton (~9 tok/s on RTX 4090) or hits shared-mem limits (#38918)
- **B200 / B300**: SM 10.x untested for Gemma 4; FlashAttention4 path on Blackwell unvalidated as of 2026-05-19

> Reuses the cluster from `gemma4-hyperpod.md` if running in the same FTP window — that spec uses p5.48xlarge (H100), so an FTP-shared run automatically lands on the H100 TP=2 fallback path. Stand up a separate p5e.48xlarge for the recommended H200 TP=1 baseline.

### 2. Model

- **Primary**: `google/gemma-4-31B-it` — 30.7B dense, head_dim=256 (sliding) / 512 (global, every 6th layer), p-RoPE on global layers (theta=1M, partial_rotary_factor=0.25)
- **Secondary (optional)**: `google/gemma-4-26B-A4B-it` — 25.2B total / 3.8B active, 128 experts (top-8)
- **NVFP4 variant**: `nvidia/Gemma-4-31B-IT-NVFP4` — runs on TP=1 with `--quantization modelopt`. Worth a Config F sub-test if NVFP4 MoE loading bug (#38912) is resolved by run time.
- **Format**: BF16 (NVFP4 optional). FP8 not validated for Gemma 4 in vLLM 0.19.
- **Context**: 32,768 max for benchmarks (256K native; truncated for compute budget)
- **Vocab**: 262,144 — large; affects logit softcap (30.0) and tied embeddings memory.
- **Deployment Card**: Run `mdc get gemma-4 --engine vllm` before benchmarking. If no card exists, create one from this spec's results.

### 3. Serving Image

`vllm/vllm-openai:latest` (≥ v0.19.0) + `pip install git+https://github.com/huggingface/transformers.git` at startup. PyPI transformers lacks `gemma4` model_type as of 2026-04. The `vllm/vllm-openai:gemma4` tag fails on H100 (subprocess NVML error) — do not use.

> **vLLM stability check (2026-05-19, verified directly against `vllm-project/vllm` issue tracker — model deployment card was incomplete)**:
>
> **Merged fixes** (re-pull `latest`):
> - PR #41991 (merged 2026-05-08): tool parser infinite loop + array boundary
> - PR #42250 (merged 2026-05-13): MoE routing closure fix (26B-A4B)
> - PR #41574 (merged): MoE activation mismatch fix
> - PR #41745 (merged 2026-05-08): Gemma 4 MTP support added — but see CRITICAL issues below
> - PR #40786 (merged): pipeline parallelism fix
>
> **CRITICAL OPEN issues that the model deployment card does NOT list** (impact T6 / Config G directly):
> - **#42261 [Bug]: Frequent crashes with gemma4 MTP enabled on H200** (updated 2026-05-19) — `device-side assert triggered`, every few hours. **This affects our recommended hardware target.**
> - **#41789 [Bug]: gemma4 31B MTP avg draft acceptance rate 0.2%** — speculative decoding is essentially useless even when it runs (1 in 500 drafts accepted vs the typical 60-75%).
> - **#42516 [Bug]: Gemma4 NVFP4 fails to start with PP=2 or TP=2 without EP** — affects T5 if we ever try TP>1.
> - **#42687 [Bug]: Gemma-4 fails to start on GPUs with <70GB memory** due to `max_num_batched_tokens < multimodal token size`. **Affects L40S (48GB), A100 40GB, etc. — must override `--max-num-batched-tokens` or disable multimodal**.
> - **#41647 [Bug]: Unable to start Gemma4 with 2 GPUs** — general TP=2 instability.
> - **#42261 / #42390**: MoE init hang and segfault still reported.
> - **#41369**: Fast Prefill Optimization degrades p95 ITL significantly.
> - **#40624**: 0% prefix cache hits with hybrid attention + DFlash — directly relevant to T1.
> - **#41452**: 31B-it cannot process images in tool messages (multimodal × tools edge case).
> - **#42995**: V1 sleep/wake leaves multimodal sender cache desynced → AssertionError.
> - **#39827 [CLOSED]**: "Gemma4 output repeated token" — same repetition class as our RCA (memory: `feedback_synthetic_specdec_repetition.md`). Closed but likely re-emerges on synthetic + spec-decode.
>
> **Still OPEN architectural blockers**: #38891 (per-layer attention backend), #38926 (g7e freeze), #38918 (Turing SM 7.5 shmem).
>
> **HF model card** (`google/gemma-4-31B-it`): silent on hardware requirements. Only mentions "consumer GPUs and workstations" for 26B/31B and "high-end phones to laptops and servers" generally. Not authoritative on SM version, head_dim implications, or specific GPU compatibility. **Use the vLLM issue tracker as the source of truth**, not the HF card.

### 3b. SGLang status

**Not supported as of 2026-05-19.** SGLang PR #21952 still open. Card and `mdc get gemma-4 --engine sglang` both confirm no working SGLang path. Cross-engine vLLM-vs-SGLang comparison (the standard for GLM-5 / Kimi K2.6) is **out of scope** for this spec.

### 4. Networking

Inherits parent spec (private subnets, EFA, VPC endpoints). Benchmarks run in-cluster via bench-runner pod — no external traffic.

---

## Optimization Tiers

Configs follow the lever priority from `domains/gpu-serving/PRACTITIONER_GUIDE.md` (sections 5, 7, 11). Each tier adds **one** independent variable on top of the prior tier so we can attribute the delta to a specific lever. Tiers that are not applicable to Gemma 4 (LMCache CPU offload, SGLang HiCache, NVIDIA Dynamo KVBM) are listed under "Skipped Tiers" with rationale.

| Tier | Lever | Configs | Workloads | Why this tier |
|------|-------|---------|-----------|---------------|
| **T0** | Baseline (no opts) | A | W5, W6, P1v-a, P1v-b | Floor measurement; validates TRITON_ATTN penalty at 31B scale |
| **T1** | Prefix caching | B vs A | W2, W4, P1v-c | Practitioner guide's #1 single optimization (50-82% TTFT reduction). Hybrid attention may erode benefit — measure. |
| **T2** | Chunked prefill | C vs B | W6, P1v-b | Mitigates TRITON_ATTN prefill stalls at long context |
| **T3** | KV cache FP8 | D vs C | W5, P1v-a | Halves KV memory → higher max concurrency. Slight quality trade-off — verify with smoke test. |
| **T4** | Higher batch ceiling | E vs D | W5, P1v-a (high QPS) | Probes throughput knee with the now-larger KV pool from T3 |
| **T5** | Weight quantization | F vs E | All workloads | NVFP4 weights → 4× memory savings. On H200 (no native FP4 cores) this is a memory/bandwidth win, not compute. |
| **T6** | Speculative decoding | G vs E | W1, W5, P1v-a | Gemma 4 MTP (PR #41745, merged 2026-05-08). Decode-bound workloads benefit most. |
| **T7** (opt) | MoE variant | H vs E | W1, W5, P1v-a | 26B-A4B dense-vs-MoE comparison on identical hardware |
| **T8** (opt) | Multi-replica | I vs E | W5 high-QPS only | 2× TP=1 replicas on the same node, round-robin proxy. Doubles prefill compute + KV capacity. |

### Skipped Tiers (with rationale)

| Lever | Why skipped |
|-------|-------------|
| **SGLang HiCache** | SGLang doesn't support Gemma 4 (PR #21952 OPEN). Re-evaluate when merged. |
| **LMCache CPU offload** | Heterogeneous head_dim (256 sliding / 512 global) breaks LMCache's KV layout (confirmed in card + parent spec). |
| **NVIDIA Dynamo KVBM** | Same heterogeneous head_dim incompatibility; Dynamo KVBM assumes uniform KV layout per layer. |
| **MLA/NSA-specific tiers** | Gemma 4 uses standard MHA on global layers (not MLA), so PR #2629 / #2951 caveats from Kimi/GLM-5 don't apply. |
| **DP+EP** | Practitioner guide §11.2: "TP=4 consistently beats DP+EP". Gemma 4 dense on TP=1 is even further from needing EP. |
| **CPU offload (`--cpu-offload-gb`)** | Model fits in HBM with massive headroom on H200 TP=1. KV cache is not the constraint. |

---

## Benchmark Configs

### Config A — T0: Baseline (BF16, no optimizations)

```yaml
extra_vllm_args:
  - --tensor-parallel-size
  - "1"
  - --dtype
  - bfloat16
  - --max-model-len
  - "32768"
  - --gpu-memory-utilization
  - "0.90"
  - --tool-call-parser
  - gemma4
  - --enable-auto-tool-choice
  - --no-enable-prefix-caching
  - --trust-remote-code
```

Establishes the floor: no prefix caching, no chunked prefill optimization, default scheduling.

> **Tool parser**: Card recommends `--tool-call-parser gemma4`. Parent hyperpod spec uses `pythonic` (battle-tested on E4B with 100% BFCL). Run the BFCL gate against **both** parsers in P0 and pick the winner for the rest of the benchmark — record the choice. PR #41991 (merged 2026-05-08) fixes the gemma4 parser infinite-loop bug, so it's worth re-evaluating.

### Config B — T1: + Prefix Caching

Config A plus `--enable-prefix-caching`. Isolates prefix caching benefit on Gemma 4's hybrid attention (sliding-window layers cannot share prefix state across full-attention layers — measure real impact). Practitioner guide reports 11.1× speedup at 16K on Gemma 4 31B (parent hyperpod blueprint result) — confirm this on H200 TP=1.

> **vLLM #40624 caveat**: 0% prefix cache hits with hybrid attention + DFlash backend. If T1 reports `vllm:prefix_cache_hits` = 0 despite shared-prefix workload, switch attention backend and retry — likely we're hitting the same code path.

### Config C — T2: + Chunked Prefill

Config B plus `--enable-chunked-prefill --max-num-batched-tokens 16384`. Tests whether chunked prefill mitigates the TRITON_ATTN prefill penalty observed on E4B (379ms TTFT at 16K vs 148ms on A10G).

### Config D — T3: + KV Cache FP8

Config C plus `--kv-cache-dtype fp8`. Halves KV memory; doubles effective concurrency at long context. Verify with a 5-prompt smoke test that output quality is unchanged (Gemma 4 hasn't been validated for kv-fp8 — first time on this model).

### Config E — T4: + Higher Batch Ceiling

Config D plus `--max-num-seqs 256` and `--max-num-batched-tokens 32768`. Probes throughput ceiling now that KV pool is larger.

### Config F — T5: NVFP4 Weights

`nvidia/Gemma-4-31B-IT-NVFP4` with `--quantization modelopt`, all other flags from Config E. Same TP=1 single-GPU topology. ~17 GB weights vs ~63 GB → 124 GB free for KV. On H200 (SM 9.0) the matmul dequantizes to FP8/BF16 (no native FP4 tensor cores), so this is a memory/bandwidth win, not compute.

**Constraints from vLLM issue tracker (2026-05-19):**
- **#38912**: NVFP4 MoE weight loading broken — config is dense-only, **do not attempt 26B-A4B NVFP4**.
- **#42516**: NVFP4 fails to start with `--tensor-parallel-size 2` or `--pipeline-parallel-size 2` (without EP). **Stay on TP=1**.

### Config G — T6: + MTP Speculative Decoding

Config E plus `--speculative-config '{"method":"gemma4_mtp","num_speculative_tokens":2}'`. PR #41745 (merged 2026-05-08) added Gemma 4 MTP.

> **⚠️ DOWNGRADED from primary to investigative status** based on vLLM issue tracker (2026-05-19):
> - **#41789**: Reported MTP draft acceptance rate is **0.2%** on 31B — effectively no speedup
> - **#42261**: MTP **crashes H200 (our target hardware) every few hours** with `device-side assert`
> - **#41967**: MTP drops first tool-call arguments in streaming multi-tool auto-tool-choice
>
> **Pre-T6 gate** (must pass to proceed):
> 1. Run a 30-min stability test with MTP enabled — abort T6 if any `device-side assert` or `EngineDeadError` (#42261).
> 2. Measure draft acceptance rate from vLLM logs (`Avg Draft acceptance rate`). If <30%, the throughput uplift is theoretical only — record the number and skip the rest of T6.
> 3. Sample 20 outputs and check for token repetition (per RCA section).
>
> If any gate fails, **report Config G as "vLLM 0.20 not viable for Gemma 4 MTP on H200 as of 2026-05-19"** with citations to the open issues. Don't pretend a number that came from a 2-min run is representative.

### Config H — T7 (optional): Gemma 4 26B-A4B MoE

Config E flags applied to `google/gemma-4-26B-A4B-it`. Validate PR #42250 (MoE routing closure fix) and #41574 (activation mismatch fix) are present in the image. ~52 GB weights still fits TP=1 on H200.

### Config I — T8 (optional): 2× Replica on Same Node

Two vLLM instances on the same p5e.48xlarge — replica 0 on GPU 0 (port 8000), replica 1 on GPU 1 (port 8001). Both with Config E flags. Round-robin client. Doubles prefill compute and KV capacity vs single replica.

---

## Benchmark Workload

Standard `benchmark-serving.py` workload sweep (W1-W6) + `vllm bench serve` phases. Same workload definitions as `qwen3-next.md` and `glm5.md` for cross-model comparison.

> **Workload cards from `proposals/001-common-benchmark-artifact/workloads/`** are the canonical source of truth. Reference by `catalog_id`; do not re-derive parameters inline. Cards used in this spec:
>
> | catalog_id | Use case | Cards-driven tier |
> |------------|---------|-------------------|
> | `chatbot-short` | Latency-sensitive interactive | T0 sanity, T3 quality smoke |
> | `chatbot-long` | 32K input prefill | T2 chunked prefill |
> | `qps-sweep` | SLO-max QPS finder (synthetic 2K/512) | T0, T3, T4 |
> | `rag-long-context` | Shared 16K+ prefix | T1 prefix caching |
> | `coding-agent` | Tool calls, longer outputs | T6 MTP (long output favors speculation) |
> | `sharegpt-production-mix` | **Real conversation distribution** | **All tiers — production cross-check** |
> | `stress-saturation` | Out of scope | Not run (per §"Mandatory rules" #3) |
> | `batch-throughput` | Out of scope | Single-replica spec; covered in T8 only if added |

### W0 (NEW): ShareGPT Production-Mix Cross-Check — `sharegpt-production-mix`

**Run on every tier (T0-T8).** Synthetic fixed-length workloads (W2/W4/W5/W6) systematically mislead because:

- `random` dataset has near-0% prefix cache hit rate; real chats hit 20-60% on shared system prompts and multi-turn context. T1 numbers from synthetic alone overstate Gemma 4's prefix-caching loss from hybrid attention.
- Fixed-length workloads under-report tail latency: every request consumes the same KV, so scheduler preemption never triggers. Real ShareGPT inputs have a long tail (p99 ≈ 8K vs p50 ≈ 240 tokens) — that's where Gemma 4's TRITON_ATTN penalty actually bites.
- Uniform output lengths (`--random-output-len 512`) create artificially clean continuous-batching step boundaries. Heavy-tailed real outputs (p99 ~2.4K) better expose decode-step efficiency.

**Execution**:

```bash
run_bench "${TIER_LABEL}_w0_sharegpt" "$VLLM_URL" sharegpt 0 0 4.0 \
  --num-prompts 500 --warmup 50
```

**Reporting policy** (per workload card's `cross_check.policy`):

> Report both synthetic and `sharegpt-production-mix` results side-by-side. When deltas exceed 25% on any p50/p99 metric, flag the synthetic result as misleading and prefer this card's numbers in headline reporting and cross-blueprint tables.

**Required validation metrics** (per card's `validation.required_metrics`):

| Metric | Why required |
|--------|--------------|
| `prefix_cache_hit_rate` | Must be ≥ 5%; synthetic baseline is ~0%. Below 5% → dataset filtering broke shared prefixes; rerun. |
| `input_length_p50_p99` | Verify the realized distribution matches card expectations (~240/7800). |
| `output_length_p50_p99` | Same; catches sampling-param truncation. |
| `per_turn_ttft_distribution` | Multi-turn must report TTFT separately for turn 1 vs turn N — turn-1 hits cold prefix cache, turn-N hits warm. Aggregate masks this. |

**Reliability flags** (per card):
- `p99_p50_ttft_ratio > 5` → flag unreliable (queue saturation, not config-attributable)
- `prefix_cache_hit_rate < 0.05` → flag unrealistic (rerun)

### Synthetic workload sweep (W1-W6)


### W1: Multi-Turn Chat

Rounds {1, 5, 10} × concurrency {1, 4, 8} × QPS {1.0, 4.0}. Measures TTFT/ITL/throughput across conversation depth.

### W2: RAG / Long Document QA

Shared document prefix {2K, 5K, 10K} tokens, warmup:query ratio {2:2, 3:1, 4:1}. Cache benefit relative to Config A baseline.

### W3: Agentic Tool Calling

Multi-turn with simulated tool latency pauses. Inherits BFCL gate from `gemma4-hyperpod.md` P0 (must hit ≥70% accuracy before running W3).

### W4: Shared System Prompt

Prompt length {2K, 8K, 16K} × concurrency {4, 8}. Isolates prefix caching hit rate on hybrid attention.

### W5: ShareGPT-Style QPS Sweep

QPS {0.5, 1.0, 2.0, 4.0, 8.0} with realistic conversation distribution. Finds throughput-latency Pareto.

### W6: Long Context Scaling

Input length {1024, 4096, 8192, 16384, 32768} × output 512. Quantifies TRITON_ATTN prefill cost as context grows.

### vllm bench phases

| Phase | Sweep | Workload |
|-------|-------|----------|
| **P1v-a** | QPS {0.5, 1.0, 2.0, 4.0, 8.0} | random 2K input / 512 output, 100 prompts |
| **P1v-b** | Context {1024, 4096, 8192, 16384, 32768} | random, 50 prompts at 1.0 QPS |
| **P1v-c** | Shared prefix {4K, 16K, 32K} | `generated-shared-prefix`, 50 prompts at 1.0 QPS |

---

## Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **TTFT p50** | Median time to first token | < 500ms at 16K input (W6) |
| **TTFT p99** | 99th percentile TTFT | < 1000ms at 32K input (P1v-b) |
| **ITL p50** | Median inter-token latency | < 50ms (P1v-a) |
| **ITL p99** | 99th percentile ITL | < 100ms |
| **Output tok/s** | Aggregate output throughput | Maximize |
| **QPS at SLO** | Max QPS meeting TTFT p99 < 2s | ≥ 4.0 (W5) |
| **Prefix cache speedup** | Cold vs warm TTFT ratio | ≥ 2x (P1v-c) |
| **RAG cache benefit** | TTFT reduction with shared prefix | ≥ 30% (W2) |
| **BFCL accuracy** | Tool-call gate (from parent spec) | ≥ 70% (P0) |
| **Error rate** | Failed requests / total | 0% |

KV cache observability via vLLM `/metrics`:

| Metric | Relevance |
|--------|-----------|
| `vllm:prefix_cache_hit_rate` | Validates W2/W4/P1v-c hit rates on hybrid attention |
| `vllm:gpu_cache_usage_perc` | Confirms TP=2 KV pressure under load |
| `vllm:num_preemptions_total` | Detects scheduler overflow |

---

## Tooling & Standards Compliance

This spec follows the repo-standard **`benchmark-runner` skill** (`.claude/skills/benchmark-runner/`) so results are directly comparable to other gpu-serving blueprints.

### Required tooling

| Tool | Path | Purpose |
|------|------|---------|
| `vllm bench serve` | container | Same client for vLLM (and SGLang when supported) — never mix tools across engines |
| `benchmark-helpers.sh` | `.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh` | Sources `run_bench`, `capture_metrics`, `capture_kv_metrics`, `prompt_restart` helpers |
| `run-benchmarks.sh.tmpl` | `.claude/skills/benchmark-runner/templates/` | Scaffold for `blueprints/gemma4-hyperpod/scripts/run-benchmarks.sh` |
| `bench-runner-pod.yaml` | `scripts/bench-runner-pod.yaml` | In-cluster benchmark client (avoids port-forward latency that inflates TTFT) |
| `benchmark-serving.py` | `scripts/benchmark-serving.py` | W1-W6 standard workload sweep |
| `validate-results.sh` | `.claude/skills/benchmark-runner/scripts/` | Pre-handoff sanity check on result JSON |

### Phase numbering reconciliation

The benchmark-runner skill defines **P0-P2** (engine selection) and **T1-T7** (custbench). This spec uses **T0-T8 optimization tiers** (single-engine; SGLang unavailable for Gemma 4). The mapping is:

| This spec | benchmark-runner skill equivalent | Notes |
|-----------|-----------------------------------|-------|
| T0 (baseline) | P0-style baseline (single-engine, no opts) | No vLLM-vs-SGLang because SGLang lacks Gemma 4 |
| T1 (prefix cache) | P2a-style (KV/cache effectiveness) | Uses `generated-shared-prefix` dataset |
| T2 (chunked prefill) | P1b-style (context scaling) | `random` dataset, ctx sweep |
| T3 (kv-fp8) | P1c-style + memory study | `random` 2K/512 QPS sweep |
| T4 (batch ceiling) | P1c-style (QPS sweep) | High-QPS tail |
| T5 (NVFP4) | Cross-quantization comparison | Whole sweep on quantized weights |
| T6 (MTP) | P1a-style (MTP isolation) | A/B with vs without speculative-config |
| T7 (MoE variant) | Cross-model variant test | — |
| T8 (multi-replica) | T6-style custbench multi-replica | Skill's custbench T6 pattern, applied to TP=1 |

### Mandatory per-tier execution rules (from skill)

1. **Same tool for both engines**: `vllm bench serve` only. Not applicable here (vLLM-only) but enforced for cross-spec comparability.
2. **Group phases to minimize restarts**: Order configs so flag changes don't require model reload where possible. T1→T2→T3→T4 share weights (Config A→E); only T5 (NVFP4 weights) and T7 (26B-A4B) require restart.
3. **Realistic QPS for config comparison**: Tiers compare at QPS 0.5-8.0. The "stress" QPS=inf path is **out of scope** — Gemma 4's TRITON_ATTN penalty makes 1000-concurrent meaningless for config attribution.
4. **Always scrape Prometheus before AND after each run** (see Metrics Capture below).
5. **Warmup 30 requests** before each measured run — JIT compile, populate prefix cache.
6. **3 repetitions per config**, report median of medians. Flag p99/p50 > 3× as unreliable.
7. **Record execution location**: in-cluster bench-runner pod (preferred). Port-forward results recorded separately and labeled.
8. **60s cooldown** between configs.

### Metrics Capture

**Client-side** (from `vllm bench serve` JSON, captured by `run_bench`):

| Metric | Source | Target |
|--------|--------|--------|
| TTFT p50/p90/p99 | JSON output | TTFT p50 < 500ms @ 16K, p99 < 1000ms @ 32K |
| ITL p50/p90/p99 | JSON output | ITL p50 < 50ms |
| TPOT p50/p90/p99 | JSON output | informational |
| Output tok/s, total tok/s | JSON output | maximize |
| Error rate | JSON output | 0% |

**Server-side** (Prometheus `/metrics` scrape via `capture_metrics` + `capture_kv_metrics`):

| Metric | PromQL | Why it matters for Gemma 4 |
|--------|--------|----------------------------|
| `vllm:kv_cache_usage_perc` | gauge | Confirms TP=1 KV pressure on H200 (expect low — that's the point) |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | counter ratio | T1 prefix cache effectiveness on hybrid attention |
| `vllm:num_preemptions_total` | counter rate | Should be 0; non-zero indicates KV oversubscription (revisit T3/T4 batch ceiling) |
| `vllm:num_requests_running` | gauge | Actual concurrency vs `--max-num-seqs` |
| `vllm:num_requests_waiting` | gauge | Queue depth — early signal of saturation |
| `vllm:gpu_cache_usage_perc` | gauge | KV pool utilization — informs T4 batch ceiling tuning |
| `vllm:e2e_request_latency_seconds` | histogram | Server-side latency vs client-measured (catches network overhead) |

**Capture pattern** in `run-benchmarks.sh` (per skill template):

```bash
source .claude/skills/benchmark-runner/scripts/benchmark-helpers.sh

# Per-tier loop
for TIER_LABEL in t0_baseline t1_prefix t2_chunked t3_kvfp8 t4_batch t5_nvfp4 t6_mtp; do
  capture_metrics "$VLLM_URL" "${RESULT_DIR}/pre_${TIER_LABEL}_metrics.txt"
  capture_kv_metrics "pre_${TIER_LABEL}" "$VLLM_URL"

  # 30-request warmup
  run_bench "warmup_${TIER_LABEL}" "$VLLM_URL" random 2048 512 1.0 --num-prompts 30

  # Tier-specific workload(s) — see Tier Execution Plan
  run_bench "${TIER_LABEL}_w5_qps2" "$VLLM_URL" random 2048 512 2.0 --num-prompts 100
  run_bench "${TIER_LABEL}_p1vc_16k" "$VLLM_URL" generated-shared-prefix 0 0 1.0 \
    --gsp-system-prompt-len 15872 --gsp-question-len 256 --gsp-output-len 512 --num-prompts 50

  capture_metrics "$VLLM_URL" "${RESULT_DIR}/post_${TIER_LABEL}_metrics.txt"
  capture_kv_metrics "post_${TIER_LABEL}" "$VLLM_URL"
  sleep 60  # cooldown
done
```

### Prometheus stack

The parent `gemma4-hyperpod.md` spec deploys a shared Prometheus that scrapes vLLM `/metrics` on the model port. This benchmark spec **inherits** that stack — no new Prometheus deployment. If running standalone (no parent FTP cluster), deploy via the standard `kube-prometheus-stack` Helm chart and add a `ServiceMonitor` selecting the vLLM pod.

### Result handoff

Per skill's post-processing pipeline:

```
run-benchmarks.sh
   └→ results/session-YYYYMMDD/*.json + pre/post_*_metrics.txt
       └→ benchmark-analyst agent → results/benchmark-report.md
           └→ visual-explainer skill → results/benchmark-visual-YYYYMMDD.html
               └→ reports/benchmark-results.md (cross-blueprint comparison row)
```

After all tiers complete, invoke the **`benchmark-analyst` agent** with the session directory as input. It writes the analysis to `blueprints/gemma4-hyperpod/results/benchmark-report.md` and updates the cross-model comparison row in `reports/benchmark-results.md`.

---

## Test Protocol

| Parameter | Value | Notes |
|-----------|-------|-------|
| Warmup requests | 30 | Sent before each measurement (skill rule #5) |
| Runs per config | 3 | Median of medians (skill rule #6) |
| Cooldown between runs | 60s | Stabilize KV cache (skill rule #8) |
| Sampling params | temperature=0.7, top_p=0.9 | Match Gemma 4 official recipe |
| Percentiles collected | p50, p90, p99 | Standard repo convention |
| Pre/post metrics scrape | every run | `capture_metrics` + `capture_kv_metrics` |
| Result format | JSON to `results/session-YYYYMMDD/` | Per skill's post-processing pipeline |
| Reliability flag | p99/p50 > 3× → mark unreliable | Per skill rule |

---

## Tier Execution Plan

Each tier is a focused A/B against the prior tier's config — minimizing run time while attributing each delta to one lever. Run tiers in order; if a tier shows no benefit (or regression), record the result and continue to the next without rolling back.

### T0 — Baseline (Config A)

W0 (sharegpt) + W1-W6 + P1v-a/b/c on Config A. Establishes floor numbers and confirms the TRITON_ATTN penalty seen on E4B is reproducible at 31B scale on H200 TP=1. **W0 is mandatory** — establishes the real-distribution baseline that every later tier's synthetic numbers will be cross-checked against.

### T1 — + Prefix Caching (Config B vs A)

**Workloads**: W0 + W2, W4, P1v-c only — workloads with shared prefixes. Quantifies how much Gemma 4's hybrid attention erodes prefix caching benefit. Expected: lower than the 11.1× speedup seen on parent hyperpod (which ran TP=2 on H100 — different baseline).

### T2 — + Chunked Prefill (Config C vs B)

**Workloads**: W0 + W6, P1v-b only — long-context workloads where prefill stalls dominate. Tests whether chunked prefill mitigates the TRITON_ATTN penalty. If #38891 merges before this tier, also re-run T0 baseline.

### T3 — + KV Cache FP8 (Config D vs C)

**Workloads**: W0 + W5, P1v-a, plus a 5-prompt quality smoke test. Halves KV memory; doubles effective concurrency. Expected: higher max QPS, no quality regression on the smoke test.

### T4 — + Higher Batch Ceiling (Config E vs D)

**Workloads**: W0 + W5 high-QPS tail (4.0, 8.0, 16.0), P1v-a. Locates the throughput knee with the larger KV pool from T3.

### T5 — NVFP4 Weights (Config F vs E)

**Workloads**: All — full W1-W6 + P1v-a/b/c. NVFP4 changes the memory profile globally, not just at high concurrency. Compare $/M-token against Config E (BF16). On H200 expect memory/bandwidth wins, not compute wins (no native FP4 cores). On B200 (future), this tier should win on compute too.

### T6 — + MTP Speculative Decoding (Config G vs E)

**Workloads**: W0 + W1 (multi-turn — speculative decoding loves repetitive contexts), W5, P1v-a. Skip if `gemma4_mtp` fails to start. Practitioner guide warns MTP only viable on NVLink/NVSwitch — H200 NVSwitch qualifies; do **not** transfer this result to g7e.

> **MANDATORY pre-run gate — degenerate output check** (see RCA below):
> Before recording any MTP throughput number, sample 20 outputs from the synthetic random workload and verify they are **not** degenerate text (e.g., "hello hello hello", "the the the", or any token repeated >5× consecutively). MTP draft heads on random-token inputs can collapse to high-acceptance trivial repetition, inflating reported tok/s by 2-5× while producing unusable output. If degenerate output detected, **discard the synthetic MTP throughput number entirely** and report only the W0 (sharegpt) MTP number as the headline.

### T7 (optional) — MoE Variant (Config H vs E)

**Workloads**: W0 + W1, W5, P1v-a only. Direct dense-vs-MoE on identical hardware. Validates that MoE routing fixes (#42250, #41574) are present in the image.

### T8 (optional) — 2× Replica (Config I vs E)

**Workloads**: W0 + W5 high-QPS tail only. Tests whether 2 replicas on the same node beat 1 replica with higher batch ceiling. Practitioner guide §11.2 found 2× TP=4 replicas beat single TP=8 for several models — analogous test on TP=1 dense.

### Tier Decision Gates

Don't run every tier blindly — these gates abort or skip based on prior results:

| Gate | If… | Then |
|------|-----|------|
| After T0 | TTFT p99 at 32K > 2× target | Investigate before running T1; likely #38891 (per-layer attention) is needed |
| After T1 | Prefix cache speedup < 1.5× | Skip T2 (chunked prefill won't help if prefix caching can't); proceed to T3 |
| After T3 | KV-FP8 smoke test fails (quality regression) | Drop kv-fp8 from T4-T8, rerun T4 with C's flags |
| After T5 | NVFP4 model loading fails | Skip T5; document #38912 status |
| After T6 | MTP fails to start | Skip T6; record vLLM version + error |
| Before T7 | T5 shows large memory headroom | Worth running T7 NVFP4 26B-A4B too if #38912 has merged |

---

## RCA: Synthetic Workloads + Speculative Decoding (Kimi K2.6 / Qwen3-235B-B300)

**What happened**: During the Kimi K2.6 and Qwen3-235B-B300 benchmark sessions, MTP speculative decoding numbers measured against `vllm bench serve --dataset-name random` looked unrealistically good. Investigation revealed the model was emitting degenerate output — repeated tokens like `"hello hello hello..."` — when fed synthetic random-token inputs. The MTP draft head trivially predicted the next repetition, hit ~100% acceptance, and the throughput counter happily reported 2-5× the real number. The output text was unusable, but `vllm bench serve` doesn't validate output content — it just counts tokens emitted per second.

**Why it happened**:

1. **Random token sequences are out-of-distribution for instruction-tuned models.** Sampled tokens from the tokenizer have no semantic structure, so the base model is uncertain everywhere and often falls into a repetition attractor (a known failure mode at high entropy / low temperature).
2. **Speculative decoding amplifies the artifact.** Once the model emits `"hello hello"`, the MTP draft head sees the trivial pattern and predicts `"hello"` again with near-certainty. Acceptance rate spikes, decode throughput scales near-linearly with `num_speculative_tokens`, and the benchmark looks "great".
3. **`vllm bench serve` is content-blind.** The tool measures emitted tokens per second; it doesn't check whether outputs are coherent or whether the same token repeats. Degenerate output and useful output produce identical headline metrics.
4. **Real distributions don't trigger this.** ShareGPT inputs have semantic structure, so the model produces varied outputs, the draft head's pattern-matching can't trivially win, acceptance drops to normal (~60-75%), and throughput regresses to the real number.

**Was it captured in lessons?** No. The blueprints' `lessons.md` files do not record this issue. It was caught and corrected mid-session (likely by sampling outputs by hand) before formal lessons were written. This is itself a guardrail gap — the institutional memory exists in the user's head, not in the repo.

**Why it could recur on Gemma 4**: Config G/T6 explicitly tests `gemma4_mtp` (PR #41745, merged 2026-05-08 — < 2 weeks old at this writing). Gemma 4 is also a fresh model where output-quality regressions are likely. If we run T6 on synthetic workloads only and don't sample outputs, we will repeat the mistake.

**Guardrails added to this spec**:

| Layer | Guardrail |
|-------|-----------|
| **Workload** | W0 (`sharegpt-production-mix`) is mandatory on **every tier**, not optional. Real-distribution data prevents the draft head from finding a trivial repetition pattern. |
| **Tier T6 pre-run gate** | Sample 20 outputs from synthetic random workload, reject if any contains a token repeated >5× consecutively. |
| **Reporting policy** | Headline metrics use the W0 (sharegpt) number. Synthetic-vs-W0 deltas >25% are flagged as misleading per the workload card's `cross_check.policy`. |
| **Validation metric** | Workload card's `validation.required_metrics` includes `output_length_p50_p99` — a degenerate "hello hello..." output usually maxes out at `max_tokens` while real outputs follow a heavy-tailed distribution. Mismatch is a red flag. |
| **Lesson capture** | If a synthetic-vs-W0 gap >25% is observed on any tier, capture it as a HIGH-severity lesson in `blueprints/gemma4-hyperpod/lessons.md` so the next blueprint inherits the knowledge. |

**Repo-level gap** (not fixed by this spec, but flagged for compound-learner):

The `benchmark-runner` skill's troubleshooting reference does not mention degenerate-output detection. Adding a "Sample N outputs and reject token repetition" step to the skill's `references/troubleshooting.md` would make the guardrail global, not Gemma-4-specific. Recommend the compound-learner agent elevate this lesson to `.claude/skills/benchmark-runner/references/troubleshooting.md` and `.claude/steering/tech-stack.md` after this benchmark session.

---

## Success Criteria

1. All target metrics in the Metrics table met or explicitly characterized as not met with root cause
2. Config B vs A delta quantifies prefix caching benefit on hybrid attention (expected: smaller than dense-attention models)
3. Config C vs B delta quantifies chunked prefill benefit at long context
4. **Every tier reports both synthetic and `sharegpt-production-mix` (W0) numbers side-by-side**. Where deltas exceed 25% on any p50/p99 metric, the W0 number is reported as the headline and the synthetic flagged as misleading.
5. W0 prefix cache hit rate ≥ 5% (validation gate from workload card; below = dataset filter broken, rerun)
6. Cross-model comparison table in `reports/benchmark-results.md` reports the **W0 number**, not a synthetic number. Past entries from other blueprints that report only synthetic numbers should be flagged for re-measurement.
7. Lessons captured to `blueprints/gemma4-hyperpod/lessons.md` with severity ratings, including any synthetic-vs-W0 gap > 25% as a HIGH-severity lesson for the broader benchmark report
8. **T6 (MTP) outputs sampled and validated** — no degenerate token repetition (>5× consecutive same token in any of 20 sampled outputs). If detected, synthetic MTP throughput is discarded and only W0 is reported.

---

## Non-Requirements

- FP8 weight quantization (not validated for Gemma 4 in vLLM 0.19; NVFP4 covered by Config G)
- 256K context evaluation (compute budget; 32K is the cap)
- Multi-node distributed inference
- LMCache integration (incompatible with heterogeneous head_dim)
- **SGLang comparison** — Gemma 4 SGLang support pending (PR #21952 OPEN as of 2026-05-19); `mdc get gemma-4 --engine sglang` returns no card. Cross-engine comparison deferred to a future spec once SGLang lands.
- Reasoning-mode benchmarks (vLLM #38855 — channel tokens stripped)
- Audio modality benchmarks (E-series only; not applicable to 31B / 26B-A4B)
- Turing / Ampere / Ada GPUs (head_dim=512 unusable; Hopper-only spec)
- g7e (RTX PRO 6000 Blackwell) — model freezes during loading (#38926)

---

## Known Limitations

Verified against the **`vllm-project/vllm` GitHub issue tracker** (synced 2026-05-19) — the model deployment card was incomplete and missed several critical bugs. The HF model card is silent on hardware. Always cross-check `mdc get gemma-4 --engine vllm` against `gh issue list --repo vllm-project/vllm --search "gemma4"` before launching.

- **head_dim=512 on global layers**: requires SM 9.0+ FlashAttention4 (Hopper). Triton fallback on Ampere/Ada → ~9 tok/s on RTX 4090. Turing (SM 7.5) hits shared-memory limits regardless of backend (#38918, OPEN).
- **g7e (RTX PRO 6000 Blackwell) freezes during loading** (#38926, OPEN, no workaround). Do not benchmark on g7e.
- **Per-layer attention backend selection** (#38891, OPEN): until merged, vLLM picks one backend for the whole model — TRITON_ATTN dominates due to the 512 global layers, costing prefill latency vs FlashAttention. Expect ~2-3x higher single-request TTFT than dense-attention models. Watch this PR — if it merges before benchmark run, re-baseline Config A.
- **Reasoning parser broken**: channel tokens stripped before parser sees them (#38855). `--reasoning-parser gemma4` separates nothing. **Out of scope** for this benchmark.
- **Tool calling streaming bugs**: invalid JSON diffs (#38945) and duplicated HTML-tag prefixes (#38910) on streaming. PR #41991 (merged 2026-05-08) fixed the infinite loop / array boundary path. Run BFCL with both `--tool-call-parser gemma4` and `pythonic`, prefer non-streaming if streaming still flakes.
- **NVFP4 MoE weight loading** (#38912, OPEN): affects 26B-A4B NVFP4. 31B dense NVFP4 should work — validate before T5.
- **Vision encoder always loads**: SigLIP tower (~550M params for 31B) loads regardless of whether prompts include images. Adds VRAM overhead even for text-only benchmarks; budget accordingly.
- **Logit softcap = 30.0**: ensure benchmark sampling params don't conflict (no manual softcap override).
- **PyPI transformers lacks gemma4**: must install from git at container startup. Adds ~30s to cold start.
- **Card-recommended max-model-len = 8192** in launch examples; we override to 32768 for context-scaling benchmarks. Verify no compile/CUDA-graph regressions at 32K.

Run `mdc prs gemma-4` immediately before launching to catch any new fixes.

---

## Verification Criteria

### Stage 4a — GPU Health
- [ ] All H100s report ECC enabled, 0 uncorrectable errors
- [ ] NVLink topology: all 8 GPUs via NVSwitch
- [ ] NCCL all-reduce > 450 GB/s for TP=2 (per parent spec)
- [ ] No Xid errors in dmesg

### Stage 5 — Serving Stack
- [ ] `/health` returns 200 within 8 minutes of pod start (includes git transformers install + model load)
- [ ] Single completion against `/v1/completions` returns valid Gemma 4 output
- [ ] No CUDA OOM in container logs
- [ ] Tool calling smoke test (parent spec P0 step 4) passes

### Stage 6 — Benchmark
- [ ] All T1-T4 runs complete with 0% error rate
- [ ] Results JSON written to `/results/` and synced to blueprint
- [ ] Cross-model comparison table updated in `reports/benchmark-results.md`

### Stage 7 — Readiness Audit
- [ ] All success criteria met or characterized
- [ ] No unresolved lessons with severity ≥ HIGH
- [ ] `mdc learn gemma-4 vllm --from blueprints/gemma4-hyperpod/lessons.md` executed

---

> **Note**: Operational artifacts (lessons, results, deployment notes) belong in
> `blueprints/gemma4-hyperpod/`, shared with the parent `gemma4-hyperpod.md` spec.
