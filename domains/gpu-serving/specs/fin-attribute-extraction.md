# Fin Attribute Extraction — Qwen3.6-35B-A3B

## Status: DRAFT (2026-06-11)

## Overview

Use case 2 of the two-part GenAI workload assessment. Given a **textual
definition** of each attribute plus a **conversation transcript**, the model
emits a short structured value (label / enum / short span). This is a
**classification-style** call, not a generative one — output is ~20 tokens and
cost is almost entirely **prefill**.

Deploy **Qwen/Qwen3.6-35B-A3B** (MoE, 35B total / **3B active**) in **BF16 and
FP8**, optimised for a tight **1s/2s E2E SLO at high request rate** (peak 70
RPS). Because the output is ~20 tokens, the levers are prefill throughput,
batching, and FP8 MoE kernels — **speculative decoding is irrelevant here**
(there is essentially no decode to accelerate).

**Workload card**: [`standards/benchmark-commons/workloads/fin-attribute-extraction.yaml`](../../../standards/benchmark-commons/workloads/fin-attribute-extraction.yaml)

### Measured workload (production tokenizer telemetry)

| Dimension | p50 | p90 | Notes |
|-----------|-----|-----|-------|
| Input (ISL) | 1,654 | 4,690 | attribute definitions + conversation transcript |
| Output (OSL) | 20 | 23 | a structured value, not a paragraph |

### Customer SLO & traffic

| Target | Value |
|--------|-------|
| E2E p50 | ≤ 1,000 ms |
| E2E p90 | ≤ 2,000 ms |
| Peak RPS | 70 |
| Average RPS | 40 |
| Target concurrency | 50 |

---

## Components

### 1. Compute
- **Platform**: EKS 1.32 (or single-node bare serving for benchmark).
- **GPU**: Model is small (3B active, ~70 GB bf16). Candidate instances:
  - **g7e.24xlarge** (4× RTX PRO 6000 Blackwell sm_120, 96 GB each, PCIe) — likely sufficient and cheapest.
  - **p6-b200 slice** (TP≤4) if g7e prefill throughput can't hold 70 RPS at the 1s/2s SLO.
- **Decision gate**: start on g7e; escalate to B200 only if P1 shows the SLO breaking before 70 RPS.

### 1a. GPU Pre-Flight
Standard pre-flight. **g7e is PCIe-only (no NVLink)** — but this is vLLM inference (custom allreduce, NCCL-Blackwell bug does NOT affect inference). Multi-GPU TP works for serving.

### 2. Model

| Precision | Source | Note |
|-----------|--------|------|
| BF16 | `Qwen/Qwen3.6-35B-A3B` | official; ~70 GB; tensor type bf16 |
| FP8 | **community quant — Stage 0 must select/verify** | **no official FP8 published** (~460 community quants exist) |

- **Architecture**: MoE, 256 experts (8 routed + 1 shared), `moe_intermediate_size=512`. Hybrid Gated-DeltaNet → MoE + Gated-Attention → MoE. **Multimodal (vision encoder present but unused here).**
- **Active params**: 3B → very fast decode, low VRAM working set.
- **Context**: 262K native (we need ≤5K).
- **Parsers**: `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` (tool use not needed here).
- **Deployment card**: `mdc get qwen3.6-35b --engine vllm` before deploying.

### 3. Serving Stack
- **vLLM** or **SGLang** (both support the model). Pin a version and smoke-test.
- **CRITICAL — disable thinking**: model defaults to `<think>` reasoning mode. With a 20-token target and 1s SLO, thinking is fatal. Set `chat_template_kwargs={"enable_thinking": False}` on every request and **verify no `<think>` tokens** in output during Stage 0.

---

## Stage 0 — Pre-Flight (fail-closed)

- [ ] **BF16 checkpoint loads** and serves.
- [ ] **FP8 variant selected & verified**: pick a community FP8/NVFP4 quant (or produce one); smoke-test output quality vs bf16 on real extraction examples. Record provenance.
- [ ] **Thinking disabled**: `enable_thinking=False` confirmed — zero `<think>` tokens, output ~20 tokens.
- [ ] **FP8 MoE TP-divisibility**: `moe_intermediate_size(512) / TP % 128 == 0`. ✅ TP1/2/4 (512/4=128). ❌ **TP8 (512/8=64) — do NOT use TP8 for FP8.** Cap TP≤4 for FP8.
- [ ] Serving-config resolver (template Stage 0c) exits 0.

---

## Optimization Grid

Output is ~20 tokens → **prefill + batching + admission**, not decode.

### Axis A — Precision
`bf16`, `fp8` (TP≤4 for fp8 per divisibility).

### Axis B — Prefill & batching knobs
| Knob | Values | Why |
|------|--------|-----|
| Chunked prefill (`max-num-batched-tokens`) | sweep {2048, 4096, 8192} | pack many short-output requests; key at 70 RPS |
| `max-num-seqs` | {50, 128, 256} | admission at 50+ concurrent |
| Attention backend | FlashInfer / TRITON | DeltaNet + Gated-Attention hybrid — verify support |
| FP8 KV cache | on/off (fp8) | tiny KV (short ctx) — minor |
| CUDA graph / torch.compile | on | decode is trivial; capture helps per-request overhead |
| TP layout | TP1, TP2, TP4 | 3B-active is small; lower TP may win on comms overhead |

### Axis C — Spec decode
**OUT OF SCOPE.** 20-token output → no meaningful decode to accelerate. Explicitly not benchmarked.

### Axis D — Prefix caching
- **Optional / conditional**: attribute *definitions* may be shared across requests (cacheable prefix), but the conversation transcript dominates ISL and is unique. **If definitions are confirmed shared**, model them as a shared prefix and measure hit rate; otherwise prefix caching has little value here. Confirm with customer whether the definition block is stable.

---

## Benchmark Plan (staged)

```
P0 ~0.5 hr · P1 ~1.5 hr · P2 ~0.5 hr  (small model, cheap)
```

### P0 — Baseline both precisions
| Step | Precision | Config | Load | Goal |
|------|-----------|--------|------|------|
| 0a | bf16 | TP2, thinking OFF | fin-attribute-extraction @ qps 40 | hold E2E p50 1s at avg RPS? |
| 0b | fp8 | TP2/TP4, thinking OFF | qps 40 | precision tradeoff |

**Gate**: E2E p50/p90 vs 1s/2s at 40 RPS. Confirm output ~20 tokens, no thinking.

### P1 — Find the SLO knee (QPS sweep)
- P1a: full `fin-attribute-extraction` QPS sweep [20, 40, 70, 100] at best P0 config.
- P1b: chunked-prefill × `max-num-seqs` at 70 RPS (peak).
- P1c: TP1 vs TP2 vs TP4 — comms overhead vs prefill parallelism.

**Gate**: **`sustained_rps_at_slo`** — max RPS holding E2E p90 ≤ 2s. Must be ≥ 70 (peak). If SLO breaks before 70 RPS on g7e → escalate to B200 slice.

### P2 — Instance/precision recommendation
- P2a: if g7e insufficient, repeat P1b on B200 slice.
- P2b: $/1M-request comparison bf16 vs fp8, g7e vs B200.

---

## Success Criteria

| Metric | Target | Condition |
|--------|--------|-----------|
| **E2E p50** | ≤ 1,000 ms | @ 40 RPS (avg) |
| **E2E p90** | ≤ 2,000 ms | @ 70 RPS (peak) — headline |
| sustained_rps_at_slo | ≥ 70 | max RPS within 2s p90 |
| Output length | ~20-23 tokens | no `<think>` leakage |
| Precision recommendation | bf16 vs fp8 | quality + $/req |
| Error rate | < 0.1% | all rates |

---

## Non-Requirements
- Speculative decoding (no decode to accelerate).
- Long context (workload ≤ 5K; model supports 262K).
- Vision / multimodal (text-only extraction).
- Tool calling (extraction returns a value, not tool calls).
- Reasoning / thinking mode (explicitly disabled).

## Known Limitations
1. **No official FP8** — must select/verify a community quant in Stage 0; quality not vendor-guaranteed.
2. **Thinking mode default ON** — must disable per-request; a regression that re-enables it blows the 1s SLO instantly.
3. **FP8 MoE TP8 forbidden** — `512/8=64` violates block_n=128 divisibility; cap TP≤4 for FP8.
4. **g7e is PCIe-only** — fine for vLLM inference (custom allreduce), but multi-GPU TP comms slower than NVLink; factor into TP-layout sweep.
5. **Tight SLO at high RPS** — 1s p50 leaves little margin; queueing at peak is the main risk, hence the QPS-sweep design.

---

## Cost Considerations

| Resource | Estimated | Notes |
|----------|-----------|-------|
| g7e.24xlarge | ~$ (spot-dependent) | likely sufficient; cheapest path |
| p6-b200 slice | ~$85-140/hr full node | only if g7e can't hold peak |
| Benchmark budget | ~2.5 hrs | small model, fast iterations |

Primary question: **can the cheapest instance (g7e) hold E2E p90 ≤ 2s at 70 RPS**, and does FP8 buy enough throughput to matter at 3B active params.

---

> Operational artifacts live in `blueprints/fin-attribute-extraction/`, not in this spec.
