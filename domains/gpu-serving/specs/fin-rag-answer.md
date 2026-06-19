# Fin RAG Answer Extraction — Nemotron-3-Super on B200

## Status: DRAFT (2026-06-11)

## Overview

Use case 1 of a two-part GenAI workload assessment for a customer-support
agent (Intercom Fin-style). This is the **most complex component**: at the end
of retrieval, the model is handed a large system prompt + 25-45 guidelines +
~10 retrieved passages + multi-turn chat history, and must **stream** a
grounded, text-message-style answer directly to the end user.

Deploy **NVIDIA Nemotron-3-Super-120B-A12B** (hybrid Mamba-2 + LatentMoE +
Select-Attention, 120B total / 12B active) on a single **p6-b200.48xlarge**
(8× B200, NVSwitch). Benchmark **both precisions** (BF16 and FP8) across a
**full optimization grid** — kernel tuning, prefix caching, speculative
decoding, and TP/disagg layouts — to find the cheapest config that holds the
streaming-answer SLO at the customer's traffic.

**Workload card**: [`standards/benchmark-commons/workloads/fin-support.yaml`](../../../standards/benchmark-commons/workloads/fin-support.yaml)
— real 5,000-prompt corpus seed + profile-matched synthetic augmentation.

### Measured workload (production tokenizer telemetry)

| Dimension | p50 | p90 | Notes |
|-----------|-----|-----|-------|
| Input (ISL) | 8,823 | 11,952 | system header + guidelines + ~10 sources + history |
| Output (OSL) | 243 | 415 | short, "feels like a text message" |

- **temperature 0.0 in the corpus**, `max_tokens` cap 2,000 (real outputs use ~5-8× less).
- **~3,050-token system header is byte-identical across 94% of requests**; ~89% of request *pairs* share it; ~27% of all KV blocks are repeats of an already-seen prefix block.
- This is a **PREFILL-DOMINATED** workload (~9K in / ~300 out): prefill throughput and queueing at 130 concurrent dominate E2E, not decode.

### Customer SLO & traffic

| Target | Value |
|--------|-------|
| E2E p50 | ≤ 6,500 ms |
| E2E p90 | ≤ 9,500 ms |
| Peak RPS | 25 |
| Average RPS | 14 |
| Target concurrency | 130 |

---

## Components

### 1. Compute
- **Platform**: EKS 1.32
- **GPU Nodes**: p6-b200.48xlarge (8× B200 183 GB HBM3e, NVSwitch NVL5+) via capacity blocks
- **System Nodes**: m6i.xlarge (control plane, Dynamo etcd/NATS if disagg used)
- **Region**: us-east-2 (B200 capacity block availability)
- **AMI**: amazon-eks-node-al2023-x86_64-nvidia-1.32 (AL2023 required for Fabric Manager on NVL5+)

### 1a. GPU & NCCL Pre-Flight
Standard pre-flight (template Stage 1a + Stage 4a). B200 NVSwitch topology proven from GLM-5 deployments. Run `gpu-infra` MCP `discover_cluster` + `check_gpu_health` + `run_nccl_test` before serving. NIXL disagg requires NVSwitch — verify NVLink topology before disaggregated runs.

### 2. Model

| Precision | Checkpoint | Disk | Notes |
|-----------|-----------|------|-------|
| BF16 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | ~240 GB | fits 8× B200 (1,464 GB) easily; no DeepGEMM JIT wait |
| FP8 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | ~124 GB | native; DeepGEMM FP8 MoE kernels; ~15 min JIT cold start |

- **Architecture**: hybrid interleaved Mamba-2 + LatentMoE + Select-Attention, `architectures: ["MambaForCausalLM"]`. 120B total, 12B active.
- **Context**: up to 1M (256K default) — far exceeds the ~12K we need.
- **Deployment card**: run `mdc get nemotron-3-super --engine vllm` and `mdc prs nemotron-3-super` before deploying.

### 3. Serving Stack

| Engine | Version guidance | Why |
|--------|------------------|-----|
| vLLM | **0.18.1 known-clean baseline**; AVOID 0.19.0 (garbled #39223), 0.15.x (FP8 regression #34356), 0.20.0 (long-ctx #41565), 0.22 (#44022). | primary engine |
| SGLang | recent cu13 nightly (#20470 spec-decode merged) | comparison / disagg |
| Dynamo | FP8 disagg recipe in-tree since PR #7216 (merged 2026-03-11); #9376 (hybrid KV routing block-size) still OPEN | disagg orchestration |

> **Version sensitivity is a hard gate** — pin and smoke-test output quality before any benchmark run (see Stage 0 below).

### 4. Storage
- **FSx Lustre** PERSISTENT_2 (holds pre-staged BF16 + FP8 weights) → **NVMe** `/mnt/nvme` serving tier. FP8 ~124 GB copies in ~15-20s; BF16 ~240 GB proportionally longer.

---

## Stage 0 — Precision & Version Pre-Flight (fail-closed)

Run BEFORE any benchmark. These guard the two biggest known risks: precision-artifact availability and vLLM version regressions.

- [ ] **Both checkpoints resolve on HF**: `nvidia/...-BF16` and `nvidia/...-FP8` download + load. (Confirmed published 2026-06; re-verify access.)
- [ ] **Confirm FP8 is FP8** (not silently NVFP4) — check `config.json` quant block; record actual quant format.
- [ ] **vLLM version smoke**: with the pinned version, generate 10 outputs on real Fin prompts; assert coherent (no garble) — guards #39223-class regressions.
- [ ] **Serving-config resolver** (template Stage 0c): `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/fin-rag-answer/benchmark.yaml --corpus-root .` exits 0.
- [ ] If FP8 MoE: confirm `moe_intermediate_size / TP % 128 == 0` for fine-grained FP8 (block_n=128) at each TP under test.

---

## Optimization Grid (FULL)

Both precisions × kernel knobs × spec-decode × parallelism. Staged P0→P2 so it stops early once SLO is met, but the full grid is in scope.

### Axis A — Precision
`bf16`, `fp8`

### Axis B — Kernel / engine tuning
| Knob | Values | Why it matters here |
|------|--------|---------------------|
| Chunked prefill (`max-num-batched-tokens`) | sweep {2048, 4096, 8192, 16384} | **#1 lever** — 9K input × 130 concurrent → head-of-line blocking without it |
| Attention backend | `TRITON_ATTN` (hybrid-safe default), FlashInfer (where select-attn permits) | select-attention may not be supported by all backends |
| Mamba-2 kernels | causal-conv1d + chunked SSM scan (engine default vs tuned) | hybrid-specific hot path |
| FP8 KV cache | `fp8_e4m3` on attention layers (fp8 precision only) | KV footprint; not the bottleneck but free headroom |
| CUDA graph / torch.compile | on (capture), measure cold-start cost | ~15 min DeepGEMM JIT on FP8 first start |
| `max-num-seqs` | {64, 130, 256} | batch admission at target concurrency |

### Axis C — Prefix caching (the RAG win)
- **Mamba-2 automatic prefix caching is MERGED** (vLLM #26201 / PR #25752): `mamba_cache_mode="all"` caches attention-layer KV **and** Mamba state for the ~3,050-token shared system header.
- [ ] Measure prefix hit rate cold vs warm; quantify TTFT improvement on the shared header.
- [ ] **Validate at TP>1** — upstream marks TP>1 prefix caching "not yet tested" (#26201). Run at both TP layouts.
- [ ] **Validate prefix-cache + MTP together** — MTP+prefix-cache crash (#39809) was real, now fixed; confirm on pinned version.

### Axis D — Speculative decoding
Secondary lever (workload is prefill-dominated, ~300-token output), but the **real prompts make acceptance measurable** rather than synthetic-meaningless.
| Variant | Config | Note |
|---------|--------|------|
| n-gram / prompt-lookup | draft from the ~9K-token context | RAG answers copy source spans verbatim → expect high acceptance |
| native MTP | model's shared-weight MTP layers | no separate draft model to host |
- [ ] **Measure acceptance rate at temperature=1.0** (model-recommended temp lowers acceptance vs greedy) — report speedup that survives at temp=1.0.

### Axis E — Parallelism / disaggregation
| Config | Layout | Note |
|--------|--------|------|
| `agg-tp2-x4` | 4 workers × TP=2 | aggregated baseline |
| `agg-tp4-x2` | 2 workers × TP=4 | check TP-divisibility for FP8 MoE |
| `disagg-4p4d` | 4 prefill + 4 decode | **now viable** — vLLM #40017 merged homogeneous (#36687) + heterogeneous (#37635) TP + conv-state layout (#37416); only Mamba PC-in-PD (#42554) open |
| `disagg-2p6d` | 2 prefill + 6 decode | decode-light here (short output) — likely prefill-heavy split wins |

> Given the prefill-dominated shape, expect a **prefill-heavy** disagg split (more prefill workers) or a well-tuned aggregated config to win — opposite of decode-heavy chat workloads.

---

## Benchmark Plan (staged)

```
P0 ~1.5 hr · P1 ~4 hr · P2 ~2 hr  (≈7.5 B200 hrs, full grid)
```

### P0 — Baseline both precisions (MUST HAVE)
| Step | Precision | Config | Load | Goal |
|------|-----------|--------|------|------|
| 0a | fp8 | agg-tp2-x4, prefix-cache on | fin-support @ conc 8, 130 | does aggregated hold E2E p50 6.5s? |
| 0b | bf16 | agg-tp2-x4, prefix-cache on | fin-support @ conc 8, 130 | precision quality/throughput tradeoff |
| 0c | winner | + chunked-prefill best value | fin-support @ 130 | confirm prefill knob impact |

**Gate**: E2E p50/p90 vs 6.5s/9.5s at 130 concurrent; prefix hit rate on shared header > 0 and TP>1-validated.

### P1 — High-impact knobs (SHOULD HAVE)
- P1a: chunked-prefill sweep {2048..16384} × `max-num-seqs` at conc 130.
- P1b: spec-decode (n-gram + MTP), acceptance @ temp=1.0, both precisions.
- P1c: FP8 KV cache on/off (fp8); attention-backend TRITON_ATTN vs FlashInfer.
- P1d: full fin-support concurrency sweep [8, 32, 128, 512] at best config.

**Gate**: identify the cheapest config meeting SLO + headroom to peak. Quantify each knob's E2E contribution.

### P2 — Parallelism & disagg (NICE TO HAVE, only if P1 lacks headroom)
- P2a: agg-tp4-x2 vs agg-tp2-x4.
- P2b: disagg-4p4d vs disagg-2p6d (prefill-heavy variants) — does disagg flatten TTFT under load vs aggregated?

**Gate**: best parallelism layout documented with rationale; disagg speedup (if any) quantified.

---

## Success Criteria

| Metric | Target | Condition |
|--------|--------|-----------|
| **E2E p50** | ≤ 6,500 ms | fin-support @ 130 concurrent (headline) |
| **E2E p90** | ≤ 9,500 ms | fin-support @ 130 concurrent |
| Sustains peak | no SLO breach @ 25 RPS | with 130-concurrent headroom |
| Prefix hit rate (shared header) | measured, TP>1-validated | cold vs warm TTFT delta reported |
| Spec-decode acceptance | measured @ temp=1.0 | report net E2E/TPOT effect |
| Precision recommendation | bf16 vs fp8 documented | quality, throughput, $/1M tokens |
| Error rate | < 0.1% | all concurrency levels |
| No OOM | pass | at conc 512 (sweep ceiling) |

---

## Non-Requirements
- 1M / 256K context benchmarking (workload tops out ~12K).
- HiCache/LMCache CPU offload (KV headroom is not the bottleneck on 8× B200).
- Multi-node (single p6-b200.48xlarge).
- Tool-calling correctness (this component only extracts the final answer; tool use is upstream).
- Production autoscaling.

## Known Limitations (refreshed 2026-06-11)
1. **vLLM version sensitivity is real** — pin 0.18.1 baseline; avoid 0.19.0 / 0.15.x / 0.20.0 / 0.22. Smoke-test output before benchmarking.
2. **Reasoning parser** now built-in `--reasoning-parser nemotron_v3` (no HF plugin) but buggy: #39103 (`--reasoning-config` nulls content), #39581 (`reasoning_effort` ignored), both OPEN. This use case likely runs **reasoning OFF** (latency-sensitive extraction) — confirm.
3. **Prefix caching at TP>1 "not yet tested" upstream** (#26201) — must validate, don't assume.
4. **Temperature=1.0** is model-recommended — lowers spec-decode acceptance; do not benchmark spec-decode greedy and claim the number for production.
5. **DeepGEMM JIT cold start ~15 min (FP8)** — set readiness probe `initialDelaySeconds >= 900s`; bf16 avoids this.
6. **block_reuse / KV-aware routing** still has hybrid block-size edge cases (Dynamo #9376 OPEN) — if using Dynamo routing, prefer round-robin until validated.
7. **No published B200 latency benchmark** for this model — closest real data is 8× RTX PRO 6000 (150ms TTFT, 3,215 tok/s burst, weaker HW). Treat aggregated-sufficiency as a hypothesis to validate in P0.

> Full upstream status in memory: `nemotron3_super_serving.md`. Old PR numbers in the prior `nemotron-super.md` spec (#19158/#18414/#19045/#19254) were fictional; real tracking issue is vLLM #40017.

---

## Cost Considerations

| Resource | Estimated | Notes |
|----------|-----------|-------|
| p6-b200.48xlarge capacity block | ~$85-140/hr | GPU dominates |
| Benchmark budget | ~7.5 hrs × $90-145 = **$675-1,090** | full grid, both precisions |

Primary cost question for the customer: **does FP8 hit SLO on fewer GPUs than BF16**, and what's the $/1M-token delta at the 130-concurrent operating point.

---

> Operational artifacts (lessons, results, benchmark.yaml sidecar) live in
> `blueprints/fin-rag-answer/`, not in this spec.
