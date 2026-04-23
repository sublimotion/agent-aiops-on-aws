# Gemma 4 31B — HyperPod Deployment Progress

## Status: P0-P1v COMPLETE (2026-04-07)

Visual report: [hyperpod-3model-visual-20260407.html](hyperpod-3model-visual-20260407.html)

### Infrastructure
- **Cluster**: mistral-sm4-eks (EKS 1.34) + mistral-sm4-hyperpod
- **Node**: ml.p5.48xlarge (8x H100 80GB NVSwitch), 2 GPUs allocated via K8s device plugin
- **Image**: vllm/vllm-openai:latest (v0.19.0) + `pip install git+transformers`
- **Model**: google/gemma-4-31B-it (ungated, ~59GB safetensors, 2 shards)
- **Config**: BF16, TP2, TRITON_ATTN (forced by heterogeneous head_dim 256/512)
- **Colocated with**: Mistral SM4 (2 GPUs, port 8001) + Gemma 4 E4B (1 GPU, port 8000)

### P0: Smoke Tests — ALL PASS
- Basic generation: PASS — coherent, accurate
- Tool calling (single): PASS — correct function + args with `pythonic` parser
- Tool calling (parallel): PASS — 2 parallel tool calls, correct JSON
- Multi-turn memory: PASS — recalls name and company

### P1v-a: QPS Sweep (input=2048, output=512)

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | Output tok/s | Total tok/s | Peak Conc |
|-----|--------------|--------------|-------------|-------------|-------------|------------|-----------|
| 1.0 | 192 | 2,374 | 17 | 152 | 470 | 2,352 | 20 |
| 4.0 | 509 | 10,817 | 33 | 313 | 1,086 | 5,430 | 93 |
| 10.0 | 2,396 | 22,221 | 34 | 450 | 1,182 | 5,910 | 100 |
| 20.0 | 5,005 | 27,085 | 34 | 591 | 1,188 | 5,938 | 100 |

**Key finding**: Saturates around QPS=4.0 with throughput plateauing at ~1,188 output tok/s (~5,938 total tok/s). At QPS=1.0, TTFT p50 is excellent (192ms) with ITL p50 of 17ms.

### P1v-b: Context Scaling (QPS=1.0, output=512)

| Context | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Output tok/s |
|---------|--------------|--------------|-------------|-------------|
| 1,024 | 109 | 181 | 16 | 438 |
| 4,096 | 376 | 1,180 | 18 | 424 |
| 8,192 | 990 | 3,204 | 27 | 388 |
| 16,384 | 1,788 | 1,809 | 19 | 69 (single-req) |

**Key finding**: TTFT scales superlinearly with context due to TRITON_ATTN. At 16K, single-request TTFT is 1.8s — well above the 500ms target. The TRITON_ATTN backend (forced by heterogeneous head_dim) is significantly slower for prefill than FlashAttention on H100.

### P1v-c: Prefix Caching (5 shared prefixes)

| Prefix Len | TTFT p50 cached (ms) | TTFT p50 no cache (ms) | Cache Speedup |
|-----------|---------------------|----------------------|--------------|
| 4,096 | 82 | 376 | 4.6x |
| 16,384 | 161 | 1,788 | 11.1x |

**Key finding**: Prefix caching is _dramatically_ effective — **11.1x speedup** at 16K prefix. This is the best prefix caching result across all models tested. The high prefill cost of TRITON_ATTN makes caching even more valuable. With caching, 16K prefix TTFT drops to just 161ms.

### Cross-Model Comparison (H100 TP2, same cluster)

| Metric | Gemma 4 31B | Mistral SM4 119B | Gemma 4 E4B (TP1) |
|--------|-------------|------------------|--------------------|
| Architecture | Dense 31B | MoE 119B (6.5B active) | Dense 4B |
| TTFT p50 @ 1K | 109ms | 93ms | 29ms |
| TTFT p50 @ 16K | 1,788ms | 539ms | 379ms |
| ITL p50 | 16-17ms | 5.9-20ms | 7-10ms |
| Max output tok/s | 1,188 | 2,160 | 5,158 |
| Peak total tok/s | 5,938 | 10,795 | 25,788 |
| Prefix cache 16K | 11.1x | 5.2x | 3.9x |
| VRAM/GPU | ~38 GiB | ~57 GiB | ~15 GiB |

**Notable**: Gemma 4 31B has the worst TTFT at long context due to TRITON_ATTN + dense architecture (all 31B params active). But it has the best prefix cache speedup (11.1x) precisely because the uncached prefill is so expensive.

---

### Success Criteria Status

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| BFCL accuracy >= 70% | P0 | 4/4 smoke (full BFCL TBD) | PASS (smoke) |
| W6 TTFT p50 at 16K < 500ms | P1 | 1,788ms | FAIL |
| vllm bench ITL p50 < 50ms | P1v | 17ms (best) | PASS |
| QPS at SLO >= 2.0 | P1v | ~1.0 (TTFT p99 > 1s at QPS=1) | MARGINAL |
| Prefix cache speedup >= 2x | P1v | 11.1x | PASS |

---

### Deployment Lessons
1. **Ungated** — no HF token required (same as E4B)
2. **Image**: `vllm/vllm-openai:latest` + `pip install git+transformers` works (same as E4B)
3. **TRITON_ATTN is the bottleneck**: Heterogeneous head_dim forces slow attention backend. TTFT at 16K is 1.8s vs 539ms for Mistral SM4 (MLA + FlashAttention) and 379ms for E4B. Dense 31B + TRITON_ATTN = worst TTFT
4. **Prefix caching is essential**: 11.1x speedup at 16K makes this model viable for RAG/agentic workloads (161ms with cache vs 1,788ms without)
5. **Tool calling works with `pythonic` parser**: Same as E4B — no need for `gemma4` parser
6. **Model load**: 30.16 GiB GPU memory (TP2), 20s weight load, ~2.5 min total startup (compile + CUDA graphs)
7. **Port 8002** avoids conflict with E4B (8000) and Mistral SM4 (8001)
8. **Generation defaults from model config**: `temperature=1.0, top_k=64, top_p=0.95` — override if needed
