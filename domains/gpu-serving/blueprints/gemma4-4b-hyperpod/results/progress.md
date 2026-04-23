# Gemma 4 E4B — HyperPod Deployment Progress

## Status: P0-P2 COMPLETE (2026-04-06)

### Infrastructure
- **Cluster**: llmd-inference-cluster (HyperPod EKS, inference-eks-v132)
- **Node**: ml.g5.4xlarge (1x A10G 24GB), us-east-1
- **Image**: vllm/vllm-openai:gemma4 (v0.18.2rc1.dev73)
- **Model**: google/gemma-4-E4B-it (ungated, ~8GB safetensors)

### Key Metrics
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| BFCL tool calling | ≥60% | **100%** (10/10) | ✓ PASS |
| W5 max QPS | ≥15 | ~2.8 actual QPS | ✗ Below target |
| W6 TTFT p50 @ 16K | <400ms | **148ms** | ✓ PASS |
| W1 multi-turn 10r TTFT p50 | <800ms | **59-113ms** | ✓ PASS |
| ITL p50 | <30ms | ~25ms (est from TPS) | ✓ PASS |
| Vision tasks | ≥3/5 | **4/4** (100%) | ✓ PASS |
| Code generation | ≥60% | **5/5** (100%) | ✓ PASS |

### W5 QPS Sweep
| QPS Target | QPS Actual | Success | TTFT p50 | TTFT p95 | TPS |
|------------|-----------|---------|----------|----------|-----|
| 0.5 | 0.44 | 15/15 | 101ms | 207ms | 40.7 |
| 2.0 | 1.5 | 39/40 | 151ms | 7147ms | 27.7 |
| 4.0 | 2.18 | 36/40 | 123ms | 158ms | 36.7 |
| 8.0 | 2.76 | 33/40 | 104ms | 158ms | 35.9 |

### W6 Long Context
| Input Length | TTFT p50 | TPS |
|-------------|----------|-----|
| 1K | 77-145ms | 30-34 |
| 4K | 75-101ms | 29-34 |
| 8K | 93-95ms | 25-30 |
| 16K | 148-150ms | 22-30 |

---

## H100 Session (2026-04-07) — Colocated on mistral-sm4-eks

### Infrastructure
- **Cluster**: mistral-sm4-eks (EKS 1.34) + mistral-sm4-hyperpod
- **Node**: ml.p5.48xlarge (8x H100 80GB NVSwitch), 1 GPU allocated via K8s device plugin
- **Image**: vllm/vllm-openai:latest (v0.19.0) + `pip install git+transformers`
- **Config**: BF16, TP1, TRITON_ATTN (forced by heterogeneous head_dim)
- **Note**: `vllm/vllm-openai:gemma4` image fails on H100 (subprocess NVML error). Latest image + git transformers works.

### P0 Smoke Tests: ALL PASS
- Basic generation, tool calling (structured), multi-turn memory

### P1v-a: QPS Sweep (input=2048, output=512, H100)

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | Output tok/s | Total tok/s | Peak Conc |
|-----|--------------|--------------|-------------|-------------|-------------|------------|-----------|
| 1.0 | 62 | 2,966 | 7.0 | 20 | 432 | 2,160 | 8 |
| 4.0 | 61 | 121 | 7.4 | 29 | 1,776 | 8,880 | 29 |
| 10.0 | 39 | 147 | 8.6 | 14 | 3,602 | 18,010 | 64 |
| 20.0 | 46 | 158 | 10.3 | 16 | 5,158 | 25,788 | 100 |

**Key finding**: H100 transforms this 4B model into a throughput monster — **5,158 output tok/s** at QPS=20, 14x faster than A10G. TTFT stays under 160ms even at QPS=20 (vs saturating at QPS=2.8 on A10G).

### P1v-b: Context Scaling (QPS=1.0, output=512, H100)

| Context | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Output tok/s |
|---------|--------------|--------------|-------------|-------------|
| 1,024 | 29 | 32 | 6.7 | 479 |
| 4,096 | 78 | 105 | 7.5 | 475 |
| 8,192 | 156 | 319 | 8.6 | 469 |
| 16,384 | 379 | 1,027 | 9.5 | 461 |

### A10G vs H100 Comparison

| Metric | A10G (g5.4xl) | H100 (p5.48xl) | Speedup |
|--------|---------------|----------------|---------|
| W6 TTFT p50 @ 16K | 148ms | 379ms | 0.4x (worse — TRITON_ATTN on H100?) |
| ITL p50 | ~25ms | 7-10ms | 2.5-3.5x |
| Max QPS at SLO | ~2.8 | >20 | >7x |
| Peak output tok/s | ~40 | 5,158 | ~129x |
| Total tok/s | ~36 | 25,788 | ~716x |

**Surprising**: TTFT is actually _slower_ on H100 at 16K context (379ms vs 148ms). Likely due to TRITON_ATTN backend (forced by heterogeneous head_dim) being less optimized for single-request prefill on H100 vs the A10G FlashAttention path. But throughput under concurrency is dramatically higher.

### P1v-c: Prefix Caching (H100)

| Prefix Len | TTFT p50 cached (ms) | TTFT p50 no cache (ms) | Cache Speedup |
|-----------|---------------------|----------------------|--------------|
| 4,096 | 45 | 78 | 1.7x |
| 16,384 | 98 | 379 | 3.9x |

### P2: Code Quality
- LRU Cache implementation: PASS — clean, correct, with docstrings and tests

---

### Deployment Lessons
1. **Model name is `gemma-4-E4B-it`** not `gemma-4-4b-it` — "E4B" = effective 4B
2. **Ungated** — no HF token required (unlike Gemma 3 which was gated)
3. **Requires `vllm/vllm-openai:gemma4` image** — standard v0.19.0 and v0.8.5 lack transformers support for `gemma4` model_type
4. **Heterogeneous head_dim** (256 local + 512 global) — vLLM forces TRITON_ATTN backend
5. **HyperPod disk pressure**: Previous image pulls can cause stale disk-pressure taints even with 50% free space. Fix: restart kubelet via `kubectl debug node/... -- chroot /host systemctl restart kubelet`
6. **HyperPod taints**: GPU nodes need tolerations for `sagemaker.amazonaws.com/RestrictedNode`, `nvidia.com/gpu`, and possibly `node.kubernetes.io/disk-pressure`
7. **Tool calling**: Works with `--tool-call-parser pythonic --enable-auto-tool-choice`. With `tool_choice: auto`, model outputs tools in content; use `tool_choice: required` for structured output
8. **Model load**: 15.16 GiB GPU memory, 53s load time, ~47s torch.compile
9. **KV cache**: 3.58 GiB available, 39K tokens capacity, ~6x concurrency at 32K context
10. **Vision**: Works with base64-encoded images. URL-based images may 403 depending on source (vLLM fetches server-side)
