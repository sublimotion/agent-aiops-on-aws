# Qwen3-Next-80B on g7e.48xlarge — Benchmark Spec Addendum

## Status: DRAFT (2026-02-25)

## Parent Spec

See [`qwen3-next.md`](./qwen3-next.md) for full model details, workload definitions, and metrics.

This addendum documents only the deltas for running on g7e.48xlarge (Blackwell GB202).

---

## Compute Delta

| Property | p5en.48xlarge (parent) | g7e.48xlarge (this spec) |
|----------|----------------------|--------------------------|
| GPU | 8x H200 (141 GB HBM3e) | 8x RTX PRO Server 6000 (96 GB GDDR7) |
| Architecture | Hopper (sm_90) | Blackwell GB202 (sm_100) |
| On-demand cost | $63.30/hr | $33.14/hr |
| Region / AZ | us-east-2c | us-east-2a |
| Interconnect | EFA v2 | NVLink (no EFA needed) |
| AMI type | AL2_x86_64_GPU | AL2023_x86_64_NVIDIA |
| NVMe | 8x 3.84 TB | Instance NVMe (generic detection) |

**Cost advantage**: g7e is 48% cheaper on-demand. If throughput is within 2x of p5en, g7e wins on $/1M tokens.

---

## Scope

Winner config only from p5en benchmarks: **vLLM TP=4, FP8, `--enable-prefix-caching`**.

No SGLang, dp8-ep, MTP, or cpu-offload testing. Those can be added in a follow-up if g7e proves viable.

---

## Benchmark Phases

| Phase | What | Runs | Purpose |
|-------|------|------|---------|
| G0 | Smoke: QPS 0.5, 1024/512 random | 3 | Model loads, FP8 works on Blackwell |
| G1 | QPS sweep: 1, 2, 4, 8 at 1024/512 | 3 each | Latency-throughput curve |
| G2 | Context: 4K, 32K, 64K at QPS 0.5 | 3 each | Context scaling on 96 GB VRAM |

---

## Infrastructure Changes

- **Separate blueprint**: `domains/gpu-serving/blueprints/qwen3-next-g7e/` (own Terraform state)
- **Reuse existing S3 model bucket**: No new model upload needed; FSx DRA points to same bucket
- **FSx**: SCRATCH_2 at 1200 GiB (sufficient for benchmarks, cheaper than PERSISTENT_2)
- **EFA available** but not needed for intra-node TP (g7e is PCIe, not NVLink)
- **`max_model_len`**: Start at 131072, reduce to 65536 if 96 GB VRAM is insufficient

---

## VRAM Budget (96 GB per GPU, TP=4)

| Component | Estimate |
|-----------|----------|
| Model weights (FP8, TP=4) | ~20 GB/GPU |
| KV cache (0.92 util) | ~70 GB/GPU |
| Activations + overhead | ~6 GB/GPU |
| **Total** | ~96 GB/GPU |

Tight fit. If 131072 context fails, fall back to `max_model_len=65536`.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| cu130 image lacks sm_100 support | Medium | Build with cu131 base or vLLM nightly |
| 96 GB VRAM insufficient for 131K | Medium | Reduce `max_model_len` to 65536 |
| AL2023_x86_64_NVIDIA AMI lacks Blackwell drivers | Medium | Check AMI driver version; use custom AMI |
| FP8 block_k on Blackwell | Low | FP8 is native on Blackwell; fall back to BF16 |

---

## Success Criteria

1. G0 smoke test passes (model loads, FP8 works on Blackwell sm_100)
2. G1 QPS sweep produces comparable format to p5en results
3. Cost comparison table: $/1M output tokens for g7e vs p5en at SLO-max QPS
4. `lessons.md` captures Blackwell-specific findings
