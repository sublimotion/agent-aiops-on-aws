# Inference First Principles

> Agent-facing roofline reasoner. Load this at **config-selection time** (spec-writer choosing parallelism/hardware; infra-deployer/autoresearch-runner before a run) to predict the bottleneck regime and what to optimize — *before* copying a config from a similar blueprint.
>
> This is the terse operational form. Full derivations, worked numbers, and the disagg analysis live in `domains/gpu-serving/PRACTITIONER_GUIDE.md` §0. Origin: arXiv:2211.05102 (Pope et al.).

## Knowledge tiers — never conflate them

A claim's half-life decides how you use it. Conflating tiers is the carryover defect class (`carryover-auditor`).

- **[T1] universal** — physics; holds across any cloud/model/year. Assert plainly.
- **[T2] environment** — provider/platform truths (AWS, EKS, instance families). Hold for years; always state the qualifier ("on AWS…").
- **[T3] release** — version-pinned (engine/driver/AMI). Weeks–months. **Must carry `<!-- stack: … | validated: YYYY-MM-DD -->`** (see `tech-stack.md`). Treat any T3 rule >90 days unrefreshed as *suspect — verify*.

**The reasoner is pure T1.** It consults T2/T3 only to check the clean prediction survives the stack.

## The rooflines [T1]

```
T = max(t_compute, t_mem)
  t_compute = (B · N_active) / FLOPs
  t_mem     = (N_total + B · L_ctx · kv_bytes) / mem_bw     # weight fetch + KV fetch
latency floor: T ≥ N_total / mem_bw   (must read all weights once)
```

- **Prefill = compute-bound** (one matmul over all prompt tokens; FLOPs are the constraint).
- **Decode = bandwidth-bound** (reads whole model + KV per token; AI ~1–2 FLOP/byte; tensor cores idle).

## Decode-bytes decomposition — the #1 error guard [T1]

```
decode bytes/token = active-weight read (N_active·dtype) + KV read (kv_bytes·dtype)
```

Attention tricks (GQA/MLA/Mamba/sparse) shrink **only the KV term**. A 32B-active MoE in FP8 reads ~16 GB weights/token → ~95% of decode bytes even with MLA. **"It's sparse so it's not the bottleneck" is false** until you confirm which term dominates.

## Optimal batch [T1]

```
B* ≈ 300 × sparsity        sparsity = N_total / N_active
```

Run 2–3× B* in practice. Depends only on sparsity, not model scale. The ~300 is FLOP/byte at the precision used — spot-check per precision **[T3]**. (DeepSeek-V3: sparsity≈8 → B*≈2,400.)

## Decision procedure (run in order)

1. **Which roofline?** Compare workload arithmetic intensity to machine balance (`FLOPs ÷ mem_bw`, ~300–630 FLOP/byte low-precision **[T3]**):

   | Regime | Optimize | No-op lever |
   |--------|----------|-------------|
   | bandwidth-bound (decode) | HBM bandwidth, shrink bytes/token, bigger batch | more FLOPs / bigger FP4 |
   | compute-bound (prefill, big-batch) | FLOPs, FP4/FP8, GEMM kernels | more bandwidth |
   | capacity-bound (won't fit) | more HBM/GPU, more GPUs, quantize, pipeline | faster kernels |
   | launch-bound (small model; SM~50/HBM~15/tensor~11 at SLO knee) | fusion, CUDA graphs, megakernels (**software**) | **a bigger chip** |

2. **B\*, and does target concurrency reach it?** Below B* the cost/token curve is still falling (wasted weight fetches). Low-QPS/"slow mode" never gets cost-efficient — compute & KV don't amortize.

3. **Fits + saturates ONE node?**
   ```
   yes → replicas + chunked prefill   (~95% of cases; disagg = over-engineering)
   no (forced to 2nd node: big weights / 100K+ ctx prefill / QPS) → only THEN consider disagg
   ```
   Parameter count is NOT the axis — a 1T MoE (32B active) fits one node. "Forced to a 2nd node" predicts everything. **[T2] AWS caveat:** EFA is SRD not true RDMA; a TCP fallback turned 355 ms TTFT into 10+ s — the moment you're forced cross-node is often when you lack the fabric disagg needs.

4. **Pipelining? Almost never for inference. [T1]** Solves weight capacity (already a surplus on a Blackwell rack) but **can't shard KV** (P stages → P micro-batches in flight → saving cancels). Recipe: **max expert-parallelism to scale-up domain size, then ~no pipelining** (DeepSeek's published setup).

5. **Scale-up domain size = bandwidth, not capacity [T1].** Bigger NVLink domain → parallel weight load → lower decode latency, longer context. Interconnect is a lever to exploit, not a bottleneck to fear.

## Parallelism for high-concurrency MoE — TP4+DP2 vs TP8 single-node sweep

**For high-concurrency MoE serving on a single 8-GPU node**, always sweep **TP4+DP2** (two TP4 replicas) vs **TP8** (one oversized replica) before accepting a throughput ceiling. TP8 funnels all concurrency into one batch that schedules inefficiently at high load; TP4+DP2 distributes the load across two smaller, better-amortized batches.

**[measured] Evidence**:
- **Kimi K2.6-NVFP4, B300 sm_103, 2026-06-17**: TP4+DP2 beat TP8 by **+19-25%** throughput (2,569→3,067 @ c=256, 2,516→3,138 @ c=512) AND lower latency (185s→142s p50), with 0 errors through c=1024. The "single-node ceiling ~2,500 tok/s" session-1 found was a TP8 artifact — true ceiling ~3,190 at TP4+DP2.
- **GLM-5.2-FP8, B300 sm_103, SGLang v0.5.13, 2026-06-27**: TP4+DP2 beat TP8 (T2-prefixcache baseline) by **+28%** throughput at the c256 knee (6,025→7,728 tok/s) AND lower TTFT (10.7s→6.66s p95). Comfortable operating knee at c320 = **9,271 tok/s**, TTFT p95 8.1s; certified to c384 (9,900 tok/s, p95 14.9s) with a distributed driver. Regime: coding-agent workload, 12K byte-identical prefix, 92% prefix-cache hit, decode-bound post-cache.

**Mechanism** ([T1], batch scheduling): TP8 admits all concurrent requests into one giant batch; at high c (e.g., c=256-512 for a ~750B-1T MoE), the batch size exceeds the scheduler's sweet spot and kernels under-amortize. TP4+DP2 splits the concurrency → 2 batches of ~128-256 each, which amortize better. The tradeoff is smaller per-replica KV pool (TP4+DP2: half the tokens/replica vs TP8), but at high-concurrency operating points that's offset by better batch utilization. Prefix cache hit stays high (~92% on GLM-5.2) in both layouts because each replica holds the full 12K prefix in its own radix cache.

**When to apply**: This is **single-node, high-concurrency MoE only** — not necessarily true for dense models (which don't have the huge routing/scheduling overhead), not for low-concurrency (where TP8's bigger KV pool helps), not for multi-node (where TP8 across nodes is a different regime). The effect was measured on **753B-FP8 and 1T-NVFP4 MoEs (~32-40B active)** at **c=128-512** with **74-92% prefix-cache hit** — exactly the coding-agent RAG workload regime. For a different regime (streaming code-gen / low cache / lower concurrency), TP8 may still win.

**Decision heuristic**: If your model is MoE, your workload is high-concurrency (c≥128), and you're fitting on one 8-GPU node → **always benchmark TP4+DP2 vs TP8** before promoting the config to production. Don't assume "more TP is always better" — batch scheduling is non-monotonic. If TP4+DP2 loses or ties, note the regime and the result (so the next deployment knows).

**Cross-blueprint evidence**: Both B300 occurrences (Kimi K2.6-NVFP4 2026-06, GLM-5.2-FP8 2026-06-27) landed +19-28%. The mechanism (batch-size amortization) is [T1]; the breakpoint (at what concurrency TP4+DP2 crosses over) is [T3] and regime-dependent. The lesson elevated here is the **sweep discipline** ("test both before declaring a ceiling"), not a fixed config. An earlier B300 INT4-QAT occurrence (kimi-k2.6-spec L20, 2026-04-22) also saw TP4+DP2 win.

## Closing rule — output a prediction, not a deploy command

Every conclusion ends:

> *"Should be `<regime>`-bound on `<hardware>` → optimize `<lever>`. **Confirm with `nvidia-smi dmon` / benchmark sweep before trusting.**"*

If measurement contradicts the prediction, you found a T2/T3 quirk → write a `lessons.md` entry. Never distrust the T1 physics; distrust the stack. Verification commands: PRACTITIONER_GUIDE §2 (preflight), §9 (sweep), §10 (regime-confirmation queries).
