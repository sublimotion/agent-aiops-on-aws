# GLM-5.2 on B200 — Full Optimization-Sweep Benchmark Requirements

## Status: DRAFT (2026-06-23)

## Overview

Benchmark `zai-org/GLM-5.2-FP8` on a **single 8-GPU B200 node**, sweeping the full T0–T5
optimization stack, on a **coding-agent workload** (primary) plus a **31K long-input variant**
(carried from the kimi-k2.6-nvfp4 customer profile). Run **vLLM and SGLang head-to-head** at
matched precision/workload and pick the most performant at the operating point, then complete the
lever sweep on the winner.

GLM-5.2 is the successor to GLM-5 (`glm_moe_dsa` family). It is a ~753B-total MoE (256 routed + 1
shared expert, top-8, ~40B active) with **MLA + DeepSeek Sparse Attention (DSA)**, a native **MTP**
speculative head, and a **1M-token** context window. This benchmark establishes the production
operating point and the measured per-tier delta stack for an agentic coding backend.

This is **not** a hardware port of an upstream card and **not** a single-stream decode chase. The
deliverable is (1) the SLO-max operating point for the coding-agent workload, (2) the long-input
(31K) knee, and (3) a complete T0–T5 Tier Stack Table with measured deltas and the winning engine.

### Target operating points

| Profile | Shape | Why |
|---------|-------|-----|
| **coding-agent** (primary) | 12K system prompt + tool defs (prefix-cacheable), 4K first-turn / 1K subsequent input, 2K output, 8 turns/session, prefix reuse ON | The model's headline use case is agentic coding; tests prefix cache + KV retention across turns + decode under moderate concurrency |
| **long-input-31k** (secondary) | 31,404-tok avg input, 1,024 output, ~74% prefix hit, byte-identical shared prefix | Carried from the kimi-k2.6-nvfp4 customer profile (long-context coding agent); a prefill-dominated stress point to compare GLM-5.2's DSA sparse attention against Kimi's MLA |

### Core questions

1. **vLLM vs SGLang** — which serves GLM-5.2 faster at the coding-agent operating point (TTFT/ITL/$),
   given vLLM's `glm47`+MTP path vs SGLang's EAGLE/DeepEP/HiCache path? Decide on measured numbers.
2. **Per-tier delta** — how much does each of T1 (already-FP8 baseline), T2 (prefix cache + HiCache),
   T3 (MTP vs EAGLE), T4 (TP8 vs TP8+DP-attn+EP), T5 (compile + FLASHINFER_MLA) actually deliver?
3. **DSA long-context behavior** — does DeepSeek Sparse Attention flatten the TTFT curve from
   coding-agent (≤16K) → 31K long-input vs a dense-attention model? This is GLM-5.2's structural bet.
4. **SLO-max + $/Mtok** at the operating point → the production recommendation.

## Components

### 1. Compute
- **Platform**: existing EKS cluster `qwen3-next-bench-eks-cluster` (us-east-2). kubectl context set.
  Serving + observability + bench-runner run as pods (adapt kimi-k2.6-nvfp4 manifests).
- **Instance**: **p6-b200.48xlarge** (8× B200 SXM6, 180 GB HBM3e/GPU, NVSwitch, sm_100) via nodegroup
  **`ai-infra-use2-b200-spot`** (SPOT, max=1, **desired=0** — scaling to 1 is the ~$18/hr billable gate).
- **AZ**: us-east-2b (use2-az2, subnet-03d03f1fb8d62d6a5).
- **AMI**: managed by the nodegroup (EKS-optimized AL2023+NVIDIA, `nodeadm` bootstrap).
- **Node targeting**: after scale-up add `blueprint=glm5.2` AND `nvidia.com/gpu.present=true`
  (no GFD/NFD on this cluster — kimi-k2.6-nvfp4 L1).
- **B200 is primary.** B300 is in scope as a **second arm, not just a capacity fallback**: its larger
  VRAM **unlocks the TP4+DP2 layout that B200 can't fit** (§1c), so run B300 if either (a) TP8 on B200
  proves KV-capacity-bound at the knee, or (b) to measure the TP4+DP2 throughput arm. If B200 TP8 is
  purely HBM-BW-bound, B300 won't lift peak — let the Stage 6 regime classification decide before paying.
  - **B300 substrate (live, confirmed 2026-06-23)**: separate EKS cluster **`qn-sglang-eks-cluster`**
    (us-west-2), nodegroup **`ai-infra-b300-spot`** (p6-b300.48xlarge, SPOT, max=1, **desired=0**,
    ~$15/hr). AZ **us-west-2b** (subnet-001db6882dbb5ac72). Node arrives labeled `ai-infra/role=b300-spot`,
    taint **`ai-infra/b300=true:NoSchedule`** (NOT `nvidia.com/gpu`). sm_103 — `-cu130` image is native
    here. Same no-GFD/NFD bring-up as B200 (manual `blueprint=glm5.2` + `nvidia.com/gpu.present=true`
    labels per scale-up). HF token must be copied cross-cluster (separate region). 4 TB node RAM.
- **Scaling**: single node, fixed (knee/sweep benchmark, not autoscaling).

### 1a. GPU & NCCL Pre-Flight
Standard Stage 4a. TP8 on B200 NVSwitch — mature path (NOT the broken Blackwell-PCIe sm_120 topology
from `devstral-sera`; that was g7e). B200 is sm_100 + NVSwitch → NCCL collectives fine for TP8.
B300 (us-west-2 arm) is sm_103 + NVSwitch — same mature NCCL path; `-cu130` image is native there.

### 1b. Model
- **Model ID**: `zai-org/GLM-5.2-FP8` (official block-FP8, `quant_method=fp8`, `fmt=e4m3`,
  `weight_block_size=[128,128]`, dynamic activation scheme; norms/gates/indexer/embeddings/MTP head
  kept bf16). **NOT** a community NVFP4 mirror — all known NVFP4 repos are REAP-pruned (smaller
  param counts, quality-unvalidated). FP8 is the deployable checkpoint.
- **Architecture**: `glm_moe_dsa` / `GlmMoeDsaForCausalLM` (DeepSeek-V3.2 lineage).
  - MoE: 256 routed + 1 shared expert, top-8 (`num_experts_per_tok=8`), sigmoid `noaux_tc` routing,
    `routed_scaling_factor=2.5`, `moe_intermediate_size=2048`, dense `intermediate_size=12288`,
    `first_k_dense_replace=3`. ~753B total, ~40B active.
  - 78 layers, `hidden_size=6144`, 64 attn heads, vocab 154,880, bf16 compute.
  - **Attention = MLA + DSA**: MLA (`kv_lora_rank=512`, `q_lora_rank=2048`, `qk_head_dim=256`,
    `v_head_dim=256`) + DeepSeek Sparse Attention indexer (`index_topk=2048`, `index_n_heads=32`,
    `index_head_dim=128`, `index_topk_freq=4`; new "IndexShare" reuses the indexer every 4 sparse layers).
  - **MTP**: `num_nextn_predict_layers=1`, `index_share_for_mtp_iteration=true` (card claims +20%
    acceptance length — verify, don't assume).
  - **Context**: `max_position_embeddings=1,048,576` (1M), `rope_theta=8e6`. We cap `max-model-len`
    at **65536** to widen the KV pool for the concurrency sweep (covers 31K input + output + headroom).
- **Engine is the PRIMARY axis — run BOTH, pick the most performant**, then sweep on the winner:
  - **vLLM** (`vllm/vllm-openai` ≥ **v0.23.0**; `glm_moe_dsa` is native in `deepseek_v2.py` —
    `trust-remote-code` likely unneeded, confirm). Recipe args (8×B200 FP8):
    `--tensor-parallel-size 8 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice
    --kv-cache-dtype fp8_e4m3 --max-model-len 65536 --max-num-seqs 32`. MTP via
    `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`. Env `VLLM_DEEP_GEMM_WARMUP=skip`
    (trades cold-start for first-request latency — decide per the cold-start gate). DeepGEMM required.
  - **SGLang** (`lmsysorg/sglang:latest`, ≥ **v0.5.13.post1**; `glm_moe_dsa` in `glm4_moe.py`).
    Low-latency: `--tp 8 --speculative-algorithm EAGLE --speculative-num-steps 5
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --mem-fraction-static 0.8
    --cuda-graph-max-bs 32 --reasoning-parser glm45 --tool-call-parser glm47 --enable-metrics`.
    High-throughput: `--tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep
    --cuda-graph-max-bs 256 --max-running-requests 256 --enable-metrics` (no spec decode).
    HiCache: `--enable-hierarchical-cache --hicache-ratio 2`.
    **`--enable-metrics` is mandatory in every SGLang command** (qwen3-235b-spec L8 — without it
    `/metrics` 404s and all TTFT histograms are permanently lost; this was the exact Kimi data-loss bug).
    **`--max-running-requests` must be set explicitly** — MTP/EAGLE silently auto-resets it to 48 if unset.
  - **Fairness rule**: matched precision (FP8), matched context/concurrency/workload, same
    token-fraction cache metric (`Σ cached_input_tokens / Σ prompt_tokens`). Note each engine's native
    advantages (SGLang HiCache/DeepEP vs vLLM glm47+MTP) rather than crippling either.
- **Deployment Card**: `mdc get glm-5.2 --engine {vllm,sglang}` (likely no card yet — create from the
  vLLM recipe / SGLang cookbook via `mdc sync`) and `mdc prs glm-5.2` before deploy.

### 1c. Memory math — TP8 forced on B200; B300 unlocks a different parallelism arm
FP8 weights ≈ **~750 GB**. Per-GPU weight residency by layout:

| Layout | Weights/GPU | Fits B200 (180 GB)? | Fits B300 (275 GB)? |
|--------|-------------|---------------------|----------------------|
| TP8×1 | ~94 GB | ✅ (tight KV: ~85 GB/GPU before graph/activation overhead) | ✅ (roomy: ~181 GB/GPU for KV) |
| **TP4+DP2** (2 replicas) | ~187 GB | ❌ (187 > 180) | ✅ (~88 GB/GPU KV) |
| TP2+DP4 | ~375 GB | ❌ | ❌ |

- **On B200, GLM-5.2-FP8 is locked to TP8 to fit.** Unlike kimi-k2.6-nvfp4 (smaller NVFP4 ckpt where
  TP4+DP2 won +19–25%), the T4 throughput lever on B200 is **TP8 + DP-attention + EP/DeepEP within the
  8 GPUs**, NOT DP replicas.
- **B300 changes T4, not just capacity.** Its 275 GB/GPU makes **TP4+DP2 fit (~187 GB weights, ~88 GB
  KV/GPU)** — i.e. B300 *unlocks the TP4×2-replica arm B200 can't run*, the same layout that beat TP8
  by +19–25% on kimi-k2.6-nvfp4 (two smaller per-replica batches schedule better than one oversized
  TP8 batch). So on B300, T4 should sweep **TP8 vs TP4+DP2** as well as DP-attn/EP.
- **Caveat (don't assume B300 wins):** TP4+DP2's advantage is real only if the B200 TP8 run is partly
  **KV-capacity-throttled** (token_usage near saturation at the knee). If TP8 is purely **HBM-BW-bound**,
  B300's extra capacity raises concurrency headroom but NOT peak throughput, and TP4+DP2 buys little.
  The Stage 6 bottleneck classification (token_usage at the knee) is the discriminator — exactly the
  kimi-k2.6-nvfp4 lesson (B200 c512 token_usage 0.95 vs B300 0.75 proved capacity was binding there).

### 2. Networking
- Existing cluster VPC. Serving + bench-runner pods: `hostNetwork: true`, tolerate
  `ai-infra/b200=true:NoSchedule` (NOT `nvidia.com/gpu` — kimi-k2.6-nvfp4 L2). In-cluster only.
  Internet-facing pods (HF/pip) need `dnsPolicy: Default` (kimi L4).

### 3. Storage
- **Weights**: stage ~750 GB FP8 to node-local `/mnt/nvme/models/glm5.2-fp8` (RAID-0 the 8×
  instance-store disks on a fresh node; root EBS too small — kimi L4). `hf download` with
  `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=0` + resilient retry loop as PID 1 (kimi L5).
  Spot reclaim wipes NVMe → re-stage per cold node (~750 GB ≈ budget the staging time).
- **HiCache tier (SGLang)**: CPU RAM + NVMe for KV offload. **EKS pod memory limit is LOAD-BEARING**
  (qwen3-235b-spec L9): HiCache allocates `host-offload × TP` GB of HOST memory; too low → pod
  **silently hangs** at "Allocating … host memory" (no OOM-kill) OR gets OOM-killed on the first rank.
  **Sizing math**: `--hicache-ratio 2` on TP8 with ~85 GB device KV/rank ≈ 170 GB host/rank × 8 =
  ~1,360 GB → set pod `resources.limits.memory: 1600Gi` (1,360 + ~240 headroom; p6-b200 has 2 TB RAM ✓).
  On B300 TP4+DP2: per TP4 replica = 4 × 170 = 680 GB → set `900Gi` (p6-b300 has 4 TB ✓). Confirm node
  RAM ≥ the limit at Stage 4-pre. vLLM has no HiCache.
- **DeepGEMM JIT cache**: persist to `/mnt/nvme/deepgemm-cache` so the ~15 min JIT cost is paid once
  per node, not per pod restart (optimization-stack T5 conflict).

### 4. Development Environment
- None. Headless benchmark.

## Non-Requirements (explicitly out of scope)
- **TP4+DP2 on B200** — does not fit (~187 GB weights/GPU > 180; see §1c). It IS in scope **on B300**
  (fits at ~88 GB KV/GPU). **TP2+DP4** does not fit on either. On B200, T4 explores TP8 vs TP8+DP-attn+EP only.
- **Context Parallel (CP)** — **upstream-blocked on Blackwell**: SGLang documents "sm100 DSA-CP FP8
  rope kernel not yet adapted" → CP disabled on b200/b300/gb300. Out of scope; re-verify if SGLang
  ships the kernel (`mdc prs glm-5.2`).
- **Wide-EP across NVL72** — single 8-GPU node only; EP capped at 8 (no NVL72 hardware).
- **P/D disaggregation** — measure the single-node knee first (kimi-k2.6-nvfp4 found 3.8× single-node
  disagg regression; same expected here). Out of scope unless the single-node knee proves it's needed.
- **Community NVFP4 checkpoints** — all known repos are REAP-pruned (different param counts);
  quality-unvalidated, not the same model. Out of scope; FP8 is the deployable artifact. (A *true*
  full-size NVFP4 ckpt, if zai-org/nvidia ship one, would be a follow-up T1 sweep.)
- **Full 1M context** — capped at 65536 for this sweep (widens KV pool). 1M is a separate stress test.
- **BF16** — ~1.5 TB, does not fit single node. FP8 is the floor precision (and is the T0 baseline here).
- **H200 / A100 / g7e** — sm90+ DeepGEMM needed; g7e can't hold 750 GB and has the sm_120 NCCL bug.
- HA/DR, multi-region, production monitoring, autoscaling, long-running stability (>1 hr).

## Security Requirements
- Spot, ephemeral. Encryption at rest on NVMe-staged weights not required for benchmark.
- **Verify node IAM S3 write EARLY** if uploading results (kimi L10 — role silently lacked S3 write;
  `kubectl cp` to workstation as the fallback).

## Cost Considerations
- B200 spot ~$18/hr (us-east-2). Full T0–T5 sweep × 2 engines × 2 workloads should fit in ≲1 day of
  node time; budget the ~750 GB staging + ~15 min DeepGEMM JIT per cold node.
- **The output number**: $/Mtok (input + output) at the SLO-max operating point, from measured
  aggregate throughput × node $/hr. Report per engine and per winning tier config.

## Known Limitations
- **Cold start ~15–16 min** (DeepGEMM JIT + torch.compile + CUDA-graph capture), inferred from GLM-5;
  not 5.2-confirmed. Persist the DeepGEMM cache; bump startup-probe timeout so the compile stall isn't
  read as a health-check failure (optimization-stack T5 conflict). `VLLM_DEEP_GEMM_WARMUP=skip` trades
  startup for first-request latency.
- **MTP auto-resets `--max-running-requests` to 48 if unset** (SGLang, documented) — set it explicitly
  or concurrency is silently capped.
- **MTP/EAGLE acceptance is workload-dependent** — synthetic uniform prompts inflate accept rate 3–5×
  vs real traffic (qwen3-235b-spec L15). Report accept length on the *coding-agent* shape, not synthetic.
- **DSA sparse-attention indexer OOM under high concurrency** was a known DeepSeek-sparse vLLM bug
  (#19412 class) — monitor `num_waiting`/preemptions under sustained QPS; SGLang's indexer differs.
- **FP8 MoE TP-divisibility**: `moe_intermediate_size / TP % 128 == 0`. Here `2048/8 = 256`, `256/128 = 2`
  ✓ at TP8 (and TP4=512/128=4 ✓, but TP4 doesn't fit anyway). Confirm in Stage 0c.
- **`glm47` tool parser + `glm45` reasoning parser** are mandatory or tool-call/CoT output won't parse.
- Model-card prose benchmark numbers are **unreliable** (future-dated arXiv links, nonexistent
  competitors) — treat as marketing; do not cite as targets. Measure our own.
- Check `mdc prs glm-5.2` for upstream PRs affecting `glm_moe_dsa` MoE / MLA / DSA / MTP paths.

## Verification Criteria

### Stage 0 — Carryover Audit (spec-design gate)
- [ ] Ran `carryover-auditor` against this spec, scanning `glm5/lessons.md`, `glm5-lmcache/lessons.md`,
      `glm5-llmd/lessons.md`, `kimi-k2.6-nvfp4/lessons.md`, `qwen3-235b-speculative/lessons.md`, and
      B200/B300 infra memory.
- [ ] Carried lessons reflected as requirements: TP8-forced by FP8 size (no TP4+DP2, §1c); cluster
      Stage 4-pre (labels + `ai-infra/b200` taint + RAID NVMe + `dnsPolicy: Default`, kimi L1/L2/L4/L12);
      SGLang `--enable-metrics` else /metrics 404 (qwen3-235b-spec L8); DeepGEMM JIT cache persist (T5);
      FlashInfer cubin race pre-clear (kimi L-Stage5); ECC gate on `volatile.total` not aggregate (kimi L3);
      HF_TOKEN export + Xet-disable (kimi L5); verify S3 write early (kimi L10); CP blocked on Blackwell;
      `glm47`/`glm45` parsers mandatory.
- [ ] No P0 carryover gap remains.

### Stage 0b — Optimization Coverage (lever ledger)
1. **Regime prediction** (`.claude/steering/inference-first-principles.md`): coding-agent at moderate
   concurrency is **prefill/KV-bound** on B200 (12K cacheable system prompt + multi-turn KV retention);
   long-input-31k is **prefill-compute-bound** (DSA should soften this). Decode is the small part
   (2K out). One line: shared-prefix-heavy agent traffic + long context → T2 (cache) and T5 (MLA kernel)
   are the high-leverage tiers; T3 (spec decode) helps per-user latency at low concurrency only.
2. **Lever ledger**:

| Tier | Lever | applied / deferred — reason |
|------|-------|------------------------------|
| T0 | Baseline (honest reference) | **applied** — FP8 is the floor (BF16 = 1.5 TB, doesn't fit). T0 = FP8, TP8, no prefix cache, no spec decode, eager, default attn backend. Documents this is an FP8-floor baseline, not BF16. |
| T1 | Quantization (weight + KV bytes) | **applied** — official block-FP8 weights (baseline) + sweep `--kv-cache-dtype fp8_e4m3` vs `auto`. NVFP4 deferred — only REAP-pruned community ckpts exist (different model); revisit if a full-size NVFP4 ships. |
| T2 | KV / prefix cache | **applied** — L1 prefix cache ON (both engines); SGLang HiCache (`--enable-hierarchical-cache --hicache-ratio 2`, pod mem sized per §3) swept on for the 12K shared prompt + 31K long input. vLLM: L1 prefix cache. **vLLM LMCache+MLA deferred — re-verify before attempting**: tech-stack.md flagged it blocked (PR #2629, validated 2026-03-07, now stale); optimization-stack says the vLLM path merged but with OPEN GLM bugs (#2774 FP8 shape, #2977 cache-hit 0). Confirm PR/issue status at Stage 0b before any LMCache run; if attempted, smoke-test cache-hit + output quality on glm_moe_dsa first. |
| T3 | Speculative decode | **applied** — native **MTP** (vLLM `method:mtp`; SGLang **EAGLE** num-steps sweep). Compare MTP-vs-EAGLE accept length on coding-agent shape. Lever for low-concurrency latency; expect aggregate hit at high concurrency. |
| T4 | Parallelism (TP/EP/DP shape) | **applied** — B200: TP8 (latency) vs **TP8 + `--enable-dp-attention` + `--moe-a2a-backend deepep`** (throughput). TP4+DP2 **deferred on B200 — does not fit** (~187 GB/GPU > 180, §1c), but **in scope on B300** (the layout that won +19–25% on kimi-k2.6-nvfp4). Wide-EP>8 deferred — no NVL72. |
| T5 | Kernel / compile | **applied** — torch.compile + **FLASHINFER_MLA** (MLA model, optimization-stack 15–25% decode); CUDA graphs (`--cuda-graph-max-bs`); SGLang overlap scheduler for agent traffic. DeepGEMM JIT cached. |

- [ ] Regime predicted with one-line reasoning ✓ (above)
- [ ] Every tier `applied` or `deferred` with reason — no blank rows ✓
- [ ] High-priority tiers for the predicted regime (T2 cache, T5 MLA kernel) are `applied` ✓
- [ ] **Re-verify every engine-blocker deferral against the live tracker** at deploy: CP-on-Blackwell
      (SGLang DSA-CP FP8 rope kernel), LMCache+MLA-on-vLLM (#2951/#2774/#2977 status), NVFP4 full-size
      ckpt existence — `mdc prs glm-5.2` + `gh pr view`/`gh issue list`. Record `validated: YYYY-MM-DD`.
- [ ] Stage 6 Tier Stack Table will carry the measured deltas for these rows.

### Stage 0c — Serving-Config Resolver (fail-closed)
- [ ] `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/glm5.2/benchmark.yaml --corpus-root .` exits 0
- [ ] `model.moe_intermediate_size: 2048` present in the sidecar so `fp8-moe-tp-divisibility` verifies
      (2048/8=256, /128=2 ✓ at TP8) rather than WARNs.
- [ ] Every `prior-failure:*` finding reviewed and noted in the deployment log.

### Stage 4-pre — Node bring-up gate (run after scale-up, BEFORE serving)
Both clusters have no GFD/NFD + a custom taint (kimi-k2.6-nvfp4 L1–L7). **B200 arm**: cluster
`qwen3-next-bench-eks-cluster` (us-east-2), nodegroup `ai-infra-use2-b200-spot`, taint `ai-infra/b200`.
**B300 arm**: cluster `qn-sglang-eks-cluster` (us-west-2), nodegroup `ai-infra-b300-spot`, taint
`ai-infra/b300`, AZ us-west-2b — **switch kubectl context + region first**, and copy the HF token cross-cluster.
- [ ] Node labeled BOTH `blueprint=glm5.2` AND `nvidia.com/gpu.present=true`; verify `allocatable.nvidia.com/gpu == 8`.
- [ ] Pods tolerate the arm's real taint — `ai-infra/b200=true:NoSchedule` (B200) or
      `ai-infra/b300=true:NoSchedule` (B300), NOT `nvidia.com/gpu`.
- [ ] NVMe RAID-0 mounted at `/mnt/nvme` (root EBS too small for 750 GB); `mountPropagation: HostToContainer` on every pod (kimi L12).
- [ ] Internet-facing pods (HF/pip) use `hostNetwork: true` + `dnsPolicy: Default`.
- [ ] DeepGEMM JIT cache dir on persistent NVMe; startup-probe timeout ≥ 20 min (cold compile).
- [ ] ECC gate: `volatile.total==0` + clean `remapped_rows` (NOT lifetime aggregate — false-fails reused spot GPUs, kimi L3).
- [ ] **DCGM PROF scrape works BEFORE serving** (kimi L8 — driver-580 combo silently exported nothing):
      apply observability, then `curl localhost:9400/metrics | grep DCGM_FI_PROF_DRAM_ACTIVE` returns
      non-empty (the exporter `-f` arg must point at the mounted PROF CSV, not the default). If empty on
      this driver, record that the Stage 6 regime classification will use engine gauges (labeled `[gauge-inferred]`).
- [ ] Node RAM ≥ the HiCache pod memory limit from §3 (p6-b200 2 TB ≥ 1600Gi ✓; p6-b300 4 TB ≥ 900Gi ✓).

### Stage 4a — GPU Health
- [ ] ECC enabled, 0 *volatile* uncorrected; `remapped_rows.{pending,failure}==No`; thermals < 85°C idle; no Xid.
- [ ] NCCL all-reduce BW recorded for TP8 (NVSwitch path) — fill from `gpu-infra card p6-b200`.

### Stage 5 — Serving Stack (per engine)
- [ ] `/health` 200; single `/v1/completions` returns valid output; no CUDA OOM; weights load at TP8.
- [ ] Cold start recorded (DeepGEMM JIT + torch.compile + CUDA-graph capture); cache populated so a
      second cold start is faster.
- [ ] **FlashInfer cubin symlink race pre-clear** — wraps EVERY server start (fires on every cold
      start, both engines; qwen3-235b-spec L16). Exact one-liner in the launch command:
      `find /usr/local/lib/python*/dist-packages/flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; exec python3 -m <sglang.launch_server|vllm ...>`.
- [ ] Baseline (T0) runs start with NO speculative flags and NO prefix cache — establish the FP8 floor
      first, then layer tiers (don't bake T2/T3 into the baseline).
- [ ] **Observability smoke-test BEFORE any benchmark** (qwen3-235b-spec L8): per engine confirm
      `/metrics` scrapes — vLLM `vllm:time_to_first_token_seconds_bucket` non-empty; **SGLang
      `--enable-metrics` set** and `curl localhost:30000/metrics` returns `sglang:*` (else TTFT data
      permanently lost). DCGM PROF fields non-empty (see Stage 6 regime gate) or fall back to engine gauges, labeled.
- [ ] **Tool-call + reasoning parse check**: one `glm47` tool-call request and one `glm45` reasoning
      request parse correctly before the agent workload runs (else functional results are invalid).

### Stage 6 — Benchmark

**Workloads** — reference canonical cards by `catalog_id`, override the operating point via sidecar:

| Card (`catalog_id`) | Role | Sidecar override |
|---------------------|------|------------------|
| `coding-agent` | **PRIMARY** — the headline use case | use card defaults (12K system prompt, 4K/1K input, 2K output, 8 turns, prefix reuse ON); sweep `concurrent_sessions` |
| `concurrency-sweep` | Find the SLO-max knee | context at coding-agent shape; sweep concurrency 1 → saturation (power-of-2) |
| `long-input-31k` (custom, from kimi profile) | Secondary — long-context stress + DSA test | input 31,404 avg (std 8000), output 1,024, **byte-identical** shared prefix targeting ~74% hit, concurrency sweep 64 → 1,024 |
| `production-mix` (ShareGPT replay) | **Ground the spec-decode accept rate** | ~200 real multi-turn conversations replayed; run alongside coding-agent whenever MTP/EAGLE is on |

> **`long-input-31k` is a custom workload** modeled on the kimi-k2.6-nvfp4 customer profile. The shared
> prefix MUST be byte-identical (RadixAttention/prefix-cache exact-match — kimi cache scar); variation
> only in the suffix, or the measured hit rate collapses to ~1.0×.
>
> **`production-mix` is mandatory whenever T3 (MTP/EAGLE) is enabled** (qwen3-235b-spec L15): synthetic
> prompts inflated EAGLE accept length 5.0 / rate 1.0 vs 1.62 / 0.156 on real ShareGPT traffic — a 3–5×
> overstatement. Report MTP/EAGLE accept length on `production-mix` AND `coding-agent`; base the T3
> go/no-go on the real-traffic number, never the synthetic one.

**Required measurements (per engine, then full sweep on the winner):**
- [ ] **coding-agent**: concurrency sweep (1 → saturation). TTFT p99 (cold) < 500 ms and warm < 100 ms
      at the operating point (card SLO); TPOT p99 < 30 ms; prefix hit rate measured (system prompt caching).
- [ ] **concurrency-sweep**: SLO-max knee identified (highest concurrency holding the coding-agent SLO).
- [ ] **long-input-31k**: concurrency sweep at 31,404 ctx; knee where TTFT p99 crosses 15,000 ms;
      per-request decode ≥ 34.8 tok/s sustained at the knee (kimi customer floor, for comparability).
- [ ] Error rate < 0.1% at all levels; no OOM (record max sustained concurrency); no timeouts.
- [ ] **DSA curve test**: plot TTFT p99 vs input length (coding-agent ≤16K → 31K long-input). Report
      whether DSA flattens the slope vs the dense expectation (the structural question).

**KV cache validation (required — agentic + long-context):**
- [ ] Prefix-cache hit rate as **token fraction** `Σ cached_input_tokens / Σ prompt_tokens` (engine-
      agnostic; vLLM `prompt_tokens_details.cached_tokens`, SGLang `meta_info.cached_tokens`) — directly
      comparable across engines. Cross-check engine native gauge for operational signal only.
- [ ] coding-agent: prefix cache disable→enable TTFT delta (the 12K system prompt is the cacheable win).
- [ ] HiCache (SGLang) toggle on/off: net effect on the 31K knee — expect help while capacity-bound,
      neutral/negative once compute/BW-bound (kimi/GLM-5 HiCache lesson). Pod memory limit sized for the tier.
- [ ] KV utilization % at the knee < 95% (record saturation point); eviction rate of the shared prefix low.

**Engine-internal metrics (Prometheus `/metrics`):**
- [ ] vLLM: `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting`,
      `vllm:spec_decode_*` (MTP accept). SGLang: `sglang:token_usage`, `num_running_reqs`,
      `num_queue_reqs`, spec accept length. **SGLang needs `--enable-metrics`** (else 404).

**Bottleneck classification at the knee (decides B200-vs-B300 + which tier to push):**
- [ ] DCGM exporter with PROF CSV + `privileged: true`; `curl localhost:9400/metrics | grep
      DCGM_FI_PROF_DRAM_ACTIVE` non-empty BEFORE the sweep (kimi L8 — the PROF CSV must actually be
      loaded by the exporter `-f` arg, not just mounted). Else classification is `[gauge-inferred]`, labeled.
- [ ] At the knee capture `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (compute), `DCGM_FI_PROF_DRAM_ACTIVE`
      (HBM BW), KV usage, queue depth. Classify: **prefill-compute-bound** (tune MNBT/chunked-prefill,
      T5 kernel) / **KV-capacity-bound** (HiCache, then B300) / **HBM-BW-bound** (B300 won't help; spec
      decode/kernel are the levers). Record the regime.
- [ ] **TTFT-share check** (benchmark-analysis.md): TTFT share = median(TTFT)/median(E2E); high share
      (>20%) confirms prefill is on the critical path — don't call it decode-bound without this.

**Tier Stack Table (required — closes the Stage 0b ledger):** one row per tier T0–T5 on the winning
engine, with measured Δ tok/s and Δ TTFT p99 vs T0, plus blocked rows.
- [ ] T0 FP8/TP8/no-cache (ref) · T1 +fp8-KV · T2 +prefix+HiCache · T3 +MTP/EAGLE · T4 +DP-attn+EP · T5 +compile+FLASHINFER_MLA.
- [ ] Every tier marked `deferred` in Stage 0b reconciled here (still deferred, or applied + measured).
- [ ] Any tier underperforming its optimization-stack typical-Δ range noted in `lessons.md`.

**Cost output (the go/no-go):**
- [ ] $/Mtok (input + output) at the SLO-max operating point per engine = (node $/hr) ÷ (agg tok/s × 3600 / 1e6).
- [ ] Recommended production config (engine + tier stack) with rationale.

**Enriched artifact**: store in `blueprints/glm5.2/results/` per `standards/benchmark-commons/PROPOSAL.md`.

### Stage 7 — Readiness Audit
- [ ] All readiness categories pass; no unresolved HIGH-severity lessons.
- [ ] Winning engine, SLO-max operating point, long-input-31k knee, DSA curve verdict, and full Tier
      Stack Table recorded with the $/Mtok recommendation.
- [ ] `mdc get glm-5.2` recommendations followed or explicitly overridden with justification.
- [ ] All criteria above checked and recorded.

---

> Operational artifacts (lessons, results, deployment notes) belong in `blueprints/glm5.2/`, not in
> this spec. Reuse the kimi-k2.6-nvfp4 manifests (cluster bring-up, observability with the **fixed**
> DCGM PROF wiring, bench-runner byte-identical-prefix driver) as the starting point.
