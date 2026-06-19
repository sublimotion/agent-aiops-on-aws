# Kimi K2.6 NVFP4 — Single-Node Knee-Finding Requirements

## Status: DRAFT

## Overview

Benchmark `nvidia/Kimi-K2.6-NVFP4` (1T MoE, 32B active) on a **single 8-GPU Blackwell node**
to find the operating-point knee for a real production workload, and decide whether the
customer's traffic can be served without P/D disaggregation.

This is **not** a hardware port of NVIDIA's released card (which validated NVFP4 accuracy on
B200, 2026-05-13) and **not** a single-stream decode-tok/s chase. Every prior Kimi benchmark we
have (`kimi-k2.6-speculative`, `qwen3-235b-speculative`) used ~219-token prompts at c≤512. This
workload is **two orders of magnitude longer in context and 2-3× higher in concurrency**, in a
prefill-dominated regime we have never measured.

### Target operating point (derived from customer 7-day stats)

| Metric | Value | Implication for us |
|--------|-------|--------------------|
| Requests/min | 1,909 avg, ~2,900 peak | ~32 req/s sustained, ~48 peak |
| Avg input context | 31,404 tok (16k peak-hours ↔ 36k quiet) | long-context prefill dominates |
| Avg output length | 1,024 tok | decode is the *small* part |
| Input:output ratio | 30-60× | prefill-bound, classic disagg candidate |
| Prompt cache hit | 74.1% | heavy shared-prefix structure |
| Avg TTFT | 3,339 ms (p95 5.8-15s at peak) | TTFT is the SLO that degrades under load |
| Avg decode throughput | 34.8 tok/s/req | the per-request target |
| Blended customer price | **$0.338/Mtok** | the $/Mtok number we must beat |
| Upstream spend | ~604B in + 19.7B out tok/wk ≈ **$1.27M/mo** | the cost ceiling to undercut |

**Concurrency (Little's law)**: request lifetime ≈ TTFT 3.34s + decode (1024/34.8) 29.4s ≈ 32.8s.
At 32 req/s → **~1,040 concurrent average, ~1,500 peak**. This is the knee we're hunting.

### Core questions this benchmark answers

1. **Where is the single-node knee?** At what concurrency (at 31K context) does p95 TTFT cross
   the customer's ~15s peak ceiling, or the KV pool saturate? Does one node hold ~1,500 concurrent?
2. **Does the prefix cache hold 74% under load?** Realizing the customer's cache economics requires
   keeping shared prefixes resident/restorable. HiCache CPU/NVMe tiering is the one lever we toggle.
3. **What is our $/Mtok at the knee** vs the $0.338 blended target → the go/no-go number.

If one node holds peak concurrency at acceptable TTFT, **disagg is not needed** — that is the
hypothesis under test. Do not build disagg before this knee is measured.

## Components

### 1. Compute
- **Platform**: **existing EKS cluster** `qwen3-next-bench-eks-cluster` (us-east-2). kubectl context
  already points at it. Serving + observability + bench-runner run as pods (adapt the working manifests
  in `qwen3-235b-speculative/k8s/`).
- **Instance Type**: **p6-b200.48xlarge** (8× B200 SXM6, 180 GB HBM3e/GPU, NVSwitch, sm_100) via the
  pre-provisioned nodegroup **`ai-infra-use2-b200-spot`** (SPOT, max=1, currently **desired=0**).
  Scaling to 1 is the billable gate (~$18/hr). NVIDIA's validated box for this NVFP4 release.
- **AZ**: us-east-2b (**use2-az2**) — the nodegroup's only subnet (`subnet-03d03f1fb8d62d6a5`).
- **AMI**: managed by the EKS nodegroup (EKS-optimized **AL2023**+NVIDIA, `nodeadm` MIME bootstrap —
  AL2 lacks `ib_umad` for NVL5+ Fabric Manager, glm5-lmcache L1). No manual AMI selection needed.
- **Node targeting**: node arrives labeled `ai-infra/role: b200-spot`. After scale-up, add
  `blueprint=kimi-k2.6-nvfp4` (keeps the `nodeSelector: {blueprint: ...}` convention of other blueprints).
- **B200 vs B300 — decided empirically, not assumed.** The two differ only in KV *capacity*
  (180 vs 275 GB/GPU), NOT HBM bandwidth. So we run on B200 and **measure which wall we hit** (see
  Stage 6 bottleneck classification):
  - Kimi K2.6 is MLA (DeepSeek-V3 arch): KV ≈ 69 KB/token → **~2.15 GB KV per resident 31K-ctx request**.
  - NVFP4 weights ~520 GB → ~65 GB/GPU at TP8. B200 KV pool ≈ 115 GB/GPU → ~920 GB → ~430 base
    resident reqs (×74% prefix sharing → ~1,000), right at the ~1,500 peak — borderline.
  - **If the knee is HBM-BW-bound, B200 == B300** and the result transfers; B300 is unnecessary.
  - **If the knee is KV-capacity-bound**, HiCache tiering is the first lever; B300 (or a 2nd node) is
    the recommendation only then. Do NOT pre-provision B300 — let the measured regime decide.
- **Fallback**: B300 (p6-b300.48xlarge, 275 GB/GPU, us-west-2b) only if B200 proves KV-capacity-bound
  and HiCache tiering doesn't recover peak. Raises the knee, doesn't change its shape.
- **Scaling**: single node, fixed (this is a knee-finding benchmark, not an autoscaling deployment)

### 1a. GPU & NCCL Pre-Flight
Standard Stage 4a (see template). TP8 on B200 NVSwitch — mature path, not the broken
Blackwell-PCIe topology from `devstral-sera/lessons.md` (that was g7e sm_120). B200 is **sm_100**:
use stock `vllm/vllm-openai:latest` (cu128); the `-cu130` tags are sm_103/B300-only. B200 needs the
AL2023 NVIDIA AMI for `ib_umad`/Fabric Manager (glm5-lmcache L1).

### 2. Model
- **Model ID**: `nvidia/Kimi-K2.6-NVFP4` (the **modelopt** checkpoint). **NOT** `RedHatAI/Kimi-K2.6-NVFP4`
  (CompressedTensors variant) — that fails to load on SGLang with `ReplicatedLinear has no attribute 'weight'`
  (SGLang issue #25331, OPEN as of 2026-06-16).
- **Format**: NVFP4 (MoE-linear weights+activations only; modelopt v0.44.0). Blackwell-only kernels.
- **Engine is the PRIMARY benchmark axis — run BOTH, pick the most performant.** Both now serve
  `nvidia/Kimi-K2.6-NVFP4` natively; the winner is genuinely unknown (SGLang NVFP4 only landed
  2026-06-10; vLLM historically had 2-3× lower TTFT on K2.6). Decide on measured TTFT/throughput/$ at
  the customer operating point, not on priors. Select the winner, then run the full optimization sweep
  on it. Both must use the **identical** workload generator and the same token-fraction cache metric.

  **SGLang** (≥ 0.5.9 — Kimi-K2.6 NVFP4 added via cookbook PR #27714, merged 2026-06-10). Familiar Kimi
  stack (HiCache, RadixAttention). **Verified launch (from SGLang Kimi-K2.6 cookbook):**
  - `--model-path nvidia/Kimi-K2.6-NVFP4`
  - `--quantization modelopt_fp4`
  - `--tp 8` (B300/H200; `--tp 4` only on GB300. **Note: TP8 puts 8 heads/GPU — fine on NVIDIA;
    AMD AITER MLA needs heads/GPU %16==0 so AMD caps at TP4. N/A for B200/B300.**)
  - `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2`
  - `--context-length 65536` (covers 36K quiet-hours peak + output + headroom; cookbook uses 128000 for
    full-context — we cap lower to widen the KV pool for the concurrency sweep)
  - `--enable-hierarchical-cache --hicache-size 200` (see Storage; this is the KV-tier lever)
  - **Spec decode is now AVAILABLE** (reverses the earlier "out of scope" call — see below): EAGLE3 MLA
    draft `lightseekorg/kimi-k2.6-eagle3.1-mla` with `--speculative-algorithm EAGLE3
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --speculative-draft-model-path lightseekorg/kimi-k2.6-eagle3.1-mla`. Treat as an optimization
    lever to sweep, NOT a baseline (see Known Limitations re: synthetic-prompt accept-rate inflation).
  **vLLM** (`vllm/vllm-openai:latest`, sm_100/cu128). Historically gave **2-3× lower TTFT than SGLang**
  on K2.6 via FLASHINFER_MLA (kimi-k2.6 lessons) — decisive if it holds at this context, since the
  workload is TTFT-bound. Args: `--tensor-parallel-size 8 --tool-call-parser kimi_k2 --reasoning-parser
  kimi_k2 --enable-auto-tool-choice --enable-prefix-caching --max-model-len 65536 --trust-remote-code`.
  vLLM KV tiering via LMCache/`--cpu-offload-gb` (NOT HiCache). **Caveat (mdc card): vLLM flags
  `[lmcache, mla]` INCOMPATIBLE for Kimi — so on vLLM, KV offload may be unavailable; if so, vLLM runs
  GPU-KV-only and its capacity knee will be lower than SGLang+HiCache. Record this as a structural
  difference, not a tuning gap.**

  **Fairness rule for the comparison**: report both engines at matched precision (NVFP4), matched
  context/concurrency/workload, and the same token-fraction cache metric. Note each engine's *native*
  advantages (SGLang HiCache tiering vs vLLM FLASHINFER_MLA TTFT) rather than crippling either to match.
- **Correction to prior carryover**: `qwen3-235b-speculative` L17 ("no native MTP → spec decode out of
  scope") was correct that there's no *MTP*, but an **external EAGLE3 MLA draft now exists upstream**,
  exactly as L17 predicted. Spec decode is therefore in scope as a lever, not excluded.
- **Deployment Card**: run `mdc get kimi-k2.6 --engine sglang` (no card yet — create from this cookbook
  via `mdc sync`) and `mdc prs kimi-k2.6` before deploy.

### 3. Networking
- **VPC**: existing cluster VPC (us-east-2, the qwen3-235b-speculative subnets). Pods use `hostNetwork: true`
  + `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` per the reference manifests.
- **Access**: in-cluster; bench-runner pod hits the serving pod on the node (hostNetwork, port 30000 SGLang
  / 8000 vLLM). No public ingress.

### 4. Storage
- **Model Storage**: stage NVFP4 weights (~520 GB) to node-local **`/mnt/nvme/models/kimi-k26-nvfp4`**
  (`hostPath` mount, as in the reference pods). Use `hf download` (not `huggingface-cli`, renamed in hub
  v1.11+) with `HF_HUB_ENABLE_HF_TRANSFER=1` and explicit `HF_TOKEN` (`kimi-k2.6-speculative` L15). Spot
  node — weights are lost on reclaim and must be re-staged; budget ~staging time per cold node.
- **HiCache tier**: CPU RAM + NVMe for KV offload. **EKS makes the pod memory limit LOAD-BEARING**
  (standalone EC2's 4 TB RAM hid this): set `resources.limits.memory: ~2160Gi` = 220 GB/rank × TP8 + 400
  headroom (`qwen3-235b-speculative` L9). Too low → HiCache **silently hangs** at "Allocating … host
  memory", no OOM-kill (kimi-k2.6 / qwen3-235b-spec). vLLM has no HiCache (and `[lmcache,mla]` incompatible),
  so the vLLM pod needs only weight+KV+CUDA-graph headroom (~256-512Gi), not the full tier budget.

### 5. Development Environment
- None. Headless benchmark.

## Non-Requirements (explicitly out of scope)
- **P/D disaggregation** — deferred until the single-node knee proves it's needed, and likely
  weak here regardless: aggregate decode (~32.5K tok/s) is ~8× smaller than fresh prefill
  (~259K tok/s). Disagg's value is scaling prefill/decode pools independently, but with decode
  this small the carved-off decode pool is tiny and NIXL/KV-transfer coordination overhead
  (see PD-disagg-single-node memory) likely eats the gain. Measure the single-node knee first.
- **Multi-node / multi-replica** — single node only.
- **Training our own EAGLE3 draft** — out of scope. (Using the *existing* upstream
  `lightseekorg/kimi-k2.6-eagle3.1-mla` draft as a sweep lever IS in scope — see Model §2.)
- **B300 as primary** — out of scope unless B200 proves KV-capacity-bound and HiCache can't recover
  peak (see Compute §1). We measure the bottleneck regime first rather than pre-provisioning B300.
- **g7e** — cannot hold 520 GB weights on 384 GB; PCIe + sm_120 NCCL issues. Not a 1T-MoE box.
- **H200 (Hopper)** — cannot run NVFP4 kernels (Blackwell-only). Would force FP8 fallback (~2× weight
  bytes, ~2× decode BW/token), defeating the cost goal. Out of scope unless a non-Blackwell fallback
  story is requested → then one FP8 reference point only.
- HA/DR, multi-region, production monitoring.

## Security Requirements
- Spot instance, ephemeral. Encryption at rest on NVMe-staged weights not required for benchmark.
- IAM: instance role needs S3 read for any staged artifacts; **verify S3 write perms** if uploading
  results (infra memory: `g7e-bench-role` silently lacked S3 write — checkpoint uploads failed).

## Cost Considerations
- B200 spot ~$18/hr (us-east-2); B300 fallback ~$15-16/hr (us-west-2). A full knee sweep (Phase 0-2)
  should fit in <1 day of node time.
- The output number that matters: **$/Mtok at the knee** vs $0.338 blended / $0.363 input / $3.76 output
  upstream. Compute from measured aggregate throughput × node $/hr at the SLO-max concurrency.
- Reference: our B300 Kimi K2.6 prior run hit $0.43/1M output tok at c=512 on spot — but that was
  219-token prompts. The 31K-context number will be very different (prefill cost dominates).

## Known Limitations
- **RadixAttention exact-match is the #1 risk to reproducing 74% on SGLang.** `kimi-k2.6/lessons.md`:
  SGLang RadixAttention showed **~1.0× (no) prefix-cache benefit** because it requires **token-level
  exact-prefix match**, and the benchmark's slight query variations broke the match. Real agent traffic
  has head variation (timestamps, request IDs, reordered context) that shatters a radix prefix. So the
  workload generator MUST place the shared scaffold as a **byte-identical leading prefix** with variation
  only in the *suffix* — otherwise we'll measure ~1.0× and wrongly conclude the stack can't cache.
  This is a generator-correctness requirement, not just an SLO. (vLLM block-level cache is also
  prefix-order-sensitive but block-aligned, slightly more forgiving — another reason to keep the vLLM
  comparison point.)
- **Prefix-cache realism is the second validity risk.** The 74.1% hit implies heavy shared prefixes.
  Random 31K prompts measure a far *worse* knee than production. Model prefix sharing (see Stage 6);
  if the customer's prefix structure is uncharacterized, report the knee as a *lower bound* and flag it.
- **HiCache helps low-concurrency, may NOT help at peak.** `kimi-k2.6/lessons.md`: HiCache improved
  single-stream TPS (~58%) but at high concurrency (c=16+) matched or trailed base SGLang — the
  bottleneck shifts from KV capacity to compute. So treat HiCache as a knee-*raiser* for the
  capacity-bound regime only; expect diminishing/negative returns once the run is compute- or BW-bound.
- FlashInfer cubin symlink race on every cold start — pre-clear inside container at launch
  (`qwen3-235b-speculative` L16).
- **NVFP4 may still trail commercial providers — verify, don't assume.** SGLang ≥0.5.9 ships FP4
  kernels (`modelopt_fp4`), so the earlier "stock is 0.5-1.0× of cutlass-3.x" caveat (qwen3-235b-spec
  L18, written against SGLang 0.5.10 lacking FP4) is **no longer assumed** — measure SGLang's NVFP4
  throughput and compare to the CoreWeave/Azure 128-144 tok/s references. If still short, that gap is
  custom-kernel work (out of scope). Don't present the baseline as the ceiling either way.
- Check `mdc prs kimi-k2.6` for upstream PRs affecting NVFP4 MoE / MLA / EAGLE3 paths.

## Verification Criteria

### Stage 0 — Carryover Audit (spec-design gate)
- [ ] Ran `carryover-auditor` against this spec, scanning `kimi-k2.6-speculative/lessons.md`,
      `qwen3-235b-speculative/lessons.md`, `kimi-k2.6/lessons.md`, `devstral-sera/lessons.md`,
      and B300/B200 infra memory.
- [ ] Carried lessons reflected as requirements: no *native* MTP but **external EAGLE3 MLA draft exists**
      → spec decode in-scope as a lever (updates L17), FlashInfer cubin race both engines (L16),
      SGLang `--enable-metrics` else /metrics 404 (L8), RadixAttention byte-identical-prefix requirement
      (kimi-k2.6), HiCache helps only while capacity-bound + host>device sizing (kimi-k2.6, L9/L14),
      EAGLE3 synthetic-accept-rate inflation (L15), HF_TOKEN export (L15-spec), VPC endpoint pre-test (L14-spec),
      S3 write perms, `-cu130`/sm_103 is B300-fallback-only (B200 is sm_100/cu128), NCCL-broken-on-PCIe
      **not applicable** (NVSwitch path), SGLang EP path broken for Kimi **avoided** (we use TP8, not EP).
- [ ] No P0 carryover gap remains.

### Stage 0c — Serving-Config Resolver (fail-closed)
- [ ] `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/kimi-k2.6-nvfp4/benchmark.yaml --corpus-root .` exits 0
- [ ] NVFP4 MoE divisibility: the `fp8-moe-tp-divisibility` rule (qwen3-235b-b300 L1) is for FP8
      block_n=128 — NVFP4 modelopt may use a different block size. Determine applicability by inspecting
      the modelopt `config.json` block scaling OR running a single-request TP8 smoke; if the assertion
      fires, record `moe_intermediate_size` and verify `/8 % block == 0`, else note N/A with the reason.
- [ ] Every `prior-failure:*` finding reviewed and noted in the deployment log.

### Stage 4-pre — Node bring-up gate (run immediately after scale-up, BEFORE Stage 4a/serving)
Cluster `qwen3-next-bench-eks-cluster` has no GPU-feature-discovery and a custom taint; these MUST be
handled or the node never serves (all discovered 2026-06-16, see `lessons.md` L1-L7):
- [ ] Node labeled BOTH `blueprint=kimi-k2.6-nvfp4` (pod nodeSelector) AND `nvidia.com/gpu.present=true`
      (else `nvidia-device-plugin` DESIRED=0, GPUs never register). Verify `allocatable.nvidia.com/gpu == 8`.
- [ ] Pods tolerate `ai-infra/b200=true:NoSchedule` (the real taint — NOT `nvidia.com/gpu`).
- [ ] Local NVMe set up: 8× raw instance-store disks are unmounted on a fresh node → RAID-0 + mkfs +
      mount at `/mnt/nvme` (root EBS is only 500 GB, too small for 575 GB weights). Re-run on every spot node.
- [ ] Any internet-facing pod (model download, pip) uses `hostNetwork: true` + `dnsPolicy: Default`
      (CoreFirst DNS fails external resolution on this cluster).
- [ ] SGLang NVFP4 uses the **`-cu130`** image (cu129 lacks the `cutlass` DSL the FP4 path needs); EAGLE3
      draft passes `--speculative-draft-model-quantization unquant`.
- [ ] ECC gate uses `volatile.total==0` + clean `remapped_rows` (NOT lifetime aggregate — false-fails reused spot GPUs).

### Stage 4a — GPU Health
- [ ] ECC enabled, 0 *volatile* uncorrected errors; `remapped_rows.{pending,failure}==No`; thermals < 85°C idle; no Xid.
      (Lifetime `aggregate` ECC may be nonzero on reused spot GPUs — that's benign, see L3.)
- [ ] NCCL all-reduce bandwidth > _____ GB/s for TP=8 (fill from `gpu-infra card` / NCCL test on NV18).

### Stage 5 — Serving Stack
- [ ] `/health` returns 200; single `/v1/completions` request returns valid output.
- [ ] No `CUDA out of memory` in logs at startup; weights load at TP8.
- [ ] Cold start < _____ min (record; NVFP4 has no DeepGEMM JIT but expect CUDA graph capture).
- [ ] **FlashInfer cubin symlink race pre-clear** (kimi-k2.6-spec L16 — fires on EVERY cold start, both
      engines): launch command wraps the server with
      `bash -c "find .../flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; exec <server>"`.
- [ ] Baseline runs (before the spec-decode sweep) start with NO speculative flags — establish the
      non-spec baseline first, then add EAGLE3 as a lever (do not bake spec decode into the baseline).
- [ ] **Observability smoke-test BEFORE any benchmark** (qwen3-235b-spec L8 — never assume metrics), per engine:
      - vLLM: `up{job="vllm"}=1`, DCGM reports 8 GPUs, `vllm:time_to_first_token_seconds_bucket` non-empty (default on, but validate scrape).
      - **SGLang: confirm `--enable-metrics` is set and `curl -s localhost:30000/metrics` returns `sglang:*` gauges**
        — WITHOUT this flag `/metrics` 404s and TTFT histograms are permanently lost (this is exactly the Kimi-spec data-loss failure).
      - Block Stage 6 on this for whichever engine is under test.

### Stage 6 — Benchmark

**Workload selection** — reference canonical cards by `catalog_id`, override the operating point via
sidecar (do not fork divergent params into the blueprint):

| Card (`catalog_id`) | Role in this spec | Sidecar override |
|---------------------|-------------------|------------------|
| `concurrency-sweep` | Find the SLO-max knee | context fixed at 31,404; sweep concurrency 64 → 2,048 (power-of-2) |
| `coding-agent` | **Primary** — the workload's true shape (shared scaffold + tool loop + short turns) | input 31,404 avg, output 1,024, request_rate 32/s |
| `shared-prefix-multitenant` | Reproduce the 74.1% cache hit | tune `shared_system_prompt_tokens` + `prefix_reuse: cross-session` to target 74% hit |

> **Workload shape is coding-agent**, not generic chat: 31K shared-context + short
> output + 74% prefix hit + diurnal bursts = an agent backend replaying a large shared
> scaffold (system prompt + tool defs + retrieved files) across turns with small completions.
> `coding-agent` is the primary card; `concurrency-sweep` finds the knee; `shared-prefix-multitenant`
> validates cache-hit reproduction.

### Full-stack optimization levers (prefill-bound, NOT decode tricks)

Because fresh prefill (~259K tok/s aggregate) dominates decode (~32.5K tok/s) ~8:1, the levers
that matter are prefill-interleave and NVFP4 MoE GEMM efficiency. Sweep these (see sidecar
`optimization:` block):
- [ ] **MNBT** (`max-num-batched-tokens`) sweep with chunked prefill — THE knob behind the
      customer's 15s p95 TTFT blowout at peak. Find the value that bounds TTFT without starving decode.
- [ ] **MoE backend identification first**, then tile/GEMM tuning *only if Triton*: confirm whether
      NVFP4 resolves to compressed-tensors / cutlass / FlashInfer-FP4 vs Triton. `benchmark_moe.py`
      tunes the **FP8 Triton** path — it is a likely **no-op for NVFP4** (qwen3-235b-spec L18, fin-rag L25).
      Record baseline-vs-tuned only if the backend is Triton; otherwise note N/A with the resolved backend.
- [ ] **`max-num-seqs`** admission-cap sweep — interacts with KV residency at 31K context.
- [ ] **KV cache dtype** sweep (`auto` vs `fp8`) — fp8 KV halves bytes/token, raising the residency knee.
- [ ] **CUDA graph** capture validated on plain NVFP4 (worth 6.4× on spec-decode per Kimi-spec L19).
- [ ] **HiCache CPU/NVMe tiering** toggle — the one structural lever that moves the knee without disagg.

**Required measurements:**
- [ ] Concurrency sweep 64 → 2,048 at fixed 31,404-token context completed.
- [ ] **Knee identified**: concurrency at which p95 TTFT crosses **15,000 ms** (customer peak ceiling).
- [ ] TTFT p50 < 2,400 ms and p95 < 15,000 ms at the target ~1,500 concurrent (customer's observed band).
- [ ] Per-request decode throughput ≥ **34.8 tok/s** sustained at the knee.
- [ ] No OOM at max sustained concurrency = _____ (record the saturation point).
- [ ] Error rate < 0.1% at all concurrency levels; no request timeouts.
- [ ] **If EAGLE3 spec decode is swept: validate accept rate on `coding-agent` (production-shaped)
      traffic, not just synthetic** (qwen3-235b-spec L15 — synthetic uniform prompts inflate accept
      rate 3-5× vs real traffic). Report accept rate per workload; base the go/no-go decode gain on
      the production-shaped number, not the synthetic one.

**KV cache validation (mandatory — this is the headline):**
- [ ] **Prefix-cache hit rate defined as `cached_input_tokens / total_input_tokens`** (token-fraction),
      computed from per-request usage fields — **NOT** the engine's native gauge. This is engine-agnostic
      so SGLang and vLLM numbers are directly comparable *to each other and to the customer's 74.1%*
      (which is a token fraction). vLLM exposes `prompt_tokens_details.cached_tokens` per response;
      SGLang exposes `cached_tokens` in its usage/meta. Sum across the run: `Σ cached / Σ prompt_tokens`.
- [ ] Tuned to reproduce **~74%** token-fraction hit (report cold-vs-warm TTFT ratio alongside).
- [ ] **Cross-check**: engine native gauge (`vllm:gpu_prefix_cache_hit_rate` / SGLang radix hit counters)
      recorded too, but only for operational signal — the token-fraction is the comparison number.
- [ ] KV cache utilization % at the knee < 95% (record where it saturates → the residency knee).
- [ ] HiCache CPU/NVMe tiering: measured net effect on the knee (toggle on/off; does it raise
      sustainable concurrency at 31K context, or just add latency?). Expect it to help only while
      capacity-bound, not under compute/BW-bound peak (kimi-k2.6 lesson).
- [ ] Eviction rate of the shared prefix under peak load (must stay low for 74% hit to hold).

**Engine-internal metrics (Prometheus `/metrics`):**
- [ ] SGLang: `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:token_usage`, radix cache
      hit counters; vLLM: `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting`.
- [ ] **SGLang requires `--enable-metrics`** or `/metrics` 404s and TTFT histograms are lost — smoke-test
      `up{job=...}=1` before any run (qwen3-235b-spec L8; the Kimi-spec data loss was exactly this).
      vLLM exposes by default but validate the scrape too.

**Bottleneck classification at the knee (decides whether B300 would even help):**
B200 vs B300 differ only in KV *capacity* (180 vs 275 GB/GPU), not HBM bandwidth. So measure which
wall we hit first and record the verdict — this makes the hardware choice empirical, not assumed.
- [ ] **PREREQUISITE (L8b fix):** the DCGM exporter MUST run with the custom `dcgm-metrics-prof` CSV
      (PROF fields) + `privileged: true`, else `DCGM_FI_PROF_*` return empty and this whole classification
      is `[inferred]` not `[measured]` (exactly what happened session 1). Verify
      `curl localhost:9400/metrics | grep DCGM_FI_PROF_DRAM_ACTIVE` is non-empty BEFORE the sweep.
- [ ] At the knee concurrency, capture DCGM `DCGM_FI_PROF_DRAM_ACTIVE` (HBM BW util) and
      `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (compute), alongside `vllm:gpu_cache_usage_perc` and
      `vllm:num_requests_waiting`. Classify the regime:
  - **HBM-BW-bound** (`DRAM_ACTIVE` >~80%, cache not full) → B300 does NOT help; **B200 result transfers to B300 unchanged.**
  - **KV-capacity-bound** (`gpu_cache_usage_perc` ~100% + `num_requests_waiting` rising, `DRAM_ACTIVE` not pegged) → the ONE case B300's extra capacity (or HiCache tiering) raises the knee. Flag result as B200-capacity-bound, NOT Blackwell-bound.
  - **Prefill-compute-bound** (`PIPE_TENSOR_ACTIVE` high) → tune MNBT/chunked-prefill, not hardware.
- [ ] Record the regime in the readiness audit + lessons. If KV-capacity-bound, re-run the
      capacity-limited concurrency points with HiCache enabled to quantify how much the tier recovers
      before recommending B300.

**Cost output (the go/no-go):**
- [ ] $/Mtok at the knee = (node $/hr) ÷ (aggregate tok/s × 3600 / 1e6), split input vs output.
- [ ] Compare to $0.338 blended / $0.363 input / $3.76 output upstream → recommendation.

**Enriched artifact**: store in `blueprints/kimi-k2.6-nvfp4/results/` per
`standards/benchmark-commons/PROPOSAL.md`.

### Stage 6c — Parallelism sweep (session-1 GAP: only TP8×1 was tested)

Session 1 swept KV/MNBT/spec/engine but **never varied the parallelism layout** — despite
`kimi-k2.6-speculative` L20 showing TP4+DP2 beat TP8 by +14% at c=256. So "single-node ceiling ~2,500
tok/s" is really "ceiling for TP8×1." Re-run the matrix below. **Verify each against LATEST upstream
before assuming a prior crash still holds** — e.g. the `forward_deepgemm_masked` EP crash (our notes,
SGLang 0.5.10) is SGLang issue #16952, **now CLOSED/fixed**, so EP may work on 0.5.13.post1. Always
`mdc prs kimi-k2.6` + check sgl-project issues for the running version, don't trust dated lessons.

**Fit constraint (520 GB NVFP4 weights, 180 GB/GPU B200):** min ~4 GPUs to hold the model.
- TP8×1 → 65 GB/GPU ✓ (baseline, done: 2,516 @ c512)
- TP4+DP2 (2 replicas) → 130 GB/GPU ✓, **35 GB KV room/GPU** (tighter — watch KV saturation)
- TP2+DP4 → 260 GB/GPU ✗ **DOES NOT FIT** — skip
- TP4+EP4 → 130 GB/GPU ✓
- EP8 / wide-EP (attn TP8 + expert-parallel) → ✓ fit

| Config | flags | hypothesis | prior evidence |
|--------|-------|-----------|----------------|
| TP4+DP2 | `--tp 4 --dp 2` | smaller per-replica batch → +14% @ high conc | Kimi-spec L20 (+14% @ c256) — **strongest** |
| 4P/4D disagg | `--disaggregation-mode` split, TP4 each. **REQUIRED or 40× TCP fallback**: pod `hostIPC: true` + env `UCX_TLS=cuda_copy,cuda_ipc` + `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on` + `IPC_LOCK` cap (pd_disagg_single_node memory) | dedicated decode GPUs remove prefill interference | risky: NVSwitch TP>1 cuda_ipc contention; 4/4 ONLY (2-GPU can't hold 1T model) |
| TP4+EP4 | `--tp 4 --expert-parallel-size 4 --moe-a2a-backend deepep` (NOT `--ep-size`/`--enable-ep-moe` — wrong on 0.5.x, L11) | expert sharding for 384-expert MoE | qwen3-235b L13: single-node EP −14 to −39%. **Do NOT combine DP+EP** — L12 `Rank 0 scheduler died` (SGLang lacks full DP-attn+EP integration). Verify upstream first (prereq below) |
| EP8 wide-EP | `--expert-parallel-size 8 --moe-a2a-backend deepep` | 384 experts spread across 8 GPUs | LMSYS: 5.2× is MULTI-node only; single-node likely null/negative — measure to confirm on NVFP4 |

- [ ] **PREREQUISITE — verify upstream before trusting ANY dated crash lesson** (applies to every EP/disagg
      row, not just one): `mdc prs kimi-k2.6` + check sgl-project issues for the RUNNING version
      (0.5.13.post1). Known-relevant: #16952 (`forward_deepgemm_masked` EP crash — CLOSED/fixed, so EP may
      now work), #25331 (CompressedTensors ReplicatedLinear — use modelopt ckpt), and the DP-attn+EP
      integration gap behind L12's `Rank 0 scheduler died`. Do NOT assume 0.5.10-era crashes still apply.
- [ ] Run each config at c=256 + c=512 (knee region) vs TP8 baseline; record agg tok/s, TTFT, **DCGM roofline** (which wall), KV `token_usage`.
- [ ] **Label each [measured] with the regime; do NOT call the matrix "exhausted" off partial data** (benchmark-analysis.md).
- [ ] EAGLE3 `s4_d4_k1` num-steps tuning (session-1 miss): only relevant if a config moves OFF compute-bound — re-test spec decode per winning layout, not just TP8.

### Stage 7 — Readiness Audit
- [ ] All readiness categories pass; no unresolved HIGH-severity lessons.
- [ ] Knee, cache-hold result, and $/Mtok recorded with the disagg go/no-go recommendation.
- [ ] All criteria above checked and recorded.

---

> Operational artifacts (lessons, results, deployment notes) belong in
> `blueprints/kimi-k2.6-nvfp4/`, not in this spec.
