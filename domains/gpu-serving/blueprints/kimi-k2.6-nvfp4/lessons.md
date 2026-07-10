---
model: "kimi-k2.6"
engine: "sglang"
hardware: "p6-b200.48xlarge"
gpu_arch: "sm_100"
deployment_date: "2026-06-16"
outcome: "success"
failure_categories: []
cards_used:
  mdc: ["kimi-k2.6"]
  gpu_infra: ["p6-b200"]
card_helped: true
benchmark:
  throughput_toks_s: 3187
  concurrent_users: 1024
  cache_hit_fraction: 0.74
  knee_concurrency: "1024"
  bottleneck: "decode-compute-saturated (TP8), scheduling-inefficiency resolved by TP4+DP2 (+25%); ~26% prefill-bound (TTFT share)"
  parallelism_winner: "TP4+DP2 (3,187 tok/s @ c=1024) vs TP8 (2,578 tok/s)"

learn_commands:
  - 'mdc learn kimi-k2.6 sglang "NVFP4 requires -cu130 image (cutlass DSL for FP4 kernels); pip install nvidia-cutlass-dsl into cu129 does NOT work"'
  - 'mdc learn kimi-k2.6 sglang "EAGLE3 bf16 draft + FP4 base requires --speculative-draft-model-quantization unquant to avoid FP4-on-bf16 ValueError"'
  - 'mdc learn kimi-k2.6 sglang "Spec decode hurts throughput on compute-bound decode (-12 to -19% c64-512) despite 3.74 accept length — draft competes with batch decode when compute saturated"'
  - 'mdc learn kimi-k2.6 sglang "NVFP4 checkpoint ships fp8 KV by default (kv_cache_scheme:fp8) + group_size 16 (not block_n 128) — fp8-KV flag no-op, fp8-moe-tp-divisibility N/A"'
  - 'mdc learn kimi-k2.6 sglang "TP4+DP2 beats TP8 by +19-25% at high concurrency (c=256-1024) for 1T NVFP4 MoE — TP8 funnels into oversized batch, TP4+DP2 schedules efficiently; always sweep both"'
  - 'mdc learn kimi-k2.6 sglang "4P/4D single-node disagg = 3.8× regression (815 vs 3,138 tok/s TP4+DP2) — SGLang disables prefix cache (disable_radix_cache=True) in disagg decode, loses the 74% cache"'
  - 'gpu-infra learn -c platform "ECC gate for reused spot GPUs: check volatile.total==0 AND remapped_rows.{pending,failure}==No AND uncorrectable==0, NOT lifetime aggregate. Blackwell uses remapped_rows.*, not retired_pages.*"'
  - 'gpu-infra learn -c platform "qwen3-next-bench-eks-cluster: no GFD/NFD -> manual nvidia.com/gpu.present=true + blueprint label per scale-up; taint ai-infra/b200 (not nvidia.com/gpu); dnsPolicy:Default for egress; NVMe raw (RAID-0); node role lacks S3 write; DCGM profiling unavailable on driver-580 (use engine gauges); mountPropagation: HostToContainer for post-kubelet NVMe mounts"'
  - 'gpu-infra learn -c inference "SGLang disagg transport: mooncake picks TCP on no-IB (ignores NVLink); use NIXL for same-node. UCX_TLS needs sm,self,tcp for control-plane, not just cuda_copy,cuda_ipc. CUDA_VISIBLE_DEVICES conflicts with --base-gpu-id (pick one)"'
---

# Kimi K2.6 NVFP4 — Lessons Learned

## Session: 2026-06-16/17 (EKS, B200 spot, session 1 + session 2 parallelism sweep)

**Hardware**: p6-b200.48xlarge spot (8× B200, sm_100, NVSwitch), us-east-2b / use2-az2
**Cluster**: qwen3-next-bench-eks-cluster, nodegroup `ai-infra-use2-b200-spot`
**Engines under test**: SGLang ≥0.5.9 (modelopt_fp4) vs vLLM latest — head-to-head, pick winner
**Model**: `nvidia/Kimi-K2.6-NVFP4` (modelopt ckpt)
**Goal**: single-node knee-finding for coding-agent workload (31K ctx, 74% cache hit, ~1,500 peak conc)

---

### CRITICAL CORRECTION — TTFT was NOT measured in sessions 1-2; re-measured session 3 (B300, standard runner)

Sessions 1-2 reported "p50/p99 latency" that was **end-to-end (full 1024-tok response), NOT TTFT** — the
hand-rolled bench was non-streaming. The customer's PRIMARY SLO is TTFT (avg 3.3s, p95 5.8-15s), which was
never measured. Session 3 fixed it with the standard `bench-standard.py` (Prometheus TTFT histograms).
**Process failure: never validated the harness emitted the brief's SLO metrics before running. Lesson: a
benchmark harness must emit the standard quartet (TTFT/ITL/TPOT/throughput) and be checked against the
spec's SLOs BEFORE the first paid run; use bench-standard.py, don't reimplement.**

Bug found in the standard runner too: `bench-standard.py` queries colon-form metric names
(`sglang:time_to_first_token_seconds_bucket`) but this Prometheus normalizes `:`→`_`
(`sglang_time_to_first_token_seconds_bucket`) → silently empty TTFT. Patched colon→underscore. (TPOT/ITL
+ DCGM-PROF still None — same colon/driver issues; TTFT + E2E land, which covers the primary SLO.)

### Session 3 — MEASURED TTFT vs customer SLO (B300 TP4+DP2, cold prompts = conservative upper bound)
| context | conc | TTFT p99 | 15s SLO | agg tok/s |
|---------|------|----------|---------|-----------|
| 16k (peak-hrs) | 64  | 5.98s | ✅ | 5,161 |
| 16k | 128 | 3.88s | ✅ | 8,981 |
| 16k | 256 | 3.97s | ✅ | 12,478 |
| 16k | 512 | 7.88s | ✅ | 14,616 |
| 31k (quiet-hrs) | 128 | 9.93s | ✅ | 5,784 |
| 31k | 256 | 7.66s | ✅ | 8,469 |
| 31k | 512 | **19.42s** | ❌ | 9,749 |
| 16k | 768  | **19.72s** | ❌ | 14,816 |
| 16k | 1024 | 19.64s | ❌ | 15,950 |
| 16k | 1536 | 37.52s | ❌ | 10,945 (collapsing) |

**Findings (CEILINGS NOW FOUND on both contexts — pressure-tested to breach):**
(1) **16k (peak-traffic context): SLO-safe to ~c512** (p99 7.9s), **breaches at c768** (19.7s). Ceiling c512-600.
(2) **31k (quiet hours): SLO-safe to ~c256** (7.7s), **breaches at c512** (19.4s). Ceiling c256-400.
(3) Beyond the knee, throughput stops helping while TTFT explodes: 16k c768 agg=14,816 (≈c512's 14,616)
but p99 19.7s; c1536 agg DROPS to 10,945 with p99 37.5s. So pushing past the SLO ceiling buys ~0 aggregate
and wrecks latency. (3) This **corrects the earlier decode-rate "scale out every ~50 users"
claim — that was wrong** (built on E2E mislabeled as TTFT). Real TTFT-bound capacity is **c256-512/node**
depending on context — much higher. (4) Context (prefill) drives TTFT as expected: 16k handles ~2× the
concurrency of 31k at the same SLO. Caveat: cold prompts (no 74% cache) = conservative upper bound; cached
production TTFT is lower. TPOT not captured (metric-name/DCGM gaps).

### L1 — Device plugin needs manual `nvidia.com/gpu.present=true` label (this cluster has no GFD/NFD)

On scaling up the B200 node, it reached `Ready` but `nvidia.com/gpu` stayed empty for >3 min.
Root cause: `nvidia-device-plugin` daemonset selects `nvidia.com/gpu.present=true`, but this cluster
runs **no GPU-feature-discovery / NFD**, so nothing auto-applies that label, and the nodegroup config
only sets `ai-infra/role=b200-spot` + instance-type. The daemonset showed DESIRED=0 (nothing matched).

**Fix**: `kubectl label node <node> nvidia.com/gpu.present=true --overwrite`. Device plugin scheduled
within ~10s and all 8 GPUs registered.

**Rule**: On this cluster, every fresh GPU node needs BOTH labels applied manually after scale-up:
`blueprint=<name>` (for pod nodeSelector) AND `nvidia.com/gpu.present=true` (for the device plugin).
Consider adding `nvidia.com/gpu.present=true` to the nodegroup's `labels` in Terraform to make it
automatic. Watch for `kubectl get ds nvidia-device-plugin -n kube-system` DESIRED=0 as the symptom.

### L2 — B200 node taint is `ai-infra/b200=true:NoSchedule`, NOT `nvidia.com/gpu`

Pods must tolerate `{key: ai-infra/b200, operator: Exists, effect: NoSchedule}` to schedule onto this
nodegroup. The reference qwen3-235b manifests tolerated `nvidia.com/gpu` — that taint isn't even present
here, so those tolerations are no-ops and pods sit Pending with
`untolerated taint {ai-infra/b200: true}`. All kimi-k2.6-nvfp4 manifests carry the `ai-infra/b200`
toleration. Confirm a node's actual taints with `kubectl get node <n> -o jsonpath='{.spec.taints}'`
before assuming the convention.

### L3 — "Zero uncorrected ECC aggregate" is the WRONG Stage 4a gate for reused spot GPUs

GPU1 showed `ecc.errors.uncorrected.aggregate.total = 143` but `volatile.total = 0`,
`remapped_rows.pending = No`, `failure = No`, `uncorrectable = 0`. The 143 is **lifetime aggregate**
(prior spot tenant); the row remapper already retired the affected rows cleanly. This is a HEALTHY GPU.
A strict "aggregate == 0" gate (as written in the spec/template Stage 4a) false-fails essentially every
reused spot Blackwell GPU.

**Rule**: Gate on `ecc.errors.uncorrected.volatile.total == 0` AND `remapped_rows.{pending,failure}==No`
AND `remapped_rows.uncorrectable == 0` — NOT on lifetime aggregate. Aggregate is informational only.
`retired_pages.*` returns `[N/A]` on B200 (Blackwell uses `remapped_rows.*` instead — don't gate on
retired_pages for sm_100).

### L4 — Two storage/network gotchas on fresh B200 spot nodes (no bootstrap automation)

(a) **Local NVMe is raw and unmounted.** The node mounts only a 500 GB EBS root at `/` (and the
reference `/mnt/nvme` hostPath pointed there → only 489 GB free, too small for 520 GB weights). The
8× 3.8 TB instance-store disks (`/dev/nvme1n1`–`nvme8n1`, 30 TB raw) were unformatted. Fix: RAID-0 them
(`mdadm --create /dev/md0 --level=0 --raid-devices=8 ...`, `mkfs.xfs`, mount at `/mnt/nvme`) via a
privileged pod that `nsenter -t 1 -m`'s into the host mount namespace so the mount is visible to
subsequent hostPath pods. Result: 28 TB at `/mnt/nvme`. Spot reclaim wipes this — re-run on every fresh node.

(b) **Pods need `dnsPolicy: Default` for external DNS on this cluster.** Default pod DNS (CoreDNS
172.20.0.10) and `ClusterFirstWithHostNet` BOTH fail external resolution from the B200 node
(`Temporary failure in name resolution` on huggingface.co/pypi.org) — only in-VPC names work, matching
the kimi-k2.6-spec L14 VPC-endpoint-interference lesson. Setting `dnsPolicy: Default` (use the node's
VPC resolver, NOT CoreDNS) fixes it. The reference bench-runner pod already used `dnsPolicy: Default` for
exactly this reason. **Rule**: any pod that must reach the public internet (HF download, pip) on this
cluster needs `hostNetwork: true` + `dnsPolicy: Default`.

### L5 — HF download: disable Xet for large multi-shard models; make the loop the pod's PID 1
huggingface_hub 1.19 ignores `HF_HUB_ENABLE_HF_TRANSFER` and uses Xet by default; on this node Xet
silently stalled at ~170/600 GB (process alive, holding `.lock`, zero IO). Fix: `HF_HUB_DISABLE_XET=1` +
`HF_HUB_ENABLE_HF_TRANSFER=0` (plain HTTPS, ~400 MB/s) + a resilient retry loop AS the pod's PID 1
(`for a in seq 30; do hf download ... && break; sleep 5; done`); clear stale `.cache/.../*.lock` first.

### L6 — Observability pod conflicts on a shared cluster (node-exporter port + prom-data perms)
node-exporter `:9100 bind: address already in use` (cluster already runs one) → drop our node-exporter,
scrape the existing one. Prometheus query_logger panic → pre-`chown 65534:65534 /mnt/nvme/prom-data`.

### L7 — SGLang Kimi NVFP4 needs the `-cu130` image (cu129 lacks the cutlass DSL)
cu129 crashes at warmup: `ModuleNotFoundError: No module named 'cutlass'` (FlashInfer FP4 path). `pip
install nvidia-cutlass-dsl` does NOT fix it. `v0.5.13.post1-cu130` bundles it and runs fine on B200/sm_100.

### L8 — DCGM profiling roofline UNAVAILABLE on this driver-580 + DCGM combo (use engine gauges)
PROF fields report "metric not enabled" (3.3.9) or export nothing (4.1.1). Classify bottleneck from SGLang
engine gauges instead — `token_usage` (KV %), `num_running_reqs` vs `num_queue_reqs` — more direct anyway.

### L9 — EAGLE3 bf16 draft from local path: needs `--speculative-draft-model-quantization unquant`
Local-path draft tripped HF repo-id validation (missing hf_quant_config) AND inheriting base modelopt_fp4
→ `Invalid quant_config ... Eagle3DeepseekV2ForCausalLM`. Fix: `--speculative-draft-model-quantization
unquant`. Do NOT inject a fake hf_quant_config.json (makes it worse).

### L10 — Node IAM role lacks S3 write (silent data-loss risk)
`ai-infra-use2-b200-node` denied `s3:PutObject`/`s3:ListBucket`. Workaround: `kubectl cp` results to the
workstation, push from local creds. Verify S3 write EARLY, not at teardown.

### L11 — (see DCGM, folded into L8)

### L12 — hostPath pods need `mountPropagation: HostToContainer` for a post-kubelet NVMe mount
RAID `/mnt/nvme` created via nsenter AFTER kubelet started → pods see `No such file or directory` (stale
containerd mount ns) unless the volumeMount sets `mountPropagation: HostToContainer`. EVERY pod mounting
`/mnt/nvme` needs it. Symptom: `df: /mnt/nvme: No such file or directory` in-pod while host has the mount.

### L13 — single-node disagg: `CUDA_VISIBLE_DEVICES` and `--base-gpu-id` CONFLICT
`CUDA_VISIBLE_DEVICES=4,5,6,7` remaps to logical 0-3, so `--base-gpu-id 4` requests a non-existent device
→ `device 4 is not visible`. Use ONE mechanism (CUDA_VISIBLE_DEVICES per container, drop --base-gpu-id).
Crash fires at GPU-init AFTER weight load (~25 min wasted) — validate with `--load-format dummy` first.

### Stage 6c lever — TP4+DP2 vs TP8: CONFIRMED WIN (+19-25%, client-measured)

Session-2 parallelism sweep (the session-1 omission, and the FIRST lever to lift the ceiling).
TP4×DP2 (2 replicas of TP4) vs TP8×1 baseline. **[measured, client-side agg decode tok/s, 0 err, cache 0.74]:**

| conc | TP8 baseline | TP4+DP2 | gain | p50 lat (TP8→DP2) |
|------|--------------|---------|------|-------------------|
| 256  | 2,569 | **3,067** | **+19%** | 102.5s → 73.1s |
| 512  | 2,516 | **3,138** | **+25%** | 185.6s → 142.4s |

**Wins on BOTH throughput AND latency.** Mechanism = kimi-spec L20: TP8 funnels all concurrency into one
oversized batch that schedules inefficiently; two TP4 replicas each run a smaller, better-amortized batch.

**Method note (verify-before-assert worked):** the instantaneous engine `gen_throughput` gauge showed
~4,040 tok/s mid-run (+57%) — but the authoritative client-side end-to-end number is +19-25%. The gauge
overstates because it excludes TTFT/queueing. I labeled the +57% preliminary and corrected to the client
number — exactly the discipline. **Always finalize on client agg tok/s, not the engine gauge.**

Tradeoff: TP4+DP2 KV pool = 535,872 tok/replica vs TP8's 2.6M (extra weight copy eats KV room). Held fine
through c512 (0 err). **Implication: the "single-node ceiling ~2,500 tok/s" from session 1 was a TP8×1
artifact — the real single-node number is ~3,140 at TP4+DP2.** Node-count for 32,500 tok/s drops 13→~11.

### Stage 6c lever — EP on NVFP4: DeepEP a2a BLOCKED upstream (#24502, OPEN), flashinfer a2a incompatible

The old EP crash (`forward_deepgemm_masked deprecated`, our notes / SGLang #16952) IS fixed on 0.5.13 —
but EP+NVFP4 hits a **new, different, currently-unfixed** wall. `--expert-parallel-size 4 --moe-a2a-backend
deepep` crashes at startup:
`NotImplementedError: Runner backend MoeRunnerBackend.FLASHINFER_TRTLLM requires a fused func for a2a
backend deepep, but none is registered.` Root cause: NVFP4 forces `moe_runner_backend=flashinfer_trtllm`,
which has **no DeepEP fused all-to-all** — confirmed live as **SGLang issue #24502 (OPEN, 2026-05-06):
"flashinfer_trtllm MoE runner has no DeepEP fused func registered — blocks EP+NVFP4 on Blackwell."**

**This is the "check upstream, don't trust the old crash signature" lesson paying off twice:** the dated
crash was fixed, but a *current* issue blocks the same config for a new reason. Tried `--moe-a2a-backend flashinfer` (matches the trtllm runner) as the alternative — **also fails**:
`AssertionError: Flashinfer MoE A2A is only supported with dp_size == tp_size and --enable-dp-attention`.
That would require TP4+DP4 (doesn't fit — 8 GPUs only, DP4×TP4=16) AND `--enable-dp-attention`, which is
the known-fragile DP-attn+EP combo that crashed `Rank 0 scheduler died` (qwen3-235b L12).

**CONCLUSION — EP/wide-EP is NOT available for Kimi NVFP4 on SGLang 0.5.13 today**, both a2a paths blocked:
- `deepep` → no fused func for flashinfer_trtllm runner (#24502 OPEN)
- `flashinfer` → requires dp==tp + dp-attention (doesn't fit 8 GPUs, fragile combo)
This is an **upstream capability gap, not a measured perf result** — we couldn't even start it. Revisit
when #24502 lands. (Aligns with prior "single-node EP doesn't pay off" but here it's a hard block, not a
regression.) **Wide-EP's real home remains multi-node at frontier scale, per LMSYS.**

### Stage 6c lever — 4P/4D single-node disagg (NIXL/NVLink): 3.8× SLOWER than TP4+DP2
After fixing 4 config traps (GPU-split L13; mooncake→NIXL; UCX_TLS needs sm,self,tcp control-plane not just
cuda data transports; backend init), disagg ran CORRECTLY (0 err, NIXL/UCX, no TCP fallback). **Result:
815 tok/s @ c512 vs TP4+DP2's 3,138 — 3.8× REGRESSION.** Causes: (1) SGLang forces `disable_radix_cache=True`
in disagg decode mode → the **74% prefix cache is GONE**; (2) decode GPUs starve (num_running=1,
token_usage=0.12) on the prefill→KV handoff. Disagg is multi-node-at-scale only; not 4P/4D on one node.
Transport notes: mooncake ignores NVLink (picks TCP w/o IB HCA → use NIXL for same-node; got 222 tok/s,
248 err on the TCP path); EFA irrelevant for same-node (NVLink/cuda_ipc is the path).

### Stage 6c — MoE tile tuning & MNBT: TP-independent null
MoE tile tuning N/A on TP4 same as TP8 — backend is quant-driven (NVFP4→flashinfer_trtllm, vendor
pre-compiled, never reads Triton tile JSON), not geometry-driven. MNBT spot-check on TP4+DP2: 32768 @ c512
= 3,148.9 vs default 16384 = 3,138.3 (+0.3%, noise) — measured-null, geometry-independent.

### Stage 6c SCOREBOARD — parallelism sweep complete (session-1 omission)
| config | best tok/s | vs TP8 | verdict |
|--------|-----------|--------|---------|
| TP8×1 (baseline) | 2,578 | — | session-1 "ceiling" was a TP8 artifact |
| **TP4+DP2** | **3,187 @ c1024** | **+19-25% across c128-1024** | **WINNER** (better latency, holds to c1024) |
| TP4+EP4 / wide-EP | — | — | BLOCKED upstream (#24502) |
| 4P/4D disagg (NIXL) | 815 | −68% | loses prefix cache + GPU starvation |

TP4+DP2 full knee: c128 2,845 / c256 3,067 / c512 3,138 / c1024 3,187 (all +19-25% vs TP8, 0 err, cache 0.74).
**New single-node ceiling ~3,190 (TP4+DP2), not 2,516. Node count for 32,500: 13→~11.**

### Stage 6d — TP4+DP2 on B300 (us-west-2): KV pool 5.6× larger, throughput pending
B300 confirmed 275,040 MiB/GPU. **[measured] KV pool: B300 TP4+DP2 = 2,992,960 tok/replica vs B200's
535,872 — 5.6× larger** (fixed ~130GB weight cost leaves proportionally more of 275GB for KV). Confirms
B300 eases concurrency/residency. Bring-up: same gotchas as B200 cluster (L1/L2/L4/L12) with `ai-infra/b300`
taint; HF token copied cross-cluster; cu130 native here.

**[measured] RESULT — B300 lifts PEAK throughput +36-43%, not just concurrency (corrected my inferred guess):**
| conc | B200 TP4+DP2 | B300 TP4+DP2 | gain | B300 token_usage |
|------|--------------|--------------|------|------------------|
| 512  | 3,138 | **4,489** | **+43%** | 0.75 (headroom) |
| 1024 | 3,187 | **4,327** | **+36%** | 0.94 (approaching knee) |

**My earlier [inferred] "B300 won't raise peak (BW-bound), node count stays ~11" was WRONG — measurement
corrected it.** Why: B200's 3,138 @ c512 was NOT a pure compute/BW ceiling — token_usage ~0.95 means it was
**partly KV-capacity-throttled** (requests queuing for KV room). B300's 5.6× KV pool runs more concurrently
(0.75 usage @ c512 = headroom) → higher aggregate. So decode wasn't as BW-bound as the B200-only data
suggested; KV capacity was a real co-constraint. **Node count for 32,500 tok/s: ~11 (B200 TP4+DP2) →
~8 (B300 TP4+DP2 @ 4,327-4,489).** Lesson: don't infer "BW-bound, capacity won't help" without the
capacity-headroom measurement — token_usage at the knee is the discriminator, and B200 c512 (0.95) vs B300
c512 (0.75) is the proof capacity was binding.
