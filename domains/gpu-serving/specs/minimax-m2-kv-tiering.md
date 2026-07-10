# MiniMax-M2 Distinct-Prefix KV-Tiering Requirements

## Status: DRAFT

## Overview

Follow-on to `minimax-m2.md`. That study measured the **single-shared-prefix** regime and found CPU KV offload **neutral** (±4%) — because one 90K prefix fits trivially in GPU HBM, so there is nothing to evict and nothing for an offload tier to relieve.

This spec measures the regime where tiered KV cache **should** pay: **many distinct codebases / prefixes, aggregate working-set > GPU HBM**, so reusable prefixes get evicted and re-referenced. In that regime:
- **gpu-only** must **recompute** a ~90K-token prefill every time an evicted prefix is re-referenced (expensive, prefill-bound).
- **cpu-offload / nvme-tiering** can **fetch** the evicted prefix's KV from the CPU/NVMe tier instead of recomputing — trading a PCIe/NVMe transfer for a 90K prefill.

**The hypothesis (from `minimax-m2/results/report.html` architecture-decision framework, [inferred — now to be MEASURED]):** in the distinct-prefix regime, tiering relieves **TTFT and the pending queue** (shorter holding time → faster slot turnover), but does **NOT** raise the decode throughput ceiling (active-decode counter-space is unchanged). And **high per-codebase reuse is structurally anti-disagg** — disagg isolates prefill *compute*, but tiering is the lever for prefix *capacity*. This run produces the data to confirm or refute that, and to size the CPU/NVMe tiers.

## The experimental knob (measured-grounded sizing)

The make-or-break variable is **the number of distinct prefixes (codebases), N**, sized relative to the GPU KV pool so eviction actually happens:

- B200 4×180GB = 720GB; weights ~228GB → **~492GB KV pool** (fp8).
- fp8 KV ≈ 124 KB/token (62 layers, 8 KV heads, head_dim 128) → pool holds **~3.87M tokens**.
- One 90K-token codebase prefix ≈ **11.4 GB** → **~43 distinct 90K prefixes fill the pool.**
- **Sweep N ∈ {22, 43, 86, 172}** = {0.5×, 1×, 2×, 4×} the fill threshold → spans under-fill (no eviction, expect tiering neutral like the prior run) → at-capacity → 2×/4× overflow (heavy eviction, where tiering must beat recompute).

Each request samples a codebase from the pool of N (zipfian by default — real fleets have hot/cold repos; uniform as a stress control), appends a small unique suffix, so **per-codebase reuse stays ~90%** but the *aggregate* working set overflows HBM.

## Components

### 1. Compute
- **Instance**: `p6-b200.48xlarge` (8× B200, SM100, NVSwitch), TP4. Same as `minimax-m2`. NVMe at `/mnt/nvme` backs the disk tier.
- **Cluster/nodegroup**: `qwen3-next-bench-eks-cluster` / **`ai-infra-use2-b200-spot-maz`** (multi-AZ spot, us-east-2 2a+2b — B200 on-demand capacity hops AZs, so a single-AZ NG whack-a-moles; a multi-subnet NG lands wherever capacity is). **Pin `--context qwen3-next-bench-eks-cluster` in every kubectl call**; label nodes by selector (tech-stack.md unattended-runner rules). The orchestrator's scaledown trap MUST target this exact NG name (a hardcoded wrong name = idle-leak).
- **Fresh-node NVMe**: a freshly-launched B200 spot node has raw, UNMOUNTED instance-store — `/mnt/nvme` does not exist and the stage Job's hostPath(type:Directory) mount fails forever. Preflight MUST format+mount one instance-store disk at `/mnt/nvme` before staging (nsenter privileged pod; idempotent).

### 2. Model & serving
- Reuse `minimax-m2` serving config verbatim: MiniMaxAI/MiniMax-M2 FP8, TP4+EP, `--moe-backend triton`, `VLLM_USE_FLASHINFER_MOE_FP8=0`, fp8 KV, minimax_m2 parsers. Same Stage-0c B200 correctness gate.
- **Engine image: `vllm/vllm-openai:v0.23.0`** (the customer's known-good image; has `/v1/messages` for code agents, which 0.19 lacks). NOT `minimax27` (stale, hyphenated-model-dir bug). Serving command MUST include the integrity-repair preamble + hyphen-free `minimax_m2` symlink (see `minimax-m2/lessons.md` "DEFINITIVE root cause").
- **Single source of truth: `domains/gpu-serving/blueprints/minimax-m2/k8s/gen-serving-manifest.sh` is the ONE canonical manifest generator.** Every runner (gate, tiering-sweep, pareto-sweep) MUST invoke that exact file — do NOT keep a second copy under `minimax-m2-kv-tiering/k8s/` (they diverge silently and you'll benchmark a stale config; see lessons "SPLIT-BRAIN BUG").
- **3 KV arms** (the comparison) — CONFIRMED against vLLM **v0.23.0** source (`vllm/v1/kv_offload/`):
  - `gpu-only`: APC only — must recompute on evict (the baseline this experiment stresses).
  - `cpu-offload`: `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":<BYTES>}}'`. **Key is `cpu_bytes_to_use` (BYTES), NOT `num_cpu_blocks`** (that was v0.11; the schema changed — using the old key fails at KV-init with "cpu_bytes_to_use must be specified"). There is NO `--kv_offloading_backend`/`--kv_offloading_size` CLI flag. Size to hold the N=172 overflow (default 200 GiB).
  - `nvme-tiering`: same connector with `"spec_name":"TieringOffloadingSpec"`, `"cpu_bytes_to_use":<BYTES>`, `"secondary_tiers":[{"type":"fs","root_dir":"/mnt/nvme/kv-cache"}]`. The `fs` tier (FileSystemTierManager) IS supported on 0.23 (registered tiers: `example`, `fs`, `obj`). Earlier `nvme_path`/`num_nvme_blocks` keys were silently ignored — do NOT use them.
- **Caveat to control**: the NVMe KV tier (`/mnt/nvme/kv-cache`) shares the physical disk with the 214GB model weights. Weights are HBM-resident after load, so the disk is mostly free during serving — but record NVMe I/O contention; a slow NVMe-fetch result may be spindle contention, not the tier's ceiling.
- **DP is NOT available**: every `--data-parallel-size` shape (tp2dp2/tp4dp2/tp2dp4) fails engine-init on M2/0.23 (DPLBAsyncMPClient path). Scale concurrency with TP4 + higher `--max-num-seqs`, not DP replicas, unless/until DP boot is root-caused.

## Non-Requirements
- P/D disaggregation — **explicitly out of scope** and predicted NOT to help here (high per-codebase reuse → prefill cheap on hit → little prefill/decode contention for disagg to isolate). This experiment is the evidence that tiering, not disagg, is the lever for this regime.
- Single-shared-prefix scenario — covered by `minimax-m2.md`.

## Verification Criteria

### Stage 0a — PRE-FLIGHT DRY RUN (mandatory; runs OFF the GPU — gate every B200 launch on this)

> Every failed B200 cycle this engagement was a bug that a cheap check would have caught BEFORE staging.
> This stage is fail-closed: **do not scale up a B200 until ALL of these pass.** (See `minimax-m2/lessons.md`
> "PROCESS / SPLIT-BRAIN / NEAR-MISS" lessons for the incidents these guard against.)

**Offline static gate (RUNNABLE — must exit 0 before any GPU spend):**
- [ ] `bash domains/gpu-serving/blueprints/minimax-m2/scripts/preflight-dry.sh` exits 0. It enforces, as executable checks (NOT prose): single/byte-identical manifest generator (no split-brain), `bash -n` on every orchestrator, no hardcoded wrong/stale NG name, manifest JSON validity + v0.23 keys (`cpu_bytes_to_use`, not `num_cpu_blocks`), `tp4ep4` shape generates, image is `v0.23.0` (not `minimax27`), and no external `pgrep` scaledown supervisor. **This script already caught a live stale-NG bug — keep extending it, don't replace it with checkboxes.**

**Cheap-node live gate (a $1 CPU/g6e pod — catches what static checks can't):**
- [ ] **Engine-config dry-load**: boot the EXACT serving command for EACH arm (gpu-only/cpu-offload/nvme-tiering/tp4ep4) far enough to pass `EngineArgs.create_engine_config` (fails fast on bad flags/keys WITHOUT a GPU). This is where a `cpu_bytes_to_use`-class schema error surfaces for ~$1 instead of a B200 cold-start.
- [ ] **Bench client end-to-end, FAIL-CLOSED**: run the ACTUAL benchmark client (same command-line as the sweep) against the dry pod. Require `err_rate < 0.5` AND `throughput > 0`. **If err=1.0 or zero-throughput, STOP and debug the client (parsing / dcgmi / metric-extraction) before staging to B200** — a `/health` 200 is necessary but NOT sufficient (this is the exact gap that zeroed the EP-compare via a `dcgmi -r` parse error while pods were healthy).

### Stage 0b — Regime prediction (roofline, validate BEFORE spending GPU time)

The hypothesis "tier-fetch beats recompute-on-evict" is a roofline claim — pre-validate the arithmetic:
- **Recompute** a 90K-token prefill: prefill is compute-bound. On TP4 B200 (~4 GPU × ~990 FP8 TFLOPS, MoE 10B active) a 90K prefill is on the order of **hundreds of ms to seconds** — and it re-occupies the GPU's prefill path, stealing SMs from concurrent decode.
- **Fetch** the same prefix's KV from the CPU tier: ~11.4 GB (90K tok × 124 KB/tok fp8) over PCIe Gen5 (~50–60 GB/s effective host↔device) ≈ **~200 ms**, off the compute path.
- **Prediction**: at N > 43 (pool overflow → eviction), fetch < recompute → tiering improves **TTFT + queue** (shorter holding time → faster slot turnover), while the **decode throughput ceiling stays flat** (active-decode counter-space is unchanged — the negative control). NVMe tier adds a slower hop (~3–7 GB/s) → expect it to help only when CPU RAM also overflows, and watch for NVMe-fetch latency becoming the new bottleneck.
- **Tier ledger** (single-variable confirmed): T0/T1 FP8, T2 prefix-cache = **the experimental tier (gpu-only/cpu/nvme arms — the ONLY thing that varies)**, T3 spec-decode deferred (tiny OSL), T4 parallelism deferred to the follow-on sweep, T5 torch.compile inherited. Only the T2 KV tier changes across arms.

### Stage 0b — Optimization objective

```yaml
optimization_objective:
  mode: fixed_grid           # NOT a hill-climb — a pre-planned measurement grid. The WORKLOAD (N, access)
                             # is the variable, not a tunable config that could trade quality for speed,
                             # so no plateau rule and no held-out quality eval needed — just a garbage screen.
  primary_question: "In the distinct-prefix / working-set>HBM regime, does a CPU/NVMe KV tier beat gpu-only (recompute-on-evict) on TTFT + queue depth, at matched concurrency?"
  axes:
    - ttft_p95_ms              # the metric tiering should improve (fetch < recompute)
    - queue_depth              # downstream of TTFT via holding time
    - output_tokens_per_sec    # the ceiling tiering should NOT change (decode counter-space) — NEGATIVE CONTROL
    - tier_hit_breakdown       # gpu_hbm / cpu / nvme / recomputed — proves WHERE prefixes are served
  attribution: "single-variable by SLICING — compare arms at FIXED (N, concurrency, access); N and concurrency are sensitivity axes, never bundled into one delta. Primary A/B: 3 arms at N=43, c=128, zipfian."
  controls:
    - N_distinct_prefixes: [22, 43, 86, 172]   # the eviction-pressure sweep (0.5x-4x pool fill)
    - concurrency: [16, 64, 128, 256]          # hold sensible; the prefix-count is the primary axis
    - kv_arm: [gpu-only, cpu-offload, nvme-tiering]
    - access_distribution: [zipfian, uniform]  # hot/cold repos vs worst-case
  quality_screen: { method: "garbage/coherence on ~10 sampled outputs per config; tool-parse + <think> intact", fail_closed: true }
  budget: { max_configs: 24, max_wall_clock_min: 420, max_usd: 800 }
  stop_rule: "Run the planned grid then STOP. Trap scales nodegroup to desired=0 on completion/cap/error (unattended)."
```

**What the data must answer:**
1. At N > 43 (pool overflow), does gpu-only TTFT/queue **degrade** (recompute on evict) while cpu-offload/nvme **hold**? (The core hypothesis.)
2. What is the **tier-hit breakdown** — fraction served from GPU-HBM vs CPU vs NVMe vs recomputed — as N grows?
3. Does the decode throughput ceiling stay **flat** across arms (confirming tiering relieves latency, not the counter-space ceiling)?
4. Where does **PCIe/NVMe fetch latency** become the new bottleneck (does NVMe fetch at high N cost more than it saves)?
5. Does CPU tier suffice, or is NVMe needed (i.e. at what N does CPU RAM also overflow)?

### Stage 0c — Build correctness (reuse minimax-m2 gate)
- `validate-serving-config.py` exits 0; TP4 applied; B200 FP8-MoE correctness smoke (no FlashInfer assertion, no garbage, tool+think round-trip). Budget ~35-40min B200 cold start.

### Stage 4a / 5 — health + serving
- Standard B200 health; `/health` 200; tool+reasoning round-trip.

### Stage 6 — Benchmark
Workload card: `standards/benchmark-commons/workloads/distinct-prefix-multitenant.yaml`. For each (N × kv_arm × concurrency × distribution), measure:
- [ ] TTFT p50/p95/p99 and **queue_depth_max** (the metrics tiering should improve)
- [ ] **tier-hit breakdown**: gpu_hbm_hit / cpu_kv_hit / nvme_kv_hit / **tokens_recomputed** (the offload COST avoided)
- [ ] **cold-vs-warm TTFT per prefix-group** (warm = prefix served from a tier; cold = recomputed)
- [ ] decode throughput (must stay ~flat across arms — the negative control)
- [ ] KV utilization, preemptions, **eviction rate** (MUST be non-zero at N>43 or the experiment didn't stress eviction — the whole premise)
- [ ] PCIe/NVMe transfer cost: DCGM `DCGM_FI_DEV_PCIE_TX/RX_BYTES` (the offload-tier transfer cost; vLLM 0.19.1rc1 doesn't expose `kv_transfer` directly) — else infer from ITL + recompute count
- [ ] **NVMe I/O contention (nvme-tiering arm)**: `iostat -x 5` %util + await on the NVMe device during each run. **Flag if %util > 80% or await > 50ms** — a slow NVMe-fetch result is then spindle contention with the model weights on the same disk, NOT the tier's true ceiling (don't misattribute)

### Stage 6b — Optimization loop (emits trajectory)
Single-variable per run; quality gate first (fail-closed); emit `results/optimization-trajectory-<date>.json`. The deliverable is **the tiering-vs-recompute crossover**: the N at which each tier starts paying, and the per-tier hit/cost curves. Reward-hacking guard: a TTFT win that drops quality/tool-calling/think is a `quality_breach`.

### Stage 7 — Readiness
- All categories pass; feed lessons (tier sizing, crossover N, PCIe-fetch ceiling) to `mdc` + compound step.

## Cost Considerations
Same $/concurrent-developer-at-SLO frame. The new question this answers: **how many distinct codebases can one node serve at acceptable TTFT before you must add the next tier (CPU→NVMe→more nodes)** — i.e. the multi-tenant capacity number.
