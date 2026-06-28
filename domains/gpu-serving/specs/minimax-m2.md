# MiniMax-M2 Coding-Agent Serving Requirements

## Status: DRAFT

## Overview

Serve **MiniMax-M2** (230B MoE, 10B active) on **4× B200** via vLLM as a self-hosted replacement for Claude Haiku/Sonnet in a customer's coding-agent workloads. The customer's **goal is goodput** — maximize concurrent developers at OK interactivity — over a **prefill-heavy, huge-context** workload (ISL 11K–94K, OSL 66–485) with a *claimed* >90% shared-prefix reuse. So this is a **KV-capacity / KV-reuse + prefill-latency** problem, not a decode problem.

> **Hardware note (B200, not H200):** the customer benchmarked on H200, but no Hopper capacity is available — we run on **4× B200 (`p6-b200.48xlarge`, NVSwitch, SM100)**. This is a deliberate substrate change with two consequences: (1) more HBM headroom (~720GB vs ~564GB) → higher concurrency ceiling, *helps the goodput goal*; (2) **our B200 numbers are NOT comparable to the customer's H200 dashboards** — we establish a *fresh B200 baseline* (per `benchmark-analysis.md` match-before-compare). State this to the customer explicitly; do not present a B200-vs-H200 ratio as a delta.

The customer has an existing config and two benchmark runs (with / without CPU KV offload). The job of this spec is to **validate the stack holistically** against the goodput objective — independently reproducing the customer's findings rather than assuming them — produce the data to defend offload-vs-disagg-vs-plain-APC as the right architecture, and decide M2 vs M2.7.

> **Source discipline (carried from `feedback_card_vs_upstream_truth`):** every engine fact below was verified against the vLLM source registry / HF config.json on 2026-06-27, not from a single secondary page. The vLLM *recipe page* and the vLLM *docs registry* disagreed on the reasoning-parser name; the registry (`vllm/reasoning/__init__.py`) is authoritative. Re-verify before deploy — these are T3 (release-pinned) facts.

## Components

### 1. Compute
- **Platform**: EKS (customer's serving framework) or direct EC2 capacity-block launch.
- **Instance**: **`p6-b200.48xlarge`** — 8× B200 (SM100, NVSwitch, 180GB/GPU). Use **4 GPUs (TP4)** for the M2 deployment, matching the customer's 4-GPU config; the other 4 are spare/second-replica.
- **AMI / bootstrap** (from prior B200 deployments): **AL2023 NVIDIA AMI** (`ami-02bb9f913067dadb1` or current) — AL2 lacks `ib_umad` for Fabric Manager on NVL5+. EKS bootstrap uses **`nodeadm`** (MIME multipart `application/node.eks.aws`), not `/etc/eks/bootstrap.sh`. Driver 580.x / CUDA 13.0.
- **Capacity**: B200 is capacity-block — launch with `--instance-market-options '{"MarketType":"capacity-block"}'`; instance termination drains ~10 min before the slot frees. Spot availability: us-east-2 AZ1/AZ2 (~$18/hr).
- **Scaling**: min=max=1 replica for the benchmark.

### 1a. GPU & NCCL Pre-Flight
- B200 = Blackwell **SM100 + NVSwitch** — mature NCCL on this topology. The NCCL-on-Blackwell bug in our memory is **PCIe-only (g7e/SM120)**; B200 NVSwitch is **unaffected**. Standard Stage 4a NCCL all-reduce check for TP4 still applies.
- vLLM inference uses custom allreduce, so NCCL collective health is necessary for TP but not the inference hot path.

### 2. Model
- **Model ID**: `MiniMaxAI/MiniMax-M2` (primary); `MiniMaxAI/MiniMax-M2.7` (staged A/B — see Stage 6b)
- **Arch** (verified, HF config.json): `MiniMaxM2ForCausalLM`, 62 layers, 256 experts / 8 active, per-layer qk_norm, MTP (3 modules, `use_mtp:true`). Context: M2 = 196K, M2.7 = 200K. **Same architecture** → memory/parallelism sizing is identical between versions.
- **Format**: FP8 (F8_E4M3) — **native**: HF `config.json` ships `quant_method: fp8`, `weight_block_size:[128,128]`. ~220GB weights + 240GB per 1M context tokens. ~57GB/GPU weights (comfortable on 180GB B200).
- **Serving**: vLLM. **Deployment card**: `cards/vllm/minimax-m2.md` (created 2026-06-27 from upstream recipe + source registry).
- **Required args** (verified): `--tool-call-parser minimax_m2`, `--reasoning-parser minimax_m2_append_think` (see §parser choice), `--enable-auto-tool-choice`, `--trust-remote-code`, TP4 + `--enable-expert-parallel`, `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'`.
- **B200 MoE-backend pin (CRITICAL)**: keep the customer's **`--moe-backend triton`** and **do NOT use FlashInfer FP8 MoE on Blackwell**. Root cause (vLLM #33543): MiniMax casts `router_logits` to float32, but the FlashInfer/TRTLLM FP8 MoE backend only supports float32 router_logits for DeepSeek-style routing → assertion failure on SM100. vLLM #37056 auto-removes `FLASHINFER_TRTLLM`/`FLASHINFER_CUTLASS` from the default candidate list on SM100+; confirm that lands in the build, or set `VLLM_USE_FLASHINFER_MOE_FP8=0` explicitly. Triton FP8 MoE is the safe Blackwell path.

#### Parser choice — a lever, not a bug
The customer set `--reasoning-parser minimax_m2`. **This is a valid parser, not an error** (both `minimax_m2` and `minimax_m2_append_think` are registered). The choice has real behavioral consequences for a multi-turn agent:
- `minimax_m2_append_think` → keeps `<think>` in the assistant message → survives into history (the HF card **requires** retaining `<think>`; dropping it degrades performance).
- `minimax_m2` → reasoning goes to a separate `reasoning_content` field → lost across turns **unless the customer's harness echoes it back**.
- **Action**: confirm whether the customer's coding-agent harness preserves `reasoning_content`. If not (likely), `append_think` is the correct default. This is A/B'd in Stage 6b.

### 3–5. Networking / Storage / Dev Env
Inherit from the customer's existing EKS serving framework. Out of scope for this validation pass.

## Non-Requirements
- Multi-region, HA/DR, autoscaling (min=max=1 for the benchmark).
- NVFP4 as a baseline — deferred to a Stage 6b arm only if it boots cleanly (open NVFP4 bugs make it unsafe as the starting quant).

## Known Limitations (verified against upstream 2026-06-27)
- **B200 FP8 is viable but NOT the FlashInfer path** — the card's blanket "broken on Blackwell" is over-broad: it conflates NVFP4/SM120 (consumer Blackwell) with FP8/SM100 (B200). FP8 is the model's native quant and there is an active upstream B200 FP8 recipe (recipes #272, open). The real pitfall is the FlashInfer FP8 MoE float32-router-logits assertion (#33543) → **pin `--moe-backend triton`** (see Model §B200 MoE-backend pin). Smoke-test correctness at Stage 0c before trusting any number.
- **NVFP4 on Blackwell is unstable** — open bugs: text degeneration (#31856), CUDA faults SM120 (#35566), DGX-Spark hangs (#41725). Keep NVFP4 out of the baseline.
- **Pure TP8 unsupported** — for >4 GPUs use DP+EP or TP+EP. On H-class, TP4+EP4 > TP8+EP8 (recipe); re-verify the EP shape on B200 (the recipe was written for H-class).
- **Corrupted-output bug (commit cf3eacfe, 2025-12-11) is FIXED in v0.23.0+** — the customer's `v0.23.0` (2026-06-15) already contains it. Not a risk on this build. (Earlier worry that v0.23.0 predated the fix was wrong.)
- **M2.7 active-param count is undocumented** — if >10B, decode cost/ITL rises. Verify empirically before quoting M2.7 latency.
- Run `mdc prs minimax-m2` and `gh issue list --repo vllm-project/vllm --search "minimax in:title" --state all` before deploy — fresh-model issue churn is high (M2.5/M3 variants exist with their own open bugs).

## Verification Criteria

### Stage 0 — Carryover Audit
- [ ] Ran `carryover-auditor`. Key prior-lesson stacks to check: any vLLM FP8 MoE deployment (TP-divisibility), large-context KV sizing, tool-call-parser mismatches.
- [ ] `feedback_card_vs_upstream_truth` honored: engine facts source-verified, not card-trusted.

### Stage 0b — Optimization Coverage (lever ledger)

1. **Regime prediction**: This is **prefill-compute-bound at the request level** (TTFT dominates: customer shows ~38s TTFT ≈ half of 76s E2E) **AND KV-capacity-bound at the serving level** (97% KV cache, 0 preemptions) on 4× H200. Reasoning: ISL 11K–94K with OSL ≤485 means the GPU spends almost all its time in prefill; the tiny output makes decode a minor term. The customer's goal is **goodput** (max concurrent developers at OK interactivity), so the operative question is *what bounds the concurrency knee* — and with a claimed >90% shared prefix, the answer hinges on whether that prefix is cacheable cross-request (→ KV-capacity/transfer-bound, where tiering may help) or not (→ prefill-bound, where it won't). **This regime must be MEASURED, not assumed** — confirm with TTFT-fraction-of-E2E + prefix-hit-rate + preemptions (per `benchmark-analysis.md`); the workload card `coding-agent-100k-shared-prefix` is built to produce exactly this evidence.

2. **Lever ledger**:

| Tier | Lever | applied / deferred — reason |
|------|-------|------------------------------|
| T0 | Baseline (honest reference) | **applied** — establish per-scenario baselines (Haiku-typical ISL~11K vs Sonnet-P75 ISL~94K are different operating points; do NOT average them on one dashboard). |
| T1 | Quantization | **applied** — FP8 weights + FP8 KV (customer already runs `--kv-cache-dtype fp8`). Keep. |
| T2 | KV / prefix cache | **applied + the central experiment**. `--enable-prefix-caching` on (baseline). The architecture question lives here: 3 KV-tier arms (GPU-only / CPU-offload / NVMe-disk) are A/B'd across cold + 90K-shared-prefix scenarios to find which holds the goodput knee. **CPU offload is NOT asserted bad** — the customer's regression was on a COLD run; re-test it under real reuse (Stage 6b H1). APC hit-rate on the shared prefix is the variable that decides whether any tiering pays. |
| T3 | Speculative decode | **deferred — low ROI**: OSL is tiny (66–485), decode is not the bottleneck. BUT the model ships native MTP (`use_mtp:true`, 3 modules) — cheap to A/B at low concurrency if a latency-mode scenario needs it. Re-verify, don't assume. |
| T4 | Parallelism | **SWEEP, do not assume** — map the full parallelism surface across 1–8 GPUs and let the data pick the optimum for throughput AND the goodput knee. Customer runs 4 GPUs, but we have 8, so measure the whole shape rather than assuming a winner. Candidates (constrained by the **FP8 block-128 rule**: expert FFN 1536/TP must %128==0 → **TP∈{1,2,4} valid; TP8 INVALID** — 1536/8=192 fails): TP4 (customer baseline), TP4+EP4, TP2+DP2, TP4+DP2 (8 GPUs, two replicas), TP2+DP4. `kimi-k2.6-nvfp4` (1T MoE, sm_100, same regime) found TP4+DP2 beat TP8 by +19–25% — a **prior to test, not a foregone conclusion**; M2 (10B active vs Kimi's 32B) may land elsewhere. Sweep in Stage 6b H4a; report the throughput-vs-concurrency Pareto. Disagg (H4b) earned only if the best parallelism shape + KV-tiers still can't hold interactivity. |
| T5 | Kernel / compile | **applied** — `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` (qk_norm fusion, PR #37045). Customer currently has NO compilation-config → this is a free lever they're leaving on the table (requires build with #37045). |
| T6 | Model / graph surgery | **deferred** — premature; exhaust T0–T5 first. |

3. **Optimization objective** (FULL PARETO — no fixed SLO gate this run; the customer hasn't given firm TTFT/ITL numbers, so we MEASURE the throughput-vs-concurrency Pareto and let them pick the knee):

```yaml
optimization_objective:
  mode: pareto_map               # NOT a single-objective hill-climb — map the surface, report the frontier
  axes:
    - output_tokens_per_sec      # throughput
    - concurrency                # offered concurrency
    - ttft_warm_p95_ms           # interactivity (reported, NOT gated this run)
    - itl_p95_ms
  report: "throughput-vs-concurrency Pareto per (scenario × KV-arm × parallelism shape); annotate where TTFT/ITL cross common bars (1s/2s/5s, 30/50ms) so the customer can pick their own knee"
  quality_screen:                # LIGHTWEIGHT — not a full eval harness (per operator decision)
    method: "automated garbage/coherence screen on ~10 sampled outputs per config: reject if repetition loop (same token >5×), empty output, or broken tool-call parse. NOT a SWE-bench pass-rate eval."
    invariants:                  # cheap structural checks, fail-closed
      - tool_call_parses          # minimax_m2 parser still yields structured tool_calls
      - think_block_present       # <think> present in output
      - no_repetition_loop
      - error_rate_max: 0.01
  budget: { max_configs: 20, max_wall_clock_min: 420, max_usd: 800 }
  plateau: none                  # Pareto MAP, not hill-climb — we want all points, so run the full planned grid (no early plateau stop)
  stop_rule: "Run the full planned config grid (parallelism shapes × KV arms × concurrency steps), then STOP. Hard caps: 420 min OR $800 wall-clock safety. On completion OR cap OR fatal error, the runner MUST scale the nodegroup to desired=0 (autonomous run — no idle B200 left billing)."
```

> **Unattended-run contract**: this loop runs without a human watching. It is a **deterministic bash/python runner, not an agent** (an agent already stalled on a watchdog once). It self-terminates and scales the node to 0 on finish/cap/error. The quality check is a cheap garbage screen, not a capability eval — we are validating *serving*, not model quality, this run.
```

> The objective is **goodput (concurrency-at-SLO), not TTFT and not throughput.** TTFT/ITL are the *guardrail* (interactivity), concurrency is what we maximize under it. A throughput-max objective would reward batching that wrecks interactivity; a TTFT-min objective would reward tiny batches that waste the GPU. The customer wants the most developers served at acceptable responsiveness — that is the knee where the interactivity SLO first breaks.

### Stage 0c — Build correctness & serving-config gate (FAIL-CLOSED, runs FIRST)

Before any latency tuning, prove the stack is correct — optimizing a latency number on a corrupted or misconfigured model is worthless. **First-start timing budget**: B200 cold start is ~16 min (DeepGEMM JIT + torch.compile + CUDA graph capture); persist `/root/.cache/vllm/` to NVMe hostPath so restarts warm-start at ~6–8 min. Don't declare a startup-probe failure before ~16 min on the first boot.

- [ ] `validate-serving-config.py --sidecar blueprints/minimax-m2/benchmark.yaml --corpus-root .` exits 0.
- [ ] **TP is actually applied**: confirm `--tensor-parallel-size 4` is injected (230B FP8 ≈ 220GB CANNOT fit one 180GB B200 — if TP defaulted to 1, the deploy isn't what the config shows).
- [ ] **B200 FP8-MoE correctness gate (the make-or-break check)**: server starts WITHOUT the FlashInfer FP8 MoE float32-router-logits assertion (#33543). Verify `--moe-backend triton` is in effect (or `VLLM_USE_FLASHINFER_MOE_FP8=0`). If startup hits the assertion → STOP, do not benchmark; fix the backend pin first.
- [ ] **Output-correctness smoke test on B200**: 20 sample completions show no garbled/corrupted output and valid `<minimax:tool_call>` parsing. The card's "garbage on Blackwell" is NVFP4/SM120-specific, but FP8/SM100 is unconfirmed for M2 — this smoke test is what confirms it. Garbage → STOP and escalate (try `minimax27` image / nightly).
- [ ] Image is `vllm/vllm-openai:minimax27` (or a build containing PR #37045 for `fuse_minimax_qk_norm` AND #37056 for the SM100 FlashInfer-MoE auto-removal). If neither lands, drop `fuse_minimax_qk_norm` and pin `--moe-backend triton` manually; note as blocked-by-build.
- [ ] Reasoning-parser choice resolved against the customer's harness (does it echo `reasoning_content`? if not → `append_think`).

### Stage 4a — GPU Health
- [ ] Standard B200 health via `gpu-infra` MCP (`discover_cluster`, `check_gpu_health`): ECC 0 uncorrected, no pending row remaps, thermals <85°C, no Xid. `run_nccl_test` all-reduce passes for TP4 across the NVSwitch domain.

### Stage 5 — Serving Stack
- [ ] `/health` returns 200; single `/v1/completions` returns valid output (no OOM at `--gpu-memory-utilization 0.95` during a 94K-ctx prefill — watch activation headroom).
- [ ] Tool call + reasoning round-trip: a request with tools returns a parsed `tool_calls` and a retained `<think>` block.

### Stage 6 — Benchmark

Workload card: **`standards/benchmark-commons/workloads/coding-agent-100k-shared-prefix.yaml`**. The benchmark is a **concurrency sweep with an interactivity-SLO gate** — for each scenario × KV-architecture arm, find the **goodput knee** (max concurrency where TTFT_warm_p95 AND ITL_p95 both hold). Run scenarios as distinct operating points — **do not average**:

- `replace-sonnet-cold` (ISL 59,128, OSL→EOS) and `replace-sonnet-p75-cold` (ISL 94,386) — the cold floor, matching the customer's existing benchmark.
- `coding-shared-prefix-90k` (90K fixed shared prefix + ~350-tok unique suffix, APC on) — **the production claim that decides the architecture.**

Across **3 KV-architecture arms** (single-variable, separate sidecars): `gpu-only` (APC baseline), `cpu-offload` (the customer's OffloadingConnector), `nvme-disk` (CPU+NVMe tiering).

**Required measurements** (full telemetry — the point is to locate the bottleneck *class*, per `benchmark-analysis.md`):
- [ ] **Goodput knee** per scenario × arm: max concurrency holding the interactivity SLO.
- [ ] **TTFT-fraction-of-E2E** per scenario — the prefill-bound gate. Never assert decode-bound without it.
- [ ] **prefix_hit_rate** on the shared-prefix scenario — validates (or refutes) the >90%-reuse claim. This is the single most decisive number.
- [ ] **KV tiering evidence**: gpu/cpu/nvme KV hit-rate, `kv_transfer_bytes_per_token`, `kv_transfer_latency_ms` — the offload COST that explains ITL inflation.
- [ ] **kv_utilization timeseries** (not just peak) — the "KV cache spikes" the customer flagged; when does it hit 97%?
- [ ] **preemptions_per_min + queue_time_p95 + running-vs-waiting** — admission/capacity-bound signals.
- [ ] **chunked_prefill_active** — CRITICAL: verify the KV connector did not silently disable chunked prefill (suspected offload-regression mechanism).
- [ ] **thinking_tokens_ratio + think_block_retained** — interpret run-to-EOS OSL vs Claude's forced 224; quality invariant.

**Tier Stack Table**: one row per tier T0–T6 vs the honest T0 baseline.

### Stage 6b — Optimization loop (in-spec, emits trajectory)

Hill-climb the **goodput** objective per `standards/benchmark-commons/OPTIMIZATION-LOOP.md`. Each candidate = **single-variable** change vs the current best; quality gate runs first (fail-closed); emit `results/optimization-trajectory-<date>.json`. The deliverable is **the data to defend offload-vs-disagg-vs-plain-APC** as the right architecture. Planned candidates (each a labeled lever):

- **H1 — KV tier A/B (RE-TEST offload under reuse, not cold).** The three KV-architecture arms (gpu-only / cpu-offload / nvme-disk) are **orthogonal A/B/C against the same T0+T1 baseline — NOT sequential stacking**; each is an independent single-variable modification (separate sidecar). The customer saw offload regress (RPM 65.9→41.1, ITL 679→~2000ms, KV% unchanged ~97%, 0 preemptions) — but **on a COLD run, where an offload tier has nothing to reuse**. Re-run all three on the `coding-shared-prefix-90k` scenario where the >90% reuse can actually feed the tier. Two mechanisms to separate: (a) "no bottleneck to relieve" — if preemptions stay 0 and cpu_kv_hit_rate is low, offload is pure PCIe cost (~64 GB/s vs ~4.8 TB/s HBM, ~75× slower, on the decode path → ITL inflation); (b) **"chunked-prefill conflict" — assert `chunked_prefill_active` stays TRUE when the OffloadingConnector engages.** A KV-transfer connector can silently disable chunked prefill (priskv-shared-prefix-cache L12 / Qwen3-Next HMA pattern), which alone would explain the ITL inflation as a *config bug*, not an architecture verdict. Check this first — it's the cheapest possible root cause. The E2E being identical (1.26 min) across the customer's two runs is suspicious (OSL mismatch / timeout clamp) — controlled re-run resolves it.
- **H2 — Admission control.** `--max-num-seqs 512` is almost certainly too high: 512 × ~90K ISL can't fit the KV pool → admits few, queues rest → fake concurrency + inflated TTFT. Sweep `--max-num-seqs` to the real KV-supported concurrency; lower `--max-num-batched-tokens` (from 131072) to interleave decode and protect the ITL ceiling under load. This is the most likely *immediate* goodput win.
- **H3 — Compilation/kernel.** Add `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` (requires #37045 build). Measure goodput-knee delta.
- **H4a — Parallelism sweep across 1–8 GPUs (MEASURE the surface, don't assume).** The customer runs 4 GPUs, but the node has 8, so map the full parallelism shape and let the data pick the optimum for **both** raw throughput and the goodput knee — report the throughput-vs-concurrency Pareto, not a single number. Candidates (FP8 block-128 valid only: **TP∈{1,2,4}**, since 1536/8=192 fails the rule): single-replica TP1/TP2/TP4, TP4+EP4, and multi-replica DP shapes filling the 8 GPUs — TP2+DP2, **TP4+DP2** (two TP4 replicas), TP2+DP4. The `kimi-k2.6-nvfp4` +19–25% TP4+DP2 result is a **prior to confirm or refute**, not an assumption — M2 (10B active) sits in a different sub-regime than Kimi (32B active) and may land elsewhere. If TP4+DP2 wins, great; if a single fatter replica or a different DP fan-out wins, that's the finding. Single-variable per run (one parallelism shape at a time). Disagg (H4b) is only on the table if the *best* shape here still can't hold interactivity.
- **H4b — P/D disaggregation (EARNED only if H1–H4a all fail).** Only if KV-tiers AND replication can't hold interactivity at target concurrency — i.e. the 90K prefill stalls decode no matter what — evaluate prefix-decode disagg. **Disagg is frontier-only** (`project_pd_disagg_frontier_only`); justify by data, not assumption. **Same-node disagg uses NVLink + UCX `cuda_ipc`, NOT EFA** — NIXL disables `cuda_ipc` by default (NVSwitch contention), so re-enable it explicitly (tech-stack.md §NIXL); EFA/SRD is only relevant for *multi-node* disagg, which this single 8-GPU node does not need.
- **H5 — Model A/B: M2 vs M2.7.** Single-variable model swap on the stabilized baseline, scored on the held-out coding-quality + think-retention gate. M2.7 is same-arch (drop-in), stronger coding, longer context — verify its active-param count first (latency risk) and that it runs on the chosen image.

Reward-hacking guard: a goodput gain that breaches the interactivity SLO or quality gate, or drops tool-calling / think-retention, is a `quality_breach` — recorded as a quality-cost lesson, never a goodput win.

### Stage 7 — Readiness Audit
- [ ] All categories pass; no unresolved HIGH lessons; deployment-card recommendations followed or overridden with justification.

## Production Scaling Recommendation (the "what do we do with the Pareto" section)

The benchmark measures the **single-replica goodput knee**. Turning that into a production system that holds TTFT under variable load is a *layered* problem — and the key insight from the data is that **TTFT explodes from unbounded QUEUE-WAIT, not from compute** (cold scenario: 2.3s→125s as offered concurrency outruns prefill throughput; Little's Law). You prevent the blowup by refusing/deferring excess load fast, not by absorbing it. Four layers, in order of who-handles-what-timescale:

1. **Prefix-cache maximization (cheapest, biggest lever).** The shared-prefix data shows TTFT stays ~flat (4s to c128) vs cold's collapse by c16 — because cached 90K prefixes make each request cheap to admit. Make the 90% reuse real: byte-identical prompts (no per-request timestamps/IDs in the prefix) + prefix-affinity routing. This *raises the concurrency at which TTFT starts to climb* — do this before anything else.

2. **Engine admission control (spike timescale, seconds).** Lower `--max-num-seqs` from 512 to the KV-supported running set so the active batch stays fast and excess **queues explicitly** instead of degrading everyone; keep chunked-prefill on. This trades TTFT for honest queue-time — it bounds the running batch, it does NOT add capacity.

3. **Load-aware gateway: routing + backpressure (spike timescale).** This is the routing layer, and it does two distinct jobs: (a) **prefix-affinity routing** — send requests sharing the 90K prefix to the *same* replica so they hit warm KV (random/round-robin routing would cold-cache and spike TTFT on the very replica you added); (b) **load-aware admission/shedding** — watch per-replica `vllm:num_requests_waiting` / `request_queue_time` and spread to the least-loaded replica, or **shed (429/backpressure) when all replicas are saturated**. For an interactive coding agent, a fast rejection beats a 125s TTFT. Routing *distributes and sheds*; it does not create throughput.

   **Runtime options for this layer (the customer recommendation):**
   - **llm-d** (Gateway API Inference Extension / EPP): K8s-native, prefix-cache-aware scorer + KV-cache-utilization + queue-aware scheduling. The natural fit on EKS — InferencePool + EPP front a set of vLLM replicas, route by prefix affinity and load. Caveat: EPP scorer set + GA API surface move fast (steering notes the v1.3.1→v1.5.0 scorer changes) — smoke-test the current scorer set on deploy. **Best fit for "vLLM replicas behind a smart K8s gateway."**
   - **NVIDIA Dynamo**: disaggregation-first runtime with its own KV-aware router, smart load balancer, and prefill/decode pools. Heavier; pays off when you also need P/D disaggregation (H4b) or wide-EP. Overkill if replicas + llm-d routing hold the SLO; the right call only if the prefill-stall (90K prefill blocking decode) survives every other layer. Grove is its K8s gang-scheduler.
   - Decision: **start with llm-d** (lighter, K8s-native, prefix-affinity is the lever this workload needs); reach for **Dynamo only if disagg becomes necessary** (frontier-only per `project_pd_disagg_frontier_only` — earned by data, not assumed).

4. **Replica autoscaling (sustained-shift timescale, NOT spikes).** Scale on **queue depth, never GPU%** — a vLLM replica sits at ~100% GPU with one request, so GPU% can't see the flood; `vllm:num_requests_waiting` (or `request_queue_time_p95` vs the TTFT budget) is the leading indicator. Wire via **KEDA Prometheus scaler** (native Prometheus, cleaner than HPA+prometheus-adapter): `replicas = ceil(num_requests_waiting / threshold)`, `maxReplicas = floor(8 GPUs / TP_degree)` = 2 at TP4 (4 at TP2) on this one node. **Hard constraint: M2 scale-out is ~8min warm-cache / ~60min cold-node — too slow to chase spikes**, so autoscaling is for business-hours ramps only; spikes are handled by layers 2–3. Pre-stage weights on NVMe + warm-pool nodes to get scale-out to boot-only time. **Scaling MUST pair with prefix-affinity routing (layer 3)** or a new replica cold-caches the shared prefix and spikes the TTFT it was added to fix.

   **Concrete manifest**: `blueprints/minimax-m2/k8s/autoscaling-production.yaml` — a customer-deliverable KEDA `ScaledObject` (queue-depth + queue-wait triggers, GPU-bounded maxReplicas, slow boot-aware cooldown) with an HPA+prometheus-adapter fallback. Thresholds are marked `<TODO customer>` pending their TTFT budget. NOT applied on the benchmark cluster (KEDA not installed there) — it's the prod-deployment template.

> **Not measured in this sweep**: it's single-replica steady-state knees. "Does 2 replicas + llm-d prefix-affinity routing hold TTFT under flood better than one fat replica" is a *separate* experiment with a gateway in front — the natural follow-on blueprint once this Pareto lands. Thresholds (the KEDA `threshold`, the shedding line) come from the customer's TTFT budget, still TODO.

## Cost Considerations
4× H200 on-demand vs the Claude API spend being replaced. The real comparison unit is **$/concurrent-developer at the interactivity SLO** (derived from the goodput knee), not $/1M-tokens (per `benchmark-analysis.md` reasoning-efficiency rule) — a self-host that can't hold the SLO at the needed concurrency has no cost advantage regardless of token price. The goodput knee × replica count = developers served; that ÷ instance cost = the number to put in front of the customer.
