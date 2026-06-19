# PrisKV Shared Prefix Cache — Co-located Replicas on PCIe-only g7e

## Status: FALSIFIED (2026-06-18)

> **Result**: On g7e (2× TP1, Qwen3-32B-FP8, 70% reuse), PrisKV shared-cache (Arm C) was **2–5×
> SLOWER** than prefix-aware routing (Arm B) and naive round-robin (Arm A), getting relatively worse
> as prefix size grew (800→4000 tok). Falsification criterion #1 triggered. The host-DRAM KV
> round-trip costs more than the avoided on-GPU prefill. Gate #4 (build/serve) PASSED but required
> ~14 undocumented fix-ups. **Steering rule: for cross-replica prefix reuse, use prefix-aware routing,
> not an external KV store. Stop evaluating AIBrix single-node-PD for our fleet.**
> Full writeup: `blueprints/priskv-shared-prefix-cache/results/RESULT-20260618.md`; integration
> lessons: `blueprints/priskv-shared-prefix-cache/lessons.md` (L1–L14).

## Hypothesis

> On a single PCIe-only node (no NVLink) running N≥2 independent vLLM replicas, a **shared
> host-DRAM KV cache (PrisKV)** lowers TTFT by ≥25% over engine-local prefix caching **at
> ≥40% shared-prefix ratio**, AND beats prefix-aware routing (llm-d) by ≥15% TTFT **when
> request arrival is bursty enough that routing can't keep same-prefix requests on one replica**.

Two thresholds because there are two competitors. Beating naive local caching is easy and
uninteresting; beating prefix-aware *routing* is the real question — PrisKV only earns its
keep (and its three open race-condition bugs) if sharing state beats routing-to-state.

## Falsification criteria

PrisKV is **falsified for our fleet** if ANY of:
1. At 70% shared-prefix ratio, PrisKV (Arm C) TTFT improvement over prefix-aware routing
   (Arm B) is <15% — i.e. routing already captures the reuse, sharing adds nothing.
2. PrisKV shows >10% TTFT regression vs engine-local (Arm A) at 0% prefix ratio — the
   host-DRAM hop overhead isn't "well-controlled" as the blog claims.
3. Any cache-hit returns **incorrect or stale output** (open issues #33/#41/#42 are cache
   races) — correctness failure voids any latency win regardless of magnitude.
4. Build/integration of `aibrix_kvcache + PrisKV` into a current vLLM on sm_120 cannot be
   made to work in <1 day of effort (maturity gate — the only prebuilt image is China-ECR,
   vLLM 0.10.2, untested on Blackwell PCIe).

## Why this matters

We have repeatedly found disaggregation and KV-tiering pay off only at frontier scale on
fast interconnect (`[[project_pd_disagg_frontier_only]]`, kimi-k2.6-nvfp4 disagg = 3.8×
regression). PrisKV is the *opposite* bet: it targets interconnect-**poor** hardware (no
NVLink, no fast RDMA) — which is exactly our cheap g7e tier. The one scenario where it could
add value is **co-located replicas sharing a prefix cache** on a PCIe-only box, for
high-shared-prefix workloads (RAG with a fixed system prompt, multi-turn agents). If it
works, the operational lever is: pack g7e with replica-per-GPU + one shared cache instead of
one big TP, and serve high-prefix-reuse workloads cheaper. If it doesn't beat prefix-aware
routing, we close the question permanently and stop revisiting AIBrix posts.

This experiment is the natural follow-on to `qwen3-32b-eks` (2026-04-03), which established
the single-replica Qwen3-32B-FP8/TP1 floor and explicitly concluded *"L1/L2 cache value
requires multiple replicas — KV sharing across replicas"* but could not test it.

## Matrix

| Axis | Values |
|------|--------|
| Model | `Qwen/Qwen3-32B-FP8` (single model — meaningful prefill cost, fits TP1 in 96GB) |
| Hardware | g7e.24xlarge (4× RTX PRO 6000, sm_120, PCIe) — **fallback g7e.12xl (2 GPU) under current quota** |
| Replicas / TP | 4× TP1 (one replica per GPU) — **fallback 2× TP1 on g7e.12xl** |
| Arm | **A** local-cache + round-robin · **B** local-cache + prefix-aware routing · **C** PrisKV shared cache + round-robin |
| Shared-prefix ratio | 0% · 40% · 70% (the discriminator sweep) |
| Arrival pattern | steady (Poisson) · bursty (batched arrivals) — bursty is where B should thrash and C should win |

Cells = 3 arms × 3 ratios × 2 arrival = **18 cells**, ×2 (cold/warm) for hit-rate stability =
**36 runs**. Run all 18 steady+warm first; add bursty only if C≈B at steady (bursty is C's
best case — if C loses there too, falsified faster).

## Baseline

"Off" = **Arm A**: 4 independent vanilla vLLM replicas, engine-local APC (`--enable-prefix-caching`,
on by default in current vLLM), round-robin in front (plain nginx/round-robin, no prefix awareness),
no external KV connector. Same model, same `--max-model-len` (24,000, matching qwen3-32b-eks),
same FP8, same node. **Arm B** adds only the router (llm-d / GAIE prefix-aware EPP or vLLM's
KV-aware router) — identical engines. **Arm C** swaps the router back to round-robin and adds the
PrisKV shared cache via `AIBrixOffloadingConnectorV1`. Only one variable changes per arm pair.

## Pre-flight / bring-up (gates before any paid run — from prior g7e lessons)

Run in order; each is a STOP gate. All cite a prior blueprint that learned it the hard way.

0. **Stage 4a — GPU health on reused spot.** SSH the node, run `nvidia-smi
   --query-gpu=ecc.errors.uncorrected.volatile.total,remapped_rows.pending,remapped_rows.failure,remapped_rows.uncorrectable
   --format=csv,noheader`. Gate on **`volatile.total==0` AND `remapped_rows.{pending,failure}==No`
   AND `uncorrectable==0`** — NOT lifetime `aggregate.total` (carries prior tenant's retired rows;
   gating on it false-fails every reused spot Blackwell). Blackwell uses `remapped_rows.*`, not
   `retired_pages.*`. (kimi-k2.6-nvfp4 L3.)
1. **Bare-EC2 node prep** (this is bare EC2, not EKS — one-off single node):
   `sudo systemctl start containerd` before any `nerdctl` (AMI doesn't auto-start it); use
   `sudo nerdctl`, `--network host` (no CNI on bare metal), `--gpus N`. NVMe is raw at boot —
   RAID-0 the instance-store disks (`mdadm --create /dev/md0 --level=0 --raid-devices=N
   /dev/nvme{1..N}n1`, `mkfs.xfs`, mount `/mnt/nvme`) for weights/images; spot reclaim wipes it,
   re-run per fresh node. (qwen3-next-g7e L5, devstral-sera, kimi-k2.6-nvfp4 L4a.)
2. **Stage 4b — observability up BEFORE serving.** Launch the Prometheus/DCGM/node-exporter
   stack (`.claude/skills/benchmark-runner/templates/observability-stack.docker-compose.yml`),
   run `observability-smoke-test.sh`, sync to S3 every 10 min. kimi-k2.6-nvfp4 permanently lost
   95 configs' TTFT because Prometheus wasn't up. Block until healthy. (tech-stack.md §observability.)
3. **Phase 0 — image build.** Base `vllm/vllm-openai:latest-cu130` (sm_120 needs cu130; confirm
   tag exists; cu129 lacks the cutlass/FP4 path — kimi-k2.6-nvfp4 L7). Install `aibrix_kvcache` +
   PrisKV on top. Validate engine boot with `--load-format dummy` before pulling 32GB weights
   (kimi-k2.6-nvfp4 L13: crashes fire at GPU-init *after* weight load). **This is falsification
   gate #4** — if the build can't be made to work on sm_120 in <1 day, stop.
4. **Stage 0c — serving-config gate.** `python3
   standards/serving-commons/resolver/validate-serving-config.py --sidecar benchmark.yaml
   --corpus-root .` (fail-closed, exit 2 blocks). Catches FP8 TP-divisibility / max-model-len
   traps. If PrisKV needs args the resolver doesn't know, waive with a cite.
5. **Harness smoke test (the kimi process-failure gate).** Before the first paid cell, fire ONE
   streaming request through `bench-standard.py` and confirm `vllm_time_to_first_token_seconds_bucket`
   actually lands in Prometheus (watch for the colon→underscore metric-name normalization bug that
   silently emptied TTFT). **No TTFT in Prometheus → do not start the matrix.** (kimi-k2.6-nvfp4
   CRITICAL CORRECTION: hand-rolled bench measured E2E not TTFT; never validated metrics first.)

## Measurement

- **Primary metric**: TTFT p50 / p99 (streaming) — where prefix reuse shows. `bench-standard.py`
  only (Prometheus histograms); never a hand-rolled harness.
- **Secondary**: aggregate output tok/s, per-replica cache-hit rate (engine + PrisKV `/metrics`),
  host-DRAM occupancy, p99 ITL (watch for PrisKV pin/unpin stalls under load).
- **Correctness gate (mandatory, every cell)**: fire the same prompt twice (cold miss, warm hit);
  compare **output token IDs**, not decoded text (text varies under sampling). Flag if Levenshtein
  distance > 5% of output length. A mismatch is **P0 — voids any TTFT win** (issues #41/#42 are
  cache races; a "hit" on stale/corrupt KV looks fast but is a bug). Verify-before-assert.
- **Workload**: `standards/benchmark-commons/workloads/shared-prefix-multitenant.yaml` (canonical
  card — don't inline divergent params, `[[feedback_canonical_workload_cards]]`). Shared-prefix
  ratio is the card's tunable; sweep it.
- **Prefix-size weighting (from the baseline)**: `qwen3-32b-eks` (2026-04-03) showed a **2K shared
  prompt warms to ~160ms TTFT within the single replica** — the per-hit saving at small prefix is
  already tiny, so a cross-replica effect there would be lost in noise. Lean the sweep on the
  card's **large fixed-system-prompt cells** (RAG-style, where a cold-replica miss costs real
  prefill time) — that's where Arm C can clear the 15%-vs-B bar. The baseline transfers as *shape*
  (cold→warm ratio), not absolute ms (it was L40S/vLLM 0.19, not g7e/sm_120).
- **Sample size**: 3 runs/cell, report median; flag any cell with >15% run-to-run TTFT variance
  (PrisKV races may show as variance, not just a mean shift).
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`.

## Fixtures

- `domains/gpu-serving/blueprints/qwen3-32b-eks/` — substrate for Qwen3-32B-FP8 serving config
  + the single-replica baseline numbers to anchor against. Reuse its vLLM args; do not duplicate.
- PrisKV integration is NOT an existing fixture — building `aibrix_kvcache + PrisKV` into a
  current vLLM on sm_120 is Phase 0 of this experiment (see falsification #4). The China-ECR
  prebuilt image (vLLM 0.10.2) is a last-resort reference only, not the test target.

## Rule the experiment would produce

**If hypothesis holds** → `tech-stack.md`: *"For PCIe-only g7e nodes serving high-shared-prefix
workloads (≥40% common prefix) with ≥2 co-located replicas, a shared host-DRAM KV cache
(PrisKV/aibrix_kvcache) lowers TTFT vs both engine-local caching and prefix-aware routing.
Use replica-per-GPU + shared cache, not one wide-TP replica. Does NOT apply to NVLink nodes
(B200/B300) — direct NVLink beats any DRAM-staging — nor cross-node (PrisKV's RC-verbs RDMA
path is dead on EFA)."*

**If falsified** → `tech-stack.md`: *"Shared external KV cache (PrisKV) adds no TTFT benefit
over prefix-aware routing on co-located g7e replicas; routing-to-state beats sharing-state.
Default to llm-d/GAIE prefix-aware routing for cross-replica reuse. Stop evaluating AIBrix
single-node-PD claims for our fleet."* — equally valuable; closes the question.

## Out of scope

- **Cross-node PrisKV** — settled by analysis: PrisKV's native RDMA transport uses RC queue
  pairs (IB/RoCE model), which **EFA does not support**; AWS has no InfiniBand. Cross-node
  would force the untested UCX/libfabric/EFA path. Not worth GPU time.
- **NVLink hardware (B200/B300)** — settled: NVLink 1.8 TB/s direct GPU→GPU beats any
  host-DRAM bounce; PrisKV is strictly a downgrade there.
- **PrisKV "zero-copy" micro-optimization** — it shaves one host-DRAM memcpy (the cheap leg);
  not the variable under test.
- LMCache / NIXL / Dynamo disagg — different techniques, separate specs.

## Carryover audit (spec-design gate)

Before running, confirm no prior-blueprint lesson was forgotten:
- [x] Ran `carryover-auditor` (2026-06-18). Found 7 gaps (3 P0, 4 P1); all folded into the
  Pre-flight/bring-up section above (Stage 4a ECC gate, bare-EC2 containerd/NVMe prep, Stage 4b
  observability-first, cu130 image, Stage 0c serving-config gate, harness smoke test, token-ID
  correctness gate). Cross-node PrisKV + NIXL confirmed correctly scoped out (N/A for this stack).
- [x] Overlapping stacks scanned:
  - `kimi-k2.6-nvfp4/lessons.md` — disagg lost prefix cache → 3.8× regression; **the cache
    IS the win** (informs why high-prefix-ratio is the regime). Hand-rolled bench measured
    E2E not TTFT → mandate `bench-standard.py`. EP/disagg blocked upstream.
  - `qwen3-32b-eks/` — single-replica floor; HyperPod LMCache no-op at 1 replica.
  - `qwen3-next-g7e/lessons.md` — g7e bring-up gotchas (sm_120, network host, nerdctl, EFA).
  - `[[infra]]` memory — g7e = no NVLink, PCIe interconnect, EFA = CPU-bounce not true RDMA;
    NCCL 2.25.1 broken on Blackwell PCIe (PrisKV uses no NCCL — likely N/A, confirm).
- [ ] Every applicable lesson reflected in matrix/fixtures/falsification, or noted N/A with cite.

## Cost estimate

- g7e.24xl spot ≈ **$13–15/hr** (Tokyo 1c, confirmed via ODCR probe 2026-06-18; 1a surge ~$30,
  avoid). g7e.12xl fallback similar per-GPU.
- Build + bring-up (Phase 0): ~0.5 day ≈ $7. Matrix (36 runs, ~10 min each + warmup): ~1 day
  serving ≈ $13×8h ≈ **$100**. Total cap: **~$150**.
- **Capacity/quota status (2026-06-18)**: g7e.24xl/48xl blocked by 64-vCPU G-instance quota
  (region-wide, spot+OD). Quota bump to 192 filed (PENDING: OD `d5028a16…`, Spot `a0bc9f4c…`).
  g7e.12xl (48 vCPU, 2 GPU) launchable NOW in Tokyo 1c — capacity confirmed by ODCR probe
  `cr-031cd06fbecee0544` (active→cancelled). Start on 12xl/2-replica; upgrade to 24xl/4-replica
  when quota lands.

## References

- Upstream: AIBrix PrisKV — https://github.com/aibrix/PrisKV ; blog
  https://aibrix.github.io/posts/2026-06-16-single-node-pd/ (skeptical review: claims "specialized
  CUDA kernels" but code is `cudaHostRegister`+`cudaMemcpy`; "shared-memory transport" not in code
  — only RDMA/UCX enums; zero-copy is open bug #33; cache races #41/#42).
- `domains/gpu-serving/blueprints/qwen3-32b-eks/results/benchmark-report-20260403.md` — baseline.
- `domains/gpu-serving/blueprints/kimi-k2.6-nvfp4/lessons.md` — disagg/cache lessons.
- `domains/gpu-serving/blueprints/glm5-llmd/` — prefix-aware routing (Arm B) prior art.
- Memory: `[[pd_disagg_single_node]]`, `[[project_pd_disagg_frontier_only]]`,
  `[[infra_g7e_capacity_chase]]`, `[[feedback_canonical_workload_cards]]`.
