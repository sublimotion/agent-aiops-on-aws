# Dynamo Snapshot — Inference Cold-Start C/R Experiment

## Status: DRAFT

## Hypothesis

NVIDIA Dynamo Snapshot (CRIU + `cuda-checkpoint`) restores a fully warmed-up single-GPU inference worker in **≤10 seconds** on g7e/p5e from FSx Lustre, bypassing weight load, kernel warmup, CUDA graph capture, torch.compile, and DeepGEMM JIT — a ≥20× cold-start reduction on workloads where JIT/compile dominates (vLLM/SGLang on Blackwell).

Storage backend matters: **FSx Lustre with O_DIRECT** matches the upstream "AIO fast path" claim within 2× of striped local NVMe; **EFS** falls back to buffered readahead and is ≥3× slower at restore than FSx.

## Falsification criteria

- Restore time on FSx Lustre > 30s for a 6 GiB artifact (Qwen3-0.6B class) → AIO/parallel-memfd path not delivering on AWS-managed FSx.
- Restored worker's TTFT or throughput regresses >5% vs a freshly warmed worker → C/R correctness issue.
- Artifact size > 2× model weights after KV-cache unmap → vLLM/SGLang `sleep()`/`torch_memory_saver` integration broken on our images.
- Cannot reproduce on EKS without modifying `runc` or the AMI kernel beyond the standard NVIDIA AL2023 AMI we already use → portability claim fails for our stack.

## Why this matters

Cold-start is our top operational pain on Blackwell:
- GLM-5 SGLang: ~15 min (DeepGEMM JIT)
- vLLM GLM-5: ~16 min (DeepGEMM + torch.compile + CUDA graphs)
- Ray Serve FT head restart: ~3 min

Existing specs (A image-pull, B model-load, C compile-cache) attack stages in isolation. Snapshot collapses **model load + JIT/compile + first-token warmup** into a single restore step. Single-GPU-only today, but that's exactly the regime for Ministral-3B, Mistral-Small-4, agent-harness verifier workers, and the cost-aware-routing small-model pool.

Positive result enables a "snapshot-restore" replica path that complements (does not replace) Spec D stacking, and gives us a credible answer for sub-30s scale-out of small/medium models on EKS.

## Stage-budget claim

Single-GPU vLLM, Ministral-3B class (~6 GiB artifact), replica-N≥2 on warm pool, FSx Lustre backend:

| Stage | Baseline (sec) | Predicted with technique (sec) | Why |
|---|---|---|---|
| Node provision | 0 (warm pool) | 0 | unchanged |
| Image pull | 5-15 (EBS prebake) | 5-15 | unchanged |
| Container start | 5-10 | 5-10 | unchanged |
| Model load | 60-120 | 0 | rolled into restore |
| JIT / compile | 30-90 | 0 | rolled into restore |
| CRIU + cuMemMap restore | — | 5-15 | new step, FSx Lustre + AIO |
| First token | 1-5 | 1-5 | unchanged |
| **Total** | **100-240** | **15-45** | ~5-10× |

For Qwen3-8B class (~26 GiB artifact, predicted restore 5-10s with optimized CRIU): **~10-15×**.

For gpt-oss-120b class with GMS + striped NVMe (when CUDA driver patch lands): NVIDIA's <5s claim, **>20×**.

Applies to **replica-N≥2**. Replica-1 still pays first-warm cost to *produce* the snapshot; that cost amortizes across the lifetime of the artifact.

## Matrix

| Axis | Values |
|------|--------|
| Models | Qwen3-0.6B (small, NVIDIA reference, sanity check), **Gemma-4-26B-A4B-it (medium-tier anchor)** — directly comparable to Modal's published AOT compile-cache HIT floor of 22s on H200 (see `blueprints/spec-c-compile-cache/references/modal-gemma4-aot-h200.md`); Mistral-Small-4 (medium, single-GPU TP=1, repo-native fixture) |
| Hardware | g7e.24xlarge (Blackwell PCIe sm_120, primary), **p5e.48xlarge single-GPU slice (H200, primary for Gemma-4 anchor cell — matches Modal's hardware)**, p5e GMS path exploratory (gated on CUDA driver patch) |
| Storage backend | FSx Lustre (with O_DIRECT), EBS gp3 + Fast Snapshot Restore (canonical snapshot → per-replica volume via EBS CSI), striped local NVMe (RAID0 across instance store) |
| Engine | vLLM (v0.20.0+ with `sleep()`/`wake_up()`), SGLang (`torch_memory_saver`) |
| Variants | baseline cold start (no snapshot), snapshot restore |

**Cells run** (12): {3 models} × {g7e for Qwen3-0.6B + Mistral-Small-4; p5e for Gemma-4-26B-A4B-it} × {FSx, EBS, NVMe} × {snapshot} + 3 baselines per storage. **Gemma-4 anchor cell** is on p5e specifically to compare against Modal's 22s floor on identical hardware. EFS dropped — no O_DIRECT, NFSv4 buffered readahead is a foregone-conclusion negative.

**EBS pattern**: one canonical EBS snapshot stored in S3 by AWS, FSR-enabled in target AZs. Each replica's PVC creates an independent gp3 volume from that snapshot via the EBS CSI driver — no Multi-Attach, no shared physical volume. Measures volume-create + attach overhead (~10-20s with FSR-warm) on top of CRIU restore. **Exploratory** (deferred): p5e + GMS path (gated on CUDA driver patch); SGLang on Mistral-Small-4 (depends on `torch_memory_saver` integration in our image).

## Stage 0: g5 correctness smoke (gating)

Before booking any g7e or p5e capacity, validate the **cuda-checkpoint correctness path** on the cheapest, most boring GPU we can rent. Hypothesis-killing failures here save the entire matrix budget (~$230).

**Why g5, not g7e**: Ampere (sm_86) has the longest cuda-checkpoint driver runway (R550+ stable for >1 year). g7e Blackwell sm_120 has a track record of edge-case bugs in our fleet (NCCL 2.25.1 broken per MEMORY.md). Validating on g5 first localizes any failure to silicon vs. tooling.

### Sequence

1. **Build the patched CRIU sidecar image** on a CPU runner (no GPU needed). Verify `criu --version` includes the NVIDIA AIO + parallel-memfd patch tags. Validate kernel sysctls (`yama.ptrace_scope`, `ns_last_pid`) and capabilities (`CAP_CHECKPOINT_RESTORE`) work via a CPU dump+restore of a sleep loop on AL2023.
2. **g5.xlarge spot** (single A10G, 24 GB), driver R555+ AMI, ~$0.30-0.50/hr.
3. Deploy Qwen3-0.6B under vLLM (fits in 24 GB), warm with one prompt.
4. `cuda-checkpoint --toggle` → `criu dump` → `criu restore` → `cuda-checkpoint --toggle`.
5. **Correctness gate**: same prompt + same seed → restored worker emits **identical token IDs** as freshly warmed worker (not just similar TTFT). Hash the first 64 generated tokens.
6. **Performance sanity**: restore wall-clock < 30s for 6 GiB artifact on local NVMe.

### Pass criteria (all three required to proceed)

- ✅ Restored worker produces byte-identical token IDs vs freshly warmed worker (logits-equivalent)
- ✅ Artifact size threshold (size-tiered, per bridging cell measurement 2026-05-30):
  - Models with weights **<3 GiB**: artifact ≤ weights + 4 GiB (fixed ~3 GiB Python/torch/vLLM/CUDA process overhead dominates)
  - Models with weights **≥3 GiB**: artifact ≤ 2× weights
  - Use vLLM `sleep(level=1)` ONLY — `level=2` reloads weights on a different CUDA context post-restore and breaks token-ID equality
- ✅ Restore < 30s on local NVMe for 6 GiB artifact (proves AIO path triggers)

### If Stage 0 fails

| Failure | Inferred cause | Action |
|---|---|---|
| Token IDs differ | cuda-checkpoint corrupts CUDA graph state | Halt experiment; file upstream; no production value yet |
| Artifact >> 2× weights | vLLM `sleep()` doesn't release as expected on our image | Fix image first, retry Stage 0 |
| Restore > 30s on NVMe | AIO/parallel-memfd patches not applied or not reachable | Rebuild CRIU image with explicit `--enable-aio`, retry |
| Driver / kernel rejection | R555+ not on the AMI, or CAP_CHECKPOINT_RESTORE missing | Switch AMI; document host prereqs |

### Stage 0 → next step

If Stage 0 passes on g5: **rerun the smoke cell on g6 (L4, sm_89)** to confirm Ada works — another ~$2 of compute. Only then promote to g7e/p5e for the full matrix. Total Stage 0 cost: **<$10**.

### Cross-family expectation

| Family | GPU | Stage 0 likelihood |
|---|---|---|
| g5 (A10G, sm_86) | Ampere | Most likely to work — primary smoke target |
| g6/g6e (L4/L40S, sm_89) | Ada | Very likely — secondary smoke target |
| p5/p5e (H100/H200, sm_90) | Hopper | Likely — Modal validates this on H200 already |
| **g7e (RTX PRO 6000, sm_120)** | Blackwell PCIe | **Riskiest** — known fleet-level Blackwell bugs |
| p6-b200 (B200, sm_100) | Blackwell NVSwitch | Untested; not in matrix |

If g5 works but g7e fails, the produced rule still has value (covers g5/g6/p5 fleets) and we file the Blackwell gap upstream.

## Baseline

For each (model, storage) pair:
- Standard blueprint deploy (e.g. `domains/gpu-serving/blueprints/ministral-3b/`), `--load-format auto`
- Cold pod, weights pulled from S3 → storage backend, full warmup including a synthetic prompt to force CUDA graph capture
- Measured pod-create → first-token-streamed

Snapshot variant uses identical blueprint + `snapshot-agent` DaemonSet + signal-file hooks added to the entrypoint.

## Measurement

- **Primary**: pod-create timestamp → first-token-streamed timestamp (seconds)
- **Cost-per-restore-second**: capture $/restored-replica-second for each storage tier (latency × $/hr storage cost amortized across replica count and lifetime). The decision rule is a **sec/min vs $/mo tradeoff** — emit both axes per cell so the produced rule has a defensible Pareto frontier.
- **Secondary**: artifact size on disk; CRIU dump time; CRIU restore time; cuMemMap restore time; steady-state TTFT P50/P99 and throughput vs freshly warmed worker (correctness gate); restore wall-clock vs storage read bandwidth (was AIO actually triggered?)
- **Sample size**: 5 cold restores per cell, drop min/max, report median + p95
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`, with a per-stage breakdown matching the budget table above
- **Tool**: extend `domains/ai-infra/shared/cold-start-profiler/` to emit a `restore_breakdown` block (CRIU dump/restore, memfd parallel read, AIO submit/complete)

## Fixtures

- `domains/gpu-serving/blueprints/ministral-3b/` — primary substrate
- `domains/gpu-serving/blueprints/mistral-small-4-hyperpod/` — single-GPU slice (override TP=1 for this experiment)
- New: `domains/ai-infra/blueprints/dynamo-snapshot/` — Helm chart for `snapshot-agent` DaemonSet, sidecar image with NVIDIA's CRIU branch (AIO + parallel-memfd patches), entrypoint wrapper that writes `/tmp/ready-for-checkpoint` and polls for `/tmp/restore-complete`

## Rule the experiment would produce

If hypothesis holds:

> For single-GPU vLLM/SGLang inference workloads on EKS where replica-N cold-start latency matters (autoscaling small/medium models, cost-aware-routing tiers), use the `dynamo-snapshot` DaemonSet pattern. Storage tier selection is a **latency vs $/month tradeoff** — pick the cheapest tier that meets the cold-start SLO:
>
> | Tier | Restore latency (50 GiB artifact) | Steady-state $/mo (1 artifact) | When to use |
> |---|---|---|---|
> | **EBS gp3 + FSR** (default) | ~10-20s volume-create + CRIU restore | ~$10-100 (FSR on-demand) to ~$550 (always-on, 1 AZ) | Production default. Multi-AZ via snapshot. Toggle FSR per AZ during scale events. |
> | **Striped local NVMe** | fastest, ephemeral | $0 (instance store) | Replica-1 produce + same-node warm-pool restore only. |
> | **FSx Lustre PERSISTENT-1000** | comparable to NVMe at scale | ~$1,095+ (1.2 TiB minimum, always-on) | Artifacts >500 GiB total, many concurrent restores across AZs, always-warm pool. |
> | **FSx Lustre PERSISTENT-125** | ~150 MB/s cap, won't hit AIO fast path | ~$146 | Avoid for this workload — saves nothing vs EBS+FSR and loses on latency. |
>
> Order of preference: NVMe (where applicable) → EBS+FSR → FSx P-1000. EBS+FSR is **3-15× cheaper than FSx P-125 and 20-100× cheaper than P-1000** for single-artifact, intermittent-burst workloads, with a ~10-20s latency penalty vs Lustre's striped throughput.
>
> Skip when model is multi-GPU TP>1 (not yet supported upstream). Do not use EFS — NFSv4 buffered readahead lacks O_DIRECT and degrades restore ≥3×. Do not skip FSR on EBS — lazy-loaded snapshot reads add 30s-2min and erase the cold-start win.

If falsified, the rule degenerates to: "Stick with Spec A+B+C stacking; snapshot's AWS-managed-FS path is not yet competitive."

## Out of scope

- Multi-GPU / multi-node restore (NCCL, NIXL quiesce) — upstream roadmap, revisit when shipped
- TensorRT-LLM (upstream roadmap)
- GMS (GPUDirect Storage) path — exploratory cell only, deferred until CUDA driver patch lands
- Replica-1 (first-ever) cold start — by construction, snapshot can only help replica-N; replica-1 stays in Spec D's domain

## Persistent caches via EBS snapshot

**Not applicable directly.** Snapshot artifacts are larger than weights (process state + filesystem overlay), per-revision (vLLM/SGLang version, model weights, hardware, driver), and benefit from shared-filesystem semantics (any node can restore). FSx Lustre is the right primitive, not EBS.

However: snapshot *artifacts themselves* can be tiered S3 (cold) → FSx (hot) using `s5cmd`, mirroring the existing slim-image and weight-staging patterns in `domains/ai-infra/`. Tag artifacts with `(model, engine_version, hardware, driver, config_hash)` exactly like the Spec C-EBS pattern.

## Cost estimate

- **Stage 0 smoke**: 1× g5.xlarge spot (A10G), ~$0.40/hr × ~6 hr + 1× g6.xlarge spot (L4), ~$0.50/hr × ~4 hr = **~$5** (gates everything below)
- 1× g7e.24xlarge spot, ~$3.50/hr × ~24 hr matrix execution = **~$85**
- 1× p5e.48xlarge on-demand single-GPU slice (capacity block), ~$10/hr × ~12 hr for Gemma-4 anchor cell + GMS exploratory = **~$120** (anchor cell runs regardless; GMS exploratory only if CUDA driver patch lands)
- FSx Lustre PERSISTENT-125 1.2 TiB, ~$0.20/hr × 30 hr = **~$6** (note: PERSISTENT-125 caps ~150 MB/s — won't validate AIO fast path; bump to PERSISTENT-1000 for the fast cells, ~$1.50/hr × 10 hr = **~$15**)
- EBS gp3 volumes (per-replica, ~50 GiB each, prorated hourly) negligible (~$1); FSR (per-snapshot per-AZ ~$0.75/hr × 10 hr × 1 AZ) = **~$8**
- S3 egress for artifact distribution, negligible
- **Total cap: ~$235** (g7e $85 + p5e Gemma anchor + GMS exploratory $120 + FSx $15 + EBS/FSR $8 + buffer)

## References

- Upstream blog: https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
- NVIDIA Dynamo project: https://github.com/ai-dynamo/dynamo
- CRIU: https://github.com/checkpoint-restore/criu (NVIDIA AIO/parallel-memfd patches pending merge)
- Related specs: `cold-start-access-profiling.md`, `cold-start-stacked.md`, `compile-cache-ebs-snapshot.md`, `model-decoupling-and-load.md`
- Memory: GLM-5 cold-start measurements (MEMORY.md → B200 section), Ray Serve FT head restart (MEMORY.md → Ray Serve FT section)
