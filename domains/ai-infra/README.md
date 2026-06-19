# AI Infrastructure Domain

Two things live here:

1. **Shared platform tooling** that other domains consume — slim serving images, the cold-start profiler, the build host that produces images, and any other reusable AI-infra plumbing.
2. **Inference-optimization experiments** — isolated technique studies that consume the shared tooling and produce *rules with conditions* that trickle down to `gpu-serving` blueprints, `mdc` deployment cards, and `.claude/steering/*.md`.

## Charter

| | gpu-serving | ai-infra |
|---|---|---|
| Unit of work | one model on one stack | shared platform tooling + one-technique experiments |
| Question | "how do we run model X?" | "what does the platform need, and when does technique T help by how much?" |
| Output | working endpoint + benchmark | reusable infra + generalizable rule + steering update |
| Lifetime | persistent deployment | tooling persists; experiments archive after their rule lands |

The domain is the **platform + lab**. `gpu-serving` and the steering files are the **factory floor**. A technique graduates from lab to factory when its rule is codified in steering and applied as default in deployment blueprints.

## What's where

| Path | Purpose |
|---|---|
| `shared/images/` | Slim Dockerfiles for vLLM and SGLang + build/compare scripts. Source of truth for serving containers used by experiments. |
| `shared/profiler.py` + `log_patterns.yaml` + `profiler_validate.py` + `stage_compare.py` | Unified cold-start profiling layer. Every experiment uses this. |
| `shared/cold_start_harness.py`, `weight_sync_harness.py` | Lightweight wrappers around the profiler for specific scenarios. |
| `blueprints/build-host/` | Spot EC2 + ECR + Terraform that builds slim images. Real infra blueprint. |
| `blueprints/<technique>/` | Individual experiment runs (created when a spec is executed). |
| `specs/_template.md` | Mandatory format for technique experiments. |
| `specs/profiler-validation.md` | Spec 0 — validates the shared profiler before any other spec runs. |
| `specs/<technique>.md` | Hypothesis-driven experiments (image-pull-acceleration, model-decoupling-and-load, compile-cache-strategies, etc.) |

## The cold-start stage budget

Every experiment must locate itself in the cold-start pipeline and report stage-time before/after. This budget is the canonical mental model for the lab.

```
node provision → image pull → container start → model load → JIT/compile → first token
```

| Stage | Typical floor (small) | Typical ceiling (frontier) | Levers |
|---|---|---|---|
| Node provision | 60-120 s | 60-120 s | Karpenter warm pool, AMI choice |
| Image pull | 30-60 s | 5-10 min | SOCI/Nydus, EBS prebake, OCI Image Volume, FUSE tuning |
| Container start | 5-10 s | 5-10 s | runtime choice (containerd default), bypass extras |
| Model load | 60-180 s | 5-15 min | Run:ai Streamer, ModelExpress P2P, model decoupling |
| JIT / compile | 0 s | 15+ min (DeepGEMM, torch.compile, CUDA graphs) | bake caches in image, mount on PVC, warmup + promote |
| First token | 1-5 s | 1-5 s | (irreducible CUDA context init) |

Stage estimates from ScaleOps's [GPU cold-start survey](https://scaleops.com/blog/reducing-gpu-cold-start-times-in-kubernetes-patterns-and-solutions/), extended with our memory's measurements on B200/B300 frontier serving.

> The frontier-model JIT stage is missing from every published taxonomy we've found. That gap motivates Spec C.

## Prior art (required reading before drafting a new spec)

| Source | What it gives | What it lacks |
|---|---|---|
| [AWS Labs ai-on-eks container startup guide](https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time) | Taxonomy: shrink image, accelerate pull. SOCI + EBS prebake recipes for Karpenter. | No benchmark numbers. No JIT. No P2P weight transfer. |
| [ScaleOps GPU cold-start patterns](https://scaleops.com/blog/reducing-gpu-cold-start-times-in-kubernetes-patterns-and-solutions/) | Stage-time budget. Warm node pool pattern. PCIe-vs-HBM bandwidth gap. | TP-as-cold-start-optimization claim is weak. No P2P. No JIT. |
| [Modal: fast lazy container loading](https://tinfoil-knight.github.io/notes/fast,-lazy-container-loading-in-modal-2024) | FUSE tunables (`read_ahead`, `max_pages`, `congestion_threshold`). Hierarchy: blob → CDN → AZ cache → node. 2.5 GiB/s with tuning. | Implementation proprietary. Stargz/SOCI users start from scratch on tuning. |
| [Modal: speeding up container launches](https://modal.com/blog/speeding-up-container-launches) | The 1,000-files-touched insight. Negative-lookup syscall cost. Latency hierarchy (~100 µs SSD vs ~2 ms NFS). | gVisor-specific runtime path doesn't apply to EKS. |

Specs must cite where they exceed or contradict prior art.

## Architectural principle: access-set minimization, not bandwidth maximization

Modal's data point: importing sklearn touches ~1,000 unique files but issues ~3,000 stat + ~1,000 openat calls. **Most container bytes are never read on the cold path.**

For LLM serving, the access ratio is plausibly 1-5% of total image bytes. This means:

- **Lazy loading wins are bounded by access ratio, not bandwidth.** A perfectly tuned eager pull still moves 20× more bytes than necessary.
- **Negative lookups (file-not-found) are a hot path for Python imports.** FUSE backends must fast-path them.
- **Tensor parallelism does not reduce cold start at the node level** — total node bytes-to-load are unchanged.
- **Replica-N+1 should never repeat replica-N's work.** P2P (ModelExpress) and node-local caches (Smart Cache DaemonSet pattern) eliminate redundant transfers.

The lab's hypothesis class: each spec should articulate *which redundant work it eliminates* and *for which replica index* (1st, Nth, or all).

## Directory layout

```
domains/ai-infra/
  specs/
    _template.md                          # mandatory hypothesis + matrix + falsification + stage-budget
    profiler-validation.md                # Spec 0 — prerequisite for everything else
    image-pull-acceleration.md            # Spec A
    model-decoupling-and-load.md          # Spec B
    compile-cache-strategies.md           # Spec C
    cold-start-stacked.md                 # Spec D
    fuse-tuning-for-snapshotters.md       # Spec E
    cold-start-access-profiling.md        # Spec F
  blueprints/
    build-host/                           # platform infra: spot EC2 + ECR for slim image builds
      terraform/
      scripts/
      README.md
    <technique>/                          # one per experiment, populated when spec runs
      hypothesis.md
      matrix.yaml
      results/<run-id>.json               # enriched artifacts (benchmark-commons format)
      analysis.md
      lessons.md                          # surprises and cross-cuts (frontmatter required)
  shared/
    images/                               # slim Dockerfiles + build/compare scripts
      Dockerfile.vllm-slim
      Dockerfile.sglang-slim
      build.sh
      compare-sizes.sh
      README.md
    profiler.py                           # canonical 14-event cold-start profiler
    log_patterns.yaml                     # version-keyed regex set for vLLM events
    profiler_validate.py                  # Spec-0 acceptance gate
    stage_compare.py                      # cross-spec stacked-bar comparison
    cold_start_harness.py                 # convenience wrapper around profiler
    weight_sync_harness.py                # learner → rollout sync (RL)
```

**Two halves coexist**: shared platform tooling (`shared/`, `blueprints/build-host/`) is long-lived infra. Per-technique experiments under `specs/` and `blueprints/<technique>/` are short-lived — once their rule lands in steering, the experiment is archived.

## Rules of the lab

1. **Every spec has a falsifiable hypothesis with a threshold and a condition.** "Investigate X" is not a spec.
2. **Every experiment has a baseline on the same infrastructure.**
3. **Every spec reports stage-time before/after** in the cold-start budget. This makes specs comparable and feeds the stacked spec naturally.
4. **Reuse `gpu-serving` blueprints as fixtures.** Don't redeploy here; point at the existing blueprint and add the experimental knob.
5. **Output is a rule, not a leaderboard.** Numbers go to `results-vault`; the conditional rule goes to steering.
6. **Falsification ends the experiment.** Negative results are the cheapest output of the lab — write them clearly and stop.
7. **Cross-cuts elevate.** When two techniques share a finding (e.g., "PCIe-only topology defeats P2P optimizations"), `compound-learner` lifts it to `.claude/steering/tech-stack.md`.

## Trickle-down workflow

1. Spec written under `specs/` with hypothesis + matrix + stage-budget claim.
2. Blueprint runs the experiment using the shared harness.
3. `analysis.md` records the rule and its conditions.
4. `lessons.md` frontmatter triggers `mdc learn` and (if applicable) `gpu-infra learn`.
5. `compound-learner` elevates cross-cutting rules to `.claude/steering/tech-stack.md`.
6. Affected `gpu-serving` blueprints get a one-line PR to flip the default.
7. New `gpu-serving` blueprints inherit the rule from steering.

## Active experiments

| Spec | Stage targeted | Technique | Status |
|------|---------------|-----------|--------|
| **0** — `profiler-validation.md` | All stages (instrumentation) | **Mandatory prerequisite.** | DONE |
| A — `image-pull-acceleration.md` | Image pull | Slim image vs upstream baseline measured; SOCI/Nydus deferred | PARTIAL |
| B — `model-decoupling-and-load.md` | Model load | RunAI Streamer measured (no win on local NVMe); ModelExpress P2P deferred | PARTIAL |
| C — `compile-cache-strategies.md` | JIT / compile | AOT cache HIT validated same-node; image-baked deferred | PARTIAL |
| **C-EBS** — `compile-cache-ebs-snapshot.md` | JIT / compile (persistent) | Bake compile cache to EBS, snapshot, restore on fresh node | **DONE — validated** |
| D — `cold-start-stacked.md` | All stages | Best-of-A + B + C combined | partial via stacked runs |
| E — `fuse-tuning-for-snapshotters.md` | Image pull (FUSE) | Modal-style tuning of stargz/SOCI/Nydus | DRAFT |
| F — `cold-start-access-profiling.md` | Image pull (measurement) | strace cold-start access ratio | DONE — 17.2% ratio measured |
| G — Cudagraph trim | CUDA graph capture | Trim `cudagraph_capture_sizes` from 51 to 10 | DONE — saves 162 s |
| H — Prefetch tuning | Weight load (page cache) | `--safetensors-load-strategy=prefetch` + thread/block sweep | DONE — saves 141 s on local NVMe |

**Run order matters.**
1. **Spec 0** first — validates the unified profiler against two existing fixtures.
2. **Spec F** next — profiles cold-start access set; informs Spec A's variant priority.
3. **Spec E** in parallel with F — cheap, no GPU needed, produces FUSE config for A.
4. **Specs A, B, C, C-EBS, G, H** in any order, each consuming the profiler's canonical artifact.
5. **Spec D last** — composes winning variants; validates additivity using the same artifact schema.

## Profiling layer

Every spec uses the same profiler (`shared/profiler.py`) and emits the same canonical artifact format. Stage boundaries are defined once in `shared/log_patterns.yaml` and enforced by `shared/profiler_validate.py`. Cross-spec comparison comes from `shared/stage_compare.py`.

Canonical 14-event timeline (T0-T13) covers: pod_create → node_assigned → image pull start/end → container created/started → python_alive → weights load start/end → JIT start/end → CUDA graphs done → health 200 → first token. Stage attribution and gap buckets are spec'd in `specs/profiler-validation.md`.

**This is non-negotiable**: a spec that doesn't emit canonical artifacts cannot be stacked into Spec D, and its findings can't be cross-compared.

## Steering rules to codify directly (no experiment needed)

These are operational consensus from prior art and our own memory. Land them as steering edits, not specs.

- **Warm node pools**: primary pool `min=1` on-demand, burst pool `min=0` spot, image pre-pull DaemonSet on primary. Source: ScaleOps.
- **Decouple model from image** for any model > 5 GB. Source: AWS Labs.
- **`hf_transfer` for HF source pulls; `runai_streamer` for S3 → GPU streaming.** Source: published benchmarks + our blueprints.
- **NCCL ≥ 2.26.2 on Blackwell PCIe-only**. Source: our memory (`devstral-sera/lessons.md`).
- **FP8 MoE TP rule**: `moe_intermediate_size / TP % 128 == 0`. Source: our memory (`b300_*` benchmarks).

## Out of scope

- Deployment of new models for production use → `gpu-serving`
- Training experiments / RL methodology → `autoresearch`
- Agent runtime tuning → `agent-runtime`
- One-shot benchmarks of an existing deployment → `gpu-serving` blueprint's `results/`
- Building Modal-shaped greenfield infrastructure (proprietary FUSE, content-addressed registry). The lab applies portable parts of that playbook to off-the-shelf EKS.
