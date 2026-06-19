# Spec B — Model Decoupling and Weight Load

## Status: DRAFT

## Hypothesis

For models ≥100 GB:

- **Decoupling weights from the container image** (init container + S3 with `s5cmd`) cuts total cold start by ≥2× vs baked-in-image.
- **Run:ai Model Streamer** (`--load-format runai_streamer`) cuts the model-load stage by 2-5× when loading from S3, and is neutral (±10%) on RAID0 NVMe.
- **ModelExpress same-node multi-pod**: 2nd+ pod on a node where weights are already in another pod's HBM completes model load in seconds via shared-memory / NVLink P2P. Expected ≥5× over Run:ai Streamer. **This is the in-scope ModelExpress cell.**
- **Without ModelExpress coordination**, every pod independently re-pulls from S3 (thundering herd). Run:ai Streamer + FSx is the ceiling; ModelExpress's contribution is replica-N optimization specifically.

Cross-node EFA P2P is **out of scope** for this spec. Rationale: EFA is kernel-bypass but still CPU-bounced (per our memory), achievable bandwidth is uncertain, and validating it requires 2× B300 simultaneously which is not guaranteed at spot pricing. Defer to a follow-up spec when the question becomes load-bearing.

## Falsification criteria

- Decoupling improvement < 1.5× → bake model into image and rely on Spec A's image-pull acceleration instead.
- Run:ai Streamer improvement on S3 < 2× → drop the flag flip; default loader is good enough.
- ModelExpress same-node improvement < 3× → operational cost (gRPC service, Redis/CRD) not justified.

## Why this matters

Model load is the single largest stage in our memory's measurements: GLM-5 ~733 GB, Kimi K2.6 ~1.1 TB. Even at 10 GB/s NVMe, that's 70-110 s minimum. Multi-replica scale-out without coordination means each pod independently re-pulls from S3 — a thundering herd that compounds with autoscale latency.

This is also the spec where the Lila critique applies most directly: their Smart Cache DaemonSet pattern + lack of P2P is the "no off-the-shelf" anti-pattern.

## Stage-budget claim

| Stage | Baseline (sec) | Decoupled + s5cmd | + Run:ai Streamer (S3) | + ModelExpress P2P (replica N≥2) | Why |
|---|---|---|---|---|---|
| Node provision | 60-120 | 60-120 | 60-120 | 60-120 | unchanged |
| Image pull | 300-600 (model in image) | 30-60 (lean image) | 30-60 | 30-60 | image is now lean |
| Container start | 5-10 | 5-10 | 5-10 | 5-10 | unchanged |
| Model load | 600-900 (S3 default) | 300-500 (s5cmd S3 → disk → loader) | 90-180 (concurrent stream) | 10-30 (HBM→HBM) | each layer compounds |
| JIT / compile | unchanged | unchanged | unchanged | unchanged | Spec C |
| First token | 1-5 | 1-5 | 1-5 | 1-5 | unchanged |

Replica index: 1st-replica for the first three columns; **Nth replica (N≥2)** for the ModelExpress column — a peer must already have weights resident.

## Matrix

| Axis | Values |
|------|--------|
| Models | (M) Qwen3 8B (~16 GB), (L) GLM-5-FP8 (~733 GB), (XL) Kimi K2.6 (~1.1 TB) |
| Storage origin | S3 (always), FSx Lustre (large only), pre-staged NVMe (large only) |
| Load mechanism | (1) baseline `--load-format auto`, (2) init container + s5cmd, lean image, default loader, (3) (2) + Run:ai Streamer, (4) (3) + ModelExpress P2P |
| Replica index | 1, 2, 4, 16 (for ModelExpress scaling validation) |
| Fabric | g7e (EFA, 2 interfaces), p5e (InfiniBand), p6-b300 (NVSwitch + EFA) |

Run a representative subset (~30 cells); prioritize XL+S3+all-load-mechanisms and the replica-index sweep on ModelExpress.

## Baseline

Model baked into container image, default `--load-format auto` (safetensors sequential), no init container, ECR direct pull. This is the "naive" pattern still common in custom-built images.

## Measurement

Reuse `shared/cold_start_harness.py` for pod-create → first-token. Add explicit instrumentation:

- Pod-create → pod-Running (image-pull stage, expected to drop sharply with decoupling).
- Pod-Running → weights-on-GPU (model-load stage; vLLM logs `weights loaded in X seconds`).
- Weights-on-GPU → first-token (warmup + JIT, controlled-for here, varied in Spec C).

For ModelExpress: also record whether the replica hit the P2P path or fell back to shared storage. Replica-1 (no peer) vs replica-N (peer available) are *different cells*, not noise.

## Fixtures

- `domains/gpu-serving/blueprints/glm5-fp8/` for the L model.
- `domains/gpu-serving/blueprints/kimi-k2.6-speculative/` for XL.
- A small Qwen3-8B blueprint (existing or trivial new) for M.
- ModelExpress server: deployed once into a shared `ai-infra` namespace, Redis backend.

## Rule the experiment would produce

> **Default model-load strategy**:
> - Always decouple weights from image when model > 5 GB. Use init container + s5cmd from S3.
> - Add `--load-format runai_streamer` to vLLM args when loading from S3 or FSx. Skip on warm NVMe (no benefit).
> - Deploy ModelExpress when serving ≥3 replicas of any model ≥100 GB on RDMA-capable fabric. Default `--load-format mx` for those deployments.
> - On non-RDMA fabric (or single-replica deployments), Run:ai Streamer + FSx Lustre is the ceiling; ModelExpress offers no benefit.

## Out of scope

- KV cache (different problem; LMCache is the answer there).
- TensorRT-LLM engine artifacts (Spec C, since they're closer to compile caches than weights).
- The actor-learner RL weight-sync use case for ModelExpress — covered in `modelexpress-rl-weight-sync.md` under autoresearch.

## Cost estimate

~$1,000-2,000 across all cells — XL models on B300 are the expensive ones. Cap before launching the matrix.

## References

- Run:ai Model Streamer: https://github.com/run-ai/runai-model-streamer
- ModelExpress: https://github.com/ai-dynamo/modelexpress
- AWS Labs: model decoupling patterns — https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time/reduce-container-image-size/decoupling-model-artifacts
- Our memory: `b300_kimi_k26_benchmark.md`, `qwen3_235b_b300_benchmark.md`, `pd_disagg_single_node.md`
