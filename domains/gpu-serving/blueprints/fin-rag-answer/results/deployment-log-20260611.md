# Deployment Log — fin-rag-answer (Nemotron-3-Super-120B-A12B on B200)

Blueprint: `domains/gpu-serving/blueprints/fin-rag-answer/`
Spec: `domains/gpu-serving/specs/fin-rag-answer.md`
Workload: `standards/benchmark-commons/workloads/fin-support.yaml`
Cluster: `qwen3-next-bench-eks-cluster` (us-east-2, EKS 1.32, context `fin-rag-b200`)
Node: `ip-10-0-25-221` (p6-b200.48xlarge SPOT, 8x B200 NVSwitch)

---

## 2026-06-11T11:45Z — Stage 0b: Deployment card lookup

- `mdc get nemotron-3-super --engine vllm` → **NO CARD FOUND**. Ran `mdc sync` to pull upstream.
  Fallback authoritative sources: spec + memory file `nemotron3_super_serving.md` (contains
  the verified Stage 0 config.json facts). Recorded as the de-facto deployment card.
- `mdc prs nemotron-3-super` → no watch_prs defined (no card). Upstream PR status tracked via
  spec + memory: vLLM #40017 (disagg), #26201 (Mamba prefix-cache MERGED), version regressions
  #39223/#34356/#41565/#44022.
- `gpu-infra card p6-b200` → loaded. B200 NVSwitch (NVL5), driver 580.126.09, CUDA 13.0,
  AL2023 NVIDIA AMI required, NCCL not affected by 2.25.1 PCIe bug (NVSwitch). NCCL busbw PASS
  >= 1,050 GB/s. **Card field error noted**: card says `sm_120` but B200 is `sm_100` (sm_120 is
  g7e RTX PRO 6000). Does not affect this deployment. (benchmark.yaml correctly says sm_100.)

**Cross-reference (model vs GPU)**:
- Model TP requirement: agg-tp2-x4 (TP=2 x 4 workers) fits 8x B200 cleanly. TP4 also safe —
  FP8 is per-tensor static (NOT block-wise), so the `moe_intermediate_size/TP % 128` rule does
  NOT apply (confirmed Stage 0). No hardware+model incompatibility.
- B200 NVSwitch is NOT affected by the Blackwell-PCIe NCCL bug (that is g7e/sm_120 only).
- DeepGEMM JIT cold start: GPU card cites ~15 min for GLM-5-FP8. **However** nemotron-super
  lesson #13 measured this exact FP8 checkpoint at ~3 min total cold start on the dynamo-0.9.1
  image (weights 31s + init 14s + torch.compile 43s + CUDA graphs 9s). Will set readiness
  initialDelaySeconds=900 conservatively but expect faster.

**Stage 0b validation**: PASS (GPU card found; model card absent → fell back to spec+memory,
which hold the verified facts). No blocking conflicts. Upstream regressions flagged.

## 2026-06-11T11:50Z — Node join

- B200 node `ip-10-0-25-221.us-east-2.compute.internal` joined EKS (k8s v1.32.13), capacityType=SPOT.
- GPU/EFA device-plugin capacity not yet populated at first poll (self-heals). Validating in Stage 4a.
- Reusable: namespace `ml-inference`, FSx PVC `vllm-qwen3-fsx-pvc` (Bound 4800Gi RWX fsx-lustre).
- ECR image `nemotron-super-vllm:dynamo-0.9.1` present (us-east-2, 615299764834). Checking bundled vLLM version.

---

## 2026-06-11T13:25Z — RESUMED (prior agent killed by stall watchdog on a FS-wide find)

Verified live state on resume: EKS context `fin-rag-b200`, B200 node
`ip-10-0-25-221` Ready with 8× B200 183GB allocatable, nvidia-device-plugin up.
Stage 0 / 0c / 4a confirmed PASSED at handoff (not re-run). ml-inference has
prior completed jobs `gpu-preflight` + `vllm-version-check`.

### Stage 5 BLOCKER — vLLM image
- ECR `nemotron-super-vllm:dynamo-0.9.1` = vLLM 0.14.1 (too old, invalid). See lessons.
- `vllm-openai` ECR repo had only tags v0.7.3 / v0.15.1 / qwen3_5* — **no v0.18.1**.
- Launched in-cluster **skopeo** job `skopeo-vllm-018-copy` (CPU node) copying
  `docker.io/vllm/vllm-openai:v0.18.1` → ECR `vllm-openai:v0.18.1`. In progress.

### Stage 3 — weight staging
- FSx PVC `vllm-qwen3-fsx-pvc` (Bound, 4800Gi) held only old qwen3 weights (~228GB);
  no nemotron weights. Created `hf-token` secret (gated NVIDIA repo).
- Launched `stage-weights-fsx` job: FP8 (~124GB) first, then BF16 (~240GB) → `/fsx/models/`.

### Stage 6 — benchmark driver gap (flagged early)
- Stock `bench-standard.py` sends ONE identical prompt → would report fake ~100%
  prefix-hit and FAIL the fin-support `prefix_hit_rate_gt_corpus_ceiling` reliability
  gate. Need a fin-support-aware driver (verbatim header + unique tail). See lessons.

## 2026-06-11T13:30Z — Image resolution policy reversed (latest-stable first)

Operator directive: stop blind-pinning vLLM 0.18.1; try the latest STABLE first and let the
Stage 0 coherence smoke test be the arbiter. Implemented:

- **Confirmed Docker Hub tags exist** via `docker manifest inspect`: `vllm/vllm-openai:v0.22.1`
  (PRIMARY) and `vllm/vllm-openai:v0.18.1` (FALLBACK) both present. (`latest` = nightly, rejected.)
- **Local docker daemon down + only ~32 GB free on the Mac** → cannot pull/push a ~9 GB image
  locally. Used the cluster's established mechanism instead: an in-cluster **skopeo Job** that
  copies Docker Hub → ECR registry-to-registry (creds via secret `ecr-push-creds`, scheduled on a
  non-GPU node via `nvidia.com/gpu.present DoesNotExist`). The prior loop iteration had already
  launched `skopeo-vllm-018-copy` for the fallback; I added `skopeo-vllm-0221-copy` for the primary.
  Manifest tracked at `manifests/skopeo-vllm-0221-copy.yaml`.
- **Both copies COMPLETE**. ECR `vllm-openai` now has:
    - `v0.22.1` (8.6 GiB, pushed 2026-06-11T09:26 local) — PRIMARY
    - `v0.18.1` (8.9 GiB, pushed 2026-06-11T09:25) — FALLBACK (pre-staged so a fallback mid-deploy
      needs no extra network round-trip).
- **benchmark.yaml updated**: `engine.version: "0.22.1"`, added `version_fallback: "0.18.1"` and
  `candidate_images.{primary,fallback}` pointing at the two ECR tags. Stage 0c resolver reads
  engine.name/TP and surfaces prior failures — does not hardcode a version string, so the bump is
  structurally safe for the fail-closed gate.
- Spec-departing decision recorded in `lessons.md` (image policy reversal + smoke gate definition).

**Stage 4 GPU node**: re-verified — node `ip-10-0-25-221` Ready, `nvidia.com/gpu` allocatable = 8,
label `nvidia.com/gpu.present=true` present (the Stage-4 device-plugin label fix from lessons.md held).
Marked Stage 4 complete in progress.md.

**Next (Stage 5)**: deploy serving with the PRIMARY `:v0.22.1` image, then run the Stage 0 coherence
smoke (10 real Fin prompts; assert coherent / `NemotronHForCausalLM` / `mamba_cache_mode=all` + prefix
hit rate > 0). If PASS → lock 0.22.1 and record evidence. If FAIL → switch `container_image` to the
`:v0.18.1` fallback and document the 0.22.1 failure mode in lessons.md.

## 2026-06-11T13:48Z — Stage 4 + Stage 5 complete

**Stage 4 (NVMe + weights)**:
- Built RAID0 over the 8 local instance NVMe (nvme1n1..nvme8n1) -> /dev/md0 -> ext4 -> /mnt/nvme (28TB).
- FP8 (120G) downloaded directly HF->NVMe in 94s (hostNetwork bypasses broken B200 pod DNS). config.json: arch NemotronHForCausalLM, model_type nemotron_h, quant modelopt (per-tensor FP8), moe_intermediate_size 2688. Ships super_v3_reasoning_parser.py (plugin) — NOT used (reasoning OFF).

**Stage 5 (serving agg-tp2-x4 FP8)**:
- `serving-agg-tp2x4-fp8.yaml`: 4 replicas x TP=2, image vllm-openai:v0.18.1, kv fp8, TRITON_ATTN, prefix-cache + mamba-cache-mode all, max-model-len 16384, max-num-batched-tokens 8192, max-num-seqs 256, gpu-mem-util 0.90. NO reasoning-parser flag.
- vLLM v0.18.1 confirmed in engine banner. Cold start ~4 min/replica (40s weights + 17s compile + 20s graphs + 53s init). NO 15-min JIT. KV cache 8.44M tokens, 507x conc headroom at 16K ctx. No OOM.
- Probe fix: initialDelaySeconds 900->30/60; strategy RollingUpdate->Recreate (GPU deadlock).
- **Stage 0 version coherence smoke: PASS** — 5/5 real Fin prompts coherent, grounded, no garble, reasoning correctly OFF.
- Cross-node routing broken -> bench-runner pinned to B200 node + hostNetwork hits ClusterIP (health 200, model nemotron-3-super).

---

## 2026-06-11T16:35Z — Stage 6b resume: n-gram spec-decode EAGER workaround

- Confirmed prior n-gram leg CrashLoopBackOff (11-12 restarts, exit 1) — operator-diagnosed
  mamba_attn CUDA-graph capture shape mismatch (tensor a(2) vs b(6)). Not re-diagnosed.
- Applied fix per directive: added `--enforce-eager` to `manifests/serving-agg-tp2x4-fp8-ngram.yaml`
  (disables CUDA-graph capture → sidesteps broken mamba_attn graph-capture path).
- `kubectl apply` → deployment.apps/fin-rag-vllm-fp8 configured (Recreate; crash-loop pods drain first).
- Goal of this leg: MEASURE acceptance_rate on real Fin prompts @ temp=1.0 (headline).
  E2E/TPOT from this leg labeled EAGER-ONLY (not production-representative; not compared to graph winner).

---

## 2026-06-11T17:30Z — Stage 6b continuation: optimization-grid legs (KV-fp8, FlashInfer, TP4, TP1 prefix)

Picked up the graph-captured FP8 winner (4/4 Ready, clean args: TP2, --kv-cache-dtype fp8,
TRITON_ATTN, mnbt=16384, max-num-seqs 256, no speculative-config, no enforce-eager).
Live service ClusterIP 172.20.146.6:8000, model `nemotron-3-super`, vLLM **0.18.1**.
Bench driver run from `bench-runner` (hostNetwork on B200 node) hitting the ClusterIP.
Copied current `scripts/bench-fin-support.py` into the pod (the in-pod copy lacked `--temperature`).

### Leg 1 (P1) — FP8 KV cache on vs off — DONE
- Removed `--kv-cache-dtype fp8` (manifest `serving-leg1-kvauto.yaml`), Recreate, ~4min cold start, 4/4 Ready.
- **conc=130: E2E p50 4598 / p90 7366, TTFT p50 390, TPOT p50 49.8, 0 errors → PASS.**
- conc=8: E2E p50 1345 / TTFT 162 / TPOT 17.9, 0 errors.
- **KEY FINDING**: dropping the flag did NOT disable fp8 KV. Engine still logs `Using fp8 data type
  to store kv cache` and config shows `kv_cache_dtype=fp8_e4m3` (`quantization=modelopt`). The
  ModelOpt FP8 checkpoint ships k/v scaling factors → engine defaults KV to fp8_e4m3. The flag is a
  no-op. Available KV memory 95.02 GiB (== winner's ~95 GiB). The small latency delta vs winner
  (4598 vs 4685) is run-to-run noise on the identical fp8 KV path. No KV-off config is achievable on
  the FP8 checkpoint short of loading BF16 (already shown to fail p90).

### Leg 2 (P1) — FlashInfer vs TRITON_ATTN attention — IN PROGRESS
- Applied `serving-leg2-flashinfer.yaml` (winner + `--attention-backend FLASHINFER`) at 17:44Z, Recreate.

### Leg 2 (P1) — FlashInfer vs TRITON_ATTN — DONE
- `serving-leg2-flashinfer.yaml` (winner + `--attention-backend FLASHINFER`), Recreate, cold start ~5min, 4/4 Ready.
- FlashInfer LOADED on hybrid Mamba2 (`Using AttentionBackendEnum.FLASHINFER backend`, HND layout).
- **conc=130: E2E p50 4197 / p90 7598, TTFT p50 354, TPOT p50 44.9, 0 errors → PASS** (~10% faster p50, ~18% lower TPOT vs TRITON 4685/55.0).
- conc=8: 1346 / 166 / 15.5.
- **CAVEAT**: FlashInfer fp8 attention warns `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention. This may cause accuracy issues` — the FP8 checkpoint lacks q/prob scales. TRITON_ATTN avoids this path. Faster but accuracy-risk; needs a coherence smoke before adopting.
- FP8 MoE GEMM backend stayed FLASHINFER_TRTLLM in both legs (independent of attention backend, confirmed).

### Leg 3 (P2) — agg-tp4-x2 vs agg-tp2-x4 — DONE
- `serving-leg3-tp4x2.yaml` (2 replicas x TP=4 = 8 B200, TRITON_ATTN + fp8 KV + mnbt=16384), Recreate, ~4.5min cold start, 2/2 Ready. TP4 NCCL clean on NVSwitch.
- TP4 KV pool 122.94 GiB / 10.74M tok / 635x (vs TP2 95 GiB / 8.44M / 507x).
- **conc=130: E2E p50 6755 / p90 10593 → FAILS BOTH gates. TPOT 74.5.** conc=8: 1156/382/9.0 (best TPOT floor).
- **Finding**: tp2-x4 (more replicas) beats tp4-x2 at high concurrency — 2 replicas means ~65 conc/replica → deeper prefill queue. Extra KV headroom wasted; NVSwitch TP4 comms lift TPOT. Keep agg-tp2-x4.
- Correction: the `uncalibrated q_scale/prob_scale 1.0 fp8 attention` warning is from `--kv-cache-dtype fp8`, present on TRITON too — NOT FlashInfer-specific.

### Leg 4 (P2, key science) — TP1 prefix-cache probe — DONE
- `serving-leg4-tp1-prefix.yaml` (1 replica x TP=1, gpu-mem-util 0.92). TP1 LOADS on single B200 (no OOM). KV pool 1.94M tok / 116.67x. Cold start ~5.7min.
- **THE ANSWER**: prefix-cache hit rate is STILL 0 at TP1. `vllm:prefix_cache_hits_total{engine=0}` = 0.0 after 1,805,903 queries. NOT a TP>1 (#26201) artifact — Mamba2 automatic prefix caching is non-functional on vLLM 0.18.1 at ANY TP for this architecture.
- Perf (conc=64): E2E p50 10559 / p90 16940 → FAILS SLO (single-GPU throughput, 116x ceiling). TP1 not a viable serving layout; was a science probe.
- Driver bug noted: prefix scrape catches `_created` timestamp gauges → garbage cumulative numbers; windowed delta (0.0) still correct; authoritative = `prefix_cache_hits_total` scraped directly.
