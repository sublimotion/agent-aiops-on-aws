---
blueprint: fin-rag-answer
model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B
engine: vllm
hardware: p6-b200.48xlarge (8x B200 NVSwitch)
status: complete
last_stage: 7
last_updated: 2026-06-11T16:35:00Z
stages:
  stage0_preflight: complete
  stage1_foundation: skipped        # cluster + FSx already exist (reuse)
  stage2_build_machine: skipped     # using prebuilt ECR image
  stage3_storage_staging: complete  # FP8 (120G) + BF16 (231G) both on NVMe
  stage4_gpu_node: complete         # B200 Ready, 8 GPU, NVMe RAID0 28TB
  stage4a_gpu_health: complete      # gpu-preflight job PASSED
  stage4b_observability: skipped    # B200 pod DNS broken; using direct /metrics scrape
  stage5_serving: complete          # FP8 agg-tp2-x4 4/4 Ready, /v1/models OK
  stage6_prebench: complete         # bench-fin-support.py driver validated
  stage6b_benchmark: in_progress    # P0+P1-chunked done; MTP+n-gram(graph) BLOCKED; n-gram EAGER applied for acceptance
  stage7_readiness: not_started
  stage8_compound: not_started
---

# fin-rag-answer — Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 0 Pre-flight | complete | vLLM **0.18.1** in ECR (skopeo). Flags verified. |
| 1 Foundation | skipped | Reuse EKS `qwen3-next-bench-eks-cluster` + FSx PVC. |
| 2 Build machine | skipped | Prebuilt image; skopeo DockerHub→ECR. |
| 3 Storage/staging | complete | FP8 120G + BF16 231G both on B200 NVMe (FSx unmountable on B200 → direct HF download). |
| 4 GPU node | complete | B200 Ready, 8 GPU. NVMe RAID0 (28TB) at /mnt/nvme. |
| 4a GPU health | complete | gpu-preflight PASSED (0 ECC, 37-42C, no Xid). |
| 4b Observability | skipped | B200 pod DNS broken → direct /metrics scrape from bench-runner. |
| 5 Serving | complete | FP8 agg-tp2-x4, mnbt=16384 (P1 winner). Probe delays fixed (60/120; MTP variant needs liveness 1500). |
| 6 Pre-bench | complete | `scripts/bench-fin-support.py`: verbatim header + unique tail, ISL lognormal, prefix-hit scrape. |
| 6b Benchmark | in_progress | P0+P1-chunked DONE. MTP=BLOCKED(TP2 shape). n-gram graph=BLOCKED(mamba_attn capture crash) → `--enforce-eager` applied to MEASURE acceptance @temp=1.0 (Recreate, draining). KV/backend + P2(tp4x2,tp1 prefix) pending. |
| 7 Readiness audit | not_started | |
| 8 Compound | not_started | |

## Results so far (FP8 agg-tp2-x4, conc=130, fin-support SLO p50<=6500/p90<=9500)

| Config | E2E p50 | E2E p90 | TTFT p50 | TPOT | SLO | Notes |
|--------|---------|---------|----------|------|-----|-------|
| FP8 mnbt=8192  | 5105 | 8878  | 454 | 55.8 | PASS | original baseline |
| FP8 mnbt=4096  | 7065 | 11933 | 684 | 83.0 | FAIL | prefill fragmentation |
| FP8 mnbt=16384 | 4685 | 8147  | 387 | 55.0 | PASS | **winner** |
| BF16 mnbt=8192 | 6128 | 10496 | 495 | 72.5 | p90 FAIL | precision comparison |

- **FP8 > BF16**: FP8 meets p90 at conc=130; BF16 does not. FP8 also faster floor (conc=8: FP8 1052/162 vs BF16 1451/215).
- **Prefix cache hit rate ~0 at TP>1** (upstream vLLM #26201 limitation, Mamba 'all' mode experimental). SLO still passes due to B200 prefill throughput + KV headroom.
- 0 errors at every concurrency level for both precisions.

## Image decision (recorded)
Using **vLLM 0.18.1** per spec hard gate. Spec AVOIDS 0.22/0.19.0/0.15.x/0.20.0.

## Blockers resolved this session
- vLLM 0.14.1 ECR image too old → skopeo-copied 0.18.1.
- NVMe RAID0 28TB built over nvme1n1..nvme8n1.
- FSx unmountable on B200 (Lustre NID parse) → direct HF download to NVMe.
- B200 pod DNS broken → hostNetwork+dnsPolicy:Default.
- Serving manifest probe delays 900/1200 reverted on re-apply (live-patch not mirrored to file) → fixed file + Recreate strategy.
- MTP spec-decode liveness OOM/SIGKILL loop (exit 137) → raised liveness initialDelay to 1500s for the longer MTP cold compile.
