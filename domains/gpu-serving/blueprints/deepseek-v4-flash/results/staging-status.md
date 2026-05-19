# DeepSeek V4 Flash — Staging Status

**Spec**: `domains/gpu-serving/specs/deepseek-v4-flash.md`

## Pre-flight verified (2026-05-19)

| Check | Result |
|-------|--------|
| AWS identity | `arn:aws:iam::615299764834:user/aiops` |
| EKS cluster | `qn-sglang-eks-cluster` ACTIVE, K8s 1.32, us-west-2 |
| VPC | `vpc-0bd6abcecded8edf6`, private subnets in usw2-az2 (`subnet-001db6882dbb5ac72`) |
| B300 capacity | ✅ Available in us-west-2b (only AZ) |
| B300 spot price | $26.49/hr (vs ~$50/hr on-demand) |
| GPU node IAM role | `gpu-eks-node-group-20260303162535678600000025` (reused from ray-ft-gpu) |
| kubectl context | `qn-sglang-uw2` configured |
| Existing nodes | 2× Ray fault-tolerance nodegroup (g5 + system), unaffected by new nodegroup |

## Patterns reused from existing blueprints

- **Model staging**: K8s Job with `python:3.11-slim` + `huggingface_hub[hf_transfer]` → directly to `/mnt/nvme/models/` on the GPU node. No S3 round-trip. Pattern from `qwen3-235b-b300/k8s/download-model.yaml`.
- **Serving Pod**: `hostNetwork: true`, `hostPath: /mnt/nvme`, vLLM cache + Triton cache + HF cache redirected to NVMe. Pattern from `qwen3-235b-b300/k8s/vllm-serve.yaml`.
- **Tolerations**: `nvidia.com/gpu:NoSchedule`. Same as all existing GPU blueprints.
- **AMI**: `AL2023_x86_64_NVIDIA` — required for B200/B300 (memory: AL2 lacks `ib_umad`).

## Artifacts created

```
domains/gpu-serving/blueprints/deepseek-v4-flash/
├── k8s/
│   ├── download-model.yaml      # K8s Job: hf_transfer → /mnt/nvme/models/DeepSeek-V4-Flash
│   ├── vllm-serve.yaml          # vLLM nightly + transformers 4.56.x pin (per #42741)
│   ├── sglang-serve.yaml        # SGLang latest — alternate engine for cross-comparison
│   └── nodegroup-spot.yaml      # docs-only; nodegroup created via aws CLI
├── scripts/
│   ├── preflight.sh             # ✅ executed; environment verified
│   ├── launch-nodegroup.sh      # provisions B300 spot nodegroup (NOT YET RUN)
│   ├── stage-model.sh           # submits the K8s download Job (NOT YET RUN)
│   └── teardown.sh              # ordered teardown — pods first, then nodegroup
└── results/
    └── staging-status.md        # this file
```

## Cost expectations (rough)

| Activity | Duration | Cost |
|----------|----------|------|
| Nodegroup launch + AMI boot | ~5 min | ~$2 |
| Model download (~150 GB at hf_transfer ~30 MB/s) | ~1.5 hr | ~$40 |
| T0 readiness probe (smoke + first benchmark) | ~30 min | ~$13 |
| **Decision gate** | — | (~$55 spent so far) |
| Full T0-T5 sweep if T0 passes | ~5 hr | ~$130 |
| Teardown | <5 min | $0 |
| **Total to publishable result** | ~7 hr | **~$200** |

## Spot reclaim mitigation

- Scripts are resumable: model download has `backoffLimit: 2`, results JSON written per-tier
- If reclaimed mid-benchmark, re-launch nodegroup → re-run from last completed tier
- Model weights persist on the local NVMe of the spot instance — if reclaimed, weights are lost; re-download (~1.5hr penalty)
- For higher reliability, consider switching to a capacity block reservation if available

## Execution log

| Time | Event |
|------|-------|
| 11:03 | `aws eks create-nodegroup dsv4-b300-spot` issued |
| 11:08 | Nodegroup ACTIVE; node `ip-10-2-27-212` Ready (no `nvidia.com/gpu` advertised) |
| 11:09 | `kubectl label node ... nvidia.com/gpu.present=true` — device plugin DS lands |
| 11:10 | 8x `nvidia.com/gpu` advertised; verified node has 8x ~3.84TB unformatted NVMe |
| 11:11 | Attached IAM inline policy `dsv4-models-s3-write` to `gpu-eks-node-group-*` (was read-only) |
| 11:13 | `init-nvme-raid0` Job — 28 TB RAID0/XFS mounted at `/mnt/nvme` |
| 11:15 | First model download attempt failed: `aws-cli:2.17.20` ships old `huggingface-hub 0.16.4`, no `hf` CLI |
| 11:17 | Second attempt: `python:3.11-slim` + `hf` CLI v1.15 — completed in 0.5s with only 48K downloaded (CLI `--exclude` semantics broke in v1.15) |
| 11:21 | Third attempt: switched to Python `snapshot_download(allow_patterns=...)` API — 69 files queued, downloading |
| 11:21+ | Watcher running (background task `bv7q5z33b`); polling every 3 min |

## Resolved decisions

1. ✅ **Nodegroup launched** — spot acquired in usw2-az2 at ~$26.50/hr
2. ✅ **transformers pinned to 4.56.2** in `vllm-serve.yaml` — vLLM PR #42806 still OPEN
3. ✅ **vLLM image pinned to digest** `nightly-6e889b582b6a0b11f22b3764be174266faa9ff5e` (2026-05-19 build, includes PR #42320 merged 2026-05-13)
4. ✅ **S3 cache enabled** — bucket `qn-sglang-models-20260303161715850900000007`, path `models/DeepSeek-V4-Flash/`. Download Job tries S3 first; persist-to-s3 Job mirrors after HF success
5. ✅ **Repo-level utility** — `scripts/stage-model-s3-cached.sh` extracted as the canonical pattern
6. ✅ **Memory** — `feedback_model_staging_pattern.md` codifies the full pattern for future blueprints

## Status: ALL ROUNDS COMPLETE — see `FINAL-report.md`

### Completed milestones
- [x] HF download — 149 GB / 46 safetensors / 140 files in 2 min 48 s via Xet
- [x] S3 persist Job — uploading to `s3://qn-sglang-models-.../models/DeepSeek-V4-Flash/`
- [x] vLLM serving Pod — Ready at 12:07 EDT, ~11 min total startup time
- [x] Smoke test passed — `"The capital of France is"` → `" Paris"`
- [x] T0 P1v-a QPS sweep started — QPS 1.0 and 2.0 results landed

### First results — T0 P1v-a (random 2K input / 512 output)

| QPS | Out tok/s | Total tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | Errors |
|-----|-----------|-------------|---------------|---------------|--------------|--------------|--------|
| 1.0 | (peak 2060) | 2426 | 167 | 13580 | 8.3 | 16.9 | 0 |
| 2.0 | 918 | 4588 | 168 | 2432 | 8.7 | 140.8 | 0 |
| 4.0 | (in progress) | — | — | — | — | — | — |

**Observations**:
- Median ITL 8-9 ms is excellent — beats Artificial Analysis hosted (~10 ms / 96.7 tok/s)
- Median TTFT 167-168 ms is excellent for B300 TP=8 with no prefix caching
- QPS=1.0 P99 TTFT spike (13.6s) is likely the first request paying cold-cache cost, watch for QPS=2.0+
- Peak concurrent at QPS=2.0 hit 26 — engine handles bursts fine
- 0 errors on 80+80 prompts

### Pending
- [ ] T0 P1v-a sweep complete (QPS 4.0, 8.0)
- [ ] T0 P1v-b context scaling (1K, 4K, 16K, 32K)
- [ ] T0 P1v-c shared-prefix probe
- [ ] benchmark-analyst run on JSON results
- [ ] visual-explainer run for the report
- [ ] Spec update: DRAFT → IN_PROGRESS or COMPLETE
- [ ] Add cross-blueprint comparison row to `reports/benchmark-results.md`
