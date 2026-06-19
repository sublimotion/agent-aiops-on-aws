# PrisKV Shared Prefix Cache — Live Node Handoff

**Spec:** `domains/ai-infra/specs/priskv-shared-prefix-cache.md`
**Started:** 2026-06-18

## Live instance (Tokyo)

| Field | Value |
|-------|-------|
| Instance ID | `i-030d90b609a2fc333` |
| Type | g7e.12xlarge (**2× RTX PRO 6000 Blackwell, sm_120, 96GB each**) |
| Region / AZ | ap-northeast-1 / ap-northeast-1c |
| Public IP | `35.78.107.205` |
| SSH | `ssh -i ~/.ssh/g7e-tokyo.pem ubuntu@35.78.107.205` |
| Spot price cap | $20/hr (1c market ~$13/hr) |
| AMI | `ami-0a1c350f68a847c64` (DLAMI Ubuntu 22.04, 2026-06-09) |

## Topology note — START AT 2 REPLICAS

Quota (64-vCPU G-instance limit, region-wide) caps us at g7e.12xl = **2 GPUs** right now, so
the run is **2× TP1** (2 replicas), not the spec's full 4× TP1. Quota bumps to 192 vCPU are
PENDING (OD `d5028a16de604f90919c9f22d9dd3df7EkCJ9u6r`, Spot `a0bc9f4c75e149e3a32eb95c3138ccc73PCCieId`).
When approved, relaunch on g7e.24xl for the full 4-replica matrix. 2 replicas is the **minimum
viable A/B/C test** and de-risks the PrisKV-on-sm_120 build (falsification gate #4) cheaply first.

## Pre-flight gates already PASSED (manual, before loop start)

- **Stage 4a GPU health**: both GPUs `volatile.total=0`, `remapped_rows.{pending,failure}=No`,
  `uncorrectable=0` → CLEAN. Driver 580.159.04, compute_cap 12.0 (sm_120).
- **Node prep**: DLAMI pre-mounts NVMe as LVM at `/opt/dlami/nvme` (3.3T free); symlinked to
  `/mnt/nvme`. **Do NOT reformat `/dev/nvme1n1`** (it's a busy LVM member — mkfs fails by design).
- **Container runtime**: Docker 29.5.3 with **nvidia runtime pre-wired and verified**
  (GPU-in-container confirmed). This DLAMI does NOT need the manual `systemctl start containerd`
  dance from the bare-metal g7e lessons — it's turnkey.

## Remaining gates the loop MUST run (per spec Pre-flight section)

2. Stage 4b — observability stack up BEFORE serving (Prometheus/DCGM/node-exporter), S3 sync.
3. Phase 0 — build `aibrix_kvcache` + PrisKV onto `vllm/vllm-openai:latest-cu130`; validate boot
   with `--load-format dummy` before pulling 32GB weights. **Falsification gate #4** (build must
   work on sm_120 in <1 day or stop).
4. Stage 0c — `validate-serving-config.py` fail-closed gate.
5. Harness smoke test — confirm `vllm_time_to_first_token_seconds_bucket` lands in Prometheus on
   ONE streaming request before the first paid cell (the kimi-k2.6-nvfp4 TTFT-loss gate).

## Model / arms (from spec)

- Model: `Qwen/Qwen3-32B-FP8` (TP1, max-model-len 24000, matching qwen3-32b-eks baseline).
- Arms: **A** local-cache + round-robin · **B** local-cache + prefix-aware routing · **C** PrisKV
  shared host-DRAM cache + round-robin. The real question is **B vs C**.
- Sweep shared-prefix ratio 0/40/70%; lean on large-prefix cells (small prefix already warms to
  ~160ms in baseline — too small to see cross-replica effect).
- Correctness gate every cell: compare output token IDs cold vs warm (PrisKV cache races #41/#42).

## Teardown

`aws ec2 terminate-instances --region ap-northeast-1 --instance-ids i-030d90b609a2fc333`
Spot one-time → no auto-restart. NVMe wiped on terminate; push results to S3 before teardown.
