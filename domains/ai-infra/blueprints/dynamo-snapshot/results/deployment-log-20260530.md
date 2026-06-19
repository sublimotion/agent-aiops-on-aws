# Deployment log — 2026-05-30

Stage 0 g5/g6 correctness smoke for `dynamo-snapshot-coldstart` spec.

## Timeline (UTC)

| Time | Event |
|---|---|
| 10:00 | Resumed Stage 0 with answered directives (CRIU build-from-source, new keypair, NVIDIA AL2023 AMI). |
| 10:05 | Fetched and parsed `https://docs.nvidia.com/dynamo/kubernetes-deployment/deployment-guide/snapshot`. Confirmed: no published prebuilt image, driver 580.xx required, vLLM-only preview, x86_64-only, privileged DaemonSet. |
| 10:10 | Sparse-cloned `github.com/ai-dynamo/dynamo` at SHA `39251bcf13f854b305331101014d2a3980a2aab6`, vendored `deploy/snapshot/` into `upstream-snapshot/`. |
| 10:15 | AWS bootstrap: keypair `dynamo-snapshot-uw2` → `~/.ssh/dynamo-snapshot-uw2.pem`. SG `sg-06821adfa7f05916f` (SSH from 108.26.230.24/32). IAM profile `dynamo-snapshot-uw2-profile` with S3RO + ECRRO + CWAgent. AMI `ami-0b8143be52d61bd07` (base NVIDIA AL2023 GPU, 2026-05-29). |
| 10:18 | Spot price check across us-west-2 AZs: g5.xlarge 2c $0.5624/hr (cheapest), 2a $0.6500, 2b $0.6928. |
| 10:25 | g5.xlarge spot launch: **InsufficientInstanceCapacity in ALL three AZs simultaneously**. Pivoted to g6.xlarge (L4 Ada sm_89) at $0.32/hr in us-west-2b — also covered by spec's cross-family expectation as the secondary smoke target. |
| 10:30 | g6.xlarge i-03716ae7f2ec40789 running on us-west-2b at 52.33.128.138. Driver `595.71.05`, CUDA 13.2 toolkit, 23 GiB L4, 232 GiB ephemeral NVMe at /mnt/nvme, 200 GiB EBS root. **Hard gate PASS: driver ≥ 580.** |
| 10:35 | Bootstrap script: dnf-installed CRIU build deps, cloned criu-dev, `make` → CRIU 4.2 (GitID `4d76d1a`). `make install-lib` failed on python3.9 wheel issue — worked around by installing only `install-criu install-cuda_plugin`. cuda-checkpoint binary 595.71.05 installed. |
| 10:50 | vLLM 0.22.0 installed in py3.11 venv. Model `Qwen/Qwen3-0.6B` confirmed available + ungated on HuggingFace. |
| 11:25 | First smoke attempt (vLLM offline mode, gpu_memory_utilization=0.85): `cuda-checkpoint --action checkpoint` → "out of memory". Root cause: cuda-checkpoint stages GPU memory through host RAM; vLLM held 19.7 GiB on a 15 GiB-RAM box. |
| 11:30 | Reduced gpu_memory_utilization to 0.25; second attempt: `cuda-checkpoint` succeeded but `criu dump` failed: `Unknown shit 600 (anon_inode:[io_uring])`. uvloop (libuv io_uring backend) is in vLLM's tree; CRIU 4.2 from criu-dev HEAD does not implement io_uring C/R. |
| 11:35 | Pivoted to a pure-pytorch smoke (`smoke-pytorch.py`) — same model, transformers `.generate()`, no asyncio/uvloop. **Full sequence succeeded.** |
| 11:39 | Stage 0 g6 result captured. Terminated instance. |

## CRIU image source decision

**Decision: build from source on the GPU host itself.** Rationale:
- The NVIDIA docs page does not advertise a prebuilt `nvcr.io` image for
  `snapshot-agent`. Published guidance is a `make docker-build-agent` flow.
- Building inside the GPU host avoided a separate CPU builder and saved time.
- Pin: `github.com/ai-dynamo/dynamo@39251bcf13f854b305331101014d2a3980a2aab6`,
  CRIU defaulted to upstream `criu-dev` HEAD (`4d76d1a`) at build time.

## Per-gate verdict (g6.xlarge L4 Ada sm_89)

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| 1. Token-ID equality | byte-identical SHA256 | `c09af54...8bb3f` pre & post | **PASS** |
| 2. Artifact ≤ 2× weights | ≤ 2.00× | 3.29 GiB / 1.41 GiB = 2.33× | **MARGINAL FAIL** (pytorch baseline; vLLM `sleep()` path expected to lower this) |
| 3. Restore < 30s on NVMe | < 30 s | 1.69 s for 3.29 GiB | **PASS** |

Plus: dump took 7.78 s (dominated by `cuda-checkpoint --action checkpoint`
GPU→host copy).

## Cost

| Item | $ |
|---|---|
| g6.xlarge spot, us-west-2b, ~1.0 hr live × $0.32/hr | 0.32 |
| 200 GiB gp3 root EBS, ~1.0 hr × $0.022/hr | 0.022 |
| Data transfer (HF model download Qwen3-0.6B 1.4 GiB) | ~0.00 (free in to AWS) |
| **Total spent** | **≈ $0.35** |

Well under the $10 target and far under the $20 cap.

## Recommendation on g7e/p5e matrix

**Conditional proceed with caveats.** The cuda-checkpoint primitive itself is
proven correct on g6 (Ada sm_89) — token IDs are byte-identical across
checkpoint+restore. Restore latency (1.69 s for 3.29 GiB on NVMe) extrapolates
favorably to the spec's ≤10 s claim for ~6 GiB Ministral-3B-class artifacts.

**Open issues that must be resolved before booking g7e or p5e capacity:**

1. **io_uring fd in vLLM/uvloop process tree**: CRIU 4.2 (criu-dev HEAD)
   cannot dump uvloop-using processes. The full Dynamo flow assumes the
   `snapshot-agent` orchestrator handles this — we did not verify that
   end-to-end. Two paths:
   - (a) Build NVIDIA's downstream CRIU fork (the Dockerfile build-arg
     `CRIU_REPO`/`CRIU_REF` allows this; the upstream blog mentions
     "AIO + parallel-memfd patches pending merge"). Test whether this fork
     also adds io_uring support.
   - (b) Configure vLLM to disable uvloop (run `python -X uvloop_disabled` or
     pin a non-uvloop async runtime in the placeholder image).

2. **Host RAM constraint**: cuda-checkpoint stages full GPU memory through
   host RAM. For larger models the matrix host instances must have
   `host_RAM ≥ 1.2 × peak_GPU_alloc`. g7e.24xlarge has 384 GiB RAM and 4×
   RTX PRO 6000 (96 GiB each) → covered for single-GPU cells. p5e.48xlarge
   has 2 TiB RAM → trivially covered. **No issue for the matrix as
   currently planned.**

3. **Gate 2 marginal fail on pytorch**: re-run with vLLM `--enable-sleep-mode`
   invoked *before* checkpoint to release activation/KV-cache memory; the
   resulting artifact should drop below the 2× threshold. This is the
   spec's intended path; the pytorch smoke was a workaround for the
   io_uring blocker, not the production target.

**Suggested next step (not done — halted per directive):** before booking
g7e/p5e capacity, write a short follow-up cell on g6 that builds the placeholder
image from `upstream-snapshot/Dockerfile` against a vLLM-runtime base, runs
the full `snapshot-agent` orchestration locally, and re-checks all three gates
with vLLM `sleep()` engaged. That's a ~$2, ~4-hr piece of work that de-risks
the entire matrix.

## Artifacts

- `results/g6-stage0/pre.json` — pre-checkpoint completion + sha256
- `results/g6-stage0/post.json` — post-restore completion + sha256
- `results/g6-stage0/smoke.log` — full smoke harness stdout
- `results/g6-stage0/criu-dump-tail.log` — last 200 lines of `criu dump -v3`
- `results/g6-stage0/criu-restore-tail.log` — last 200 lines of `criu restore -v3`
- `upstream-snapshot/` — vendored Dynamo `deploy/snapshot/` at
  pinned SHA `39251bcf13f854b305331101014d2a3980a2aab6`

## Cleanup state

- Spot instance `i-03716ae7f2ec40789` terminated (shutting-down at 11:42 UTC).
- Keypair `dynamo-snapshot-uw2` retained; private key at
  `~/.ssh/dynamo-snapshot-uw2.pem` (mode 600).
- IAM role `dynamo-snapshot-uw2-role` and instance profile
  `dynamo-snapshot-uw2-profile` retained for the next stage.
- Security group `sg-06821adfa7f05916f` retained (will need IP refresh on
  next session).
- No FSx, ECR, S3 buckets created — all build artifacts on the (now
  terminated) instance.

## Bridging cell resume attempt (16:30 UTC)

- Operator IP refresh skipped — instance no longer exists.
- `aws ec2 describe-instances i-06f07e11372c73da1`: empty Reservations.
- `aws ec2 describe-spot-instance-requests sir-6rrfjy3h`:
  - State: `closed`
  - Status code: `instance-terminated-no-capacity`
  - UpdateTime: 2026-05-30T13:09:11Z
  - Launched at 11:44:11Z → reclaim at 13:09:11Z (85 min uptime)
- Cancelled spot request explicitly (was already closed; `aws ec2
  cancel-spot-instance-requests` returned state=`cancelled`).
- All on-host bridging-cell work (CRIU build + PR #3021 merge, vendored
  Dynamo, NVIDIA driver setup) lost with the instance.
- No new spot launched — re-running the build from scratch on a fresh
  spot exposes the same reclaim risk. Halt and document.
- Lessons appended: g6 spot reclaim pattern + AMI-bake remediation;
  bridging cell status logged as inconclusive (gates with vLLM sleep()
  never ran).

## Halt summary

| Gate | Status with vLLM `sleep()` engaged |
|---|---|
| 1 — token-id byte equality | not measured (instance gone before run) |
| 2 — artifact ≤ 2× weights | not measured (the gate this cell exists to clear) |
| 3 — restore < 30 s on local NVMe | not measured |

Stage 0 baseline (pytorch, no vLLM sleep) results from earlier in the day
remain valid: gates 1+3 PASS, gate 2 marginal at 2.33×.

**Spend so far**: ~$0.50 (Stage 0 g6 ~$0.32 + EBS + bridging cell startup
before reclaim). Well under $10 cap.

**Go/no-go for p5e (Hopper) — primary anchor for Gemma-4-26B-A4B-it**:
**NO-GO** until the bridging cell completes a clean Gate 2 PASS with
vLLM `--enable-sleep-mode` + `sleep()` engaged. p5e capacity blocks are
expensive and time-boxed; opening one with the gate-2 question still
unanswered would risk burning a window on a primitive that may still
emit > 2× artifacts. Re-run the bridging cell first, this time baking an
AMI after the CRIU build phase to survive spot reclaim.

