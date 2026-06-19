# dynamo-snapshot — p5e Gemma-4 anchor cell deployment log

Date: 2026-05-30 (post-bridging cell)
Goal: Run anchor cell on p5e (H200 sm_90) for `google/gemma-4-26B-A4B-it` to compare
restored TTFT vs Modal's published 22 s AOT-compile-cache HIT floor.

## Stage 0 — pre-launch reconnaissance (UTC ~19:25)

- Operator IP: `108.26.230.24` (already in SG `sg-06821adfa7f05916f`).
- Keypair: `~/.ssh/dynamo-snapshot-uw2.pem` (mode 600), AWS-side key `dynamo-snapshot-uw2` present.
- AMI `ami-041914c9e9b61b15e` — present, `available`, tagged `criu-pr=3021`,
  `dynamo-sha=39251bcf`, `role=bridging-base`, `keep=true`.
- HF `google/gemma-4-26B-A4B-it`: public, ungated, sha
  `6e6f6edea8c52db2094dca3086e4b963a0034dfc`. Drafter
  `google/gemma-4-26B-A4B-it-assistant` will be checked once on host.
- Capacity blocks: `describe-capacity-block-offerings` returned 0 offerings for
  p5e.48xlarge × 24 h in us-west-2 → no short-notice CB available.
- p5e.48xlarge AZ availability: only **us-west-2c** offers the type at all.
- Spot price (us-west-2c, last 24 h): **$15.05 → $16.89/hr** rising.
- On-demand listed at ~$40.77/hr — directive said "fall back to on-demand for
  ~2 hr (~$20)" which is off by ~4× (it's $80/2hr). With a $50 cumulative cap
  and current spend ~$1.84, **on-demand violates the halt condition**.

### Decision: spot p5e.48xlarge in us-west-2c

- Spot at $16.89/hr × 2 hr ≈ **$33.78**, cumulative ≈ $35.62 — under $50 cap.
- AMI-bake recovery proven on Ada (bridging cell), 30 s spot fulfilment + ~4 min
  re-stage cycle. For Gemma-4 26 GiB weights, re-staging from HF would be
  ~3-5 min; if reclaim happens we still fit one full retry under cap.
- Subnet: `subnet-0b38824e7811cc7ca` (us-west-2c, public, vpc-0a705e8b01d91a9f8).
- Security group: `sg-06821adfa7f05916f` (operator IP intact).
- IAM profile: `dynamo-snapshot-uw2-profile`.

### Risk register for this cell

| Risk | Mitigation |
|---|---|
| AMI baked on Ada (sm_89), driver may not enumerate H200 (sm_90) | If `nvidia-smi` fails to show H200 → fall back to fresh NVIDIA AL2023 GPU AMI + re-bootstrap (~25 min, ~$7 on-demand build host). |
| Spot reclaim mid-gate | AMI-bake pattern; second spot launch from same AMI in <2 min. |
| Gemma-4 needs >141 GiB host RAM during cuda-checkpoint | p5e.48xlarge has ~2 TiB RAM — trivially covered. |
| `gemma-4-26B-A4B-it-assistant` (drafter) gated/missing | Skip MTP if missing, run baseline single-model first. |
| Gate 2 fail despite >3 GiB weights — root-cause as activation memory anomaly | Document, do NOT extrapolate to g7e cell. |
| Restored TTFT > 22 s | Still useful data; do not halt; document margin. |


## Stage 1 — instance launch + verification (UTC ~20:38)

- Spot request: SIR `sir-3xdqksgg`, fulfilled in 30 s at ceiling $20/hr. Actual fill price ~$16.89/hr.
- Instance: `i-075d128c2e99849c5`, public IP 35.89.25.92, AZ us-west-2c.
- SSH up in 50 s (port 22 reachable from operator IP).
- AMI Hopper compatibility: **PASSED**. `nvidia-smi` enumerated all 8× H200 (143771 MiB each, sm_90, driver 595.71.05).
- CRIU 4.2 GitID `2da963e` confirmed (= `criu-dev` HEAD + nv parallel-memfd PR #3021 merged at AMI bake).
- cuda-checkpoint 595.71.05 functional. cuda_plugin.so present. seccomp-wrap present.
- Mounted /dev/nvme1n1 (3.5 TiB local NVMe) at /mnt/nvme.
- Symlinked `/usr/lib/criu/cuda_plugin.so → /usr/local/lib/criu/cuda_plugin.so` (per bridging-cell lesson).

## Stage 2 — model + venv staging (UTC 20:40 → 20:48)

- Built Python 3.12 venv at `/mnt/nvme/venv` (system python3.9 too old for vLLM 0.21.0 which requires >=3.10).
- Installed vLLM 0.21.0 + hf_transfer + transformers 5.9.0 + torch 2.11.0 + flashinfer 0.6.8.
- Downloaded `google/gemma-4-26B-A4B-it` rev `6e6f6edea8c52db2094dca3086e4b963a0034dfc`: 49.9 GiB shard 1 + 1.7 GiB shard 2 = 51.6 GiB on disk (Modal log: 48.07 GiB on 9P; we measured slightly larger because we count the `blobs/` content fully, not just safetensors).
- Downloaded `google/gemma-4-26B-A4B-it-assistant`: ~830 MiB.

## Stage 3 — gate run attempts (UTC 20:52 → 21:12)

Four attempts, all failing at the same `criu dump` step:

| Attempt | Variant | Result |
|---|---|---|
| orch-A  | default vLLM 0.21 settings, `tensor_parallel_size=1` | dump fail: `Can't dump file 9 of that type [20666] (chr 195:255)` (= /dev/nvidiactl) |
| orch-A2 | + `CUDA_VISIBLE_DEVICES=0` on the harness | same fail |
| orch-A3 | (re-attempt to stabilize) | same fail |
| orch-A4 | + `--external dev[195/255]:nvidiactl --external dev[195/0]:nvidia0 --external dev[195/254]:nvidia-uvm` | same fail (the `dev[m/n]:NAME` syntax doesn't apply to character-device file fds) |

**Cold-baseline data captured** (`pre.json`):
- Model load (offline `vllm.LLM(... enforce_eager=True, dtype=bf16)`): **45.14 s**
- Warmup gen (single deterministic prompt → 4 tokens, eager): **0.36 s**
- vLLM `sleep(level=1)` succeeded (CuMemAllocator: freed 119.88 GiB, 5.18 GiB still in use, 48.59 GiB backed up to CPU; "It took 44.86 seconds to fall asleep.").
- cuda_plugin pause/checkpoint stages succeeded per dump.log; failure happens after, at fd-walking stage.

## Halt condition triggered

Per spec halt-conditions:
> Gate 1 fails on Hopper → halt, document (this would be a Hopper-specific cuda-checkpoint bug)

The actual failure is **NOT a Hopper-specific cuda-checkpoint bug**. cuda-checkpoint and cuda_plugin's CHECKPOINT_DEVICES hook ran successfully on Hopper. The blocker is **CRIU's lack of `HANDLE_DEVICE_FD` plugin support** combined with the **NVIDIA driver opening device files for ALL GPUs in a multi-GPU node** even with `CUDA_VISIBLE_DEVICES=0`. The bridging cell on Ada g6.xlarge (1 GPU only) didn't expose this because there were only 3 device-file fds total (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and they happened to be VMA-backed by Qwen3-0.6B's tensor allocations — the cuda_plugin's `HANDLE_DEVICE_VMA` hook claimed them.

**Implication**: this is a structural gap in the open-stack (criu-dev + cuda_plugin) for multi-GPU instances on bare EC2, NOT a Hopper hardware incompatibility and NOT a Gemma-4-specific issue. Replicates with a 5-line `torch.cuda.init() + randn` test on the same instance.

## Anchor comparison (partial — restore not measured)

| Phase | Measured this cell | Modal H200 reference |
|---|---|---|
| Container/process create | n/a (bare EC2) | ~14 min Modal-side (proprietary L0/L1/L2) |
| APIServer init | n/a (offline LLM, no server) | 32 s |
| Engine bootstrap | folded into LLM constructor | 9 s |
| Model weight load (FP16/BF16) | **45.14 s** total constructor incl bootstrap (NVMe local) | 28 s (9P, ~1.94 GB/s) |
| Compile+warmup (no AOT cache, eager mode) | n/a — eager bypasses torch.compile | (Modal AOT cache HIT: ~22 s) |
| First-token (warmup gen) | **0.36 s** (eager) | 5.34 s (post-compile profile/warmup) |
| **Restore-to-first-token (anchor goal)** | **NOT MEASURED — dump failed** | n/a — Modal does not snapshot/restore |

**Cold baseline (load + warmup) on H200 single-GPU eager mode = ~45.5 s**. Modal's 22 s reference is for a *compiled* Gemma-4 worker reaching steady state from an AOT cache HIT — different code path. To produce an apples-to-apples anchor, we need: (a) snapshot a compiled+warmed worker, (b) measure restore. Both are blocked on the CRIU device-fd issue above.

## Cleanup

- Instance `i-075d128c2e99849c5` terminated 21:18 UTC (~40 min uptime).
- Spot bill: 40/60 hr × $16.89/hr ≈ **$11.26**.
- AMI `ami-041914c9e9b61b15e` retained (`keep=true`) — still useful for the containerized retry.
- Subnet/SG/keypair/IAM intact.

## Spend update

| Item | $ |
|---|---|
| Carry-over (bridging+earlier) | 1.84 |
| p5e spot 40 min × $16.89/hr | 11.26 |
| **Total this session** | **~$13.10** |
| Cumulative cap | $50 |

Well under cap.

## Go/no-go decisions

- **Go for re-running the anchor cell in containerized mode**: vLLM in an `nvidia-container-runtime` container with `--gpus '"device=0"'`. The container's mount namespace will only have /dev/nvidia0+nvidiactl+nvidia-uvm, eliminating the bare-fd deluge. Reuse this AMI; bake the container image first on a build host.
- **No-go for direct g7e Blackwell PCIe extrapolation**: the multi-GPU device-file issue would also apply to g7e.24xlarge (4× RTX PRO 6000) on bare EC2. The g7e cell needs the same containerized-runtime fix before booking. **The riskiest cell remains gated on this fix**, not on Blackwell-specific bugs.
- **Halt condition met**: documented as a CRIU/cuda_plugin gap, not a Hopper hardware bug.
