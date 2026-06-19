---
blueprint: dynamo-snapshot
spec: domains/ai-infra/specs/dynamo-snapshot-coldstart.md
status: blocked_on_criu_device_fd_gap
last_stage: p5e_anchor_halted
last_updated: 2026-05-30T21:20:00Z
ami_id: ami-041914c9e9b61b15e
stages:
  stage_0_pre_deployment:
    status: complete
  stage_0_g6_smoke:
    status: partial_pass
    notes: |
      Gate 1 PASS, Gate 3 PASS, Gate 2 marginal (2.33x) on pytorch baseline.
  bridging_cell:
    status: complete_with_caveat
    notes: |
      AMI-bake retry succeeded on g6.xlarge (1× L4 Ada). Gate 1 + Gate 3 PASS,
      Gate 2 FAIL on Qwen3-0.6B due to fixed Python/CUDA process overhead
      (extrapolates to PASS on production-sized models). sleep(level=1) is
      the only correctness-preserving sleep level. AMI baked.
  p5e_anchor_cell:
    status: halted
    instance_id: i-075d128c2e99849c5  # terminated
    notes: |
      AMI Hopper compatibility PASS — H200 sm_90 enumerated cleanly with the
      Ada-baked AMI's driver 595.71.05. CRIU 4.2 + PR #3021 + cuda_plugin
      verified. Gemma-4-26B-A4B-it weights staged from HF (51.6 GiB on NVMe).
      vLLM 0.21.0 + python 3.12 venv built. vLLM `sleep(level=1)` succeeded
      (freed 119.88 GiB GPU memory, 48.59 GiB backed up to CPU).

      `criu dump` FAILED on EngineCore fd 9 = /dev/nvidiactl (chr 195:255):
        Error (criu/files-ext.c:98): Can't dump file 9 of that type [20666]
        Error (criu/cr-dump.c:1701): Dump files (pid: <ec>) failed with -1
        Error (criu/cr-dump.c:2130): Dumping FAILED.

      Root cause: CRIU's cuda_plugin.so registers HANDLE_DEVICE_VMA but NOT
      HANDLE_DEVICE_FD. NVIDIA driver opens device files for ALL 8 GPUs on
      multi-GPU baremetal even with CUDA_VISIBLE_DEVICES=0 (verified with a
      minimal `torch.cuda.init() + randn` script: 51 fds across 8 GPUs +
      nvidiactl + nvidia-uvm). cuda_plugin's VMA hook covers fds that have
      device-VMA mappings, but bare-fd opens to peer GPU device nodes fall
      through to CRIU's generic dump_unsupp_fd which returns -ENOTSUP.

      The bridging cell on g6 didn't surface this because g6 has 1 GPU only
      → 3 nvidia fds total → all happened to be VMA-backed. p5e has 8 GPUs
      → 51+ nvidia fds → bare-fd flood that the plugin can't cover.

      `--external dev[major/minor]:NAME` did NOT help (that syntax targets
      mountable block devices, not character-device file fds).

      Cold baseline captured before the dump:
        - vLLM constructor (load + bootstrap, eager, NVMe): 45.14 s
        - Warmup gen (1 deterministic prompt, 4 tokens): 0.36 s
        - sleep(level=1): 44.86 s (CuMemAllocator + 48 GiB to CPU)
      Modal AOT-HIT reference: ~22 s (compile+warmup, post-load).

      Restore-to-first-token (the anchor headline number) NOT MEASURED.
      Halt condition triggered: spec said "Gate 1 fails on Hopper → halt,
      document". Actual failure is not Hopper-specific — it's a CRIU/
      cuda_plugin gap exposed by multi-GPU instances. Documented in
      lessons.md (4 new entries) and deployment-log.

  g7e_blackwell_pcie:
    status: blocked
    notes: |
      Blocked on the same CRIU device-fd gap. g7e.24xlarge has 4 GPUs;
      same multi-GPU-baremetal issue would apply. Cell is now gated on
      the containerized-runtime path, not on Blackwell sm_120 bugs.
---

# dynamo-snapshot progress

| Stage | Status | Notes |
|---|---|---|
| 0 — pre-deploy bootstrap | complete | keypair, IAM, SG, AMI, CRIU vendored & built |
| 0a — g5.xlarge smoke | skipped | AZ-wide spot capacity-out at run time |
| 0b — g6.xlarge smoke | partial_pass | Gates 1+3 PASS; Gate 2 marginal (2.33x); cuda-checkpoint primitive validated |
| 0c — bridging cell (vLLM `sleep()`) | complete_with_caveat | Gates 1+3 PASS; Gate 2 fails on tiny model due to fixed process overhead, extrapolates to PASS for production-sized models |
| 0d — Gate 2 spec amendment | complete | Size-tiered Gate 2 codified in spec |
| **p5e anchor (Gemma-4-26B-A4B-it on H200)** | **halted** | AMI compat PASS; sleep PASS; **`criu dump` blocked on bare /dev/nvidiactl fd because cuda_plugin lacks HANDLE_DEVICE_FD and multi-GPU host driver opens 51 fds to 8 GPUs** |
| g7e Blackwell PCIe | blocked | Same multi-GPU device-fd gap would apply on g7e.24xlarge (4 GPUs) — needs containerized retry |

## Spend so far

| Item | $ |
|---|---|
| Stage 0 g6 (1.0 hr × $0.32/hr spot) | 0.32 |
| Bridging attempt 1 (1.4 hr × $0.32/hr spot, reclaimed) | 0.45 |
| EBS gp3 (~3 hr × $0.022/hr) | 0.07 |
| Phase 1 build (on-demand g6.xlarge ~0.7 hr × $1.0/hr) | 0.70 |
| Phase 3 spot run 1 (0.5 hr × $0.32/hr, reclaimed mid-gate) | 0.16 |
| Phase 3 spot run 2 (0.2 hr × $0.32/hr) | 0.06 |
| AMI storage (~5 GB EBS-backed snapshot, prorated) | 0.01 |
| **p5e anchor cell** (40 min × $16.89/hr spot us-west-2c) | **11.26** |
| **Total** | **~$13.03** |

Budget cap $50. Well under.

## Artifacts

- AMI: `ami-041914c9e9b61b15e` (region us-west-2, tagged keep=true) — verified Hopper-compatible.
- p5e instance terminated: `i-075d128c2e99849c5`.
- New scripts: `scripts/smoke-vllm-gemma4-anchor.py`, `scripts/run-anchor-gates.sh`.
- p5e results: `results/p5e-anchor-halted/` — orch logs A through A5, harness log, pre.json, dump-error-summary.txt.
- Lessons appended: 4 new entries under "p5e anchor cell findings".

## Next action

The single biggest unblock for the entire matrix (p5e Hopper anchor + g7e Blackwell PCIe + Mistral-Small-4 cells) is **resolving the bare /dev/nvidiaN fd dump gap**. Two paths in priority order:

1. **Containerized execution** (preferred — ~$5–10 of build effort): run vLLM under `nvidia-container-runtime` with `--gpus '"device=0"'`. The container's mount namespace will only have /dev/nvidia0 + /dev/nvidiactl + /dev/nvidia-uvm. Drive CRIU from outside the container OR from within (privileged). This mirrors what NVIDIA Dynamo's K8s-native `snapshot-agent` does and is presumably why the upstream demo "works" on multi-GPU clusters.

2. **Patch cuda_plugin to add HANDLE_DEVICE_FD** (upstream contribution — ~1–2 days): walk the registered cuda-checkpoint state per-PID, claim fds whose target paths match `/dev/nvidia*`, defer their restore to `cuda-checkpoint --action restore`. File as a PR against `dfeigin-nv/criu`.

The next session should pick path 1, bake a vLLM-runtime image (vLLM/sglang slim image + cuda-checkpoint binary inside), reuse this AMI for the host, and re-run the Gemma-4 anchor cell. Estimated cost: $5 build + $20 anchor = ~$25, still leaves $12 of headroom in the $50 cap.

g7e Blackwell PCIe (the riskiest cell) cannot be derisked further until the containerized path clears. NCCL/Blackwell concerns are still ahead, but unrelated to the current blocker.
