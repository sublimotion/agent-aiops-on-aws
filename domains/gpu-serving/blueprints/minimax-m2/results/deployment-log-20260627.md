# MiniMax-M2 Deployment Log — 2026-06-27

Spec: `domains/gpu-serving/specs/minimax-m2.md`
Cluster: `qwen3-next-bench-eks-cluster` (us-east-2)
Node: `ip-10-0-27-134.us-east-2.compute.internal` (p6-b200.48xlarge SPOT, AL2023 NVIDIA AMI, 8 GPU, labeled blueprint=minimax-m2)
Target: MiniMax-M2 FP8, vLLM, TP4 + EP4, 4× B200 (SM100 NVSwitch).
NOTE: BILLABLE spot B200 (~$18/hr) — clock running.

---

## Stage 0 — Pre-deployment gate

### 0b — Card lookup
- `mdc get minimax-m2 --engine vllm`: loaded. Two cards (M2.5 distinct; M2/M2.7 shared arch). Recipe: TP4+EP4, image `vllm/vllm-openai:minimax27`, compilation mode:3 + fuse_minimax_qk_norm (PR #37045), sampling temp=1.0/top_p=0.95/top_k=40. Tool parser `minimax_m2`; reasoning parser `minimax_m2_append_think` (multi-turn default).
- **CONFLICT (resolved in favor of spec)**: card says `broken_on:[B200]` / "garbage output on B200 / do NOT deploy". Spec (audited, source-verified) overrides: card is over-broad, conflates NVFP4/SM120 with FP8/SM100. Real pitfall = FlashInfer FP8 MoE float32-router-logits assertion (#33543) → pin `--moe-backend triton`. Proceed with Triton pin + hard Stage 0c correctness gate. Logged to lessons.md.
- `mdc prs minimax-m2`: synced. Mostly M2.5/M3 + SGLang/AMD PRs; nothing that changes the vLLM B200 FP8 path. M2.7 same-arch.
- `gpu-infra card p6-b200`: AL2023 NVIDIA AMI mandatory (ib_umad/Fabric Manager), nodeadm bootstrap, driver 580.x/CUDA 13.0, NVSwitch (NCCL PCIe bug N/A), cold start ~15min (DeepGEMM JIT). NOTE: card prints `arch: sm_120` — this is the known mislabel; B200 NVSwitch is SM100 (per spec + memory). Sidecar pins sm_100.

### 0c — Serving-config gate (fail-closed)
- Fetched HF config.json (2026-06-27): `intermediate_size=1536` (MoE expert FFN, shared_intermediate_size=0), FP8 block-quant `weight_block_size:[128,128]`, max_position_embeddings=196608, 62 layers, 256/8 experts, head_dim 128, use_qk_norm per_layer, use_mtp 3 modules.
- Divisibility: TP4 → 384 (%128==0) OK; TP8 → 192 (%128!=0) FAIL. This is the mechanism behind "pure TP8 unsupported." Logged to lessons.md.
- `validate-serving-config.py --sidecar benchmark.yaml --corpus-root .` → **EXIT 0, "no applicable rules fired — config is clean."** (FP8-MoE divisibility verified at TP4, max-model-len==max_position_embeddings, AL2023, sm_100 not sm_120 so no cu131 warn, no MLA so no lmcache-incompat.)

**Stage 0 verdict: PASS.** Both cards noted; card-vs-spec B200 conflict flagged and resolved in favor of spec; resolver exit 0. Hard B200 FP8-MoE correctness gate deferred to runtime (post-start).

---
<!-- append Stage 1+ entries below with timestamps -->

## Stage 3 — Storage & model staging

- Launched `k8s/stage-model.yaml` (hf download MiniMaxAI/MiniMax-M2 → /mnt/nvme/models/minimax-m2).
- **ISSUE**: first two attempts failed `socket.gaierror: Temporary failure in name resolution`. Root cause: B200 node has no CNI/kube-proxy DNS path; both pod-network and `hostNetwork+ClusterFirstWithHostNet` (CoreDNS ClusterIP 172.20.0.10) are unreachable from this node.
- **FIX**: `hostNetwork: true` + `dnsPolicy: Default` → inherits node resolv.conf (VPC resolver 10.0.0.2), resolves huggingface.co. Verified via dns-probe. Logged to lessons.md (platform).
- NOTE: huggingface-hub 1.21.0 dropped the `hf_transfer` extra (now Xet high-perf transfer); harmless deprecation warning, download proceeds.
- Download in progress (~450GB FP8). ⚠️ spot reclaim wipes NVMe → re-stage on fresh node.

## Stage 4a — GPU health
- gpu-infra MCP tools not exposed as callable functions in this session; running equivalent checks via kubectl diag pod (nvidia-smi, ECC, Xid) + vLLM's own NCCL init at TP4 startup as the collective-health proof.

## Stage 4b — Observability
- `k8s/observability.yaml` applied: Prometheus :9090 + DCGM :9400 (PROF metrics) on the node, scraping vLLM :8000 (histograms appear post-Stage-5), node-exporter :9100 (cluster's existing). external_labels blueprint=minimax-m2.

### Stage 4a results — PASS
8× B200 visible, 183359 MiB (180GB) each. Temps 31-34C idle (<<85C throttle). ECC uncorrected aggregate = 0 on all 8. No pending row remaps (Pending Page Blacklist N/A, remapped correctable=0, no uncorrected). No Xid. NCCL TP4 all-reduce health proven by vLLM NCCL init at TP4 startup (Stage 5); B200 NVSwitch=SM100, g7e PCIe NCCL bug N/A.

### Stage 4b results
- DCGM_FI_PROF_* profiling counters NOT enabled on this node's driver (matches kimi L11) — pruned the CSV to working non-PROF gauges (power/temp/FB/util/XID). Sufficient for health + cost; HBM-BW/tensor-active roofline classification unavailable from DCGM here (would need a different driver/profiler).
- Prometheus crashed `permission denied: /prometheus/queries.active` (uid 65534 vs root-owned hostPath /mnt/nvme/prom-data). An initContainer chown hung (NVMe I/O saturated by the 450GB download). Resolved by running prometheus as root (runAsUser:0) on the ephemeral bench pod. Logged to lessons.md.
- Image pulls/container-create slowed by NVMe+network contention from concurrent staging download (non-blocking — serving can't start until download completes anyway).

---

## Stage 6b — Detached Pareto-sweep runner: BUILD, INCIDENT, RECOVERY (append 14:00–15:00Z)

### Built (the deliverable)
- `k8s/bench-runner.yaml` — ConfigMap `minimax-bench-scripts` (bench.py: streaming TTFT/ITL, byte-identical
  shared-prefix vs cold, per-config KV-tier breakdown from vLLM `prompt_tokens_by_source`/`external_prefix_cache_*`,
  cheap garbage screen) + hostNetwork/dnsPolicy:Default bench-runner pod.
- `k8s/gen-serving-manifest.sh` — deterministic per-(shape×KV-arm) manifest generator. Shapes: tp4, tp4ep4,
  tp2dp2, tp4dp2, tp2dp4 (TP∈{2,4}; TP8 INVALID per FP8 block-128). KV arms: gpu-only / cpu-offload / nvme-tiering.
- `k8s/run-pareto-sweep.sh` — the unattended runner: full grid (30 configs), per-config concurrency sweep
  [1,4,8,16,32,64,128] with error-rate stop, Prometheus harvest (vLLM-0.19.1rc1-verified metric names),
  pareto-<date>.json + optimization-trajectory-<date>.json, STATUS file, sweep.log.
- `k8s/launch-detached.sh` — nohup+timeout(420m) detached launcher.

### Plumbing VALIDATED on the original node (tp4 gpu-only, cold, c=1 + c=8)
- bench.py streamed TTFT (0.15s @c1), ITL, e2e, quality screen PASS; Prometheus harvest returned live
  TTFT p95 / ITL p95. Trap-scaledown dry-run fired on all 4 exit paths (normal/error/fatal/SIGINT). Interlock verified.

### INCIDENT (root-caused, see lessons.md "kubectl context drift")
- macOS `setsid` missing → first launch failed, left a stray runner subshell.
- `pkill` of that stray fired its EXIT trap → us-east-2 b200 nodegroup desired=0.
- Compounding: kubectl default context had DRIFTED to `qn-sglang-usw2` (us-west-2); relaunched run targeted
  the WRONG cluster while the trap's `aws --region us-east-2` hit the REAL b200.
- Result: original node `i-07aed6e57df2a31cd` (10.0.27.134) drained + TERMINATED; serving/observability pods lost.

### RECOVERY (complete for infra; serving rebuild in progress)
- Restored nodegroup desired=1 → ASG launched replacement `i-025e793b9ec81fcea` → node `ip-10-0-16-155` joined.
- Manually labeled new node `blueprint=minimax-m2` + `nvidia.com/gpu.present=true` (no GFD/gpu-operator in cluster;
  labels are applied manually). Device plugin scheduled → **8 GPUs allocatable**. Taint `ai-infra/b200=true` present from nodegroup.
- Relaunched model staging Job (214GB FP8, ~25min) — downloading. Redeployed observability (3/4 targets up; vllm down until serving).

### HARDENING landed in the runner (so the incident cannot recur)
- `KCTX=(kubectl --context qn-bench-use2)` — every kubectl call context-pinned (no ambient-context dependence).
- PREFLIGHT: verify context→cluster mapping AND a Ready blueprint=minimax-m2 node BEFORE acting; mismatch → abort.
- SCALEDOWN INTERLOCK: `PREFLIGHT_OK` gates scale_to_zero — a misdirected/never-started run exits WITHOUT scaling the node.
- Internal wall-clock watchdog as a second cap enforcer if `timeout` is absent.

### NEXT (to relaunch the sweep)
1. Wait for staging Job Complete (verify 130 shards + config.json on /mnt/nvme/models/minimax-m2).
2. Deploy `k8s/vllm-baseline.yaml`; wait ~35min first boot; re-run Stage 0c correctness gate (20-sample output + tool-parse).
3. `VALIDATE_ONLY=1 bash k8s/launch-detached.sh` to re-confirm plumbing on the new node, THEN `bash k8s/launch-detached.sh` for the full grid.
