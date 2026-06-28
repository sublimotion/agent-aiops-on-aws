---
blueprint: minimax-m2
status: in_progress
last_stage: 3-rebuild
last_updated: 2026-06-27T15:00:00Z
stages:
  stage_0_pre_deploy: complete
  stage_0c_serving_gate: complete
  stage_1_foundation: skipped        # infra pre-provisioned (cluster + node already up)
  stage_2_build_machine: skipped     # no custom image build
  stage_3_storage_staging: in_progress   # RE-STAGING on replacement node i-025e793b9ec81fcea (incident)
  stage_4_gpu_node: complete         # node already joined, labeled, 8 GPU allocatable
  stage_4a_gpu_health: complete
  stage_4b_observability: complete   # redeployed on new node; 3/4 targets up (vllm down until serving)
  stage_5_serving: not_started       # must redeploy + re-pass Stage 0c on replacement node
  stage_6_benchmark: in_progress     # detached Pareto sweep launched (full grid, unattended)
---

# MiniMax-M2 Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 0 pre-deploy | complete | cards loaded; card-vs-spec B200 conflict flagged + resolved (spec wins, triton pin) |
| 0c serving gate | complete | resolver exit 0; FP8-MoE TP4 divisibility verified (1536/4=384, %128==0) |
| 1 foundation | skipped | cluster + VPC pre-provisioned |
| 2 build machine | skipped | stock vllm image |
| 3 storage/staging | complete | 214GB FP8 / 130 shards on /mnt/nvme/models/minimax-m2 |
| 4 gpu node | complete | REPLACEMENT node ip-10-0-16-155 (i-025e793b9ec81fcea) Ready, labeled blueprint+gpu.present, 8 GPU (orig node terminated by incident) |
| 4a gpu health | complete | (live state verified prior to this session) |
| 4b observability | complete | prometheus+dcgm+node 4/4 targets up; vLLM histograms present |
| 5 serving | not_started | must redeploy baseline + re-pass Stage 0c on replacement node after re-stage |
| 6 benchmark | in_progress | Sweep launched then SIGTERM'd at +3.5min (0 configs run, trap scaled nodegroup→0). Cache-hit telemetry HARDENED before relaunch: bench.py now emits per-config `kv_tier` breakdown (GPU-HBM vs offload-tier vs recompute, from vLLM `prompt_tokens_by_source`/`external_prefix_cache_*` deltas) + cold-vs-warm TTFT ratio, alongside the headline Σcached/Σprompt token fraction. harvest_prom carries the same tier counters as cross-check. kv_transfer bytes/latency NOT exposed by this build (recorded as a finding). READY for relaunch via launch-detached.sh. |
