# GLM-5.2-FP8 on B200 — Full Optimization Sweep

Spec: `domains/gpu-serving/specs/glm5.2.md`

GLM-5.2-FP8 (`glm_moe_dsa`, ~753B MoE, MLA+DSA, native MTP, 1M ctx) on a single 8-GPU B200 node.
vLLM vs SGLang head-to-head on **coding-agent** (primary) + **long-input-31k** (from the kimi customer
profile), full **T0–T5** lever sweep. **TP8 is forced on B200** (~750 GB FP8 doesn't fit TP4);
**TP4+DP2 is a B300-only arm** (us-west-2, the layout that won +19–25% on kimi-k2.6-nvfp4).

## Sweep / run order

1. **Stage 4-pre** (us-east-2): scale `ai-infra-use2-b200-spot` → 1; label node `blueprint=glm5.2` +
   `nvidia.com/gpu.present=true`; RAID-0 NVMe at `/mnt/nvme`; ECC gate (`volatile.total==0`).
2. **Observability FIRST** — `kubectl apply -f k8s/observability.yaml`, then smoke-test:
   `curl -s localhost:9400/metrics | grep DCGM_FI_PROF_DRAM_ACTIVE` (non-empty) before any bench.
3. **Stage GLM-5.2-FP8** (~750 GB) → `/mnt/nvme/models/glm5.2-fp8` (Xet disabled + retry loop).
4. **Engine head-to-head** — SGLang (`k8s/sglang-glm52-t0-baseline.yaml`) then vLLM
   (`k8s/vllm-glm52-baseline.yaml`); `/health` 200, glm47/glm45 parser checks, record cold start.
5. **T0→T5 on the winning engine** — isolate one tier per run:
   - T0 `sglang-glm52-t0-baseline.yaml` (FP8 floor, radix off)
   - T1 +`--kv-cache-dtype fp8_e4m3`
   - T2 +prefix cache, +HiCache (`--enable-hierarchical-cache --hicache-ratio 2`, pod mem 1600Gi)
   - T3 +EAGLE (SGLang) / +MTP (vLLM) — **ground accept on production-mix, not synthetic**
   - T4 +`--enable-dp-attention --moe-a2a-backend deepep` (B200); TP4+DP2 on B300
   - T5 full stack: `sglang-glm52-fullstack.yaml`
6. **Bottleneck classify at the knee** (PROF DRAM/TENSOR + token_usage); fill the Tier Stack Table.
7. **B300 arm (optional)** — if B200 TP8 KV-capacity-bound: switch to `qn-sglang-eks-cluster`
   (us-west-2), scale `ai-infra-b300-spot`, run `sglang-glm52-tp4dp2-b300.yaml`.

## Manifests

| File | Purpose |
|------|---------|
| `k8s/observability.yaml` | DCGM (PROF wiring **fixed** vs kimi parent) + Prometheus |
| `k8s/sglang-glm52-t0-baseline.yaml` | T0 FP8 floor, SGLang TP8 |
| `k8s/sglang-glm52-fullstack.yaml` | T1+T2+T3+T5 stacked, SGLang TP8 |
| `k8s/vllm-glm52-baseline.yaml` | vLLM head-to-head, TP8 + MTP |
| `k8s/sglang-glm52-tp4dp2-b300.yaml` | B300-only TP4+DP2 arm (us-west-2) |
| `k8s/bench-runner.yaml` | concurrency driver (coding-agent + long-input modes) |
| `benchmark.yaml` | sidecar (engine compare, T0–T5, workloads, SLOs) |

> Operational artifacts (lessons, results) land here after the run.
