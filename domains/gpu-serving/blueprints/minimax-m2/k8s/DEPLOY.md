# MiniMax-M2 — EKS Deploy Runbook (4× B200, vLLM)

Cluster `qwen3-next-bench-eks-cluster` (us-east-2), nodegroup `ai-infra-use2-b200-spot`
(p6-b200.48xlarge SPOT, AL2023 NVIDIA AMI). Node `ip-10-0-27-134.us-east-2.compute.internal`,
labeled `blueprint=minimax-m2`, taint `ai-infra/b200=true:NoSchedule`, 8 allocatable GPUs.
Spec: `domains/gpu-serving/specs/minimax-m2.md`. ⚠️ BILLABLE spot B200 (~$18/hr).

Serving: MiniMaxAI/MiniMax-M2 FP8 (native, block-quant), TP4 + EP4, **`--moe-backend triton`** (CRITICAL).

## 0. Pre-flight (done)
- [x] `kubectl config current-context` → `...qwen3-next-bench-eks-cluster`
- [x] Node Ready, labeled `blueprint=minimax-m2`, `nvidia.com/gpu.present=true`, 8 GPU.
- [x] Stage 0c resolver: `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar domains/gpu-serving/blueprints/minimax-m2/benchmark.yaml --corpus-root .` → exit 0.

## 1. Stage model weights to NVMe (~450GB FP8) — ⚠️ spot reclaim wipes NVMe
```bash
kubectl apply -f k8s/stage-model.yaml
kubectl logs -f job/stage-minimax-m2          # wait for "config.json OK" + shard count + tokenizer OK
```
Uses the `hf-token` secret (key `token`) and HF_HUB_ENABLE_HF_TRANSFER=1. Lands at /mnt/nvme/models/minimax-m2.

## 2. Stage 4a — GPU health (gpu-infra MCP)
- [ ] `discover_cluster`, `check_gpu_health` (ECC 0, no pending row remaps, thermals <85C, no Xid).
- [ ] `run_nccl_test` TP4 all-reduce across the NVSwitch domain (PASS >= ~1050 GB/s busbw). B200 NVSwitch is SM100 — NCCL PCIe bug (g7e/SM120) does NOT apply.

## 3. Stage 4b — Observability (BEFORE serving)
```bash
kubectl apply -f k8s/observability.yaml
# verify Prometheus :9090 up, DCGM :9400 reports 4+ GPUs. Engine histograms appear after Stage 5.
```

## 4. Stage 5 — Serve baseline + B200 FP8-MoE CORRECTNESS GATE (make-or-break)
```bash
kubectl apply -f k8s/vllm-baseline.yaml
kubectl logs -f vllm-minimax-m2-baseline       # first boot ~16 min (DeepGEMM JIT). Do NOT call failure before then.
```
Gate (FAIL-CLOSED — STOP and escalate if any fails):
- [ ] Server starts WITHOUT the FlashInfer FP8 MoE float32-router-logits assertion (#33543). Confirm `--moe-backend triton` in effect.
- [ ] `curl localhost:8000/health` → 200.
- [ ] Single `/v1/completions` returns valid output (no OOM at 0.90 util on a long prefill).
- [ ] **20 sample completions: non-garbled output + valid `<minimax:tool_call>` parsing.** Garbage → STOP, escalate (minimax27 image / nightly).
- [ ] Tool + reasoning round-trip: parsed `tool_calls` returned AND `<think>` retained in content (append_think parser).
- [ ] Post-start: rerun observability smoke — vLLM histograms present in Prometheus.

## 5. STOP after Stage 5 — report before benchmarking
Do NOT run Stage 6 / 6b until the correctness gate is confirmed. The spec says benchmarking
corrupted output is worthless; escalate instead.

## Teardown — STOP SPEND
```bash
# snapshot Prometheus TSDB to S3 first (Kimi-spec lost all data on termination)
kubectl delete pod vllm-minimax-m2-baseline observability-bench --ignore-not-found
aws eks update-nodegroup-config --cluster-name qwen3-next-bench-eks-cluster \
  --nodegroup-name ai-infra-use2-b200-spot --region us-east-2 \
  --scaling-config minSize=0,maxSize=1,desiredSize=0
```
