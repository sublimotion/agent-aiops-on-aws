# Kimi-K2.6-NVFP4 — EKS Deploy Runbook

Cluster `qwen3-next-bench-eks-cluster` (us-east-2), nodegroup `ai-infra-use2-b200-spot`
(p6-b200.48xlarge SPOT, max=1, us-east-2b / use2-az2). Reference manifests adapted from
`qwen3-235b-speculative/k8s/`.

## 0. Pre-flight (free, no node yet)
- [ ] `kubectl config current-context` → `...cluster/qwen3-next-bench-eks-cluster`
- [ ] Nodegroup ACTIVE, desired=0: `aws eks describe-nodegroup --cluster-name qwen3-next-bench-eks-cluster --nodegroup-name ai-infra-use2-b200-spot --region us-east-2 --query 'nodegroup.scalingConfig'`
- [ ] Stage 0c resolver passes: `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar domains/gpu-serving/blueprints/kimi-k2.6-nvfp4/benchmark.yaml --corpus-root .`

## 1. Scale up the B200 node — ⚠️ BILLABLE (~$18/hr spot) — requires explicit go
```bash
aws eks update-nodegroup-config --cluster-name qwen3-next-bench-eks-cluster \
  --nodegroup-name ai-infra-use2-b200-spot --region us-east-2 \
  --scaling-config minSize=0,maxSize=1,desiredSize=1
# wait for node Ready (~few min if spot capacity available; may fail if no spot)
kubectl get nodes -l ai-infra/role=b200-spot -w
```
- [ ] Label the node — BOTH labels required (this cluster has no GFD/NFD, see lessons L1):
```bash
NODE=$(kubectl get nodes -l ai-infra/role=b200-spot -o name | head -1)
kubectl label "$NODE" blueprint=kimi-k2.6-nvfp4 --overwrite          # pod nodeSelector
kubectl label "$NODE" nvidia.com/gpu.present=true --overwrite         # else device plugin DESIRED=0, GPUs never register
# verify: kubectl get node "${NODE##*/}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'  → 8
```
- [ ] Node taint is `ai-infra/b200=true:NoSchedule` (NOT nvidia.com/gpu) — all manifests already tolerate it (L2).

## 2. Stage 4a — GPU health (gpu-infra MCP)
- [ ] `discover_cluster`, `check_gpu_health` (ECC, row remap, thermals), `run_nccl_test` (TP8 NV18 all-reduce BW)
- [ ] Record NCCL all-reduce BW into the spec Stage 4a blank.

## 3. Stage 4b — Observability (MUST pass before any serving)
```bash
kubectl apply -f observability.yaml
```
- [ ] Prometheus `up` has no zeros once a serving pod is up; DCGM reports 8 GPUs; `DCGM_FI_PROF_*` non-empty.

## 4. Stage model weights to node NVMe (~520 GB)
On the node (via a staging pod or ssm): `hf download nvidia/Kimi-K2.6-NVFP4 --local-dir /mnt/nvme/models/kimi-k26-nvfp4`
with `HF_HUB_ENABLE_HF_TRANSFER=1` and `export HF_TOKEN=...`. For the full-stack arm also stage the EAGLE3
draft: `hf download lightseekorg/kimi-k2.6-eagle3.1-mla --local-dir /mnt/nvme/models/kimi-k26-eagle3-mla`.
⚠️ Spot reclaim wipes NVMe — re-stage on any fresh node.

## 5. Stage 5 — Serve (head-to-head; one engine at a time, hostNetwork=one per node)
```bash
# Arm A: SGLang baseline (no spec, no HiCache)
kubectl apply -f sglang-nvfp4-baseline.yaml
# smoke: curl -s localhost:30000/health ; curl -s localhost:30000/metrics | grep -c sglang:
# ... bench (Stage 6) ... then delete and bring up next arm:
kubectl delete pod sglang-kimi-nvfp4-baseline
# Arm B: vLLM baseline (FLASHINFER_MLA, GPU-KV-only)
kubectl apply -f vllm-nvfp4-baseline.yaml
# Arm C: SGLang full-stack (HiCache + EAGLE3) — winner's optimization sweep
kubectl apply -f sglang-nvfp4-fullstack.yaml
```
- [ ] SGLang `/metrics` returns `sglang:*` (proves `--enable-metrics`); else TTFT data is lost.
- [ ] FlashInfer cubin pre-clear ran (baked into the launch command).

## 6. Stage 6 — Benchmark
```bash
kubectl apply -f ../../qwen3-235b-speculative/k8s/bench-runner.yaml  # or a kimi-specific copy
```
Run the concurrency sweep (64→2048 at 31,404 ctx), coding-agent shape, and shared-prefix cache validator.
Headline cache metric = Σ cached_input_tokens / Σ prompt_tokens (engine-agnostic). Classify the knee
regime from DCGM (HBM-BW-bound vs KV-capacity-bound vs prefill-compute-bound).

## 7. Teardown — STOP SPEND
```bash
aws eks update-nodegroup-config --cluster-name qwen3-next-bench-eks-cluster \
  --nodegroup-name ai-infra-use2-b200-spot --region us-east-2 \
  --scaling-config minSize=0,maxSize=1,desiredSize=0
```
⚠️ Snapshot Prometheus TSDB to S3 BEFORE teardown (Kimi-spec lost all data on termination).
