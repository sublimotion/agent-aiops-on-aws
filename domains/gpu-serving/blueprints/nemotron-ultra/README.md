# Nemotron-3-Ultra-550B-A55B-NVFP4 — Serving Benchmark Blueprint

Deploys NVIDIA **Nemotron-3-Ultra-550B-A55B** in native **NVFP4** on **p6-b300.48xlarge
spot** (us-west-2b / usw2-az2), via vLLM `v0.22.0-cu130`. Hybrid Mamba-2 + LatentMoE +
Select-Attention with native MTP spec-decode. Spec: `domains/gpu-serving/specs/nemotron-ultra.md`.

> **Current scope**: deployment is gated at the **Stage 5 / P0 smoke test**. P1-P4
> benchmark phases are deferred until the brand-new model is confirmed to load and serve.

## Reused infrastructure

- EKS cluster `qn-sglang-eks-cluster` (us-west-2, v1.32) — attached via Terraform data sources.
- Managed node group `ai-infra-b300-spot` — p6-b300.48xlarge, AL2023 NVIDIA AMI, SPOT,
  usw2-az2, taint `ai-infra/b300=true:NoSchedule`, scaled 0->1 for the GPU node.
- `nvidia-device-plugin` DaemonSet (self-heals when the GPU node joins).
- S3 staging bucket `qn-sglang-models-20260303161715850900000007`.

## Deploy sequence

```bash
cd domains/gpu-serving/blueprints/nemotron-ultra

# 0. Stage 0c gate (fail-closed)
python3 ../../../../standards/serving-commons/resolver/validate-serving-config.py \
  --sidecar benchmark.yaml --corpus-root ../../../..

# 1. Stage weights HF -> S3 (~335 GB NVFP4; uses snapshot_download, not huggingface-cli)
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 ./scripts/stage-model.sh

# 2. Bring up the GPU node (~$27/hr B300 spot)
./scripts/scale-node.sh 1

# 3. Deploy the serving stack (init container syncs S3 -> /mnt/nvme, then vLLM starts)
terraform init && terraform apply -var-file=nemotron-ultra-b300.tfvars

# 4. Smoke gate (6 items)
kubectl -n ai-infra port-forward svc/nemotron-ultra 8000:8000 &
BASE=http://localhost:8000 MODEL=nvidia/nemotron-3-ultra ./scripts/smoke-test.sh

# 5. Tear down (cost discipline)
terraform destroy -var-file=nemotron-ultra-b300.tfvars
./scripts/scale-node.sh 0
```

## Cost

**~$27/hr** B300 spot (the spec's ~$15/hr is stale — see `lessons.md`). Scale the node
group to 0 immediately on a hard blocker or when finished.

## Key risks (Stage 0c findings)

1. **MTP + prefix caching** on this mamba-hybrid may need `--no-enable-prefix-caching`
   (set `enable_prefix_caching=false`). First attempt uses the verbatim card config.
2. **NVFP4 is brand-new** — vLLM load-time error is the authoritative arch check; the
   resolver's FP8 divisibility rule does not cover fp4.
3. **B300 = sm_103 -> `-cu130` image tags only.**
