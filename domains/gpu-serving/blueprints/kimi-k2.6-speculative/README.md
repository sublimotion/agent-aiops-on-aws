# Kimi K2.6 Speculative Decode + Dynamic MLA — Blueprint

Phase 0-5 EAGLE3 + dynamic MLA/MHA routing study on B300 spot. Spec: [`../../specs/kimi-k2.6-speculative.md`](../../specs/kimi-k2.6-speculative.md).

## Reusable Artifacts

| Artifact | Location | Notes |
|---|---|---|
| **AWS region** | `us-east-1` | Spot placement score 9/10 in `use1-az6` |
| **EKS cluster** | `inference-eks-v132` (K8s 1.32) | Hosts benchmark runner pod |
| **VPC** | `vpc-07ff0e6bdc3cac475` (10.192.0.0/16) | Shared with HyperPod stack |
| **GPU subnet** | `subnet-05569398360910f46` (us-east-1c / use1-az6, 10.192.12.0/24) | Associated with RT that has IGW + S3 VPC endpoint |
| **Runner subnet** | `subnet-09c51a62517440cab` or `subnet-0ee4f9c19d21d5532` | az2/az4, already has routing |
| **Security group** | `sg-0c026989a9a5246e3` (hyperpod-eks-no-ingress-sg) | Self-referencing; add explicit ingress on 30000, 8000, 22 |
| **Model bucket** | `s3://kimi-k2-bench-models-20260216163240701700000006` | Stage K2.6 + draft under `models/Kimi-K2.6/` and `models/kimi-k2.6-eagle3/` |
| **Results bucket** | `s3://kimi-k2-bench-results-20260216163240701700000007` | Benchmark JSON archive |
| **ECR vLLM** | `615299764834.dkr.ecr.us-east-1.amazonaws.com/vllm-openai` | Push new tag `k26-eagle3-cu130-<date>` |
| **ECR SGLang** | (to create) `615299764834.dkr.ecr.us-east-1.amazonaws.com/sglang` | Push `v0.5.10-cu130-<date>` |
| **ECR bench runner** | (to create) `615299764834.dkr.ecr.us-east-1.amazonaws.com/bench-runner` | aiperf + custbench |
| **DLAMI** | `ami-027c3ae8019fc0d3a` | Deep Learning Base OSS NVIDIA AL2023 20260512 |
| **SSH key** | `g7e-bench` | Private at `~/.ssh/g7e-bench.pem` |

## Directory Layout

```
kimi-k2.6-speculative/
├── README.md                        ← this file
├── scripts/
│   ├── stage-weights.sh             ← helper EC2: HF → S3
│   ├── launch-helper-ec2.sh         ← c6i.8xlarge in us-east-1b
│   ├── launch-gpu-node.sh           ← p6-b300 spot in us-east-1c
│   ├── bootstrap-gpu-node.sh        ← userdata: NVMe RAID, S3 sync, image pull
│   ├── phase-0-roofline.sh
│   ├── phase-1-sglang-eagle3.sh
│   ├── phase-2-vllm-eagle3.sh
│   ├── phase-3-dynamic-mla.sh
│   ├── phase-4-fullstack.sh
│   └── phase-5-frontier.sh
├── manifests/
│   ├── bench-runner-pod.yaml        ← CPU pod on EKS for driving benchmarks
│   └── k26-vllm-eagle3.yaml         ← (optional) GPU pod spec if joining EKS
├── configs/
│   ├── sglang-eagle3.env            ← flag set for Phase 1
│   ├── vllm-eagle3.env              ← Phase 2
│   ├── vllm-dynamic-mla.env         ← Phase 3
│   └── fullstack.env                ← Phase 4
├── terraform/
│   └── (optional — VPC endpoints, IAM, bucket policies if needed)
├── results/
│   ├── progress.md                  ← stage tracker (YAML frontmatter)
│   └── (populated as phases complete)
├── docs/
│   ├── roofline-explainer.html
│   └── roofline-explainer.svg
└── lessons.md                       ← mid-flight capture per CLAUDE.md §5
```

## Runbook

```bash
# 0. Stage weights (background, ~2h; run once, artifact survives)
./scripts/stage-weights.sh

# 1. Launch GPU spot node (waits for weights to be in S3)
./scripts/launch-gpu-node.sh
# Outputs GPU_NODE_IP — record in lessons.md

# 2. Deploy bench runner on EKS
kubectl apply -f manifests/bench-runner-pod.yaml

# 3. Kick off RALPH loop
/ralph-loop:ralph-loop Deploy domains/gpu-serving/specs/kimi-k2.6-speculative.md

# ... Phases 0-5 execute ...

# 4. Compound + teardown
./scripts/teardown.sh  # terminates spot node, keeps S3 artifacts
```

## Budget

Spot p6-b300 at ~$26.41/hr × ~13 hrs = **~$343** baseline for full sweep. Weight staging + image builds + EKS runner: **~$15-20**. Results bucket + ECR storage: negligible.
