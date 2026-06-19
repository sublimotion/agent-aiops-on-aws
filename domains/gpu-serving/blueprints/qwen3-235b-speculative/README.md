# Qwen3-235B Speculative Decode + Optimization Tier — Blueprint

Phase 0-5 EAGLE3 + HiCache + TP4/DP2 study on B300 spot. Spec: [`../../specs/qwen3-235b-speculative.md`](../../specs/qwen3-235b-speculative.md).

Mirrors the methodology of `kimi-k2.6-speculative/` — same hardware (p6-b300), same phase structure, same observability mandate (Prometheus + DCGM from Stage 4b), same v1 envelope output.

## Reusable Artifacts

| Artifact | Location | Notes |
|---|---|---|
| **AWS region** | `us-west-2` | B300 spot $25.47/hr in **usw2-az2** (2026-05-14) |
| **Subnet / VPC** | (to provision) | Need public subnet with IGW in usw2-az2 — do NOT reuse the Kimi us-east-1 infra |
| **Security group** | (to provision) | SSH/SGLang 30000 from controller + self-ref |
| **IAM role** | `qwen3-gpu` (to create) | S3 read model bucket + write results bucket + SSM read HF token |
| **Model bucket** | (to provision) | Stage Qwen3-235B FP8 + EAGLE3 draft |
| **Results bucket** | (to provision) | Prometheus snapshots + v1 envelopes |
| **ECR SGLang** | `lmsysorg/sglang:v0.5.10-cu130` (public) | Same as Kimi — no custom image needed |
| **AMI** | AL2023 NVIDIA DLAMI (us-west-2) | Must verify current id in us-west-2 region |
| **SSH key** | `qwen3-bench` (to create) or reuse existing | Private at `~/.ssh/qwen3-bench.pem` |
| **HF Token** | Use existing `~/.cache/huggingface/token` — push to SSM at bootstrap per Kimi L14/L15 | Required for draft download (gated repo) |

## Directory Layout

```
qwen3-235b-speculative/
├── README.md                     ← this file
├── benchmark.yaml                ← v1 sidecar (authoritative metadata)
├── scripts/
│   ├── bootstrap-gpu-node.sh    ← NVMe RAID, HF staging, image pull
│   ├── launch-gpu-node.sh       ← p6-b300 spot in us-west-2b
│   ├── phase0-nccl-node.sh      ← NCCL + topology
│   ├── stage-weights.sh         ← direct-HF download with hf_transfer
│   ├── run-phase1b-sweep.sh     ← 13-config EAGLE3 sweep
│   ├── run-phase4-fullstack.sh  ← winner + HiCache
│   ├── run-phase5.sh            ← frontier variants
│   └── teardown.sh              ← terminate spot, preserve S3
├── configs/
│   ├── sglang-eagle3-default.env
│   ├── sglang-eagle3-sweep.env
│   └── sglang-fullstack.env
├── results/
│   ├── progress.md              ← YAML frontmatter stage tracker
│   ├── standard/                ← v1 envelopes (populated by bench-standard.py)
│   ├── phase-0-roofline/
│   ├── phase-1b/
│   ├── phase-4/
│   └── phase-5/
├── docs/
│   └── benchmark-report.html    ← visual report (generated post-session)
└── lessons.md                    ← mid-flight capture per CLAUDE.md §5
```

## Runbook

```bash
# 0. Deploy card lookup
mdc get qwen3-235b --engine sglang
mdc prs qwen3-235b
gpu-infra card p6-b300

# 1. Provision networking (new — us-west-2)
# TODO: create subnet, security group, IAM role, SSH key

# 2. Push HF token to SSM
aws ssm put-parameter --name /qwen3/hf-token --type SecureString \
  --value "$(cat ~/.cache/huggingface/token)" --region us-west-2 --overwrite

# 3. Launch GPU spot node
./scripts/launch-gpu-node.sh

# 4. Install observability (MANDATORY — Stage 4b)
ssh -i ~/.ssh/qwen3-bench.pem ec2-user@<pub-ip> \
  "bash /opt/benchmark-runner/scripts/bootstrap-observability.sh <results-bucket> qwen3-235b-speculative"
ssh -i ~/.ssh/qwen3-bench.pem ec2-user@<pub-ip> \
  "bash /opt/benchmark-runner/scripts/observability-smoke-test.sh"

# 5. Stage weights (target ~235 GB + draft ~2 GB, ~5-15 min on 3200 Gbps)
./scripts/stage-weights.sh

# 6. Phase 0 roofline
./scripts/phase0-nccl-node.sh

# 7. Phase 1b sweep (launches SGLang via bench-standard.py per config)
./scripts/run-phase1b-sweep.sh

# 8. Phase 4 fullstack (reads WINNER.env from Phase 1b)
./scripts/run-phase4-fullstack.sh

# 9. Phase 5 frontier
./scripts/run-phase5.sh

# 10. Post-session
./scripts/teardown.sh
python3 ../../results-vault/rebuild-index.py
python3 ../../results-vault/build-dashboard.py
```

## Budget

Spot p6-b300 at $25.47/hr × ~7 hrs = **~$180**. Weight staging + images + buffer: **~$10**. Total: **~$190** (well under $350 ceiling).

## Key Differences from Kimi K2.6-spec Blueprint

| | Kimi K2.6 | Qwen3-235B |
|---|---|---|
| Model size | 1T total / 32B active | 235B / 22B |
| Quantization | FP8 block-scaled, MLA | FP8 block-scaled, GQA |
| TP constraint | TP8 optimal | **TP4** (FP8 block_n=128 divisibility) |
| Draft model | `lightseekorg/kimi-k2.6-eagle3` (accept len 5.0) | `lmsys/Qwen3-235B-A22B-EAGLE3` (accept len 3.0-3.5 per model card) |
| Context | 131,072 | **40,960** (NO YaRN in FP8 variant) |
| Region | us-east-1c | us-west-2b |
| Spot price | $25.65/hr | **$25.47/hr** |
| Expected winner `num_steps` | 4 (accept length ceiling = 5) | **2 or 3** (accept length ceiling ~3.5) |
| Phase 2 (vLLM EAGLE3) | Blocked by L13 | N/A — no first-party vLLM draft |
| Phase 3 (dynamic MLA) | Skipped — low delta | N/A — Qwen3 uses GQA, not MLA |

## References

- Spec: `domains/gpu-serving/specs/qwen3-235b-speculative.md`
- Baseline: `domains/gpu-serving/blueprints/qwen3-235b-b300/` (vLLM TP4 / TP2+DP4+EP, 2026-04-22)
- Methodology template: `domains/gpu-serving/blueprints/kimi-k2.6-speculative/`
- Observability stack: `.claude/skills/benchmark-runner/` (Prometheus + DCGM + `bench-standard.py`)
- Draft model card: https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3
