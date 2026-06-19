# Launch Checklist — self-coding-agent-loop on p4de spot

**Target**: p4de.24xlarge spot in us-east-1c (zone ID `use1-az6`), ~$13/hr median.

**All prerequisites already exist in the account** — no resources need to be created. Updated 2026-05-10 after account audit.

## Reusable resources (no creation needed)

| Variable | Value | Source |
|---|---|---|
| `REGION` | `us-east-1` | target |
| `AZ` | `us-east-1c` | zone name for use1-az6 |
| `AMI_ID` | `ami-091f07e77f51e6b42` | Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04) 20260505 |
| `AWS_KEYPAIR_NAME` | `g7e-bench` | ~/.ssh/g7e-bench.pem |
| `AWS_SUBNET_ID` | `subnet-00951f2f48d1eebc2` | default VPC public subnet in us-east-1c (172.31.32.0/20) |
| `AWS_SECURITY_GROUP_ID` | `sg-0de21eebeb0b8c70a` | `coderforge-training` SG — SSH:22 open, all egress; default VPC |
| `AWS_INSTANCE_PROFILE` | `arn:aws:iam::615299764834:instance-profile/g7e-bench-profile` | IAM role `g7e-bench-role` — S3 rw on `agent-aiops-artifacts`, SSM, Bedrock |

## Sibling infrastructure also available

- **`i-02b3e99702834e4a9`** (`swebench-eval`, m7i.4xlarge, stopped in us-east-1a) — preconfigured VP SWE-bench Docker eval box. Start it up for Round 1's Docker gold eval instead of launching a new m7i:
  ```bash
  aws ec2 start-instances --region us-east-1 --instance-ids i-02b3e99702834e4a9
  aws ec2 describe-instances --region us-east-1 --instance-ids i-02b3e99702834e4a9 \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
  # then: export M7I_HOST=ubuntu@<that-ip>
  ```
  Note: m7i.4xlarge is 4× smaller than the budgeted m7i.16xlarge → eval wall-clock is ~40-80hr for 4K tasks instead of ~15hr. For Round 1's evals (300 control + 300 drift_audit = 600 tasks), that's ~12-24hr — acceptable. For later rounds with more trajectories, may want to scale up.

## Pre-launch verification

```bash
# Is p4de available in us-east-1c RIGHT NOW?
aws ec2 describe-spot-price-history --region us-east-1 \
  --instance-types p4de.24xlarge --product-descriptions "Linux/UNIX" \
  --availability-zone us-east-1c --max-items 3 \
  --query 'SpotPriceHistory[*].{ts:Timestamp,price:SpotPrice}' --output table

# Is the Nebius dataset still at the expected path?
ls -lh domains/autoresearch/blueprints/self-coding-agent-loop/data/nebius/trajectories.parquet

# Is Gen0 adapter on S3?
aws s3 ls s3://agent-aiops-artifacts/self-coding-agent-loop/gen0/
```

## Launch command (single line — all vars populated from existing resources)

```bash
cd /Users/phi/Documents/workbench/agent-aiops-on-aws
AWS_SUBNET_ID=subnet-00951f2f48d1eebc2 \
AWS_SECURITY_GROUP_ID=sg-0de21eebeb0b8c70a \
AWS_INSTANCE_PROFILE=arn:aws:iam::615299764834:instance-profile/g7e-bench-profile \
bash domains/autoresearch/blueprints/self-coding-agent-loop/scripts/launch_p4de.sh
```

The script will:
1. Snapshot repo → S3
2. `aws ec2 run-instances` with spot market options, user-data, tags
3. Wait for `running` state
4. Print SSH command + bootstrap-monitor command + teardown command

## After instance is running

```bash
# Wait for cloud-init bootstrap to complete (~10-15 min for pip install heavy torch/vllm)
IP=<from launch output>
until ssh -o ConnectTimeout=5 -i ~/.ssh/g7e-bench.pem ubuntu@$IP \
    test -f /mnt/nvme/self-coding-agent-loop/.bootstrap-complete 2>/dev/null; do
    sleep 30; echo waiting...; done

# Kick off Round 1
ssh -i ~/.ssh/g7e-bench.pem ubuntu@$IP \
  'cd /mnt/nvme/self-coding-agent-loop/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop/scripts && \
   tmux new -d -s round1 "bash run_round.sh 1 2>&1 | tee /mnt/nvme/self-coding-agent-loop/runs/round1.log"'

# Watch progress from laptop
ssh -i ~/.ssh/g7e-bench.pem ubuntu@$IP 'tail -f /mnt/nvme/self-coding-agent-loop/runs/round1.log'

# Or watch via S3 (updated every 10 min by s3-sync daemon)
watch -n 120 'aws s3 ls --recursive s3://agent-aiops-artifacts/self-coding-agent-loop/runs/round_1/ | tail -20'
```

## Cost guardrails

- p4de.24xlarge spot at ~$13/hr median in us-east-1c
- Round 1 expected wall-clock: ~20-24 hours (train + control eval + drift_audit eval + verifier recalibrate)
- Expected Round 1 cost: ~$260-310 on p4de + ~$30 m7i Docker eval = **~$300 for Round 1**

## Teardown

```bash
aws ec2 terminate-instances --region us-east-1 --instance-ids <instance-id>
```

S3 artifacts persist in `s3://agent-aiops-artifacts/self-coding-agent-loop/` and survive teardown.

## Pitfalls discovered during setup

- **AZ name ≠ AZ ID**: "az6" is the zone ID. The zone name in us-east-1 is `us-east-1c`. Launching with `us-east-1f` (which I originally hardcoded) would have gone to a different AZ that doesn't even offer p4de.
- **AMI mismatch**: original spec assumed AL2023 NVIDIA; no AL2023 NVIDIA AMI exists in us-east-1 at this cutoff. Using Ubuntu DL Base OSS Nvidia AMI instead. `user_data.sh` was updated from `dnf` → `apt`.
- **EKS node roles don't have S3**: three existing `*-gpu-node-*` instance profiles in the account are EKS-only (worker node + CNI + ECR-read). None can write to S3. Created a purpose-built role.
- **No SSH-capable SG**: existing aiops VPC SGs are VPC endpoints, SageMaker, or EKS cluster/node. None allow SSH:22 ingress. Created a purpose-built SG.
- **p4de only in 2 AZs**: us-east-1b ($21/hr) and us-east-1c ($13/hr). Don't try other AZs.
