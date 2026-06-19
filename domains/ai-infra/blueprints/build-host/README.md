# Build Host — Slim Image Builder

Spot c7i.4xlarge that builds the slim vLLM and SGLang containers and pushes them to private ECR. Tear down with `terraform destroy` when not in use.

## Why

Builds eat 30-60 min CPU + tens of GB disk. We don't want that on a laptop. Source-of-truth is the Dockerfiles in `domains/ai-infra/shared/images/` plus the `build.sh` and version pins there — this blueprint just provisions an ephemeral environment to run them.

## What it provisions

- **EC2 spot c7i.4xlarge** in an existing EKS cluster's VPC public subnet. CPU-bound, no GPU. ~$0.15-0.25/hr spot in us-east-1.
- **200 GB gp3 root volume**, 3000 IOPS, 250 MB/s throughput.
- **Two private ECR repos**: `ai-infra/vllm-slim`, `ai-infra/sglang-slim`. Lifecycle policy keeps last 10 tags.
- **IAM role** with ECR push + SSM (so you can session-manager in if SSH is firewalled).
- **Security group** with SSH from a single CIDR (your IP).

## Prerequisites

- An existing EKS cluster (any of the gpu-serving blueprints). Used only for VPC lookup.
- An existing EC2 key pair in the same region.
- AWS CLI credentials with permissions to provision EC2/ECR/IAM in the account.

## Workflow

```bash
cd domains/ai-infra/blueprints/build-host/terraform

terraform init
terraform apply \
    -var "eks_cluster_name=<your-existing-eks-cluster>" \
    -var "ssh_key_name=<your-keypair>" \
    -var "ssh_ingress_cidr=$(curl -s ifconfig.me)/32"

# Wait ~3 min for instance + cloud-init.
ssh -i ~/.ssh/<your-keypair>.pem ubuntu@$(terraform output -raw public_ip)

# On the host:
cd /opt/build/agent-aiops-on-aws
./domains/ai-infra/blueprints/build-host/scripts/pull-and-build.sh

# Or one engine + variant:
./domains/ai-infra/blueprints/build-host/scripts/pull-and-build.sh vllm cu128

# When the queue empties:
terraform destroy
```

The `pull-and-build.sh` script does:

1. `git pull` to pick up new Dockerfiles.
2. `ecr-login` to refresh Docker credentials.
3. Run `shared/images/build.sh` to build.
4. Tag + push to ECR.
5. Print compressed size table from ECR.

## What's not included

- **No remote Terraform state.** Stack is small, single-operator, ephemeral. Local state is fine.
- **No CI integration.** Two reasons: (a) builds are infrequent — only when upstream vLLM/SGLang releases or our base Dockerfile changes, and (b) reproducing CI builds locally with `build.sh` is straightforward enough that a CI job adds complexity without much value. If we end up bumping more than monthly, revisit.
- **No caching beyond Docker buildx local cache.** First build is slow (~30 min); subsequent builds reuse layers. If a host is destroyed, the cache goes with it. For a build that runs a few times a quarter, that's fine.
- **No multi-arch builds.** EKS GPU fleet is amd64-only. `linux/amd64` only.

## Costs

- Spot c7i.4xlarge: ~$0.15-0.25/hr (varies). A typical build session is 1-2 hours including the first cold build; ~$0.50 per build cycle.
- ECR storage: $0.10/GB-month. Four images at ~7 GB each = ~$3/month if left running.
- Data transfer: ECR pulls into the same region are free. Pulling pre-existing layers from Docker Hub (CUDA base, etc.) is the only network cost during build, ~$0.10 per build.

Total realistic spend: under $5/month for moderate use, $0 when destroyed.

## Operational notes

- **Spot interruption**: persistent spot with `instance_interruption_behavior = "stop"`. If interrupted mid-build, the instance stops; bring it back with `aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)`. The Docker layer cache is preserved on EBS.
- **First build is slow**: cold Docker layer cache pulls ~10 GB of CUDA base images. Subsequent variant builds (cu128 → cu130) share most layers and complete in ~10-15 min each.
- **Disk pressure**: if you build many tags, prune occasionally with `docker image prune -af` and `docker buildx prune -af`. The 200 GB root disk holds ~6-8 tagged variants comfortably.
- **No GPU testing**: the build host can't run the images it builds (no GPU). Validation runs against EKS GPU nodes via the profiler, not here.

## Files

- `terraform/main.tf`, `variables.tf`, `outputs.tf` — infra
- `scripts/bootstrap.sh` — cloud-init userdata, runs once at first boot
- `scripts/pull-and-build.sh` — invoked manually per build cycle
- Source Dockerfiles + `build.sh`: `domains/ai-infra/shared/images/`
