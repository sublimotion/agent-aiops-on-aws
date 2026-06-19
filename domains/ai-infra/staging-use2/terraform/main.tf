###############################################################################
# AI-Infra Lab Staging — us-west-2
#
# Adds to the existing qn-sglang-eks-cluster:
#   1. Two ECR repos for slim images (vllm-slim, sglang-slim).
#   2. A B300 spot managed nodegroup pinned to us-west-2b (where spot lives).
#      Starts at desired_size=0; we scale to 1 only when a B300 experiment
#      cell is about to run, then back to 0.
#   3. IAM service account for pods to read the kimi-k2.6 model bucket.
#
# Why us-west-2b: spot price history shows B300 spot consistently in 2b at
# ~$26.30/hr (vs ~$40-60 on-demand). Pinning to one subnet prevents the
# nodegroup from drifting to other AZs and missing the spot pool.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

locals {
  cluster_name     = "qwen3-next-bench-eks-cluster"
  region           = "us-east-2"
  spot_az          = "us-east-2b"
  spot_subnet_id   = "subnet-03d03f1fb8d62d6a5"
  kimi_bucket_name = "kimi-k2-bench-models-20260216163240701700000006"
  project          = "ai-infra-use2"
  instance_type    = "p6-b200.48xlarge"
}

data "aws_eks_cluster" "this" {
  name = local.cluster_name
}

data "aws_caller_identity" "current" {}

###############################################################################
# ECR repos for slim images.
###############################################################################

resource "aws_ecr_repository" "vllm_slim" {
  name                 = "ai-infra-use2/vllm-slim"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  tags = { Project = local.project, Engine = "vllm" }
}

resource "aws_ecr_repository" "sglang_slim" {
  name                 = "ai-infra-use2/sglang-slim"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  tags = { Project = local.project, Engine = "sglang" }
}

resource "aws_ecr_lifecycle_policy" "vllm_slim" {
  repository = aws_ecr_repository.vllm_slim.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "sglang_slim" {
  repository = aws_ecr_repository.sglang_slim.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

###############################################################################
# IAM role for the B300 nodegroup. The cluster's existing nodegroup role
# can be reused, but we make a dedicated one so we can scope ECR + S3
# precisely without touching ray-ft-gpu's role.
###############################################################################

resource "aws_iam_role" "b200_node" {
  name = "${local.project}-b200-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = { Project = local.project }
}

resource "aws_iam_role_policy_attachment" "b300_eks_worker" {
  role       = aws_iam_role.b200_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "b300_eks_cni" {
  role       = aws_iam_role.b200_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "b300_ecr" {
  role       = aws_iam_role.b200_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "b300_ssm" {
  role       = aws_iam_role.b200_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Direct read on kimi-k2.6 model bucket for the init container.
resource "aws_iam_role_policy" "b300_kimi_s3" {
  name = "kimi-models-read"
  role = aws_iam_role.b200_node.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${local.kimi_bucket_name}",
        "arn:aws:s3:::${local.kimi_bucket_name}/*",
      ]
    }]
  })
}

###############################################################################
# B300 spot managed nodegroup, pinned to us-west-2b.
#
# Default desired_size=0 — costs nothing while idle. Scale to 1 via:
#   aws eks update-nodegroup-config --nodegroup-name ai-infra-b200-spot \
#     --scaling-config minSize=0,maxSize=1,desiredSize=1 ...
###############################################################################

resource "aws_eks_node_group" "b200_spot" {
  cluster_name    = local.cluster_name
  node_group_name = "${local.project}-b200-spot"
  node_role_arn   = aws_iam_role.b200_node.arn
  subnet_ids      = [local.spot_subnet_id]

  # Spot, single instance type so AWS doesn't pick something else.
  capacity_type  = "SPOT"
  instance_types = ["p6-b200.48xlarge"]

  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = 1
  }

  # Allow brief over-provision during scale-up; SDS to ignore tag drift.
  update_config {
    max_unavailable = 1
  }

  # Bigger root disk: container layers + model staging area.
  disk_size = 500

  # AL2023 with NVIDIA — must use AL2023 per memory (kernel ib_umad needed
  # for B300 Fabric Manager). Latest AMI ID via SSM at apply-time.
  ami_type = "AL2023_x86_64_NVIDIA"

  labels = {
    "ai-infra/role"     = "b200-spot"
    "node.kubernetes.io/instance-type" = "p6-b200.48xlarge"
  }

  taint {
    key    = "ai-infra/b200"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Project           = local.project
    AvailabilityZone  = local.spot_az
    "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

###############################################################################
# Outputs.
###############################################################################

output "vllm_ecr_uri" {
  value = aws_ecr_repository.vllm_slim.repository_url
}

output "sglang_ecr_uri" {
  value = aws_ecr_repository.sglang_slim.repository_url
}

output "registry_prefix" {
  value = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${local.region}.amazonaws.com/ai-infra"
}

output "b200_nodegroup_name" {
  value = aws_eks_node_group.b200_spot.node_group_name
}

output "scale_up_command" {
  value = <<-EOT
    aws eks update-nodegroup-config \
      --cluster-name ${local.cluster_name} \
      --nodegroup-name ${aws_eks_node_group.b200_spot.node_group_name} \
      --scaling-config minSize=0,maxSize=1,desiredSize=1 \
      --region ${local.region}
  EOT
}

output "scale_down_command" {
  value = <<-EOT
    aws eks update-nodegroup-config \
      --cluster-name ${local.cluster_name} \
      --nodegroup-name ${aws_eks_node_group.b200_spot.node_group_name} \
      --scaling-config minSize=0,maxSize=1,desiredSize=0 \
      --region ${local.region}
  EOT
}
