###############################################################################
# Ray Serve Fault Tolerance — Infrastructure
# Reuses existing qn-sglang-eks-cluster, adds:
#   1. ElastiCache Serverless (Redis) for GCS state persistence
#   2. System nodegroup (m6i.xlarge x2) for head pod + KubeRay operator
#   3. GPU nodegroup (g5.xlarge x2) for YOLO replicas
#
# Note: Ray's C++ Redis client does NOT support TLS. ElastiCache Serverless
# enforces TLS. Each pod runs a stunnel sidecar that proxies
# localhost:6380 (plain TCP) → ElastiCache endpoint (TLS).
# The endpoint is hardcoded in k8s/stunnel.yaml — update it after
# terraform apply using: terraform output -raw redis_endpoint
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
  region = var.region
}

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

data "aws_eks_cluster_auth" "this" {
  name = var.eks_cluster_name
}

# Look up the private subnets from the existing VPC
data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_eks_cluster.this.vpc_config[0].vpc_id]
  }
  filter {
    name   = "tag:Name"
    values = ["*private*"]
  }
}

###############################################################################
# ElastiCache Serverless (Redis) for GCS Fault Tolerance
###############################################################################

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = data.aws_eks_cluster.this.vpc_config[0].vpc_id
  description = "ElastiCache Serverless Redis for Ray GCS FT"

  ingress {
    description     = "Redis from EKS nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [data.aws_eks_cluster.this.vpc_config[0].cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-redis"
    Project = var.project_name
  }
}

resource "aws_elasticache_serverless_cache" "gcs" {
  engine = "redis"
  name   = "${var.project_name}-gcs"

  cache_usage_limits {
    data_storage {
      maximum = 1
      unit    = "GB"
    }
    ecpu_per_second {
      maximum = 1000
    }
  }

  security_group_ids = [aws_security_group.redis.id]
  subnet_ids         = data.aws_subnets.private.ids

  tags = {
    Project = var.project_name
  }
}

###############################################################################
# EKS Nodegroups
###############################################################################

# Reuse the existing node role from qn-sglang (created by Terraform)
data "aws_iam_role" "system_node" {
  name = var.system_node_role_name
}

data "aws_iam_role" "gpu_node" {
  name = var.gpu_node_role_name
}

# System nodegroup — head pod + KubeRay operator
resource "aws_eks_node_group" "system" {
  cluster_name    = var.eks_cluster_name
  node_group_name = "${var.project_name}-system"
  node_role_arn   = data.aws_iam_role.system_node.arn
  subnet_ids      = data.aws_subnets.private.ids

  instance_types = ["m6i.xlarge"]
  ami_type       = "AL2023_x86_64_STANDARD"
  disk_size      = 50  # runtime_env pip install needs ephemeral storage

  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 3
  }

  labels = {
    role = "system"
  }

  tags = {
    Name    = "${var.project_name}-system"
    Project = var.project_name
  }
}

# GPU nodegroup — YOLO replicas on g5.xlarge (1x A10G each)
resource "aws_eks_node_group" "gpu" {
  cluster_name    = var.eks_cluster_name
  node_group_name = "${var.project_name}-gpu"
  node_role_arn   = data.aws_iam_role.gpu_node.arn
  subnet_ids      = [data.aws_subnets.private.ids[0]] # single AZ for simplicity

  instance_types = ["g5.xlarge"]
  ami_type       = "AL2023_x86_64_NVIDIA"
  disk_size      = 100  # GPU images are large; default 20GB causes disk pressure

  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 2
  }

  labels = {
    role                      = "gpu"
    "nvidia.com/gpu.present"  = "true"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Name    = "${var.project_name}-gpu"
    Project = var.project_name
  }
}

###############################################################################
# Outputs
###############################################################################

output "redis_endpoint" {
  description = "ElastiCache Serverless endpoint — use in k8s/stunnel.yaml 'connect' field"
  value       = aws_elasticache_serverless_cache.gcs.endpoint[0].address
}

output "redis_port" {
  description = "ElastiCache Redis port (TLS-only, accessed via stunnel sidecar)"
  value       = aws_elasticache_serverless_cache.gcs.endpoint[0].port
}

output "stunnel_connect" {
  description = "Ready-to-paste value for stunnel.conf 'connect' line"
  value       = "${aws_elasticache_serverless_cache.gcs.endpoint[0].address}:${aws_elasticache_serverless_cache.gcs.endpoint[0].port}"
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = data.aws_eks_cluster.this.endpoint
}

output "gpu_nodegroup_status" {
  description = "GPU nodegroup status"
  value       = aws_eks_node_group.gpu.status
}

output "system_nodegroup_status" {
  description = "System nodegroup status"
  value       = aws_eks_node_group.system.status
}
