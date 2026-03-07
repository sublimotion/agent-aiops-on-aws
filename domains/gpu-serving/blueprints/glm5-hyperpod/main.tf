# GLM-5 on SageMaker HyperPod EKS Blueprint
# Deploys HyperPod cluster with p5e/p6-b200 GPU nodes, FSx Lustre, and SGLang for GLM-5 inference
# Uses upstream hyperpod-eks-tf module — handles VPC, EKS, HyperPod, FSx, observability

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = "~> 1.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "glm5-hyperpod"
      Environment = var.environment
      ManagedBy   = "terraform"
      Blueprint   = "glm5-hyperpod"
    }
  }
}

provider "awscc" {
  region = var.aws_region
}

locals {
  tags = {
    Project     = "glm5-hyperpod"
    Environment = var.environment
  }

  # ECR URI construction
  account_id = data.aws_caller_identity.current.account_id
  ecr_uri    = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

data "aws_caller_identity" "current" {}

# =============================================================================
# HyperPod EKS Module (upstream)
# Handles: VPC, EKS, HyperPod cluster, FSx, observability, inference operator
# =============================================================================

module "hyperpod" {
  source = "git::https://github.com/aws-samples/awsome-distributed-training.git//1.architectures/7.sagemaker-hyperpod-eks/terraform-modules/hyperpod-eks-tf?ref=main"

  # Core
  kubernetes_version    = var.kubernetes_version
  eks_cluster_name      = var.eks_cluster_name
  hyperpod_cluster_name = var.hyperpod_cluster_name
  resource_name_prefix  = var.resource_name_prefix
  aws_region            = var.aws_region

  # GPU instance group with Training Plan support
  instance_groups = var.instance_groups

  # Operators
  create_hyperpod_inference_operator_module = true
  create_hyperpod_training_operator_module  = false
  create_task_governance_module             = true

  # FSx Lustre
  create_fsx_module         = true
  create_new_fsx_filesystem = true
  fsx_storage_capacity      = var.fsx_storage_capacity
  fsx_throughput            = var.fsx_throughput

  # Observability — managed AMP/AMG
  create_observability_module      = true
  accelerated_compute_metric_level = "ADVANCED"
  network_metric_level             = "ADVANCED"
  node_metric_level                = "ADVANCED"
  cluster_metric_level             = "ADVANCED"
  logging_enabled                  = true

  # Closed network (air-gap)
  closed_network              = var.closed_network
  eks_endpoint_public_access  = !var.closed_network
  eks_endpoint_private_access = true
  create_s3_endpoint          = true
  create_ec2_endpoint         = true
  create_ecr_api_endpoint     = true
  create_ecr_dkr_endpoint     = true
  create_sts_endpoint         = true
  create_logs_endpoint        = true
  create_monitoring_endpoint  = true
  create_ssm_endpoint         = true
  create_ssmmessages_endpoint = true
  create_ec2messages_endpoint = true
  create_eks_auth_endpoint    = true

  # Deep health checks + auto-recovery
  enable_deep_health_check = true
  enable_job_auto_restart  = true
  auto_node_recovery       = true

  # Helm chart path (required for closed network)
  helm_repo_path = var.helm_repo_path
}

# =============================================================================
# S3 Buckets
# =============================================================================

# Model weights — GLM-5-FP8 (756 GB)
resource "aws_s3_bucket" "model_weights" {
  bucket_prefix = "${var.resource_name_prefix}-models-"
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "model_weights" {
  bucket = aws_s3_bucket.model_weights.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_weights" {
  bucket = aws_s3_bucket.model_weights.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# Benchmark results
resource "aws_s3_bucket" "benchmark_results" {
  bucket_prefix = "${var.resource_name_prefix}-results-"
  force_destroy = true
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "benchmark_results" {
  bucket = aws_s3_bucket.benchmark_results.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "benchmark_results" {
  bucket = aws_s3_bucket.benchmark_results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# =============================================================================
# ECR Repositories
# =============================================================================

resource "aws_ecr_repository" "sglang_glm5" {
  name                 = "${var.resource_name_prefix}/sglang-glm5"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.tags
}

resource "aws_ecr_repository" "bench_runner" {
  name                 = "${var.resource_name_prefix}/bench-runner"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.tags
}
