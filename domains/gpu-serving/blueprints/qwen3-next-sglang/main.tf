# Qwen3-Next SGLang + HiCache Blueprint
# Deploys EKS with g7e.24xlarge GPU nodes (4x Blackwell GB202), FSx Lustre, and SGLang
# Reuses existing S3 model bucket from qwen3-next blueprint
#
# Benchmark-focused: configs/ scripts run via nerdctl on GPU node, not K8s deployments.
# Terraform provides the infrastructure; serving happens in direct containers.

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Blueprint   = "qwen3-next-sglang"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }

  # Private ECR registry flag
  use_ecr = var.ecr_registry != ""

  # Image resolution: use ECR when set, otherwise default public registries.
  images = {
    fsx_csi_driver            = local.use_ecr ? "${var.ecr_registry}/fsx-csi-driver:v1.2.0" : "public.ecr.aws/fsx-csi-driver/aws-fsx-csi-driver:v1.2.0"
    csi_livenessprobe         = local.use_ecr ? "${var.ecr_registry}/csi-livenessprobe" : "public.ecr.aws/eks-distro/kubernetes-csi/livenessprobe"
    csi_node_driver_registrar = local.use_ecr ? "${var.ecr_registry}/csi-node-driver-registrar" : "public.ecr.aws/eks-distro/kubernetes-csi/node-driver-registrar"
    csi_external_provisioner  = local.use_ecr ? "${var.ecr_registry}/csi-external-provisioner" : "public.ecr.aws/eks-distro/kubernetes-csi/external-provisioner"
    csi_external_resizer      = local.use_ecr ? "${var.ecr_registry}/csi-external-resizer" : "public.ecr.aws/eks-distro/kubernetes-csi/external-resizer"
    dcgm_exporter             = local.use_ecr ? "${var.ecr_registry}/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04" : "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"
    prometheus                = local.use_ecr ? "${var.ecr_registry}/prometheus:v2.54.1" : "quay.io/prometheus/prometheus:v2.54.1"
    grafana                   = local.use_ecr ? "${var.ecr_registry}/grafana:11.2.0" : "docker.io/grafana/grafana:11.2.0"
  }

  # Filter subnets for GPU nodes
  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]

  # Single subnet for FSx (same AZ as GPU for latency)
  fsx_subnet_id = local.gpu_subnet_ids[0]
}

# Networking
module "networking" {
  source = "../../modules/networking"

  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  enable_nat_gateway = var.enable_nat_gateway
  tags               = local.tags
}

# EKS Cluster
module "eks" {
  source = "../../modules/eks-cluster"

  project_name       = var.project_name
  cluster_version    = var.eks_cluster_version
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnets

  # System nodes
  system_instance_types = ["m6i.xlarge"]
  system_desired_size   = 2

  # GPU nodes - g7e.24xlarge (4x Blackwell, PCIe)
  enable_gpu_nodes   = true
  gpu_instance_types = var.gpu_instance_types
  gpu_ami_type       = var.gpu_ami_type
  gpu_subnet_ids     = local.gpu_subnet_ids
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size
  gpu_volume_size    = var.gpu_volume_size

  # Post-bootstrap: NVMe RAID0 + FSx mount
  gpu_post_bootstrap_user_data = templatefile("${path.module}/templates/post_bootstrap.sh.tftpl", {
    fsx_dns        = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : ""
    fsx_mount_name = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : ""
  })

  nvidia_device_plugin_image = var.nvidia_device_plugin_image

  tags = local.tags
}

# FSx for Lustre (model storage)
# SCRATCH_2 — cheaper, sufficient for benchmark workloads
module "fsx_lustre" {
  source = "../../modules/fsx-lustre"
  count  = var.enable_fsx_lustre ? 1 : 0

  project_name                = var.project_name
  vpc_id                      = module.networking.vpc_id
  subnet_id                   = local.fsx_subnet_id
  storage_capacity_gib        = var.fsx_storage_capacity
  deployment_type             = var.fsx_deployment_type
  per_unit_storage_throughput = var.fsx_per_unit_throughput
  efa_enabled                 = false # g7e.24xl has no EFA; no GDS support
  data_compression_type       = "LZ4"
  file_system_type_version    = "2.15"
  allowed_security_group_ids  = [module.eks.node_security_group_id]
  tags                        = local.tags
}

# IRSA for FSx CSI Driver
module "fsx_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  count = var.enable_fsx_lustre ? 1 : 0

  role_name = "${var.project_name}-fsx-csi-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:fsx-csi-controller-sa"]
    }
  }

  role_policy_arns = {
    fsx = aws_iam_policy.fsx_csi[0].arn
  }

  tags = local.tags
}

resource "aws_iam_policy" "fsx_csi" {
  count = var.enable_fsx_lustre ? 1 : 0

  name_prefix = "${var.project_name}-fsx-csi-"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "fsx:DescribeFileSystems",
          "fsx:DescribeVolumes",
          "fsx:CreateVolume",
          "fsx:DeleteVolume",
          "fsx:TagResource"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeVpcs"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.tags
}

# FSx CSI Driver
resource "helm_release" "fsx_csi_driver" {
  count = var.enable_fsx_lustre ? 1 : 0

  name       = "aws-fsx-csi-driver"
  repository = "https://kubernetes-sigs.github.io/aws-fsx-csi-driver"
  chart      = "aws-fsx-csi-driver"
  namespace  = "kube-system"
  version    = "1.9.0"

  set {
    name  = "controller.serviceAccount.create"
    value = "true"
  }

  set {
    name  = "controller.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.fsx_csi_irsa[0].iam_role_arn
  }

  set {
    name  = "image.repository"
    value = split(":", local.images.fsx_csi_driver)[0]
  }

  set {
    name  = "image.tag"
    value = split(":", local.images.fsx_csi_driver)[1]
  }

  set {
    name  = "sidecars.livenessProbe.image.repository"
    value = local.images.csi_livenessprobe
  }

  set {
    name  = "sidecars.nodeDriverRegistrar.image.repository"
    value = local.images.csi_node_driver_registrar
  }

  set {
    name  = "sidecars.provisioner.image.repository"
    value = local.images.csi_external_provisioner
  }

  set {
    name  = "sidecars.resizer.image.repository"
    value = local.images.csi_external_resizer
  }

  # Tolerate GPU nodes for FSx node daemonset
  set {
    name  = "node.tolerations[0].key"
    value = "nvidia.com/gpu"
  }

  set {
    name  = "node.tolerations[0].operator"
    value = "Exists"
  }

  set {
    name  = "node.tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [module.eks]
}

# ML Namespace
resource "kubernetes_namespace" "ml" {
  metadata {
    name = "ml-inference"
    labels = {
      name = "ml-inference"
    }
  }

  depends_on = [module.eks]
}

# Model bucket: reference existing or create new in this region
data "aws_s3_bucket" "models_existing" {
  count  = var.model_s3_bucket_id != "" ? 1 : 0
  bucket = var.model_s3_bucket_id
}

resource "aws_s3_bucket" "models_new" {
  count         = var.model_s3_bucket_id == "" ? 1 : 0
  bucket_prefix = "${var.project_name}-models-"
  force_destroy = true
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "models_new" {
  count  = var.model_s3_bucket_id == "" ? 1 : 0
  bucket = aws_s3_bucket.models_new[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models_new" {
  count  = var.model_s3_bucket_id == "" ? 1 : 0
  bucket = aws_s3_bucket.models_new[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

locals {
  models_bucket_id = var.model_s3_bucket_id != "" ? data.aws_s3_bucket.models_existing[0].id : aws_s3_bucket.models_new[0].id
}

# Data Repository Association: model bucket <-> FSx auto-sync
resource "aws_fsx_data_repository_association" "models" {
  count = var.enable_fsx_lustre ? 1 : 0

  file_system_id       = module.fsx_lustre[0].file_system_id
  data_repository_path = "s3://${local.models_bucket_id}"
  file_system_path     = "/models"

  s3 {
    auto_import_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
    auto_export_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
  }

  batch_import_meta_data_on_create = true

  tags = local.tags
}

# S3 Bucket for benchmark results
resource "aws_s3_bucket" "benchmark_results" {
  bucket_prefix = "${var.project_name}-results-"
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

# No K8s serving deployment — benchmarks run via nerdctl containers directly.
# See configs/sglang-baseline.sh, configs/sglang-hicache-l2.sh, configs/sglang-hicache-nvme.sh
