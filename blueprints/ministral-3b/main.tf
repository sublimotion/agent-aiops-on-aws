# Ministral-3B Blueprint
# Deploys EKS with GPU nodes, SageMaker Studio, and vLLM serving Ministral-3B

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
      Blueprint   = "ministral-3b"
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

  # Filter subnets for GPU nodes (exclude AZs with limited capacity)
  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]
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
  system_instance_types = ["m6i.large"]
  system_desired_size   = 2

  # GPU nodes
  enable_gpu_nodes   = var.enable_gpu_nodes
  gpu_instance_types = var.gpu_instance_types
  gpu_subnet_ids     = local.gpu_subnet_ids
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size

  # SageMaker access
  access_entries = var.enable_sagemaker ? {
    sagemaker = {
      principal_arn = module.sagemaker[0].execution_role_arn
      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  } : {}

  cluster_security_group_additional_rules = var.enable_sagemaker ? {
    ingress_sagemaker = {
      description              = "Allow SageMaker Studio to access EKS API"
      protocol                 = "tcp"
      from_port                = 443
      to_port                  = 443
      type                     = "ingress"
      source_security_group_id = module.sagemaker[0].security_group_id
    }
  } : {}

  tags = local.tags
}

# SageMaker Studio (optional)
module "sagemaker" {
  source = "../../modules/sagemaker-studio"
  count  = var.enable_sagemaker ? 1 : 0

  project_name     = var.project_name
  domain_name      = "${var.project_name}-studio"
  vpc_id           = module.networking.vpc_id
  vpc_cidr         = var.vpc_cidr
  subnet_ids       = module.networking.private_subnets
  eks_cluster_name = module.eks.cluster_name
  tags             = local.tags
}

# FSx for Lustre (optional - for KV cache offloading benchmarks)
module "fsx_lustre" {
  source = "../../modules/fsx-lustre"
  count  = var.enable_fsx_lustre ? 1 : 0

  project_name               = var.project_name
  vpc_id                     = module.networking.vpc_id
  subnet_id                  = local.gpu_subnet_ids[0]
  storage_capacity_gib       = var.fsx_storage_capacity_gib
  deployment_type            = var.fsx_deployment_type
  allowed_security_group_ids = [module.eks.node_security_group_id]
  tags                       = local.tags

  depends_on = [module.eks]
}

# FSx CSI Driver (required when FSx is enabled)
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

  depends_on = [module.eks]
}

# KV cache configuration lookup
locals {
  kv_configs = {
    "none" = {
      gpu_util       = var.vllm_gpu_memory_utilization
      cpu_offload_gb = 0
      swap_space_gb  = 0
      memory_request = "16Gi"
    }
    "cpu-light" = {
      gpu_util       = 0.85
      cpu_offload_gb = 10
      swap_space_gb  = 0
      memory_request = "32Gi"
    }
    "cpu-aggressive" = {
      gpu_util       = 0.7
      cpu_offload_gb = 30
      swap_space_gb  = 0
      memory_request = "48Gi"
    }
    "fsx-swap" = {
      gpu_util       = 0.85
      cpu_offload_gb = 0
      swap_space_gb  = 50
      memory_request = "24Gi"
    }
    "hybrid" = {
      gpu_util       = 0.7
      cpu_offload_gb = 20
      swap_space_gb  = 50
      memory_request = "48Gi"
    }
  }
  active_kv_config = local.kv_configs[var.kv_cache_config]

  # Determine if model needs mistral-specific args
  is_mistral_model = can(regex("(?i)mistral|ministral", var.vllm_model_id))
  mistral_args = local.is_mistral_model ? [
    "--tokenizer_mode", "mistral",
    "--config_format", "mistral",
    "--load_format", "mistral"
  ] : []
}

# vLLM Deployment
module "vllm" {
  source = "../../modules/vllm"
  count  = var.enable_vllm ? 1 : 0

  project_name    = var.project_name
  namespace       = "ml-inference"
  deployment_name = "vllm-inference"

  # Model configuration (supports Ministral or Nemotron)
  model_id   = var.vllm_model_id
  vllm_image = var.vllm_image
  extra_args = local.mistral_args

  gpu_memory_utilization = local.active_kv_config.gpu_util
  max_model_len          = var.vllm_max_model_len

  # KV Cache offloading
  cpu_offload_gb  = local.active_kv_config.cpu_offload_gb
  swap_space_gb   = local.active_kv_config.swap_space_gb
  swap_space_path = var.enable_fsx_lustre && local.active_kv_config.swap_space_gb > 0 ? "/mnt/fsx/swap" : "/tmp"

  # FSx mount (when enabled and config uses swap)
  enable_fsx_mount = var.enable_fsx_lustre && local.active_kv_config.swap_space_gb > 0
  fsx_dns_name     = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : ""
  fsx_mount_name   = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : ""

  # Resources - sized for g6e.xlarge (4 vCPU, 16GB, ~3.9 allocatable CPU, ~14GB allocatable mem)
  replicas       = 1
  gpu_count      = 1
  cpu_request    = "2"
  memory_request = local.active_kv_config.memory_request
  enable_pvc     = true
  pvc_size       = "100Gi"

  # Services
  enable_load_balancer   = true
  load_balancer_internal = false

  # HPA disabled for prototyping
  enable_hpa = false

  tags = local.tags

  depends_on = [module.eks, helm_release.fsx_csi_driver]
}
