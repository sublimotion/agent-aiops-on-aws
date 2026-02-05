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

# vLLM Deployment
module "vllm" {
  source = "../../modules/vllm"
  count  = var.enable_vllm ? 1 : 0

  project_name    = var.project_name
  namespace       = "ml-inference"
  deployment_name = "vllm-ministral"

  # Ministral-3B specific configuration
  model_id   = var.vllm_model_id
  vllm_image = var.vllm_image
  extra_args = [
    "--tokenizer_mode", "mistral",
    "--config_format", "mistral",
    "--load_format", "mistral"
  ]

  gpu_memory_utilization = var.vllm_gpu_memory_utilization
  max_model_len          = var.vllm_max_model_len

  # Resources
  replicas   = 1
  gpu_count  = 1
  enable_pvc = true
  pvc_size   = "50Gi"

  # Services
  enable_load_balancer   = true
  load_balancer_internal = false

  # HPA disabled for prototyping
  enable_hpa = false

  tags = local.tags

  depends_on = [module.eks]
}
