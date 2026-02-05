# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-eks-cluster"
  cluster_version = var.eks_cluster_version

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster access
  enable_cluster_creator_admin_permissions = true

  # Access entries for SageMaker (only when enabled)
  access_entries = var.enable_sagemaker_studio ? {
    sagemaker = {
      principal_arn = aws_iam_role.sagemaker_execution_role.arn
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

  # Node groups
  eks_managed_node_groups = {
    # System node group for core workloads
    system = {
      name           = "system"
      instance_types = ["m6i.large"]
      capacity_type  = "ON_DEMAND"

      min_size     = 2
      max_size     = 4
      desired_size = 2

      labels = {
        role = "system"
      }

      tags = {
        Name = "${var.project_name}-system-node"
      }
    }

    # GPU node group for ML inference
    gpu = {
      name           = "gpu"
      instance_types = var.eks_gpu_instance_types
      capacity_type  = "ON_DEMAND"
      ami_type       = "AL2_x86_64_GPU"

      # Use specific AZs with GPU capacity (us-east-1b has limited g6e availability)
      subnet_ids = [for i, az in var.availability_zones : module.vpc.private_subnets[i] if contains(var.gpu_availability_zones, az)]

      min_size     = var.enable_gpu_nodes ? var.eks_gpu_min_size : 0
      max_size     = var.enable_gpu_nodes ? var.eks_gpu_max_size : 0
      desired_size = var.enable_gpu_nodes ? var.eks_gpu_desired_size : 0

      labels = {
        role                     = "gpu"
        "nvidia.com/gpu.present" = "true"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      # GPU instances need larger root volume for model caching
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }

      tags = {
        Name = "${var.project_name}-gpu-node"
      }
    }
  }

  # Security group rules for nodes
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
  }

  # Security group rules for cluster (API server)
  cluster_security_group_additional_rules = var.enable_sagemaker_studio ? {
    ingress_sagemaker = {
      description              = "Allow SageMaker Studio to access EKS API"
      protocol                 = "tcp"
      from_port                = 443
      to_port                  = 443
      type                     = "ingress"
      source_security_group_id = aws_security_group.sagemaker_studio[0].id
    }
  } : {}

  tags = {
    Name = "${var.project_name}-eks-cluster"
  }
}

# IRSA for EBS CSI Driver
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.project_name}-ebs-csi-role"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = {
    Name = "${var.project_name}-ebs-csi-role"
  }
}

# Install NVIDIA Device Plugin
resource "helm_release" "nvidia_device_plugin" {
  count = var.enable_gpu_nodes ? 1 : 0

  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  namespace  = "kube-system"
  version    = "0.14.5"

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }

  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }

  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [module.eks]
}

# Kubernetes namespace for ML workloads
resource "kubernetes_namespace" "ml" {
  metadata {
    name = "ml-inference"
    labels = {
      name = "ml-inference"
    }
  }

  depends_on = [module.eks]
}
