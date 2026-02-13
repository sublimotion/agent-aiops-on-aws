# EKS Cluster Module - Cluster, Node Groups, Addons, NVIDIA Plugin

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-eks-cluster"
  cluster_version = var.cluster_version

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

  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnet_ids

  # Cluster access
  enable_cluster_creator_admin_permissions = true

  # Access entries (e.g., for SageMaker)
  access_entries = var.access_entries

  # Node groups
  eks_managed_node_groups = merge(
    # System node group
    {
      system = {
        name           = "system"
        instance_types = var.system_instance_types
        capacity_type  = "ON_DEMAND"

        min_size     = var.system_min_size
        max_size     = var.system_max_size
        desired_size = var.system_desired_size

        labels = {
          role = "system"
        }

        tags = {
          Name = "${var.project_name}-system-node"
        }
      }
    },
    # GPU node group (conditional)
    var.enable_gpu_nodes ? {
      gpu = {
        name           = "gpu"
        instance_types = var.gpu_instance_types
        capacity_type  = "ON_DEMAND"
        ami_type       = var.gpu_ami_type

        # Use specific subnets for GPU nodes (AZ filtering)
        subnet_ids = var.gpu_subnet_ids != null ? var.gpu_subnet_ids : var.private_subnet_ids

        min_size     = var.gpu_min_size
        max_size     = var.gpu_max_size
        desired_size = var.gpu_desired_size

        labels = merge(
          {
            role                     = "gpu"
            "nvidia.com/gpu.present" = "true"
          },
          var.enable_efa ? { "vpc.amazonaws.com/efa.present" = "true" } : {}
        )

        taints = [
          {
            key    = "nvidia.com/gpu"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        ]

        # GPU instances need larger root volume
        block_device_mappings = {
          xvda = {
            device_name = "/dev/xvda"
            ebs = {
              volume_size           = var.gpu_volume_size
              volume_type           = "gp3"
              encrypted             = true
              delete_on_termination = true
            }
          }
        }

        # EFA configuration for p5e instances
        network_interfaces = var.enable_efa ? [
          {
            delete_on_termination       = true
            device_index                = 0
            associate_public_ip_address = false
            interface_type              = "efa"
          }
        ] : []

        tags = {
          Name = "${var.project_name}-gpu-node"
        }
      }
    } : {}
  )

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
  cluster_security_group_additional_rules = var.cluster_security_group_additional_rules

  tags = var.tags
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

  tags = var.tags
}

# Install NVIDIA Device Plugin
resource "helm_release" "nvidia_device_plugin" {
  count = var.enable_gpu_nodes ? 1 : 0

  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  namespace  = "kube-system"
  version    = "0.14.5"

  values = [
    yamlencode({
      image = {
        repository = split(":", var.nvidia_device_plugin_image)[0]
        tag        = split(":", var.nvidia_device_plugin_image)[1]
      }
      nodeSelector = {
        "nvidia.com/gpu.present" = "true"
      }
      tolerations = [
        {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      ]
    })
  ]

  depends_on = [module.eks]
}

# Storage class for gp3 volumes
resource "kubernetes_storage_class" "gp3" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "true"
    }
  }

  storage_provisioner = "ebs.csi.aws.com"
  reclaim_policy      = "Delete"
  volume_binding_mode = "WaitForFirstConsumer"

  parameters = {
    type      = "gp3"
    encrypted = "true"
  }

  depends_on = [module.eks]
}
