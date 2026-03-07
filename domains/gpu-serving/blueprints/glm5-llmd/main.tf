# GLM-5 llm-d Blueprint
# Multi-replica vLLM + LMCache on EKS with llm-d prefix-cache-aware routing
# Phase 2: Multi-replica, intelligent routing via Gateway API + EPP

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
      Blueprint   = "glm5-llmd"
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

  use_ecr = var.ecr_registry != ""

  images = {
    fsx_csi_driver            = local.use_ecr ? "${var.ecr_registry}/fsx-csi-driver:v1.2.0" : "public.ecr.aws/fsx-csi-driver/aws-fsx-csi-driver:v1.2.0"
    csi_livenessprobe         = local.use_ecr ? "${var.ecr_registry}/csi-livenessprobe" : "public.ecr.aws/eks-distro/kubernetes-csi/livenessprobe"
    csi_node_driver_registrar = local.use_ecr ? "${var.ecr_registry}/csi-node-driver-registrar" : "public.ecr.aws/eks-distro/kubernetes-csi/node-driver-registrar"
    csi_external_provisioner  = local.use_ecr ? "${var.ecr_registry}/csi-external-provisioner" : "public.ecr.aws/eks-distro/kubernetes-csi/external-provisioner"
    csi_external_resizer      = local.use_ecr ? "${var.ecr_registry}/csi-external-resizer" : "public.ecr.aws/eks-distro/kubernetes-csi/external-resizer"
    efa_device_plugin         = local.use_ecr ? "${var.ecr_registry}/aws-efa-k8s-device-plugin:v0.5.7" : "public.ecr.aws/eks/aws-efa-k8s-device-plugin:v0.5.7"
    dcgm_exporter             = local.use_ecr ? "${var.ecr_registry}/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04" : "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"
  }

  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]

  fsx_subnet_id = local.gpu_subnet_ids[0]

  app_label = "vllm-glm5"

  # vLLM launch args
  # Ref: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
  vllm_args = concat(
    [
      "--model", var.model_path,
      "--served-model-name", "glm5",
      "--tensor-parallel-size", tostring(var.tp_size),
      "--enable-prefix-caching",
      "--max-model-len", tostring(var.max_model_len),
      "--swap-space", "32",
      "--gpu-memory-utilization", tostring(var.gpu_memory_utilization),
      "--speculative-config.method", "mtp",
      "--speculative-config.num_speculative_tokens", "1",
      "--tool-call-parser", "glm47",
      "--reasoning-parser", "glm45",
      "--enable-auto-tool-choice",
      "--port", "8000",
      "--disable-log-requests",
    ],
    var.enable_lmcache ? [
      "--kv-transfer-config",
      jsonencode({
        kv_connector = "LMCacheConnectorV1"
        kv_role      = "kv_both"
      })
    ] : []
  )
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

  system_instance_types = ["m6i.24xlarge"]
  system_desired_size   = 2

  enable_gpu_nodes   = true
  gpu_instance_types = var.gpu_instance_types
  gpu_subnet_ids     = local.gpu_subnet_ids
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size
  gpu_volume_size    = var.gpu_volume_size

  nvidia_device_plugin_image = var.nvidia_device_plugin_image

  tags = local.tags
}

# FSx for Lustre
module "fsx_lustre" {
  source = "../../modules/fsx-lustre"
  count  = var.enable_fsx_lustre ? 1 : 0

  project_name                = var.project_name
  vpc_id                      = module.networking.vpc_id
  subnet_id                   = local.fsx_subnet_id
  storage_capacity_gib        = var.fsx_storage_capacity
  deployment_type             = "PERSISTENT_2"
  per_unit_storage_throughput = var.fsx_per_unit_throughput
  efa_enabled                 = false
  data_compression_type       = "LZ4"
  file_system_type_version    = "2.15"
  allowed_security_group_ids  = [module.eks.node_security_group_id]
  tags                        = local.tags

  depends_on = [module.eks]
}

# IRSA for FSx CSI Driver
module "fsx_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"
  count   = var.enable_fsx_lustre ? 1 : 0

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
  count       = var.enable_fsx_lustre ? 1 : 0
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

# FSx StorageClass
resource "kubernetes_storage_class" "fsx_lustre" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name = "fsx-lustre"
  }

  storage_provisioner = "fsx.csi.aws.com"
  reclaim_policy      = "Retain"
  volume_binding_mode = "Immediate"

  parameters = {
    subnetId                 = local.fsx_subnet_id
    securityGroupIds         = module.eks.node_security_group_id
    deploymentType           = "PERSISTENT_2"
    perUnitStorageThroughput = tostring(var.fsx_per_unit_throughput)
    dataCompressionType      = "LZ4"
  }

  depends_on = [helm_release.fsx_csi_driver]
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

# IRSA for serving engine
module "serving_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"
  count   = var.enable_serving ? 1 : 0

  role_name = "${var.project_name}-serving-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["ml-inference:vllm-sa"]
    }
  }

  role_policy_arns = var.enable_fsx_lustre ? {
    fsx_access = module.fsx_lustre[0].iam_policy_arn
  } : {}

  tags = local.tags
}

# Service Account
resource "kubernetes_service_account" "vllm" {
  count = var.enable_serving ? 1 : 0

  metadata {
    name      = "vllm-sa"
    namespace = kubernetes_namespace.ml.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = module.serving_irsa[0].iam_role_arn
    }
  }
}

# FSx PV
resource "kubernetes_persistent_volume" "fsx" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name = "glm5-llmd-fsx-pv"
  }

  spec {
    capacity = {
      storage = "${var.fsx_storage_capacity}Gi"
    }
    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = "fsx-lustre"

    persistent_volume_source {
      csi {
        driver        = "fsx.csi.aws.com"
        volume_handle = module.fsx_lustre[0].file_system_id
        volume_attributes = {
          dnsname   = module.fsx_lustre[0].dns_name
          mountname = module.fsx_lustre[0].mount_name
        }
      }
    }
  }

  depends_on = [kubernetes_storage_class.fsx_lustre]
}

# FSx PVC
resource "kubernetes_persistent_volume_claim" "fsx" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name      = "glm5-llmd-fsx-pvc"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = "fsx-lustre"
    volume_name        = kubernetes_persistent_volume.fsx[0].metadata[0].name

    resources {
      requests = {
        storage = "${var.fsx_storage_capacity}Gi"
      }
    }
  }
}

# vLLM Deployment (multi-replica)
resource "kubernetes_deployment" "vllm_glm5" {
  count            = var.enable_serving ? 1 : 0
  wait_for_rollout = false

  metadata {
    name      = local.app_label
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app   = local.app_label
      model = "glm5"
    }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        app = local.app_label
      }
    }

    template {
      metadata {
        labels = {
          app   = local.app_label
          model = "glm5"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.vllm[0].metadata[0].name

        affinity {
          node_affinity {
            required_during_scheduling_ignored_during_execution {
              node_selector_term {
                match_expressions {
                  key      = "nvidia.com/gpu.present"
                  operator = "In"
                  values   = ["true"]
                }
              }
            }
          }
          pod_anti_affinity {
            required_during_scheduling_ignored_during_execution {
              label_selector {
                match_labels = {
                  app = local.app_label
                }
              }
              topology_key = "kubernetes.io/hostname"
            }
          }
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        container {
          name  = "vllm"
          image = var.vllm_image

          security_context {
            run_as_user                = 0
            run_as_non_root            = false
            allow_privilege_escalation = true
          }

          args = local.vllm_args

          env {
            name  = "HF_HUB_OFFLINE"
            value = "1"
          }

          env {
            name  = "TRANSFORMERS_OFFLINE"
            value = "1"
          }

          env {
            name  = "NCCL_TIMEOUT"
            value = "1800"
          }

          env {
            name  = "VLLM_ATTENTION_BACKEND"
            value = "FLASHINFER"
          }

          env {
            name  = "FI_PROVIDER"
            value = "efa"
          }

          env {
            name  = "FI_EFA_USE_DEVICE_RDMA"
            value = "1"
          }

          env {
            name  = "NCCL_PROTO"
            value = "Simple"
          }

          dynamic "env" {
            for_each = var.enable_lmcache ? [1] : []
            content {
              name  = "LMCACHE_USE_EXPERIMENTAL"
              value = "True"
            }
          }

          dynamic "env" {
            for_each = var.enable_lmcache && var.lmcache_local_cpu ? [1] : []
            content {
              name  = "LMCACHE_LOCAL_CPU"
              value = "True"
            }
          }

          dynamic "env" {
            for_each = var.enable_lmcache && var.lmcache_local_cpu ? [1] : []
            content {
              name  = "LMCACHE_MAX_LOCAL_CPU_SIZE"
              value = tostring(var.lmcache_max_cpu_size)
            }
          }

          dynamic "env" {
            for_each = var.enable_lmcache && var.lmcache_redis_url != "" ? [1] : []
            content {
              name  = "LMCACHE_REMOTE_URL"
              value = var.lmcache_redis_url
            }
          }

          env {
            name = "HUGGING_FACE_HUB_TOKEN"
            value_from {
              secret_key_ref {
                name     = "hf-token"
                key      = "token"
                optional = true
              }
            }
          }

          port {
            container_port = 8000
            name           = "http"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = 8
            }
            requests = {
              "nvidia.com/gpu" = 8
              cpu              = "32"
              memory           = "192Gi"
            }
          }

          volume_mount {
            name       = "model-cache"
            mount_path = "/root/.cache"
          }

          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }

          dynamic "volume_mount" {
            for_each = var.enable_fsx_lustre ? [1] : []
            content {
              name       = "fsx-lustre"
              mount_path = "/mnt/fsx"
            }
          }

          volume_mount {
            name       = "nvme"
            mount_path = "/mnt/nvme"
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 600
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 300
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 5
          }
        }

        volume {
          name = "model-cache"
          empty_dir {}
        }

        volume {
          name = "shm"
          empty_dir {
            medium     = "Memory"
            size_limit = "64Gi"
          }
        }

        dynamic "volume" {
          for_each = var.enable_fsx_lustre ? [1] : []
          content {
            name = "fsx-lustre"
            persistent_volume_claim {
              claim_name = kubernetes_persistent_volume_claim.fsx[0].metadata[0].name
            }
          }
        }

        volume {
          name = "nvme"
          host_path {
            path = "/mnt/nvme"
            type = "Directory"
          }
        }
      }
    }
  }

  depends_on = [module.eks, kubernetes_namespace.ml]
}

# Headless Service (for EPP pod discovery)
resource "kubernetes_service" "vllm_headless" {
  count = var.enable_serving ? 1 : 0

  metadata {
    name      = local.app_label
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = local.app_label
    }

    port {
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
      name        = "http"
    }

    cluster_ip = "None"
  }
}

# NodePort Service (for direct access)
resource "kubernetes_service" "vllm_nodeport" {
  count = var.enable_serving ? 1 : 0

  metadata {
    name      = "${local.app_label}-nodeport"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = local.app_label
    }

    port {
      port        = 8000
      target_port = 8000
      node_port   = var.node_port
      protocol    = "TCP"
    }

    type = "NodePort"
  }
}

# Security group for GPU NodePort
resource "aws_security_group" "gpu_nodeport" {
  name_prefix = "${var.project_name}-gpu-nodeport-"
  vpc_id      = module.networking.vpc_id
  description = "Allow NodePort access to vLLM on GPU nodes"

  ingress {
    description = "vLLM NodePort"
    from_port   = var.node_port
    to_port     = var.node_port
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-gpu-nodeport"
  })
}

# EFA Device Plugin
resource "helm_release" "efa_device_plugin" {
  count = var.enable_efa ? 1 : 0

  name       = "aws-efa-k8s-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  namespace  = "kube-system"

  set {
    name  = "image.repository"
    value = split(":", local.images.efa_device_plugin)[0]
  }

  set {
    name  = "image.tag"
    value = split(":", local.images.efa_device_plugin)[1]
  }

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

# DCGM Exporter
resource "helm_release" "dcgm_exporter" {
  count = var.enable_monitoring ? 1 : 0

  name       = "dcgm-exporter"
  repository = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart      = "dcgm-exporter"
  namespace  = "monitoring"
  wait       = false

  create_namespace = true

  set {
    name  = "image.repository"
    value = split(":", local.images.dcgm_exporter)[0]
  }

  set {
    name  = "image.tag"
    value = split(":", local.images.dcgm_exporter)[1]
  }

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

  set {
    name  = "serviceMonitor.enabled"
    value = "true"
  }

  depends_on = [module.eks]
}

# GPU Node IAM Role
resource "aws_iam_role" "gpu_node" {
  name_prefix = "${var.project_name}-gpu-node-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "gpu_node_eks" {
  role       = aws_iam_role.gpu_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "gpu_node_ecr" {
  role       = aws_iam_role.gpu_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "gpu_node_cni" {
  role       = aws_iam_role.gpu_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "gpu_node_ssm" {
  role       = aws_iam_role.gpu_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu_node" {
  name_prefix = "${var.project_name}-gpu-node-"
  role        = aws_iam_role.gpu_node.name
  tags        = local.tags
}

locals {
  gpu_user_data = base64encode(templatefile("${path.module}/templates/user_data.sh.tftpl", {
    cluster_name   = module.eks.cluster_name
    fsx_dns        = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : ""
    fsx_mount_name = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : ""
  }))
}

# S3 Buckets
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

resource "aws_s3_bucket" "models" {
  bucket_prefix = "${var.project_name}-models-"
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# S3 → FSx Data Repository Association
resource "aws_fsx_data_repository_association" "models" {
  count = var.enable_fsx_lustre ? 1 : 0

  file_system_id       = module.fsx_lustre[0].file_system_id
  data_repository_path = "s3://${aws_s3_bucket.models.id}"
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
  tags                             = local.tags
}
