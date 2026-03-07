# GLM-5 LMCache Blueprint
# Deploys EKS with p5e.48xlarge, FSx Lustre, SGLang + LMCache for GLM-5 FP8 inference
# Phase 1: Single instance, 3-tier KV cache benchmark

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
      Blueprint   = "glm5-lmcache"
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
    prometheus                = local.use_ecr ? "${var.ecr_registry}/prometheus:v2.54.1" : "quay.io/prometheus/prometheus:v2.54.1"
  }

  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]

  fsx_subnet_id = local.gpu_subnet_ids[0]

  app_label = "sglang-glm5"

  # SGLang launch args
  sglang_base_args = [
    "--model-path", var.model_path,
    "--tp-size", tostring(var.tp_size),
    "--context-length", tostring(var.context_length),
    "--chunked-prefill-size", "32768",
    "--max-running-requests", tostring(var.max_running_requests),
    "--mem-fraction-static", tostring(var.mem_fraction_static),
    "--served-model-name", "glm5",
    "--port", "30000",
  ]

  lmcache_args = var.enable_lmcache ? ["--enable-lmcache"] : []

  sglang_args = concat(local.sglang_base_args, local.lmcache_args)
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

  system_instance_types = ["m6i.xlarge"]
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
      namespace_service_accounts = ["ml-inference:sglang-sa"]
    }
  }

  role_policy_arns = var.enable_fsx_lustre ? {
    fsx_access = module.fsx_lustre[0].iam_policy_arn
  } : {}

  tags = local.tags
}

# Service Account
resource "kubernetes_service_account" "sglang" {
  count = var.enable_serving ? 1 : 0

  metadata {
    name      = "sglang-sa"
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
    name = "glm5-fsx-pv"
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
    name      = "glm5-fsx-pvc"
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

# LMCache ConfigMap
resource "kubernetes_config_map" "lmcache" {
  count = var.enable_serving ? 1 : 0

  metadata {
    name      = "lmcache-config"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  data = {
    "config.yaml" = var.lmcache_config_content
  }
}

# SGLang Deployment
resource "kubernetes_deployment" "sglang_glm5" {
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
    replicas = 1

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
          "prometheus.io/port"   = "30000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.sglang[0].metadata[0].name

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
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        container {
          name  = "sglang"
          image = var.sglang_image

          security_context {
            run_as_user                = 0
            run_as_non_root            = false
            allow_privilege_escalation = true
          }

          command = ["python3", "-m", "sglang.launch_server"]
          args    = local.sglang_args

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
              name  = "LMCACHE_CONFIG_FILE"
              value = "/etc/lmcache/config.yaml"
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
            container_port = 30000
            name           = "http"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = 8
            }
            requests = {
              "nvidia.com/gpu" = 8
              cpu              = "32"
              memory           = "128Gi"
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

          dynamic "volume_mount" {
            for_each = var.enable_lmcache ? [1] : []
            content {
              name       = "lmcache-config"
              mount_path = "/etc/lmcache"
              read_only  = true
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 30000
            }
            initial_delay_seconds = 600
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 30000
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

        dynamic "volume" {
          for_each = var.enable_lmcache ? [1] : []
          content {
            name = "lmcache-config"
            config_map {
              name = kubernetes_config_map.lmcache[0].metadata[0].name
            }
          }
        }
      }
    }
  }

  depends_on = [module.eks, kubernetes_namespace.ml]
}

# ClusterIP Service
resource "kubernetes_service" "sglang" {
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
      port        = 30000
      target_port = 30000
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

# NodePort Service
resource "kubernetes_service" "sglang_nodeport" {
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
      port        = 30000
      target_port = 30000
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
  description = "Allow NodePort access to SGLang on GPU nodes"

  ingress {
    description = "SGLang NodePort"
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
    value = "false"
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
    instance_type  = var.gpu_instance_types[0]
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

  timeouts {
    create = "20m"
  }
}
