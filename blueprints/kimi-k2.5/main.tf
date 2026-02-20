# Kimi K2.5 Blueprint
# Deploys EKS with p5e.48xlarge GPU nodes, FSx Lustre, and vLLM for Kimi K2.5 inference

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
      Blueprint   = "kimi-k2.5"
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
  # Repo names must match those created by scripts/stage-images-ecr.sh.
  images = {
    fsx_csi_driver             = local.use_ecr ? "${var.ecr_registry}/fsx-csi-driver:v1.2.0" : "public.ecr.aws/fsx-csi-driver/aws-fsx-csi-driver:v1.2.0"
    csi_livenessprobe          = local.use_ecr ? "${var.ecr_registry}/csi-livenessprobe" : "public.ecr.aws/eks-distro/kubernetes-csi/livenessprobe"
    csi_node_driver_registrar  = local.use_ecr ? "${var.ecr_registry}/csi-node-driver-registrar" : "public.ecr.aws/eks-distro/kubernetes-csi/node-driver-registrar"
    csi_external_provisioner   = local.use_ecr ? "${var.ecr_registry}/csi-external-provisioner" : "public.ecr.aws/eks-distro/kubernetes-csi/external-provisioner"
    csi_external_resizer       = local.use_ecr ? "${var.ecr_registry}/csi-external-resizer" : "public.ecr.aws/eks-distro/kubernetes-csi/external-resizer"
    efa_device_plugin          = local.use_ecr ? "${var.ecr_registry}/aws-efa-k8s-device-plugin:v0.5.7" : "public.ecr.aws/eks/aws-efa-k8s-device-plugin:v0.5.7"
    dcgm_exporter              = local.use_ecr ? "${var.ecr_registry}/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04" : "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"
    prometheus                 = local.use_ecr ? "${var.ecr_registry}/prometheus:v2.54.1" : "quay.io/prometheus/prometheus:v2.54.1"
    prometheus_config_reloader = local.use_ecr ? "${var.ecr_registry}/prometheus-config-reloader:v0.76.0" : "quay.io/prometheus-operator/prometheus-config-reloader:v0.76.0"
    prometheus_operator        = local.use_ecr ? "${var.ecr_registry}/prometheus-operator:v0.76.0" : "quay.io/prometheus-operator/prometheus-operator:v0.76.0"
    grafana                    = local.use_ecr ? "${var.ecr_registry}/grafana:11.2.0" : "docker.io/grafana/grafana:11.2.0"
    grafana_sidecar            = local.use_ecr ? "${var.ecr_registry}/grafana-sidecar:1.27.5" : "quay.io/kiwigrid/k8s-sidecar:1.27.5"
    kube_state_metrics         = local.use_ecr ? "${var.ecr_registry}/kube-state-metrics:v2.13.0" : "registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.13.0"
    alertmanager               = local.use_ecr ? "${var.ecr_registry}/alertmanager:v0.27.0" : "quay.io/prometheus/alertmanager:v0.27.0"
    node_exporter              = local.use_ecr ? "${var.ecr_registry}/node-exporter:v1.8.2" : "quay.io/prometheus/node-exporter:v1.8.2"
  }

  # Filter subnets for GPU nodes (p5e capacity block AZ)
  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]

  # Single subnet for FSx (same AZ as GPU for latency)
  fsx_subnet_id = local.gpu_subnet_ids[0]

  # Kimi K2.5 specific vLLM args (always enabled, no toggle)
  kimi_extra_args = [
    "--trust-remote-code",
    "--enforce-eager",
    "--mm-encoder-tp-mode", "data",
    "--tool-call-parser", "kimi_k2",
    "--reasoning-parser", "kimi_k2",
  ]

  # Tensor parallel and prefix caching args
  tp_args = var.vllm_tensor_parallel_size > 1 ? [
    "--tensor-parallel-size", tostring(var.vllm_tensor_parallel_size)
  ] : []

  prefix_caching_args = var.vllm_enable_prefix_caching ? [
    "--enable-prefix-caching"
  ] : []

  # Swap space for MoE memory variability (zero overhead when not triggered)
  # Default 32GB is safe for MoE models — only activates during preemptions
  swap_args = ["--swap-space", var.vllm_swap_space_gb > 0 ? tostring(var.vllm_swap_space_gb) : "32"]

  # Combined extra args for the vLLM module
  vllm_extra_args = concat(
    local.kimi_extra_args,
    local.tp_args,
    local.prefix_caching_args,
    local.swap_args,
    ["--disable-log-requests"]
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

  # System nodes
  system_instance_types = ["m6i.xlarge"]
  system_desired_size   = 2

  # GPU nodes - p5e.48xlarge managed via capacity blocks
  enable_gpu_nodes   = true
  gpu_instance_types = var.gpu_instance_types
  gpu_subnet_ids     = local.gpu_subnet_ids
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size
  gpu_volume_size    = var.gpu_volume_size
  gpu_ami_type       = var.gpu_ami_type
  enable_efa         = var.enable_efa

  nvidia_device_plugin_image = var.nvidia_device_plugin_image

  tags = local.tags
}

# FSx for Lustre (model storage and KV cache)
# PERSISTENT_2 with EFA for GPUDirect Storage, LZ4 compression, version 2.15 for GDS
module "fsx_lustre" {
  source = "../../modules/fsx-lustre"
  count  = var.enable_fsx_lustre ? 1 : 0

  project_name                = var.project_name
  vpc_id                      = module.networking.vpc_id
  subnet_id                   = local.fsx_subnet_id
  storage_capacity_gib        = var.fsx_storage_capacity
  deployment_type             = var.fsx_deployment_type
  per_unit_storage_throughput = var.fsx_per_unit_throughput
  efa_enabled                 = true
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

  # Container images (resolved via local.images — uses ECR when ecr_registry is set)
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
    deploymentType           = var.fsx_deployment_type
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

# IRSA for vLLM (S3 + FSx access)
module "vllm_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  count = var.enable_vllm ? 1 : 0

  role_name = "${var.project_name}-vllm-role"

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

# Service Account for vLLM
resource "kubernetes_service_account" "vllm" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-sa"
    namespace = kubernetes_namespace.ml.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = module.vllm_irsa[0].iam_role_arn
    }
  }
}

# FSx PV (static provisioning)
resource "kubernetes_persistent_volume" "fsx" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name = "vllm-kimi-fsx-pv"
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
    name      = "vllm-kimi-fsx-pvc"
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

# Model cache PVC (local EBS for HuggingFace cache)
resource "kubernetes_persistent_volume_claim" "model_cache" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-kimi-model-cache"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = "gp3"

    resources {
      requests = {
        storage = "500Gi"
      }
    }
  }

  wait_until_bound = false
}

# vLLM Deployment - Kimi K2.5
resource "kubernetes_deployment" "vllm_kimi" {
  count            = var.enable_vllm ? 1 : 0
  wait_for_rollout = false

  metadata {
    name      = "vllm-kimi-k2"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app   = "vllm-kimi-k2"
      model = "kimi-k2-5"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "vllm-kimi-k2"
      }
    }

    template {
      metadata {
        labels = {
          app   = "vllm-kimi-k2"
          model = "kimi-k2-5"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.vllm[0].metadata[0].name

        # GPU node affinity
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

        # GPU toleration
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

          args = concat(
            [
              "--model", var.vllm_model_id,
              "--port", "8000",
              "--gpu-memory-utilization", tostring(var.vllm_gpu_memory_utilization),
              "--max-model-len", tostring(var.vllm_max_model_len),
            ],
            var.vllm_cpu_offload_gb > 0 ? ["--cpu-offload-gb", tostring(var.vllm_cpu_offload_gb)] : [],
            local.vllm_extra_args
          )

          # Kimi K2.5 and vLLM environment variables
          env {
            name  = "NCCL_TIMEOUT"
            value = "1800"
          }

          env {
            name  = "HF_HUB_OFFLINE"
            value = "1"
          }

          env {
            name  = "TRANSFORMERS_OFFLINE"
            value = "1"
          }

          env {
            name  = "VLLM_ATTENTION_BACKEND"
            value = "FLASHINFER"
          }

          # EFA transport for NCCL (p5e uses EFA v2)
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
              "nvidia.com/gpu" = var.vllm_gpu_count
            }
            requests = {
              "nvidia.com/gpu" = var.vllm_gpu_count
              cpu              = var.vllm_cpu_request
              memory           = var.vllm_memory_request
            }
          }

          # Model cache mount
          volume_mount {
            name       = "model-cache"
            mount_path = "/root/.cache"
          }

          # Shared memory
          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }

          # FSx Lustre mount
          dynamic "volume_mount" {
            for_each = var.enable_fsx_lustre ? [1] : []
            content {
              name       = "fsx-lustre"
              mount_path = "/mnt/fsx"
            }
          }

          # NVMe RAID0 mount (14x faster model loading than FSx)
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

        # Model cache volume
        volume {
          name = "model-cache"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.model_cache[0].metadata[0].name
          }
        }

        # Shared memory volume (large for TP=8)
        volume {
          name = "shm"
          empty_dir {
            medium     = "Memory"
            size_limit = "64Gi"
          }
        }

        # FSx Lustre volume
        dynamic "volume" {
          for_each = var.enable_fsx_lustre ? [1] : []
          content {
            name = "fsx-lustre"
            persistent_volume_claim {
              claim_name = kubernetes_persistent_volume_claim.fsx[0].metadata[0].name
            }
          }
        }

        # NVMe RAID0 hostPath (ephemeral local storage on GPU node)
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

# vLLM ClusterIP Service
resource "kubernetes_service" "vllm" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-kimi-k2"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = "vllm-kimi-k2"
    }

    port {
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

# vLLM NodePort Service (for direct access via capacity block instance)
resource "kubernetes_service" "vllm_nodeport" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-kimi-k2-nodeport"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = "vllm-kimi-k2"
    }

    port {
      port        = 8000
      target_port = 8000
      node_port   = var.vllm_node_port
      protocol    = "TCP"
    }

    type = "NodePort"
  }
}

# Security group for GPU nodes (allow NodePort access)
resource "aws_security_group" "gpu_nodeport" {
  name_prefix = "${var.project_name}-gpu-nodeport-"
  vpc_id      = module.networking.vpc_id
  description = "Allow NodePort access to vLLM on GPU nodes"

  ingress {
    description = "vLLM NodePort"
    from_port   = var.vllm_node_port
    to_port     = var.vllm_node_port
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

# Prometheus monitoring stack
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0

  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "62.7.0"

  create_namespace = true

  set {
    name  = "prometheus.prometheusSpec.scrapeInterval"
    value = var.prometheus_scrape_interval
  }

  set {
    name  = "prometheus.prometheusSpec.scrapeTimeout"
    value = var.prometheus_scrape_interval
  }

  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  # Container images (resolved via local.images — uses ECR when ecr_registry is set)
  set {
    name  = "prometheus.prometheusSpec.image.repository"
    value = split(":", local.images.prometheus)[0]
  }

  set {
    name  = "prometheus.prometheusSpec.image.tag"
    value = split(":", local.images.prometheus)[1]
  }

  set {
    name  = "prometheusOperator.image.repository"
    value = split(":", local.images.prometheus_operator)[0]
  }

  set {
    name  = "prometheusOperator.image.tag"
    value = split(":", local.images.prometheus_operator)[1]
  }

  set {
    name  = "prometheusOperator.prometheusConfigReloader.image.repository"
    value = split(":", local.images.prometheus_config_reloader)[0]
  }

  set {
    name  = "prometheusOperator.prometheusConfigReloader.image.tag"
    value = split(":", local.images.prometheus_config_reloader)[1]
  }

  set {
    name  = "alertmanager.alertmanagerSpec.image.repository"
    value = split(":", local.images.alertmanager)[0]
  }

  set {
    name  = "alertmanager.alertmanagerSpec.image.tag"
    value = split(":", local.images.alertmanager)[1]
  }

  set {
    name  = "kube-state-metrics.image.repository"
    value = split(":", local.images.kube_state_metrics)[0]
  }

  set {
    name  = "kube-state-metrics.image.tag"
    value = split(":", local.images.kube_state_metrics)[1]
  }

  set {
    name  = "prometheus-node-exporter.image.repository"
    value = split(":", local.images.node_exporter)[0]
  }

  set {
    name  = "prometheus-node-exporter.image.tag"
    value = split(":", local.images.node_exporter)[1]
  }

  set {
    name  = "grafana.enabled"
    value = "true"
  }

  set {
    name  = "grafana.adminPassword"
    value = var.grafana_admin_password
  }

  set {
    name  = "grafana.image.repository"
    value = split(":", local.images.grafana)[0]
  }

  set {
    name  = "grafana.image.tag"
    value = split(":", local.images.grafana)[1]
  }

  set {
    name  = "grafana.sidecar.image.repository"
    value = split(":", local.images.grafana_sidecar)[0]
  }

  set {
    name  = "grafana.sidecar.image.tag"
    value = split(":", local.images.grafana_sidecar)[1]
  }

  depends_on = [module.eks]
}

# AWS EFA Device Plugin (exposes EFA interfaces to pods)
resource "helm_release" "efa_device_plugin" {
  count = var.enable_efa ? 1 : 0

  name       = "aws-efa-k8s-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  namespace  = "kube-system"

  # Container image (resolved via local.images — uses ECR when ecr_registry is set)
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

# NVIDIA DCGM Exporter (GPU telemetry: utilization, memory, power, thermal)
resource "helm_release" "dcgm_exporter" {
  count = var.enable_monitoring ? 1 : 0

  name       = "dcgm-exporter"
  repository = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart      = "dcgm-exporter"
  namespace  = "monitoring"

  create_namespace = true

  # Container image (resolved via local.images — uses ECR when ecr_registry is set)
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

  set {
    name  = "serviceMonitor.interval"
    value = var.prometheus_scrape_interval
  }

  depends_on = [module.eks, helm_release.prometheus]
}

# ServiceMonitor for vLLM metrics
resource "kubernetes_manifest" "vllm_service_monitor" {
  count = var.enable_vllm && var.enable_monitoring ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "vllm-kimi-k2"
      namespace = kubernetes_namespace.ml.metadata[0].name
      labels = {
        app = "vllm-kimi-k2"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app = "vllm-kimi-k2"
        }
      }
      endpoints = [
        {
          port     = "http"
          interval = var.prometheus_scrape_interval
          path     = "/metrics"
        }
      ]
    }
  }

  depends_on = [helm_release.prometheus, kubernetes_service.vllm]
}

# IAM Instance Profile for capacity block EC2 instances
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

# Capacity block EC2 instance (launched manually or via CLI)
# user_data bootstraps the node into EKS, sets up NVMe RAID, mounts FSx with tuning, installs GDS
locals {
  gpu_user_data = base64encode(templatefile("${path.module}/templates/user_data.sh.tftpl", {
    cluster_name   = module.eks.cluster_name
    fsx_dns        = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : ""
    fsx_mount_name = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : ""
  }))
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

# S3 Bucket for model storage (source of truth, AZ-independent)
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

# Data Repository Association: S3 <-> FSx auto-sync
# Models uploaded to S3 are auto-imported to FSx /models/ path
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

  tags = local.tags
}
