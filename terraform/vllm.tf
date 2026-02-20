# vLLM Deployment on EKS for ML inference
# Supports: Ministral-3B, Nemotron-3-8B, KV cache offloading, FSx for Lustre

locals {
  # GPU subnet IDs (filtered for AZs with capacity)
  gpu_subnet_ids = [for i, az in var.availability_zones : module.vpc.private_subnets[i] if contains(var.gpu_availability_zones, az)]

  # Compute model path based on source (S3 or HuggingFace Hub)
  vllm_model_path = var.vllm_use_s3_model ? "s3://${aws_s3_bucket.models.id}/${var.vllm_s3_model_path}" : var.vllm_model_id

  # Image selection based on KV cache backend
  vllm_image = var.kv_cache_backend == "lmcache" ? "lmcache/vllm-openai:latest" : "vllm/vllm-openai:latest"

  # LMCache configuration presets
  lmcache_configs = {
    "cpu" = {
      local_device = "cpu"
      max_cache    = var.lmcache_max_cache_size_gb
    }
    "disk" = {
      local_device = "file:///tmp/lmcache/"
      max_cache    = var.lmcache_max_cache_size_gb
    }
    "fsx" = {
      local_device = "file:///mnt/fsx/lmcache/"
      max_cache    = var.lmcache_max_cache_size_gb
    }
  }
  active_lmcache_config = local.lmcache_configs[var.lmcache_config]

  # Whether LMCache is enabled
  use_lmcache = var.kv_cache_backend == "lmcache"

  # Whether LMCache needs disk storage (disk or fsx config)
  lmcache_needs_disk = local.use_lmcache && var.lmcache_config == "disk"
  lmcache_needs_fsx  = local.use_lmcache && var.lmcache_config == "fsx"

  # KV cache configuration lookup (for native vLLM offloading)
  # Resource requests sized for g6e.xlarge (4 vCPU, 16GB RAM → ~3.9 CPU, ~14GB allocatable)
  kv_configs = {
    "none" = {
      gpu_util       = var.vllm_gpu_memory_utilization
      cpu_offload_gb = 0
      swap_space_gb  = 0
      memory_request = "12Gi"
      cpu_request    = "2"
    }
    "cpu-light" = {
      gpu_util       = 0.85
      cpu_offload_gb = 4
      swap_space_gb  = 0
      memory_request = "12Gi"
      cpu_request    = "2"
    }
    "cpu-aggressive" = {
      gpu_util       = 0.7
      cpu_offload_gb = 8
      swap_space_gb  = 0
      memory_request = "12Gi"
      cpu_request    = "2"
    }
    "fsx-swap" = {
      gpu_util       = 0.85
      cpu_offload_gb = 0
      swap_space_gb  = 20
      memory_request = "12Gi"
      cpu_request    = "2"
    }
    "hybrid" = {
      gpu_util       = 0.7
      cpu_offload_gb = 4
      swap_space_gb  = 20
      memory_request = "12Gi"
      cpu_request    = "2"
    }
  }
  active_kv_config = local.kv_configs[var.kv_cache_config]

  # Determine if model needs mistral-specific args
  is_mistral_model = can(regex("(?i)mistral|ministral", var.vllm_model_id))

  # Base args for vLLM
  vllm_base_args = [
    "--model", local.vllm_model_path,
    "--port", "8000",
    "--gpu-memory-utilization", tostring(local.active_kv_config.gpu_util),
    "--max-model-len", tostring(var.vllm_max_model_len)
  ]

  # Mistral-specific args (only for Mistral/Ministral models)
  vllm_mistral_args = local.is_mistral_model ? [
    "--tokenizer_mode", "mistral",
    "--config_format", "mistral",
    "--load_format", "mistral"
  ] : []

  # KV cache offloading args (only for native vLLM, not LMCache)
  vllm_cpu_offload_args = !local.use_lmcache && local.active_kv_config.cpu_offload_gb > 0 ? [
    "--cpu-offload-gb", tostring(local.active_kv_config.cpu_offload_gb)
  ] : []

  vllm_swap_args = !local.use_lmcache && local.active_kv_config.swap_space_gb > 0 ? [
    "--swap-space", tostring(local.active_kv_config.swap_space_gb)
  ] : []

  # LMCache requires --enforce-eager flag
  vllm_lmcache_args = local.use_lmcache ? [
    "--enforce-eager"
  ] : []

  # Additional args for S3 loading via Run:ai Model Streamer (overrides load_format)
  vllm_s3_args = var.vllm_use_s3_model ? [
    "--load-format", "runai_streamer",
  ] : []

  # Combined args
  vllm_args = concat(
    local.vllm_base_args,
    local.vllm_mistral_args,
    local.vllm_cpu_offload_args,
    local.vllm_swap_args,
    local.vllm_lmcache_args,
    local.vllm_s3_args
  )

  # Determine if FSx mount is needed (native swap or LMCache FSx)
  use_fsx_mount = var.enable_fsx_lustre && (local.active_kv_config.swap_space_gb > 0 || local.lmcache_needs_fsx)
}

# IRSA for vLLM (to access models from S3 via Run:ai Model Streamer)
module "vllm_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  count = var.enable_vllm_deployment ? 1 : 0

  role_name = "${var.project_name}-vllm-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["ml-inference:vllm-sa"]
    }
  }

  role_policy_arns = {
    s3_read = aws_iam_policy.vllm_s3_policy[0].arn
  }

  tags = {
    Name = "${var.project_name}-vllm-role"
  }
}

# S3 policy for model storage access (specific bucket + KMS)
resource "aws_iam_policy" "vllm_s3_policy" {
  count = var.enable_vllm_deployment ? 1 : 0

  name        = "${var.project_name}-vllm-s3-policy"
  description = "Policy for vLLM to access model storage via Run:ai Model Streamer"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ModelBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket",
          "s3:HeadObject"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.models.arn
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-vllm-s3-policy"
  }
}

# LMCache ConfigMap (when using LMCache backend)
resource "kubernetes_config_map" "lmcache" {
  count = var.enable_vllm_deployment && local.use_lmcache ? 1 : 0

  metadata {
    name      = "lmcache-config"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  data = {
    "lmcache_config.yaml" = yamlencode({
      chunk_size           = var.lmcache_chunk_size
      local_device         = local.active_lmcache_config.local_device
      max_local_cache_size = local.active_lmcache_config.max_cache
    })
  }

  depends_on = [kubernetes_namespace.ml]
}

# ServiceAccount for vLLM
resource "kubernetes_service_account" "vllm" {
  count = var.enable_vllm_deployment ? 1 : 0

  metadata {
    name      = "vllm-sa"
    namespace = kubernetes_namespace.ml.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = module.vllm_irsa[0].iam_role_arn
    }
  }

  depends_on = [kubernetes_namespace.ml]
}

# PersistentVolumeClaim for model cache
resource "kubernetes_persistent_volume_claim" "vllm_cache" {
  count = var.enable_vllm_deployment ? 1 : 0

  metadata {
    name      = "vllm-model-cache"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = "gp3"

    resources {
      requests = {
        storage = "50Gi"
      }
    }
  }

  # Don't wait for the PVC to be bound since storage class is WaitForFirstConsumer
  wait_until_bound = false

  depends_on = [kubernetes_namespace.ml]
}

# vLLM Deployment
resource "kubernetes_deployment" "vllm" {
  count = var.enable_vllm_deployment ? 1 : 0

  # Don't wait for rollout - GPU pods take time to pull image and load model
  wait_for_rollout = false

  metadata {
    name      = "vllm-ministral"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = "vllm-ministral"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "vllm-ministral"
      }
    }

    template {
      metadata {
        labels = {
          app = "vllm-ministral"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.vllm[0].metadata[0].name

        # GPU node selector and tolerations
        node_selector = {
          "nvidia.com/gpu.present" = "true"
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        # Pod-level security context
        security_context {
          run_as_non_root = false # vLLM requires root for GPU access
          fs_group        = 1000
        }

        container {
          name  = "vllm"
          image = local.vllm_image

          # Container-level security context
          security_context {
            read_only_root_filesystem  = false # vLLM needs write access for model cache
            allow_privilege_escalation = false
            capabilities {
              drop = ["ALL"]
              add  = ["SYS_PTRACE"] # Required for GPU profiling
            }
          }

          args = local.vllm_args

          port {
            container_port = 8000
            name           = "http"
            protocol       = "TCP"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = "1"
              memory           = local.active_kv_config.memory_request
              cpu              = "4"
            }
            requests = {
              "nvidia.com/gpu" = "1"
              memory           = local.active_kv_config.memory_request
              cpu              = local.active_kv_config.cpu_request
            }
          }

          volume_mount {
            name       = "model-cache"
            mount_path = "/root/.cache/huggingface"
          }

          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }

          # FSx mount for KV cache swap (when enabled)
          dynamic "volume_mount" {
            for_each = local.use_fsx_mount ? [1] : []
            content {
              name       = "fsx-swap"
              mount_path = "/mnt/fsx"
            }
          }

          # LMCache config volume mount
          dynamic "volume_mount" {
            for_each = local.use_lmcache ? [1] : []
            content {
              name       = "lmcache-config"
              mount_path = "/etc/lmcache"
              read_only  = true
            }
          }

          # LMCache local disk cache (for disk backend)
          dynamic "volume_mount" {
            for_each = local.lmcache_needs_disk ? [1] : []
            content {
              name       = "lmcache-disk"
              mount_path = "/tmp/lmcache"
            }
          }

          env {
            name  = "HUGGING_FACE_HUB_TOKEN"
            value = "" # Set via secret if needed for gated models
          }

          # AWS region for S3 access via IRSA
          env {
            name  = "AWS_DEFAULT_REGION"
            value = var.aws_region
          }

          env {
            name  = "AWS_REGION"
            value = var.aws_region
          }

          # Run:ai Model Streamer configuration
          dynamic "env" {
            for_each = var.vllm_use_s3_model ? [1] : []
            content {
              name  = "RUNAI_STREAMER_MEMORY_LIMIT"
              value = "4Gi" # Limit memory used for streaming buffer
            }
          }

          # LMCache configuration file path
          dynamic "env" {
            for_each = local.use_lmcache ? [1] : []
            content {
              name  = "LMCACHE_CONFIG_FILE"
              value = "/etc/lmcache/lmcache_config.yaml"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 600 # 10 min for large model loading
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 300 # 5 min for model loading
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 6
          }
        }

        volume {
          name = "model-cache"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.vllm_cache[0].metadata[0].name
          }
        }

        volume {
          name = "shm"
          empty_dir {
            medium     = "Memory"
            size_limit = "16Gi"
          }
        }

        # FSx volume for KV cache swap (when enabled)
        dynamic "volume" {
          for_each = local.use_fsx_mount ? [1] : []
          content {
            name = "fsx-swap"
            persistent_volume_claim {
              claim_name = kubernetes_persistent_volume_claim.fsx[0].metadata[0].name
            }
          }
        }

        # LMCache config volume
        dynamic "volume" {
          for_each = local.use_lmcache ? [1] : []
          content {
            name = "lmcache-config"
            config_map {
              name = kubernetes_config_map.lmcache[0].metadata[0].name
            }
          }
        }

        # LMCache local disk cache volume
        dynamic "volume" {
          for_each = local.lmcache_needs_disk ? [1] : []
          content {
            name = "lmcache-disk"
            empty_dir {
              size_limit = "${var.lmcache_max_cache_size_gb}Gi"
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_namespace.ml,
    helm_release.nvidia_device_plugin
  ]
}

# Service for vLLM
resource "kubernetes_service" "vllm" {
  count = var.enable_vllm_deployment ? 1 : 0

  metadata {
    name      = "vllm-ministral"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = "vllm-ministral"
    }
  }

  spec {
    selector = {
      app = "vllm-ministral"
    }

    port {
      name        = "http"
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }

  depends_on = [kubernetes_namespace.ml]
}

# Optional: LoadBalancer service for external access
resource "kubernetes_service" "vllm_lb" {
  count = var.enable_vllm_deployment ? 1 : 0

  metadata {
    name      = "vllm-ministral-lb"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = "vllm-ministral"
    }
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-type"            = "nlb"
      "service.beta.kubernetes.io/aws-load-balancer-scheme"          = "internal"
      "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
    }
  }

  spec {
    selector = {
      app = "vllm-ministral"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 8000
      protocol    = "TCP"
    }

    type = "LoadBalancer"
  }

  depends_on = [kubernetes_namespace.ml]
}

# HorizontalPodAutoscaler for vLLM (optional)
resource "kubernetes_horizontal_pod_autoscaler_v2" "vllm" {
  count = var.enable_vllm_deployment ? 1 : 0

  metadata {
    name      = "vllm-ministral-hpa"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.vllm[0].metadata[0].name
    }

    min_replicas = 1
    max_replicas = 3

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }

  depends_on = [kubernetes_deployment.vllm]
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
