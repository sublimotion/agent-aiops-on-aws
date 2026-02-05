# vLLM Deployment Module - Kubernetes resources for vLLM inference

locals {
  # Merge default args with custom args
  vllm_args = concat(
    [
      "--model", var.model_id,
      "--port", "8000",
      "--gpu-memory-utilization", tostring(var.gpu_memory_utilization),
      "--max-model-len", tostring(var.max_model_len)
    ],
    var.extra_args
  )
}

# Namespace for ML workloads
resource "kubernetes_namespace" "ml" {
  metadata {
    name = var.namespace
    labels = {
      name = var.namespace
    }
  }
}

# IRSA for vLLM (S3 access for model storage)
module "vllm_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  count = var.enable_s3_access ? 1 : 0

  role_name = "${var.project_name}-vllm-role"

  oidc_providers = {
    main = {
      provider_arn               = var.oidc_provider_arn
      namespace_service_accounts = ["${var.namespace}:${var.service_account_name}"]
    }
  }

  role_policy_arns = var.s3_policy_arn != null ? {
    s3_read = var.s3_policy_arn
  } : {}

  tags = var.tags
}

# Service Account
resource "kubernetes_service_account" "vllm" {
  metadata {
    name      = var.service_account_name
    namespace = kubernetes_namespace.ml.metadata[0].name
    annotations = var.enable_s3_access ? {
      "eks.amazonaws.com/role-arn" = module.vllm_irsa[0].iam_role_arn
    } : {}
  }
}

# PVC for model cache (optional)
resource "kubernetes_persistent_volume_claim" "vllm_cache" {
  count = var.enable_pvc ? 1 : 0

  metadata {
    name      = "${var.deployment_name}-cache"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = var.storage_class_name

    resources {
      requests = {
        storage = var.pvc_size
      }
    }
  }

  wait_until_bound = false
}

# vLLM Deployment
resource "kubernetes_deployment" "vllm" {
  wait_for_rollout = false

  metadata {
    name      = var.deployment_name
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = var.deployment_name
    }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        app = var.deployment_name
      }
    }

    template {
      metadata {
        labels = {
          app = var.deployment_name
        }
      }

      spec {
        service_account_name = kubernetes_service_account.vllm.metadata[0].name

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

          args = local.vllm_args

          port {
            container_port = 8000
            name           = "http"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = var.gpu_count
            }
            requests = {
              "nvidia.com/gpu" = var.gpu_count
              cpu              = var.cpu_request
              memory           = var.memory_request
            }
          }

          dynamic "volume_mount" {
            for_each = var.enable_pvc ? [1] : []
            content {
              name       = "model-cache"
              mount_path = "/root/.cache"
            }
          }

          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 120
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 60
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }
        }

        dynamic "volume" {
          for_each = var.enable_pvc ? [1] : []
          content {
            name = "model-cache"
            persistent_volume_claim {
              claim_name = kubernetes_persistent_volume_claim.vllm_cache[0].metadata[0].name
            }
          }
        }

        volume {
          name = "shm"
          empty_dir {
            medium     = "Memory"
            size_limit = "16Gi"
          }
        }
      }
    }
  }
}

# ClusterIP Service
resource "kubernetes_service" "vllm" {
  metadata {
    name      = var.deployment_name
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = var.deployment_name
    }

    port {
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

# LoadBalancer Service (optional)
resource "kubernetes_service" "vllm_lb" {
  count = var.enable_load_balancer ? 1 : 0

  metadata {
    name      = "${var.deployment_name}-lb"
    namespace = kubernetes_namespace.ml.metadata[0].name
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-type"            = "nlb"
      "service.beta.kubernetes.io/aws-load-balancer-scheme"          = var.load_balancer_internal ? "internal" : "internet-facing"
      "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
    }
  }

  spec {
    selector = {
      app = var.deployment_name
    }

    port {
      port        = 80
      target_port = 8000
      protocol    = "TCP"
    }

    type = "LoadBalancer"
  }
}

# HPA (optional)
resource "kubernetes_horizontal_pod_autoscaler_v2" "vllm" {
  count = var.enable_hpa ? 1 : 0

  metadata {
    name      = "${var.deployment_name}-hpa"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.vllm.metadata[0].name
    }

    min_replicas = var.hpa_min_replicas
    max_replicas = var.hpa_max_replicas

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_target_cpu
        }
      }
    }
  }
}
