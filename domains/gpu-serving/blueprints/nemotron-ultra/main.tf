# Nemotron-3-Ultra-550B-A55B-NVFP4 Blueprint
# Attaches to the EXISTING qn-sglang-eks-cluster (us-west-2) and the existing
# ai-infra-b300-spot managed node group. Adds only the K8s serving deployment +
# services for the vLLM TP4 single-replica smoke test.
#
# Reuses: VPC, EKS, ai-infra-b300-spot node group (AL2023 NVIDIA AMI, p6-b300 spot,
#         usw2-az2), nvidia-device-plugin DaemonSet, S3 staging bucket.
# Adds:   vLLM serving deployment (NVFP4, MTP, mamba flags — verbatim from HF card),
#         init container that pulls weights S3 -> NVMe hostPath, ClusterIP + NodePort.
#
# Model weights are staged out-of-band by scripts/stage-model.sh (HF -> S3 -> NVMe).
# The init container only copies S3 -> NVMe if not already present.

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
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Blueprint   = "nemotron-ultra"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_eks_cluster" "existing" {
  name = var.eks_cluster_name
}

data "aws_eks_cluster_auth" "existing" {
  name = var.eks_cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.existing.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.existing.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.existing.token
}

data "kubernetes_namespace" "target" {
  metadata {
    name = var.namespace
  }
}

locals {
  labels = {
    app     = "nemotron-ultra"
    model   = "nemotron-3-ultra-550b-nvfp4"
    backend = "vllm"
  }

  s3_uri = "s3://${var.s3_model_bucket}/${var.s3_model_prefix}"

  # vLLM launch args — VERBATIM from the HF model card (spec Track A, TP4).
  # Order preserved; flag spellings unchanged. enable-prefix-caching is gated by var
  # (Stage 0c mamba-mtp-prefix-cache WARN — drop it if the engine fails at startup).
  vllm_args = concat(
    [
      "/model",
      "--host", "0.0.0.0",
      "--port", "8000",
      "--served-model-name", var.served_model_name,
      "--trust-remote-code",
      "--tensor-parallel-size", tostring(var.tp_size),
      "--enable-expert-parallel",
      "--kv-cache-dtype", "fp8",
      "--max-model-len", tostring(var.max_model_len),
      "--gpu-memory-utilization", tostring(var.gpu_memory_utilization),
      "--max-num-seqs", tostring(var.max_num_seqs),
      "--max-num-batched-tokens", tostring(var.max_num_batched_tokens),
      "--enable-chunked-prefill",
    ],
    var.enable_prefix_caching ? ["--enable-prefix-caching"] : [],
    [
      "--reasoning-parser", "nemotron_v3",
      "--enable-auto-tool-choice",
      "--tool-call-parser", "qwen3_coder",
      "--mamba-ssm-cache-dtype", "float16",
      "--mamba-backend", "flashinfer",
      "--enable-mamba-cache-stochastic-rounding",
      "--mamba-cache-philox-rounds", "5",
      "--speculative-config", "{\"method\": \"nemotron_h_mtp\", \"num_speculative_tokens\": 5}",
      "--model-loader-extra-config", "{\"enable_multithread_load\": true, \"num_threads\": 96}",
    ]
  )
}

# ============================================================================
# Serving Deployment — vLLM TP4 single replica
# ============================================================================

resource "kubernetes_deployment" "nemotron_ultra" {
  wait_for_rollout = false

  metadata {
    name      = "nemotron-ultra-vllm"
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "nemotron-ultra"
      }
    }

    template {
      metadata {
        labels = local.labels
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = var.service_account_name
        host_network         = true # g7e/B-series bare metal: no CNI for serving port
        dns_policy           = "ClusterFirstWithHostNet"

        node_selector = var.gpu_node_label_selector

        # Tolerate the B300 node group taint + the standard nvidia GPU taint.
        toleration {
          key      = var.gpu_node_taint_key
          operator = "Exists"
          effect   = "NoSchedule"
        }
        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        # Init container: stage weights S3 -> NVMe hostPath if not already present.
        init_container {
          name  = "stage-model"
          image = "amazon/aws-cli:2.15.40"

          command = ["/bin/sh", "-c"]
          args = [<<-EOT
            set -e
            DEST="/host-nvme/models/nemotron-3-ultra-nvfp4"
            if [ -f "$DEST/config.json" ]; then
              echo "Model already staged on NVMe at $DEST"
              exit 0
            fi
            echo "Syncing ${local.s3_uri} -> $DEST"
            mkdir -p "$DEST"
            aws s3 sync ${local.s3_uri} "$DEST" --region ${var.aws_region}
            test -f "$DEST/config.json" && echo "config.json present — staging OK"
          EOT
          ]

          volume_mount {
            name       = "nvme"
            mount_path = "/host-nvme"
          }
        }

        container {
          name  = "vllm"
          image = var.serving_image

          security_context {
            run_as_user                = 0
            run_as_non_root            = false
            allow_privilege_escalation = true
          }

          args = local.vllm_args

          # GPUs 0-3 (one TP4 replica). device_ids pinned via env; resource limit caps count.
          env {
            name  = "VLLM_WORKER_MULTIPROC_METHOD"
            value = "spawn"
          }
          env {
            name  = "SAFETENSORS_FAST_GPU"
            value = "1"
          }
          env {
            name  = "NVIDIA_TF32_OVERRIDE"
            value = "1"
          }
          env {
            name  = "VLLM_LOGGING_LEVEL"
            value = "INFO"
          }
          env {
            name  = "HF_HUB_OFFLINE"
            value = "1"
          }
          env {
            name  = "TRANSFORMERS_OFFLINE"
            value = "1"
          }
          # AOT compile cache on NVMe — persists across restarts/spot reclaim.
          env {
            name  = "VLLM_TORCH_COMPILE_CACHE"
            value = "/mnt/nvme/vllm-cache"
          }
          env {
            name  = "TRITON_CACHE_DIR"
            value = "/mnt/nvme/triton-cache"
          }

          port {
            container_port = 8000
            name           = "http"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = var.tp_size
            }
            requests = {
              "nvidia.com/gpu" = var.tp_size
              cpu              = "48"
              memory           = "512Gi"
            }
          }

          volume_mount {
            name       = "nvme"
            mount_path = "/mnt/nvme"
          }
          volume_mount {
            name       = "model"
            mount_path = "/model"
            read_only  = true
          }
          volume_mount {
            name       = "shm"
            mount_path = "/dev/shm"
          }

          # Cold start can take several minutes (NVFP4 load + torch.compile + CUDA graphs);
          # generous startup window before liveness kicks in.
          startup_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 120
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 40 # up to ~20 min cold start
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            period_seconds    = 30
            timeout_seconds   = 10
            failure_threshold = 5
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            period_seconds    = 15
            timeout_seconds   = 5
            failure_threshold = 5
          }
        }

        volume {
          name = "nvme"
          host_path {
            path = "/mnt/nvme"
            type = "DirectoryOrCreate"
          }
        }
        volume {
          name = "model"
          host_path {
            path = "/mnt/nvme/models/nemotron-3-ultra-nvfp4"
            type = "DirectoryOrCreate"
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

# ============================================================================
# Services
# ============================================================================

resource "kubernetes_service" "nemotron_ultra" {
  metadata {
    name      = "nemotron-ultra"
    namespace = var.namespace
  }

  spec {
    selector = {
      app = "nemotron-ultra"
    }
    port {
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
    }
    type = "ClusterIP"
  }
}

resource "kubernetes_service" "nemotron_ultra_nodeport" {
  metadata {
    name      = "nemotron-ultra-nodeport"
    namespace = var.namespace
  }

  spec {
    selector = {
      app = "nemotron-ultra"
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
