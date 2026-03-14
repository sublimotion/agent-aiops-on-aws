# Nemotron-3-Super Blueprint
# Deploys Dynamo + vLLM/SGLang workers on existing glm5-lmcache-b200 EKS cluster
# Reuses: VPC, EKS, FSx Lustre, GPU node group, capacity block infra
# Adds: Nemotron-specific K8s deployments, Dynamo frontend/router, etcd, NATS

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
      Blueprint   = "nemotron-super"
    }
  }
}

# Connect to existing EKS cluster
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

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.existing.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.existing.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.existing.token
  }
}

locals {
  ecr_registry = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
  namespace    = var.namespace

  # Backend image selection
  backend_images = {
    vllm   = "${local.ecr_registry}/nemotron-super-vllm:dynamo-0.9.1"
    sglang = "${local.ecr_registry}/nemotron-super-sglang:dynamo-0.9.1"
  }

  serving_image = local.backend_images[var.serving_backend]

  # Common env vars for all workers
  common_env = {
    HF_HUB_OFFLINE       = "1"
    TRANSFORMERS_OFFLINE = "1"
    NCCL_TIMEOUT         = "1800"
    TRUST_REMOTE_CODE    = "1"
  }

  # vLLM worker args (aggregated mode)
  vllm_base_args = [
    "--model", var.model_path,
    "--served-model-name", "nemotron-3-super",
    "--dtype", "auto",
    "--kv-cache-dtype", "fp8",
    "--tensor-parallel-size", tostring(var.tp_size),
    "--swap-space", "0",
    "--trust-remote-code",
    "--attention-backend", "TRITON_ATTN",
    "--gpu-memory-utilization", tostring(var.gpu_memory_utilization),
    "--enable-chunked-prefill",
    "--max-num-seqs", tostring(var.max_num_seqs),
    "--async-scheduling",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser-plugin", "${var.model_path}/super_v3_reasoning_parser.py",
    "--reasoning-parser", "super_v3",
  ]

  # SGLang worker args (aggregated mode)
  sglang_base_args = [
    "--model-path", var.model_path,
    "--served-model-name", "nemotron-3-super",
    "--host", "0.0.0.0",
    "--port", "30000",
    "--trust-remote-code",
    "--tp", tostring(var.tp_size),
    "--ep", "1",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "nano_v3",
  ]

  serving_args = var.serving_backend == "vllm" ? local.vllm_base_args : local.sglang_base_args
  serving_port = var.serving_backend == "vllm" ? 8000 : 30000

  worker_labels = {
    app     = "nemotron-super"
    model   = "nemotron-3-super-120b"
    backend = var.serving_backend
  }
}

data "aws_caller_identity" "current" {}

# ============================================================================
# Namespace (reuse existing ml-inference)
# ============================================================================

# Use existing namespace — don't create new one
data "kubernetes_namespace" "ml" {
  metadata {
    name = local.namespace
  }
}

# ============================================================================
# Serving Worker Deployment
# ============================================================================

resource "kubernetes_deployment" "nemotron_worker" {
  wait_for_rollout = false

  metadata {
    name      = "nemotron-${var.serving_backend}"
    namespace = local.namespace
    labels    = local.worker_labels
  }

  spec {
    replicas = var.num_workers

    selector {
      match_labels = {
        app = "nemotron-super"
      }
    }

    template {
      metadata {
        labels = merge(local.worker_labels, {
          "dynamo.nvidia.com/role" = "worker"
        })
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = tostring(local.serving_port)
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = var.service_account_name

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
          name  = var.serving_backend
          image = local.serving_image

          security_context {
            run_as_user                = 0
            run_as_non_root            = false
            allow_privilege_escalation = true
          }

          command = var.serving_backend == "vllm" ? ["vllm", "serve"] : ["python3", "-m", "sglang.launch_server"]
          args    = local.serving_args

          dynamic "env" {
            for_each = local.common_env
            content {
              name  = env.key
              value = env.value
            }
          }

          port {
            container_port = local.serving_port
            name           = "http"
          }

          resources {
            limits = {
              "nvidia.com/gpu" = var.tp_size
            }
            requests = {
              "nvidia.com/gpu" = var.tp_size
              cpu              = tostring(var.worker_cpu)
              memory           = var.worker_memory
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

          volume_mount {
            name       = "nvme"
            mount_path = "/mnt/nvme"
          }

          liveness_probe {
            http_get {
              path = var.serving_backend == "vllm" ? "/health" : "/health"
              port = local.serving_port
            }
            initial_delay_seconds = 900
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          readiness_probe {
            http_get {
              path = var.serving_backend == "vllm" ? "/health" : "/health"
              port = local.serving_port
            }
            initial_delay_seconds = 600
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 5
          }
        }

        # Init container to install Lustre client, mount FSx, and stage model to NVMe
        # AL2023 NVIDIA AMI lacks lustre-client userspace tools — install via chroot
        init_container {
          name  = "stage-model"
          image = "amazonlinux:2023"

          security_context {
            privileged = true
          }

          command = ["/bin/bash", "-c"]
          args = [<<-EOT
            set -e
            MODEL_DIR="/host-nvme/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

            if [ -f "$MODEL_DIR/config.json" ]; then
              echo "Model already staged on NVMe"
              exit 0
            fi

            echo "Installing Lustre client on host via chroot..."
            chroot /host bash -c '
              if ! command -v mount.lustre &>/dev/null; then
                dnf install -y lustre-client 2>/dev/null || yum install -y lustre-client 2>/dev/null || true
              fi
              mkdir -p /mnt/fsx
              if ! mountpoint -q /mnt/fsx; then
                echo "Mounting FSx Lustre..."
                mount -t lustre ${var.fsx_ip}@tcp:/${var.fsx_mount_name} /mnt/fsx
              fi
              echo "FSx contents:"
              ls /mnt/fsx/models/ 2>/dev/null || echo "No models dir"
            '

            echo "Copying model from FSx to NVMe..."
            mkdir -p "$MODEL_DIR"
            cp -r /host/mnt/fsx/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/* "$MODEL_DIR/"
            echo "Staging complete"
            ls "$MODEL_DIR/config.json" && echo "config.json verified"
          EOT
          ]

          volume_mount {
            name       = "host-root"
            mount_path = "/host"
          }

          volume_mount {
            name       = "nvme"
            mount_path = "/host-nvme"
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

        volume {
          name = "nvme"
          host_path {
            path = "/mnt/nvme"
            type = "DirectoryOrCreate"
          }
        }

        volume {
          name = "host-root"
          host_path {
            path = "/"
            type = "Directory"
          }
        }
      }
    }
  }
}

# ============================================================================
# Services
# ============================================================================

resource "kubernetes_service" "nemotron" {
  metadata {
    name      = "nemotron-super"
    namespace = local.namespace
  }

  spec {
    selector = {
      app = "nemotron-super"
    }

    port {
      port        = local.serving_port
      target_port = local.serving_port
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_service" "nemotron_nodeport" {
  metadata {
    name      = "nemotron-super-nodeport"
    namespace = local.namespace
  }

  spec {
    selector = {
      app = "nemotron-super"
    }

    port {
      port        = local.serving_port
      target_port = local.serving_port
      node_port   = var.node_port
      protocol    = "TCP"
    }

    type = "NodePort"
  }
}

# ============================================================================
# S3 Bucket for Benchmark Results
# ============================================================================

resource "aws_s3_bucket" "benchmark_results" {
  bucket_prefix = "${var.project_name}-results-"
  force_destroy = true
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
