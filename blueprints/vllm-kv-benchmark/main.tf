# vLLM KV Cache Benchmark Blueprint
# Deploys EKS with GPU nodes, FSx Lustre, Prometheus/Grafana, and vLLM
# for KV cache offloading benchmarking per specs/vllm-kv-cache-benchmark.md

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
      Blueprint   = "vllm-kv-benchmark"
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

  # Filter subnets for GPU nodes (g7e availability)
  gpu_subnet_ids = [
    for i, az in var.availability_zones : module.networking.private_subnets[i]
    if contains(var.gpu_availability_zones, az)
  ]

  # vLLM extra args for KV cache configuration
  vllm_extra_args = concat(
    # KV cache config
    var.vllm_enable_prefix_caching ? ["--enable-prefix-caching"] : [],
    ["--disable-log-requests"],
    # Multi-GPU tensor parallelism
    var.vllm_tensor_parallel_size > 1 ? ["--tensor-parallel-size", tostring(var.vllm_tensor_parallel_size)] : [],
    var.vllm_cpu_offload_gb > 0 ? ["--cpu-offload-gb", tostring(var.vllm_cpu_offload_gb)] : [],
    var.vllm_swap_space_gb > 0 ? ["--swap-space", tostring(var.vllm_swap_space_gb)] : []
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
  system_instance_types = ["m6i.large"]
  system_desired_size   = 2

  # GPU nodes - p5e.48xlarge with 8x H100 + EFA for GDS benchmarks
  enable_gpu_nodes   = var.enable_gpu_nodes
  gpu_instance_types = var.gpu_instance_types
  gpu_subnet_ids     = local.gpu_subnet_ids
  gpu_desired_size   = var.gpu_desired_size
  gpu_min_size       = var.gpu_min_size
  gpu_max_size       = var.gpu_max_size
  gpu_volume_size    = var.gpu_volume_size
  gpu_ami_type       = var.gpu_ami_type
  enable_efa         = var.enable_efa

  # Use public NVIDIA image (NAT Gateway enabled)
  nvidia_device_plugin_image = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"

  tags = local.tags
}

# FSx Lustre for KV cache offloading
resource "aws_fsx_lustre_file_system" "kv_cache" {
  count = var.enable_fsx_lustre ? 1 : 0

  storage_capacity            = var.fsx_storage_capacity
  subnet_ids                  = [module.networking.private_subnets[0]]
  deployment_type             = var.fsx_deployment_type
  security_group_ids          = [aws_security_group.fsx[0].id]
  per_unit_storage_throughput = var.fsx_deployment_type == "PERSISTENT_1" ? 200 : null

  tags = merge(local.tags, {
    Name = "${var.project_name}-fsx-lustre"
  })
}

resource "aws_security_group" "fsx" {
  count = var.enable_fsx_lustre ? 1 : 0

  name_prefix = "${var.project_name}-fsx-"
  description = "Security group for FSx Lustre file system"
  vpc_id      = module.networking.vpc_id

  ingress {
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Lustre client traffic"
  }

  ingress {
    from_port   = 1018
    to_port     = 1023
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Lustre inter-node communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic for FSx operations"
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-fsx-sg"
  })
}

# FSx CSI Driver for Kubernetes (using private ECR images)
resource "helm_release" "fsx_csi_driver" {
  count = var.enable_fsx_lustre ? 1 : 0

  name       = "aws-fsx-csi-driver"
  repository = "https://kubernetes-sigs.github.io/aws-fsx-csi-driver"
  chart      = "aws-fsx-csi-driver"
  namespace  = "kube-system"
  version    = "1.9.0"

  values = [
    yamlencode({
      image = {
        repository = "public.ecr.aws/fsx-csi-driver/aws-fsx-csi-driver"
        tag        = "v1.2.0"
      }
      sidecars = {
        livenessProbe = {
          image = {
            repository = "public.ecr.aws/eks-distro/kubernetes-csi/livenessprobe"
            tag        = "v2.12.0-eks-1-29-5"
          }
        }
        nodeDriverRegistrar = {
          image = {
            repository = "public.ecr.aws/eks-distro/kubernetes-csi/node-driver-registrar"
            tag        = "v2.10.0-eks-1-29-5"
          }
        }
        provisioner = {
          image = {
            repository = "public.ecr.aws/eks-distro/kubernetes-csi/external-provisioner"
            tag        = "v4.0.0-eks-1-29-5"
          }
        }
        resizer = {
          image = {
            repository = "public.ecr.aws/eks-distro/kubernetes-csi/external-resizer"
            tag        = "v1.10.0-eks-1-29-5"
          }
        }
      }
      controller = {
        serviceAccount = {
          annotations = {
            "eks.amazonaws.com/role-arn" = module.fsx_csi_irsa[0].iam_role_arn
          }
        }
      }
      node = {
        tolerations = [
          {
            key      = "nvidia.com/gpu"
            operator = "Exists"
            effect   = "NoSchedule"
          }
        ]
      }
    })
  ]

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

# FSx Storage Class for Kubernetes
resource "kubernetes_storage_class" "fsx_lustre" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name = "fsx-lustre"
  }

  storage_provisioner = "fsx.csi.aws.com"
  reclaim_policy      = "Delete"

  parameters = {
    subnetId         = module.networking.private_subnets[0]
    securityGroupIds = aws_security_group.fsx[0].id
    deploymentType   = var.fsx_deployment_type
    fileSystemId     = aws_fsx_lustre_file_system.kv_cache[0].id
  }

  depends_on = [helm_release.fsx_csi_driver]
}

# Prometheus monitoring (using private ECR images to avoid NAT Gateway requirement)
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0

  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "prometheus"
  namespace        = "monitoring"
  create_namespace = true
  version          = "25.27.0"

  values = [
    yamlencode({
      server = {
        image = {
          repository = "quay.io/prometheus/prometheus"
          tag        = "v2.54.1"
        }
        global = {
          scrape_interval     = var.prometheus_scrape_interval
          scrape_timeout      = var.prometheus_scrape_interval
          evaluation_interval = var.prometheus_scrape_interval
        }
        retention = "7d"
        persistentVolume = {
          enabled      = true
          size         = "50Gi"
          storageClass = "gp3"
        }
        service = {
          type     = "NodePort"
          nodePort = 30090
        }
      }
      alertmanager = {
        enabled = false
      }
      kube-state-metrics = {
        enabled = false
      }
      prometheus-node-exporter = {
        enabled = false
      }
      prometheus-pushgateway = {
        enabled = false
      }
      configmapReload = {
        prometheus = {
          image = {
            repository = "quay.io/prometheus-operator/prometheus-config-reloader"
            tag        = "v0.76.0"
          }
        }
      }
      serverFiles = {
        "prometheus.yml" = {
          scrape_configs = [
            {
              job_name        = "vllm"
              scrape_interval = var.prometheus_scrape_interval
              static_configs = [
                {
                  targets = ["vllm-benchmark.ml-inference.svc.cluster.local:8000"]
                  labels = {
                    app = "vllm-benchmark"
                  }
                }
              ]
            }
          ]
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# ML namespace
resource "kubernetes_namespace" "ml" {
  metadata {
    name = "ml-inference"
    labels = {
      name = "ml-inference"
    }
  }

  depends_on = [module.eks]
}

# ServiceMonitor not needed with simple prometheus - using static scrape config

# vLLM Deployment with KV cache configuration
resource "kubernetes_deployment" "vllm" {
  count = var.enable_vllm ? 1 : 0

  wait_for_rollout = false

  metadata {
    name      = "vllm-benchmark"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = "vllm-benchmark"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "vllm-benchmark"
      }
    }

    template {
      metadata {
        labels = {
          app = "vllm-benchmark"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
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

          # vllm-openai image args (uses python -m vllm.entrypoints.openai.api_server)
          args = concat(
            [
              "--model", var.vllm_model_id,
              "--port", "8000",
              "--gpu-memory-utilization", tostring(var.vllm_gpu_memory_utilization),
              "--max-model-len", tostring(var.vllm_max_model_len)
            ],
            local.vllm_extra_args
          )

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
              mount_path = "/fsx"
            }
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
            initial_delay_seconds = 120
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
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

        dynamic "volume" {
          for_each = var.enable_fsx_lustre ? [1] : []
          content {
            name = "fsx-lustre"
            persistent_volume_claim {
              claim_name = kubernetes_persistent_volume_claim.fsx_lustre[0].metadata[0].name
            }
          }
        }
      }
    }
  }
}

# PVC for model cache
resource "kubernetes_persistent_volume_claim" "vllm_cache" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-benchmark-cache"
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

  wait_until_bound = false
}

# Static PV for existing FSx Lustre filesystem
resource "kubernetes_persistent_volume" "fsx_lustre" {
  count = var.enable_fsx_lustre ? 1 : 0

  metadata {
    name = "fsx-lustre-pv"
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
        volume_handle = aws_fsx_lustre_file_system.kv_cache[0].id
        volume_attributes = {
          dnsname   = aws_fsx_lustre_file_system.kv_cache[0].dns_name
          mountname = aws_fsx_lustre_file_system.kv_cache[0].mount_name
        }
      }
    }
  }

  depends_on = [helm_release.fsx_csi_driver, aws_fsx_lustre_file_system.kv_cache]
}

# FSx Lustre PVC for KV cache offloading (bound to static PV)
resource "kubernetes_persistent_volume_claim" "fsx_lustre" {
  count = var.enable_fsx_lustre && var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-fsx-cache"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = "fsx-lustre"
    volume_name        = kubernetes_persistent_volume.fsx_lustre[0].metadata[0].name

    resources {
      requests = {
        storage = "${var.fsx_storage_capacity}Gi"
      }
    }
  }

  wait_until_bound = false

  depends_on = [kubernetes_persistent_volume.fsx_lustre, kubernetes_storage_class.fsx_lustre]
}

# ClusterIP Service
resource "kubernetes_service" "vllm" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-benchmark"
    namespace = kubernetes_namespace.ml.metadata[0].name
    labels = {
      app = "vllm-benchmark"
    }
  }

  spec {
    selector = {
      app = "vllm-benchmark"
    }

    port {
      port        = 8000
      target_port = 8000
      protocol    = "TCP"
      name        = "http"
    }

    type = "ClusterIP"
  }
}

# NodePort Service for local access (spec: localhost:30080)
resource "kubernetes_service" "vllm_nodeport" {
  count = var.enable_vllm ? 1 : 0

  metadata {
    name      = "vllm-benchmark-nodeport"
    namespace = kubernetes_namespace.ml.metadata[0].name
  }

  spec {
    selector = {
      app = "vllm-benchmark"
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
