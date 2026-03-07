variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "benchmark"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "qwen3-next-bench"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-2a", "us-east-2b", "us-east-2c"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (p5en capacity block availability)"
  type        = list(string)
  default     = ["us-east-2c"]
}

variable "capacity_reservation_id" {
  description = "Capacity reservation ID for p5en.48xlarge capacity block"
  type        = string
  default     = ""
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for public registry access"
  type        = bool
  default     = true
}

# EKS
variable "eks_cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.32"
}

variable "gpu_instance_types" {
  description = "GPU instance type - p5en.48xlarge with 8x H200"
  type        = list(string)
  default     = ["p5en.48xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired GPU nodes (0 = launch via capacity block CLI)"
  type        = number
  default     = 0
}

variable "gpu_min_size" {
  type    = number
  default = 0
}

variable "gpu_max_size" {
  type    = number
  default = 1
}

variable "gpu_volume_size" {
  description = "Root volume size for GPU nodes (GB)"
  type        = number
  default     = 500
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes"
  type        = string
  default     = "AL2023_x86_64_NVIDIA"
}

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for NVLink TP communication"
  type        = bool
  default     = true
}

# FSx Lustre
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for model storage and KV cache swap"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB. Min 4800 for PERSISTENT_2 at 1000 MB/s/TiB."
  type        = number
  default     = 4800
}

variable "fsx_deployment_type" {
  description = "FSx Lustre deployment type. PERSISTENT_2 required for high throughput."
  type        = string
  default     = "PERSISTENT_2"
}

variable "fsx_per_unit_throughput" {
  description = "FSx throughput per TiB in MB/s. PERSISTENT_2: 125, 250, 500, 1000."
  type        = number
  default     = 1000
}

# Serving Engine
variable "serving_engine" {
  description = "Serving engine: vllm or sglang"
  type        = string
  default     = "vllm"

  validation {
    condition     = contains(["vllm", "sglang"], var.serving_engine)
    error_message = "serving_engine must be 'vllm' or 'sglang'."
  }
}

variable "enable_vllm" {
  description = "Enable serving engine deployment"
  type        = bool
  default     = true
}

variable "vllm_model_id" {
  description = "Local model path. Prefer NVMe for fastest loading."
  type        = string
  default     = "/mnt/nvme/models/qwen3-next-fp8"
}

variable "vllm_image" {
  description = "vLLM container image. Pin to a release tag; avoid :latest."
  type        = string
  default     = "vllm/vllm-openai:qwen3_5-x86_64-cu130"
}

variable "sglang_image" {
  description = "SGLang container image. Pin to a release tag; avoid :latest."
  type        = string
  default     = "lmsysorg/sglang:v0.5.2-cu130"
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image (use private ECR for air-gapped deployments)"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization (overridden by kv_cache_config preset)"
  type        = number
  default     = 0.92
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 131072
}

variable "vllm_tensor_parallel_size" {
  description = "Tensor parallel size (TP=4 default due to FP8 block_k=128 incompatibility with TP=8)"
  type        = number
  default     = 4
}

variable "vllm_gpu_count" {
  description = "Number of GPUs (4 for TP=4 baseline, 8 for TP=8/dp8-ep)"
  type        = number
  default     = 4
}

variable "vllm_cpu_request" {
  description = "CPU request for serving pod"
  type        = string
  default     = "32"
}

variable "vllm_memory_request" {
  description = "Memory request for serving pod (overridden by kv_cache_config preset)"
  type        = string
  default     = "128Gi"
}

variable "vllm_enable_prefix_caching" {
  description = "Enable prefix caching"
  type        = bool
  default     = true
}

variable "vllm_cpu_offload_gb" {
  description = "CPU memory for KV cache offloading in GB (0 = disabled, overridden by kv_cache_config)"
  type        = number
  default     = 0
}

# KV Cache Config Preset
variable "kv_cache_config" {
  description = "KV cache preset: baseline or cpu-light (fsx-swap/hybrid removed — unnecessary on H200)"
  type        = string
  default     = "baseline"

  validation {
    condition     = contains(["baseline", "cpu-light"], var.kv_cache_config)
    error_message = "kv_cache_config must be one of: baseline, cpu-light."
  }
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

variable "prometheus_scrape_interval" {
  description = "Prometheus scrape interval"
  type        = string
  default     = "1s"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

# Container image registry override for air-gapped deployments.
variable "ecr_registry" {
  description = "Private ECR registry URI. When set, all Helm chart images use this registry. Run scripts/stage-images-ecr.sh first."
  type        = string
  default     = ""
}

variable "vllm_node_port" {
  description = "NodePort for serving engine service"
  type        = number
  default     = 30080
}
