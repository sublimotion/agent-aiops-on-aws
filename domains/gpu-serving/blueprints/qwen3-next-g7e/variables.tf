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
  default     = "qwen3-next-g7e-bench"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-2a", "us-east-2b", "us-east-2c"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (g7e capacity)"
  type        = list(string)
  default     = ["us-east-2a"]
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
  description = "GPU instance type - g7e.48xlarge with 8x RTX PRO Server 6000 (Blackwell GB202)"
  type        = list(string)
  default     = ["g7e.48xlarge"]
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes (AL2023_x86_64_NVIDIA for Blackwell)"
  type        = string
  default     = "AL2023_x86_64_NVIDIA"
}

variable "gpu_desired_size" {
  description = "Desired GPU nodes"
  type        = number
  default     = 1
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

# Model storage — reuse existing S3 bucket from qwen3-next blueprint
variable "model_s3_bucket_id" {
  description = "Existing S3 bucket ID containing model weights (from qwen3-next blueprint output)"
  type        = string
}

# FSx Lustre
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for model storage"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB. SCRATCH_2 minimum 1200."
  type        = number
  default     = 1200
}

variable "fsx_deployment_type" {
  description = "FSx Lustre deployment type. SCRATCH_2 sufficient for benchmarks."
  type        = string
  default     = "SCRATCH_2"
}

variable "fsx_per_unit_throughput" {
  description = "FSx throughput per TiB in MB/s. Only applies to PERSISTENT_2."
  type        = number
  default     = 200
}

# Serving Engine
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

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization"
  type        = number
  default     = 0.92
}

variable "vllm_max_model_len" {
  description = "Maximum model context length (may need 65536 if 96 GB VRAM insufficient for 131K)"
  type        = number
  default     = 131072
}

variable "vllm_tensor_parallel_size" {
  description = "Tensor parallel size"
  type        = number
  default     = 4
}

variable "vllm_gpu_count" {
  description = "Number of GPUs for serving pod"
  type        = number
  default     = 4
}

variable "vllm_cpu_request" {
  description = "CPU request for serving pod"
  type        = string
  default     = "32"
}

variable "vllm_memory_request" {
  description = "Memory request for serving pod"
  type        = string
  default     = "128Gi"
}

variable "vllm_enable_prefix_caching" {
  description = "Enable prefix caching"
  type        = bool
  default     = true
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

variable "prometheus_scrape_interval" {
  description = "Prometheus scrape interval (must be > scrapeTimeout; Prometheus default timeout is 10s)"
  type        = string
  default     = "15s"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

# Container image registry override for air-gapped deployments.
variable "ecr_registry" {
  description = "Private ECR registry URI. When set, all Helm chart images use this registry."
  type        = string
  default     = ""
}

variable "vllm_node_port" {
  description = "NodePort for serving engine service"
  type        = number
  default     = 30080
}
