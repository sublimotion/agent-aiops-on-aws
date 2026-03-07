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
  default     = "glm5-lmcache"
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
  description = "AZs for GPU nodes (capacity block availability)"
  type        = list(string)
  default     = ["us-east-2b"]
}

variable "capacity_reservation_id" {
  description = "Capacity reservation ID for p5e.48xlarge capacity block"
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
  description = "GPU instance type — p6-b200.48xlarge with 8x B200 or p5e.48xlarge with 8x H200"
  type        = list(string)
  default     = ["p6-b200.48xlarge"]
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

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter"
  type        = bool
  default     = true
}

# FSx Lustre
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for model storage and KV cache"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB"
  type        = number
  default     = 4800
}

variable "fsx_per_unit_throughput" {
  description = "FSx throughput per TiB in MB/s (PERSISTENT_2: 125, 250, 500, 1000)"
  type        = number
  default     = 500
}

# Serving Engine
variable "enable_serving" {
  description = "Enable SGLang serving deployment"
  type        = bool
  default     = true
}

variable "sglang_image" {
  description = "SGLang container image. Use latest or cu128+ for Blackwell (sm_100) support."
  type        = string
  default     = "lmsysorg/sglang:latest"
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "model_path" {
  description = "Local model path on GPU node"
  type        = string
  default     = "/mnt/nvme/models/GLM-5-FP8"
}

variable "model_id" {
  description = "HuggingFace model ID for staging"
  type        = string
  default     = "zai-org/GLM-5-FP8"
}

variable "tp_size" {
  description = "Tensor parallel size"
  type        = number
  default     = 8
}

variable "context_length" {
  description = "Maximum context length"
  type        = number
  default     = 131072
}

variable "max_running_requests" {
  description = "Maximum concurrent running requests"
  type        = number
  default     = 256
}

variable "mem_fraction_static" {
  description = "Static memory fraction for KV cache"
  type        = number
  default     = 0.85
}

# LMCache
variable "enable_lmcache" {
  description = "Enable LMCache integration with SGLang"
  type        = bool
  default     = false
}

variable "lmcache_config_content" {
  description = "LMCache YAML config content (mounted as ConfigMap)"
  type        = string
  default     = <<-EOT
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 400
    use_layerwise: true
  EOT
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable DCGM exporter"
  type        = bool
  default     = true
}

# Container image registry override
variable "ecr_registry" {
  description = "Private ECR registry URI"
  type        = string
  default     = ""
}

variable "node_port" {
  description = "NodePort for SGLang service"
  type        = number
  default     = 30080
}
