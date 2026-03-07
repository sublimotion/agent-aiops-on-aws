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
  default     = "glm5-llmd"
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
  description = "AZs for GPU nodes (p5e capacity block availability)"
  type        = list(string)
  default     = ["us-east-2c"]
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
  description = "GPU instance type — p6-b200.48xlarge (8x B200) or p5e.48xlarge (8x H200)"
  type        = list(string)
  default     = ["p6-b200.48xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_min_size" {
  type    = number
  default = 0
}

variable "gpu_max_size" {
  description = "Max GPU nodes (2+ for multi-replica)"
  type        = number
  default     = 4
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
  description = "FSx throughput per TiB in MB/s"
  type        = number
  default     = 500
}

# Serving Engine
variable "enable_serving" {
  description = "Enable vLLM serving deployment"
  type        = bool
  default     = true
}

variable "replicas" {
  description = "Number of vLLM replicas (1 per GPU node)"
  type        = number
  default     = 2
}

variable "vllm_image" {
  description = "vLLM container image — official GLM-5 image with DeepGEMM + tool calling"
  type        = string
  default     = "vllm/vllm-openai:glm5"
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

variable "max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 131072
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization for KV cache"
  type        = number
  default     = 0.85
}

# LMCache
variable "enable_lmcache" {
  description = "Enable LMCache via vLLM KVConnector"
  type        = bool
  default     = true
}

variable "lmcache_local_cpu" {
  description = "Enable LMCache L1 CPU offload"
  type        = bool
  default     = true
}

variable "lmcache_max_cpu_size" {
  description = "Max CPU memory for LMCache L1 (GB)"
  type        = number
  default     = 400
}

variable "lmcache_redis_url" {
  description = "Redis URL for LMCache L2 cross-replica cache (e.g., redis://redis:6379)"
  type        = string
  default     = "redis://redis-glm5.ml-inference.svc.cluster.local:6379"
}

# llm-d
variable "enable_llmd" {
  description = "Enable llm-d inference scheduler (Gateway API + EPP)"
  type        = bool
  default     = true
}

variable "llmd_epp_image" {
  description = "llm-d EPP container image"
  type        = string
  default     = "ghcr.io/llm-d/llm-d-inference-scheduler:latest"
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
  description = "NodePort for vLLM service"
  type        = number
  default     = 30080
}
