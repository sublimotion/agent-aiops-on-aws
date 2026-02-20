variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "aiops"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU node group (some AZs have limited GPU capacity)"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1c"] # us-east-1b excluded due to limited g6e capacity
}

# EKS Configuration
variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.29"
}

variable "eks_gpu_instance_types" {
  description = "Instance types for GPU node group (multiple types for better capacity availability)"
  type        = list(string)
  default     = ["g6e.xlarge", "g6.xlarge", "g6e.2xlarge", "g6.2xlarge"]
}

variable "eks_gpu_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "eks_gpu_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "eks_gpu_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 3
}

# SageMaker Configuration
variable "sagemaker_domain_name" {
  description = "Name for SageMaker Studio domain"
  type        = string
  default     = "aiops-studio"
}

# vLLM Configuration
variable "vllm_model_id" {
  description = "HuggingFace model ID for vLLM (used when loading from HF Hub)"
  type        = string
  default     = "mistralai/Ministral-3-3B-Instruct-2512"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization for vLLM (0.0-1.0)"
  type        = number
  default     = 0.9
}

variable "vllm_use_s3_model" {
  description = "Load model from S3 using Run:ai Model Streamer instead of HuggingFace Hub"
  type        = bool
  default     = false # Using HuggingFace Hub for now - S3 model is Ministral which isn't supported yet
}

variable "vllm_s3_model_path" {
  description = "S3 path to model within the models bucket (e.g., 'models/ministral-3b')"
  type        = string
  default     = "models/ministral-3b"
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 4096
}

# Feature Toggles
variable "enable_gpu_nodes" {
  description = "Enable GPU node group in EKS"
  type        = bool
  default     = true
}

variable "enable_sagemaker_studio" {
  description = "Enable SageMaker Studio deployment"
  type        = bool
  default     = true
}

variable "enable_vllm_deployment" {
  description = "Enable vLLM deployment on EKS"
  type        = bool
  default     = true
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

# FSx for Lustre Configuration
variable "enable_fsx_lustre" {
  description = "Enable FSx for Lustre for KV cache offloading benchmarks"
  type        = bool
  default     = false
}

variable "fsx_storage_capacity_gib" {
  description = "FSx storage capacity in GiB (minimum 1200 for SCRATCH_2)"
  type        = number
  default     = 1200
}

variable "fsx_deployment_type" {
  description = "FSx deployment type: SCRATCH_2 recommended for benchmarks"
  type        = string
  default     = "SCRATCH_2"
}

# KV Cache Offloading Configuration
variable "kv_cache_config" {
  description = "KV cache offload config: none, cpu-light, cpu-aggressive, fsx-swap, hybrid"
  type        = string
  default     = "none"
}

variable "vllm_cpu_offload_gb" {
  description = "CPU memory for KV cache offload (overrides kv_cache_config if set > 0)"
  type        = number
  default     = 0
}

variable "vllm_swap_space_gb" {
  description = "Swap space for KV cache on FSx (overrides kv_cache_config if set > 0)"
  type        = number
  default     = 0
}

# LMCache Configuration
variable "kv_cache_backend" {
  description = "KV cache backend: native (vLLM built-in) or lmcache (LMCache integration)"
  type        = string
  default     = "native"

  validation {
    condition     = contains(["native", "lmcache"], var.kv_cache_backend)
    error_message = "kv_cache_backend must be 'native' or 'lmcache'."
  }
}

variable "lmcache_config" {
  description = "LMCache preset: cpu (memory), disk (local NVMe), fsx (FSx Lustre)"
  type        = string
  default     = "cpu"

  validation {
    condition     = contains(["cpu", "disk", "fsx"], var.lmcache_config)
    error_message = "lmcache_config must be 'cpu', 'disk', or 'fsx'."
  }
}

variable "lmcache_chunk_size" {
  description = "LMCache chunk size for KV cache blocks"
  type        = number
  default     = 256
}

variable "lmcache_max_cache_size_gb" {
  description = "Maximum LMCache size in GB"
  type        = number
  default     = 20
}
