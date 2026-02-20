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

# Networking
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
  description = "Availability zones for GPU nodes (some AZs have limited capacity)"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1c"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway"
  type        = bool
  default     = true
}

# EKS
variable "eks_cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node group"
  type        = bool
  default     = true
}

variable "gpu_instance_types" {
  description = "GPU instance types (in priority order)"
  type        = list(string)
  default     = ["g6e.xlarge", "g6.xlarge", "g6e.2xlarge", "g6.2xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_min_size" {
  description = "Minimum GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum GPU nodes"
  type        = number
  default     = 3
}

# SageMaker
variable "enable_sagemaker" {
  description = "Enable SageMaker Studio"
  type        = bool
  default     = true
}

# vLLM
variable "enable_vllm" {
  description = "Enable vLLM deployment"
  type        = bool
  default     = true
}

variable "vllm_model_id" {
  description = "HuggingFace model ID for vLLM"
  type        = string
  default     = "mistralai/Ministral-3-3B-Instruct-2512"
}

variable "vllm_image" {
  description = "vLLM container image"
  type        = string
  default     = "vllm/vllm-openai:latest"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization"
  type        = number
  default     = 0.9
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 4096
}

# KV Cache Benchmark Configuration
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

variable "kv_cache_config" {
  description = "KV cache offload config: none, cpu-light, cpu-aggressive, fsx-swap, hybrid"
  type        = string
  default     = "none"

  validation {
    condition     = contains(["none", "cpu-light", "cpu-aggressive", "fsx-swap", "hybrid"], var.kv_cache_config)
    error_message = "kv_cache_config must be one of: none, cpu-light, cpu-aggressive, fsx-swap, hybrid"
  }
}

variable "vllm_cpu_offload_gb" {
  description = "CPU memory for KV cache offload (auto-set by kv_cache_config)"
  type        = number
  default     = 0
}

variable "vllm_swap_space_gb" {
  description = "Swap space for KV cache (auto-set by kv_cache_config)"
  type        = number
  default     = 0
}
