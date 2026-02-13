variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "benchmark"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "vllm-kv-bench"
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
  default     = ["us-east-1a", "us-east-1b"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (g7e availability)"
  type        = list(string)
  default     = ["us-east-1a"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for public registry access (requires EIP quota)"
  type        = bool
  default     = false
}

# EKS
variable "eks_cluster_version" {
  description = "Kubernetes version (1.31+ required per spec)"
  type        = string
  default     = "1.31"
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node group"
  type        = bool
  default     = true
}

variable "gpu_instance_types" {
  description = "GPU instance types - g6e/g7e with L40S (48GB VRAM) for high concurrency benchmarks"
  type        = list(string)
  default     = ["g6e.xlarge", "g6e.2xlarge"]
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
  default     = 1
}

# FSx Lustre for KV cache offloading (uses private ECR images)
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for KV cache offloading"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB (1200 = 1.2 TiB per spec)"
  type        = number
  default     = 1200
}

variable "fsx_deployment_type" {
  description = "FSx Lustre deployment type (SCRATCH_2 per spec)"
  type        = string
  default     = "SCRATCH_2"
}

# vLLM
variable "enable_vllm" {
  description = "Enable vLLM deployment"
  type        = bool
  default     = true
}

variable "vllm_model_id" {
  description = "Model path for vLLM (HuggingFace ID or S3 path)"
  type        = string
  default     = "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/"
}

variable "vllm_image" {
  description = "vLLM container image (using ECR for private subnets)"
  type        = string
  default     = "615299764834.dkr.ecr.us-east-1.amazonaws.com/vllm-openai:v0.15.1"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization"
  type        = number
  default     = 0.9
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 32768 # Ministral 3B supports up to 32K
}

# KV Cache offloading configuration
variable "vllm_enable_prefix_caching" {
  description = "Enable prefix caching (required for KV cache benchmarks)"
  type        = bool
  default     = true
}

variable "vllm_cpu_offload_gb" {
  description = "CPU memory for KV cache offloading in GB (0 = disabled)"
  type        = number
  default     = 0
}

variable "vllm_swap_space_gb" {
  description = "Swap space for KV cache in GB (0 = disabled)"
  type        = number
  default     = 0
}

# Monitoring (uses private ECR images)
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

# NodePort for local access
variable "vllm_node_port" {
  description = "NodePort for vLLM service (spec: 30080)"
  type        = number
  default     = 30080
}
