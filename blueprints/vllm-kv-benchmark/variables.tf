variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-2" # Ohio - has p5e.48xlarge availability
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
  default     = ["us-east-2a", "us-east-2b"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (p5e availability)"
  type        = list(string)
  default     = ["us-east-2a"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for public registry access (required for public vLLM images)"
  type        = bool
  default     = true # Required for pulling public images
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
  description = "GPU instance types - p5e.48xlarge with 8x H100 (640GB VRAM) + EFA for GDS benchmarks"
  type        = list(string)
  default     = ["p5e.48xlarge"]
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

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for GPU nodes (required for p5e + GDS)"
  type        = bool
  default     = true # Required for p5e with GDS + FSx
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes"
  type        = string
  default     = "AL2_x86_64_GPU"
}

variable "gpu_volume_size" {
  description = "Root volume size for GPU nodes (GB) - p5e needs more for local NVMe"
  type        = number
  default     = 500
}

# FSx Lustre for KV cache offloading (uses private ECR images)
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for KV cache offloading"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB (500000 = 500 TiB for max throughput with GDS)"
  type        = number
  default     = 500000 # 500 TiB - maximizes throughput for GDS benchmarks
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
  default     = "mistralai/Ministral-3B-Instruct-2412" # Use HF directly for us-east-2
}

variable "vllm_image" {
  description = "vLLM container image"
  type        = string
  default     = "vllm/vllm-openai:v0.6.6.post1" # Public image, requires NAT gateway
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

variable "vllm_gpu_count" {
  description = "Number of GPUs for vLLM (p5e.48xlarge has 8x H100)"
  type        = number
  default     = 8
}

variable "vllm_tensor_parallel_size" {
  description = "Tensor parallel size for multi-GPU inference"
  type        = number
  default     = 8
}

variable "vllm_cpu_request" {
  description = "CPU request for vLLM pod"
  type        = string
  default     = "32"
}

variable "vllm_memory_request" {
  description = "Memory request for vLLM pod"
  type        = string
  default     = "256Gi"
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
