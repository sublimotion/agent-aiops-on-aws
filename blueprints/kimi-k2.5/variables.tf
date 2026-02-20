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
  default     = "kimi-k2-bench"
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
  description = "Availability zones for GPU nodes (p5e capacity block availability)"
  type        = list(string)
  default     = ["us-east-2c"]
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
  description = "GPU instance type - p5e.48xlarge with 8x H200"
  type        = list(string)
  default     = ["p5e.48xlarge"]
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
  description = "AMI type for GPU nodes (AL2023 for GDS support)"
  type        = string
  default     = "AL2023_x86_64_NVIDIA"
}

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for GDS and future multi-node"
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
  description = "FSx Lustre storage capacity in GiB. Min 4800 for PERSISTENT_2 at 1000 MB/s/TiB with EFA. Override to 100800 for 100 TiB production."
  type        = number
  default     = 4800
}

variable "fsx_deployment_type" {
  description = "FSx Lustre deployment type. PERSISTENT_2 required for EFA/GDS."
  type        = string
  default     = "PERSISTENT_2"
}

variable "fsx_per_unit_throughput" {
  description = "FSx throughput per TiB in MB/s. PERSISTENT_2: 125, 250, 500, 1000."
  type        = number
  default     = 1000
}

# vLLM - Kimi K2.5 specific
variable "enable_vllm" {
  description = "Enable vLLM deployment"
  type        = bool
  default     = true
}

variable "vllm_model_id" {
  description = "Local model path. Prefer NVMe for 14x faster loading (~2 min vs ~25 min from FSx)."
  type        = string
  default     = "/mnt/nvme/models/Kimi-K2.5"
}

variable "vllm_image" {
  description = "vLLM container image. Use 'nightly' (CUDA 12.9) for LMCache+GDS — cu130-nightly lacks CUDA dev headers needed for LMCache source build. Use 'cu130-nightly' for baseline/Dynamo (no LMCache dependency). See lessons.md #15."
  type        = string
  default     = "vllm/vllm-openai:nightly"
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image (use private ECR for air-gapped deployments)"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "vllm_gpu_memory_utilization" {
  description = "GPU memory utilization (leave headroom for MoE routing)"
  type        = number
  default     = 0.85
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 32768
}

variable "vllm_tensor_parallel_size" {
  description = "Tensor parallel size (must match GPU count)"
  type        = number
  default     = 8
}

variable "vllm_gpu_count" {
  description = "Number of GPUs (p5e.48xlarge = 8x H200)"
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

variable "vllm_enable_prefix_caching" {
  description = "Enable prefix caching"
  type        = bool
  default     = true
}

variable "vllm_cpu_offload_gb" {
  description = "CPU memory for KV cache offloading in GB (0 = disabled)"
  type        = number
  default     = 0
}

variable "vllm_swap_space_gb" {
  description = "Swap space for KV cache in GB (0 = default 32GB, safe for MoE — zero overhead when not triggered)"
  type        = number
  default     = 0
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
# Set to your private ECR registry (e.g., "123456789012.dkr.ecr.us-east-2.amazonaws.com")
# to pull all images from ECR instead of public registries.
# When empty, uses the default public registry for each image.
variable "ecr_registry" {
  description = "Private ECR registry URI. When set, all Helm chart images use this registry. Run scripts/stage-images-ecr.sh first."
  type        = string
  default     = ""
}

variable "vllm_node_port" {
  description = "NodePort for vLLM service"
  type        = number
  default     = 30080
}
