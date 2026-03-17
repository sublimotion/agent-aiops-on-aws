variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "benchmark"
}

variable "project_name" {
  description = "Project name used for resource naming (keep <=12 chars for IAM limits)"
  type        = string
  default     = "qn-sglang"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.2.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2d"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (g7e capacity)"
  type        = list(string)
  default     = ["us-west-2d"]
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
  description = "GPU instance type. g7e.24xlarge (4x RTX PRO 6000 PCIe, $16.57/hr) or g7e.48xlarge (8x RTX PRO 6000 PCIe, $110.30/hr)"
  type        = list(string)
  default     = ["g7e.24xlarge"]
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

# Model storage — create local bucket in same region; sync model after deploy
variable "model_s3_bucket_id" {
  description = "Existing S3 bucket ID containing model weights. Leave empty to create a new bucket."
  type        = string
  default     = ""
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

# SGLang serving
variable "sglang_image" {
  description = "SGLang container image. Nightly required for --fp8-gemm-backend flag on Blackwell."
  type        = string
  default     = "lmsysorg/sglang:nightly-dev-cu13-20260303-145ae518"
}

variable "sglang_tp_size" {
  description = "Tensor parallel size. 4 for g7e.24xl, 4 or 8 for g7e.48xl."
  type        = number
  default     = 4
}

variable "sglang_gpu_count" {
  description = "Number of GPUs. 4 for g7e.24xl, 8 for g7e.48xl."
  type        = number
  default     = 4
}

variable "sglang_context_length" {
  description = "Context length. Start with 65536; try 131072 if VRAM allows."
  type        = number
  default     = 65536
}

variable "sglang_mem_fraction" {
  description = "Fraction of GPU memory for KV cache"
  type        = number
  default     = 0.90
}

variable "model_path" {
  description = "Local model path on GPU node (NVMe preferred)"
  type        = string
  default     = "/mnt/nvme/models/qwen3-next-fp8"
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = false
}

# Container image registry override for air-gapped deployments.
variable "ecr_registry" {
  description = "Private ECR registry URI. When set, all Helm chart images use this registry."
  type        = string
  default     = ""
}
