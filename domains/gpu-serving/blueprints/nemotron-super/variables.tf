# Nemotron-3-Super Blueprint Variables

# ============================================================================
# AWS / Provider
# ============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "nemotron-super"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

# ============================================================================
# EKS Cluster (existing)
# ============================================================================

variable "eks_cluster_name" {
  description = "Name of the existing EKS cluster to deploy into"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for workloads"
  type        = string
  default     = "ml-inference"
}

variable "service_account_name" {
  description = "Kubernetes service account for GPU workloads"
  type        = string
  default     = "default"
}

# ============================================================================
# Model Configuration
# ============================================================================

variable "model_path" {
  description = "Path to model weights (NVMe-staged location)"
  type        = string
  default     = "/mnt/nvme/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
}

variable "model_fsx_path" {
  description = "Path to model weights on FSx Lustre"
  type        = string
  default     = "models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
}

# ============================================================================
# Serving Backend
# ============================================================================

variable "serving_backend" {
  description = "Serving backend: vllm or sglang"
  type        = string
  default     = "vllm"

  validation {
    condition     = contains(["vllm", "sglang"], var.serving_backend)
    error_message = "serving_backend must be 'vllm' or 'sglang'"
  }
}

variable "tp_size" {
  description = "Tensor parallelism degree per worker"
  type        = number
  default     = 2
}

variable "num_workers" {
  description = "Number of serving worker replicas"
  type        = number
  default     = 4
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization fraction (vLLM only)"
  type        = number
  default     = 0.90
}

variable "max_num_seqs" {
  description = "Max concurrent sequences per worker (vLLM only)"
  type        = number
  default     = 256
}

# ============================================================================
# Worker Resources
# ============================================================================

variable "worker_cpu" {
  description = "CPU request per worker (cores)"
  type        = number
  default     = 32
}

variable "worker_memory" {
  description = "Memory request per worker"
  type        = string
  default     = "128Gi"
}

# ============================================================================
# Storage
# ============================================================================

variable "fsx_pvc_name" {
  description = "Name of existing FSx Lustre PVC (unused if fsx_ip set)"
  type        = string
  default     = "fsx-lustre-pvc"
}

variable "fsx_ip" {
  description = "FSx Lustre MGS IP address (for direct mount on AL2023 NVIDIA AMI)"
  type        = string
}

variable "fsx_mount_name" {
  description = "FSx Lustre mount name (e.g. gsyc5b4v)"
  type        = string
}

# ============================================================================
# Networking
# ============================================================================

variable "node_port" {
  description = "NodePort for external access"
  type        = number
  default     = 30080
}
