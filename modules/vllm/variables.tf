variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for vLLM"
  type        = string
  default     = "ml-inference"
}

variable "deployment_name" {
  description = "Name for the vLLM deployment"
  type        = string
  default     = "vllm"
}

variable "service_account_name" {
  description = "Name for the service account"
  type        = string
  default     = "vllm-sa"
}

# Model configuration
variable "model_id" {
  description = "HuggingFace model ID or S3 path"
  type        = string
}

variable "vllm_image" {
  description = "vLLM container image"
  type        = string
  default     = "vllm/vllm-openai:latest"
}

variable "extra_args" {
  description = "Extra arguments for vLLM (e.g., --tokenizer_mode mistral)"
  type        = list(string)
  default     = []
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization (0.0-1.0)"
  type        = number
  default     = 0.9
}

variable "max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 4096
}

# Resource configuration
variable "replicas" {
  description = "Number of replicas"
  type        = number
  default     = 1
}

variable "gpu_count" {
  description = "Number of GPUs per pod"
  type        = number
  default     = 1
}

variable "cpu_request" {
  description = "CPU request"
  type        = string
  default     = "4"
}

variable "memory_request" {
  description = "Memory request"
  type        = string
  default     = "16Gi"
}

# Storage configuration
variable "enable_pvc" {
  description = "Enable PVC for model cache"
  type        = bool
  default     = true
}

variable "pvc_size" {
  description = "PVC size for model cache"
  type        = string
  default     = "50Gi"
}

variable "storage_class_name" {
  description = "Storage class name"
  type        = string
  default     = "gp3"
}

# S3/IRSA configuration
variable "enable_s3_access" {
  description = "Enable S3 access via IRSA"
  type        = bool
  default     = false
}

variable "oidc_provider_arn" {
  description = "OIDC provider ARN for IRSA"
  type        = string
  default     = ""
}

variable "s3_policy_arn" {
  description = "IAM policy ARN for S3 access"
  type        = string
  default     = null
}

# Service configuration
variable "enable_load_balancer" {
  description = "Enable LoadBalancer service"
  type        = bool
  default     = true
}

variable "load_balancer_internal" {
  description = "Make LoadBalancer internal"
  type        = bool
  default     = false
}

# HPA configuration
variable "enable_hpa" {
  description = "Enable HorizontalPodAutoscaler"
  type        = bool
  default     = false
}

variable "hpa_min_replicas" {
  description = "HPA minimum replicas"
  type        = number
  default     = 1
}

variable "hpa_max_replicas" {
  description = "HPA maximum replicas"
  type        = number
  default     = 3
}

variable "hpa_target_cpu" {
  description = "HPA target CPU utilization"
  type        = number
  default     = 80
}

variable "tags" {
  description = "Tags to apply to AWS resources"
  type        = map(string)
  default     = {}
}

# KV Cache Offloading Configuration
variable "cpu_offload_gb" {
  description = "Amount of CPU memory (GB) to use for KV cache offloading. Set to 0 to disable."
  type        = number
  default     = 0
}

variable "swap_space_gb" {
  description = "Amount of swap space (GB) for KV cache. Requires swap_space_path if using FSx."
  type        = number
  default     = 0
}

variable "swap_space_path" {
  description = "Path to swap space directory (default /tmp, or /mnt/fsx/swap for FSx)"
  type        = string
  default     = "/tmp"
}

# FSx Lustre Mount Configuration
variable "enable_fsx_mount" {
  description = "Enable FSx Lustre mount for swap space"
  type        = bool
  default     = false
}

variable "fsx_dns_name" {
  description = "FSx Lustre DNS name for mounting"
  type        = string
  default     = ""
}

variable "fsx_mount_name" {
  description = "FSx Lustre mount name"
  type        = string
  default     = ""
}

variable "fsx_mount_path" {
  description = "Mount path for FSx Lustre in the container"
  type        = string
  default     = "/mnt/fsx"
}
