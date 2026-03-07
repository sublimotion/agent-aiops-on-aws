# GLM-5 HyperPod Blueprint Variables

# =============================================================================
# AWS / Project
# =============================================================================

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

variable "resource_name_prefix" {
  description = "Prefix for all resource names (keep short — IAM role names have 64-char limit)"
  type        = string
  default     = "glm5"
}

# =============================================================================
# HyperPod / EKS
# =============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.32"
}

variable "eks_cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "glm5-hyperpod-eks"
}

variable "hyperpod_cluster_name" {
  description = "SageMaker HyperPod cluster name"
  type        = string
  default     = "glm5-cluster"
}

# =============================================================================
# Instance Groups (with Training Plan support)
# =============================================================================

variable "instance_groups" {
  description = "HyperPod instance groups. Each group can reference a Training Plan ARN for managed capacity."
  type = list(object({
    name                      = string
    instance_type             = string
    instance_count            = number
    ebs_volume_size_in_gb     = number
    threads_per_core          = number
    enable_stress_check       = bool
    enable_connectivity_check = bool
    lifecycle_script          = string
    availability_zone_id      = string
    training_plan_arn         = optional(string)
  }))
  default = [
    {
      name                      = "glm5-p5e"
      instance_type             = "ml.p5e.48xlarge"
      instance_count            = 1
      ebs_volume_size_in_gb     = 500
      threads_per_core          = 2
      enable_stress_check       = true
      enable_connectivity_check = true
      lifecycle_script          = "on_create.sh"
      availability_zone_id      = "usw2-az2"
      training_plan_arn         = null
    }
  ]
}

# =============================================================================
# FSx Lustre
# =============================================================================

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB (min 4800 for PERSISTENT_2)"
  type        = number
  default     = 4800
}

variable "fsx_throughput" {
  description = "FSx throughput per TiB in MB/s (125, 250, 500, 1000)"
  type        = number
  default     = 500
}

# =============================================================================
# Closed Network
# =============================================================================

variable "closed_network" {
  description = "Enable closed network (air-gap) mode — no outbound internet"
  type        = bool
  default     = true
}

variable "helm_repo_path" {
  description = "Local path to sagemaker-hyperpod-cli Helm charts (required for closed network)"
  type        = string
  default     = ""
}

# =============================================================================
# Model Configuration
# =============================================================================

variable "model_s3_prefix" {
  description = "S3 key prefix for model weights"
  type        = string
  default     = "GLM-5-FP8"
}

variable "sglang_image_tag" {
  description = "Tag for the custom SGLang container in ECR"
  type        = string
  default     = "v0.5.2"
}

variable "sglang_base_image" {
  description = "Base SGLang image for building the custom container"
  type        = string
  default     = "lmsys/sglang:v0.5.2-cu124"
}

# =============================================================================
# Serving Configuration
# =============================================================================

variable "sglang_tp_size" {
  description = "Tensor parallel size (8 for both H200 and B200)"
  type        = number
  default     = 8
}

variable "sglang_context_length" {
  description = "Maximum context length"
  type        = number
  default     = 131072
}

variable "sglang_max_running_requests" {
  description = "Maximum concurrent running requests (256 for H200, 512 for B200)"
  type        = number
  default     = 256
}

variable "sglang_chunked_prefill_size" {
  description = "Chunked prefill token batch size"
  type        = number
  default     = 32768
}

variable "sglang_mem_fraction_static" {
  description = "Static GPU memory fraction for KV cache"
  type        = string
  default     = "0.85"
}

variable "sglang_served_model_name" {
  description = "Model name exposed via API"
  type        = string
  default     = "glm-5-fp8"
}

variable "sglang_tool_call_parser" {
  description = "Tool call parser for the model"
  type        = string
  default     = "glm47"
}

variable "sglang_reasoning_parser" {
  description = "Reasoning parser for the model"
  type        = string
  default     = "glm45"
}
