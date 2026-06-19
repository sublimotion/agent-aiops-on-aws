# Nemotron-3-Ultra Blueprint Variables

# ============================================================================
# AWS / Provider
# ============================================================================

variable "aws_region" {
  description = "AWS region (B300 long-context leg)"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "nemotron-ultra"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

# ============================================================================
# EKS Cluster (existing — reused)
# ============================================================================

variable "eks_cluster_name" {
  description = "Name of the existing EKS cluster to deploy into"
  type        = string
  default     = "qn-sglang-eks-cluster"
}

variable "namespace" {
  description = "Kubernetes namespace for workloads"
  type        = string
  default     = "ai-infra"
}

variable "service_account_name" {
  description = "Kubernetes service account for GPU workloads"
  type        = string
  default     = "default"
}

# ============================================================================
# GPU Node Group (existing managed node group — scaled by aws CLI, not TF)
# ============================================================================

variable "gpu_node_group_name" {
  description = "Existing managed node group of p6-b300.48xlarge spot nodes"
  type        = string
  default     = "ai-infra-b300-spot"
}

variable "gpu_node_taint_key" {
  description = "Taint key on the B300 node group (pods must tolerate)"
  type        = string
  default     = "ai-infra/b300"
}

variable "gpu_node_label_selector" {
  description = "Node label selecting the B300 spot pool"
  type        = map(string)
  default = {
    "ai-infra/role" = "b300-spot"
  }
}

# ============================================================================
# Model Configuration
# ============================================================================

variable "model_id" {
  description = "HuggingFace model id"
  type        = string
  default     = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
}

variable "model_path" {
  description = "Path to model weights on NVMe (mounted into the serving container)"
  type        = string
  default     = "/mnt/nvme/models/nemotron-3-ultra-nvfp4"
}

variable "served_model_name" {
  description = "Model name registered in the OpenAI API"
  type        = string
  default     = "nvidia/nemotron-3-ultra"
}

variable "s3_model_bucket" {
  description = "S3 bucket holding the staged NVFP4 weights"
  type        = string
  default     = "qn-sglang-models-20260303161715850900000007"
}

variable "s3_model_prefix" {
  description = "S3 key prefix for the staged weights"
  type        = string
  default     = "nemotron-3-ultra-nvfp4"
}

# ============================================================================
# Serving (vLLM TP4 single replica — smoke config)
# ============================================================================

variable "serving_image" {
  description = "vLLM container image (B300 sm_103 requires -cu130 tag)"
  type        = string
  default     = "vllm/vllm-openai:v0.22.0-cu130"
}

variable "tp_size" {
  description = "Tensor parallelism degree (TP4 = NVIDIA's documented unit)"
  type        = number
  default     = 4
}

variable "max_model_len" {
  description = "Max context length (== max_position_embeddings = 262144)"
  type        = number
  default     = 262144
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization fraction"
  type        = number
  default     = 0.90
}

variable "max_num_seqs" {
  description = "Max concurrent sequences"
  type        = number
  default     = 16
}

variable "max_num_batched_tokens" {
  description = "Max batched tokens"
  type        = number
  default     = 32768
}

variable "enable_prefix_caching" {
  description = "Enable prefix caching. WARNING (Stage 0c mamba-mtp-prefix-cache): MTP + prefix caching may conflict with vLLM mamba 'align' mode on this hybrid model. Set false if the engine fails at startup."
  type        = bool
  default     = true
}

# ============================================================================
# Networking
# ============================================================================

variable "node_port" {
  description = "NodePort for external access"
  type        = number
  default     = 30090
}
