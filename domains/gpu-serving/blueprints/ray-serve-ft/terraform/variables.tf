variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name prefix for resources (keep short for IAM limits)"
  type        = string
  default     = "ray-ft"
}

variable "eks_cluster_name" {
  description = "Existing EKS cluster name"
  type        = string
  default     = "qn-sglang-eks-cluster"
}

variable "system_node_role_name" {
  description = "Existing IAM role name for system nodegroup"
  type        = string
  default     = "system-eks-node-group-20260303162535678500000024"
}

variable "gpu_node_role_name" {
  description = "Existing IAM role name for GPU nodegroup"
  type        = string
  default     = "gpu-eks-node-group-20260303162535678600000025"
}
