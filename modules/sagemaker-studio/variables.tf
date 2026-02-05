variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "domain_name" {
  description = "Name for SageMaker Studio domain"
  type        = string
}

variable "user_profile_name" {
  description = "Name for default user profile"
  type        = string
  default     = "default-user"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for SageMaker"
  type        = list(string)
}

variable "default_instance_type" {
  description = "Default instance type for SageMaker apps"
  type        = string
  default     = "ml.t3.medium"
}

variable "eks_cluster_name" {
  description = "EKS cluster name for IAM permissions (optional)"
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
