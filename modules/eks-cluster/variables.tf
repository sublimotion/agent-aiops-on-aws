variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.29"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs"
  type        = list(string)
}

# System node group configuration
variable "system_instance_types" {
  description = "Instance types for system node group"
  type        = list(string)
  default     = ["m6i.large"]
}

variable "system_min_size" {
  description = "Minimum number of system nodes"
  type        = number
  default     = 2
}

variable "system_max_size" {
  description = "Maximum number of system nodes"
  type        = number
  default     = 4
}

variable "system_desired_size" {
  description = "Desired number of system nodes"
  type        = number
  default     = 2
}

# GPU node group configuration
variable "enable_gpu_nodes" {
  description = "Enable GPU node group"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "Instance types for GPU node group (in priority order)"
  type        = list(string)
  default     = ["g6e.xlarge", "g6.xlarge", "g6e.2xlarge", "g6.2xlarge"]
}

variable "gpu_subnet_ids" {
  description = "Specific subnet IDs for GPU nodes (for AZ filtering). If null, uses all private subnets."
  type        = list(string)
  default     = null
}

variable "gpu_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 3
}

variable "gpu_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_volume_size" {
  description = "Root volume size for GPU nodes (GB)"
  type        = number
  default     = 200
}

# Access configuration
variable "access_entries" {
  description = "Map of access entries for the EKS cluster"
  type        = any
  default     = {}
}

variable "cluster_security_group_additional_rules" {
  description = "Additional security group rules for the cluster"
  type        = any
  default     = {}
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin container image (use ECR for private subnets)"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for GPU nodes (required for p5e instances)"
  type        = bool
  default     = false
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes (AL2_x86_64_GPU or AL2023_x86_64_NVIDIA)"
  type        = string
  default     = "AL2_x86_64_GPU"
}

variable "capacity_reservation_id" {
  description = "Capacity reservation ID for GPU instances (e.g., for capacity blocks)"
  type        = string
  default     = null
}
