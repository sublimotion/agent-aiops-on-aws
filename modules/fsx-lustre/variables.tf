variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for security group"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for FSx Lustre (single AZ)"
  type        = string
}

variable "allowed_security_group_ids" {
  description = "Security group IDs allowed to access FSx (e.g., EKS node security group)"
  type        = list(string)
}

variable "storage_capacity" {
  description = "Storage capacity in GiB (must be multiple of 1200 for SCRATCH_2, 2400 for PERSISTENT)"
  type        = number
  default     = 1200
}

variable "deployment_type" {
  description = "FSx deployment type: SCRATCH_1, SCRATCH_2, PERSISTENT_1, or PERSISTENT_2"
  type        = string
  default     = "SCRATCH_2"
}

variable "per_unit_storage_throughput" {
  description = "Throughput per TiB for PERSISTENT deployments (50, 100, 200)"
  type        = number
  default     = 200
}

variable "log_level" {
  description = "Logging level: DISABLED, WARN_ONLY, ERROR_ONLY, WARN_ERROR"
  type        = string
  default     = "WARN_ERROR"
}

variable "log_destination" {
  description = "CloudWatch log group ARN for FSx logs (null to disable)"
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
