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

variable "storage_capacity_gib" {
  description = "Storage capacity in GiB. Minimums: 1200 (SCRATCH_2), 2400 (PERSISTENT_1), 4800 (PERSISTENT_2 at 1000 MB/s/TiB with EFA)"
  type        = number
  default     = 1200
}

variable "deployment_type" {
  description = "FSx deployment type: SCRATCH_1, SCRATCH_2, PERSISTENT_1, or PERSISTENT_2"
  type        = string
  default     = "SCRATCH_2"
}

variable "per_unit_storage_throughput" {
  description = "Throughput per TiB for PERSISTENT deployments. PERSISTENT_1: 50, 100, 200. PERSISTENT_2: 125, 250, 500, 1000"
  type        = number
  default     = 250
}

variable "efa_enabled" {
  description = "Enable Elastic Fabric Adapter (EFA) and GPUDirect Storage (GDS). Requires PERSISTENT_2. Must be set at creation time."
  type        = bool
  default     = false
}

variable "data_compression_type" {
  description = "Data compression configuration: LZ4 or NONE"
  type        = string
  default     = null
}

variable "file_system_type_version" {
  description = "Lustre version (e.g. 2.15 for GDS compatibility with AL2023). Null uses provider default."
  type        = string
  default     = null
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
