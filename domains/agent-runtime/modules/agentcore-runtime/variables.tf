variable "name" {
  description = "Name prefix for all resources in this module"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where AgentCore Runtime resources will be deployed"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Private subnet IDs for AgentCore Runtime networking"
  type        = list(string)
  default     = []
}

variable "execution_role_arn" {
  description = "External IAM role ARN (unused — module creates its own role)"
  type        = string
  default     = ""
}

variable "idle_session_ttl_seconds" {
  description = "Seconds of inactivity before a session is terminated"
  type        = number
  default     = 1800
}

variable "foundation_model_id" {
  description = "Bedrock foundation model ID for the agent"
  type        = string
}

variable "efs_file_system_arn" {
  description = "ARN of the EFS file system used for inter-agent file coordination"
  type        = string
}

variable "s3_output_bucket_arn" {
  description = "ARN of the S3 bucket for research output files"
  type        = string
}

variable "search_secrets_arns" {
  description = "Secrets Manager ARNs for search API keys (Tavily, Brave) that the agent container loads at startup"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
