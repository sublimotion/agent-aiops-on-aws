variable "name" {
  description = "Name prefix for all resources in this module"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where AgentCore Runtime resources will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "Private subnet IDs for AgentCore Runtime networking"
  type        = list(string)
}

variable "execution_role_arn" {
  description = "IAM role ARN that AgentCore Runtime assumes to invoke foundation models"
  type        = string
}

variable "idle_session_ttl_seconds" {
  description = "Seconds of inactivity before a session is terminated"
  type        = number
  default     = 600
}

variable "foundation_model_id" {
  description = "Bedrock foundation model ID for the agent (e.g. anthropic.claude-3-5-sonnet-20241022-v2:0)"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
