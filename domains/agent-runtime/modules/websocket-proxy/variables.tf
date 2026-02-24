variable "name" {
  description = "Name prefix for WebSocket proxy resources"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for ECS cluster and service"
  type        = string
}

variable "subnet_ids" {
  description = "Private subnet IDs for ECS tasks"
  type        = list(string)
}

variable "image_uri" {
  description = "Full ECR image URI (including digest or tag) for the agent container"
  type        = string
}

variable "agent_id" {
  description = "Bedrock AgentCore Runtime agent ID"
  type        = string
}

variable "agent_alias_id" {
  description = "Bedrock AgentCore Runtime agent alias ID"
  type        = string
}

variable "cognito_user_pool_id" {
  description = "Cognito user pool ID for JWT validation (informational; validated in server.py)"
  type        = string
}

variable "cognito_app_client_id" {
  description = "Cognito app client ID for JWT validation"
  type        = string
}

variable "efs_file_system_id" {
  description = "EFS file system ID to mount at /app/files in the container"
  type        = string
}

variable "s3_output_bucket_arn" {
  description = "ARN of the S3 bucket for research output files"
  type        = string
}

variable "s3_output_bucket_name" {
  description = "Name of the S3 bucket for research output files (injected as env var)"
  type        = string
}

variable "search_api_secret_arn" {
  description = "Secrets Manager ARN for the Brave search API key"
  type        = string
}

variable "tavily_api_key_secret_arn" {
  description = "Secrets Manager ARN for the Tavily search API key"
  type        = string
}

variable "session_table_arn" {
  description = "DynamoDB session table ARN"
  type        = string
}

variable "session_table_name" {
  description = "DynamoDB session table name (injected as env var)"
  type        = string
}

variable "container_port" {
  description = "Port the agent server listens on inside the container"
  type        = number
  default     = 8080
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "cpu" {
  description = "ECS task CPU units (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 2048
}

variable "memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 4096
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
