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
  description = "Full ECR image URI (including digest or tag) for the proxy container"
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
  description = "Cognito user pool ID for JWT validation"
  type        = string
}

variable "cognito_app_client_id" {
  description = "Cognito app client ID for JWT validation"
  type        = string
}

variable "container_port" {
  description = "Port the proxy listens on inside the container"
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
  default     = 512
}

variable "memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 1024
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
