variable "name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "research-agent"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for subnets"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "tavily_api_key" {
  description = "Tavily Search API key (primary web search backend)"
  type        = string
  sensitive   = true
}

variable "brave_api_key" {
  description = "Brave Search API key (fallback web search backend)"
  type        = string
  sensitive   = true
}

variable "container_image_tag" {
  description = "Docker image tag to deploy (set after ECR push)"
  type        = string
  default     = "latest"
}

variable "ecs_cpu" {
  description = "ECS task CPU units"
  type        = number
  default     = 2048
}

variable "ecs_memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 4096
}

variable "foundation_model_id" {
  description = "Bedrock model ID for all agents"
  type        = string
  default     = "us.anthropic.claude-opus-4-6-20250514-v1:0"
}

variable "tags" {
  description = "Tags applied to all resources"
  type        = map(string)
  default = {
    project   = "research-agent"
    blueprint = "agent-runtime"
    managed   = "terraform"
  }
}
