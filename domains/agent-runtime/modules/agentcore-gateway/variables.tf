variable "name" {
  description = "Name prefix for AgentCore Gateway resources"
  type        = string
}

variable "agent_id" {
  description = "Bedrock agent ID to expose via the gateway"
  type        = string
}

variable "agent_alias_id" {
  description = "Bedrock agent alias ID to expose via the gateway"
  type        = string
}

variable "tool_name" {
  description = "MCP tool name exposed to clients (e.g. 'research')"
  type        = string
  default     = "research"
}

variable "tool_description" {
  description = "Human-readable description of the MCP tool"
  type        = string
  default     = "Run a multi-agent research pipeline and return a PDF report URL."
}

variable "cognito_user_pool_arn" {
  description = "Cognito user pool ARN for gateway JWT authorisation"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
