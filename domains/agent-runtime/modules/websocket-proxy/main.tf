# WebSocket Proxy module
# Deploys a Node.js WebSocket proxy on ECS Fargate (ARM64) that forwards
# authenticated client connections to the Bedrock AgentCore Runtime.
# Stub — fill in resource definitions during first RALPH loop for an agent-runtime blueprint.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}
