# Agent Memory module
# Provisions a DynamoDB table for AgentCore session state storage.
# Stub — fill in resource definitions during first RALPH loop for an agent-runtime blueprint.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}
