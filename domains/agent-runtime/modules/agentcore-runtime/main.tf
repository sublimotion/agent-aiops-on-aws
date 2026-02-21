# AgentCore Runtime module
# Provisions a Bedrock AgentCore Runtime resource with VPC networking and IAM roles.
# Stub — fill in resource definitions during first RALPH loop for an agent-runtime blueprint.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}
