# Cognito App Auth module
# Provisions a Cognito user pool and app client for agent authentication.
# Stub — fill in resource definitions during first RALPH loop for an agent-runtime blueprint.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}
