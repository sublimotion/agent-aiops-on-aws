# Agent Memory module
# Provisions a DynamoDB table for AgentCore session state storage.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

resource "aws_dynamodb_table" "sessions" {
  name         = "${var.name}-sessions"
  billing_mode = var.billing_mode
  hash_key     = "session_id"

  attribute {
    name = "session_id"
    type = "S"
  }

  ttl {
    attribute_name = var.ttl_attribute
    enabled        = true
  }

  tags = var.tags
}
