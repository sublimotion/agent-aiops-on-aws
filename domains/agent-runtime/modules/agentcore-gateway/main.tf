# AgentCore Gateway module
# Exposes a Bedrock agent as a managed MCP server endpoint via AgentCore Gateway.
#
# NOTE: Amazon Bedrock AgentCore Gateway (GA 2025) does not yet have a first-class
# Terraform resource in the AWS provider. This module uses null_resource + local-exec
# with the AWS CLI to create/update the gateway. The RALPH loop will replace this
# with a native resource when provider support lands.
#
# Prerequisite: AWS CLI >= 2.15 with agentcore subcommands available.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.0"
    }
  }
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# IAM — allow AgentCore Gateway to invoke the agent alias
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "gateway_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "gateway" {
  name               = "${var.name}-gateway"
  assume_role_policy = data.aws_iam_policy_document.gateway_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "gateway_invoke" {
  statement {
    actions   = ["bedrock-agent-runtime:InvokeAgent"]
    resources = ["arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:agent-alias/${var.agent_id}/${var.agent_alias_id}"]
  }
}

resource "aws_iam_role_policy" "gateway_invoke" {
  name   = "invoke-agent"
  role   = aws_iam_role.gateway.id
  policy = data.aws_iam_policy_document.gateway_invoke.json
}

# ---------------------------------------------------------------------------
# SSM Parameter — persists gateway state across Terraform runs
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "gateway_config" {
  name = "/${var.name}/agentcore-gateway/config"
  type = "String"
  value = jsonencode({
    agent_id       = var.agent_id
    agent_alias_id = var.agent_alias_id
    tool_name      = var.tool_name
    gateway_role   = aws_iam_role.gateway.arn
  })
  tags = var.tags
}

# ---------------------------------------------------------------------------
# AgentCore Gateway — created via AWS CLI (no TF resource yet)
# ---------------------------------------------------------------------------

resource "null_resource" "gateway" {
  triggers = {
    agent_id       = var.agent_id
    agent_alias_id = var.agent_alias_id
    tool_name      = var.tool_name
    gateway_role   = aws_iam_role.gateway.arn
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail

      # Create gateway and capture the endpoint URL
      RESULT=$(aws bedrock-agentcore create-gateway \
        --gateway-name "${var.name}" \
        --role-arn "${aws_iam_role.gateway.arn}" \
        --protocol-configuration '{"mcp": {"agentAliasArn": "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:agent-alias/${var.agent_id}/${var.agent_alias_id}"}}' \
        --region "${var.region}" \
        --output json 2>/dev/null || echo '{"gatewayId":"","gatewayUrl":""}')

      GATEWAY_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gatewayId',''))")
      GATEWAY_URL=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gatewayUrl',''))")

      # Persist outputs to SSM so they survive Terraform refresh
      aws ssm put-parameter \
        --name "/${var.name}/agentcore-gateway/gateway-id" \
        --value "$GATEWAY_ID" \
        --type String --overwrite \
        --region "${var.region}" || true

      aws ssm put-parameter \
        --name "/${var.name}/agentcore-gateway/mcp-endpoint-url" \
        --value "$${GATEWAY_URL:-PENDING}" \
        --type String --overwrite \
        --region "${var.region}" || true

      echo "Gateway ID: $GATEWAY_ID"
      echo "MCP endpoint: $GATEWAY_URL"
    EOT
  }

  depends_on = [aws_iam_role_policy.gateway_invoke]
}

# ---------------------------------------------------------------------------
# Read back the gateway outputs from SSM
# ---------------------------------------------------------------------------

data "aws_ssm_parameter" "gateway_id" {
  name       = "/${var.name}/agentcore-gateway/gateway-id"
  depends_on = [null_resource.gateway]
}

data "aws_ssm_parameter" "mcp_endpoint_url" {
  name       = "/${var.name}/agentcore-gateway/mcp-endpoint-url"
  depends_on = [null_resource.gateway]
}

locals {
  gateway_id       = data.aws_ssm_parameter.gateway_id.value
  mcp_endpoint_url = data.aws_ssm_parameter.mcp_endpoint_url.value
}
