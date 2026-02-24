# AgentCore Runtime module
# Provisions IAM role, Bedrock AgentCore Runtime agent + alias.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# IAM — execution role that AgentCore Runtime assumes
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "agent" {
  name               = "${var.name}-agentcore-exec"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "bedrock" {
  statement {
    sid     = "InvokeFoundationModel"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:*:*:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "bedrock" {
  name   = "bedrock-invoke"
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.bedrock.json
}

data "aws_iam_policy_document" "s3" {
  statement {
    sid     = "ResearchOutputBucket"
    actions = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
    resources = [
      var.s3_output_bucket_arn,
      "${var.s3_output_bucket_arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "s3" {
  name   = "s3-output"
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.s3.json
}

data "aws_iam_policy_document" "secrets" {
  count = length(var.search_secrets_arns) > 0 ? 1 : 0
  statement {
    sid       = "SecretsRead"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = var.search_secrets_arns
  }
}

resource "aws_iam_role_policy" "secrets" {
  count  = length(var.search_secrets_arns) > 0 ? 1 : 0
  name   = "secrets-read"
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.secrets[0].json
}

data "aws_iam_policy_document" "efs" {
  statement {
    sid = "EFSMount"
    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
      "elasticfilesystem:ClientRootAccess",
    ]
    resources = [var.efs_file_system_arn]
  }
}

resource "aws_iam_role_policy" "efs" {
  name   = "efs-mount"
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.efs.json
}

# ---------------------------------------------------------------------------
# Bedrock AgentCore Runtime agent
# ---------------------------------------------------------------------------

resource "aws_bedrockagent_agent" "this" {
  agent_name                  = var.name
  agent_resource_role_arn     = aws_iam_role.agent.arn
  foundation_model            = var.foundation_model_id
  instruction                 = "You are a research orchestrator. Decompose the user query into subtopics, spawn parallel researcher subagents, coordinate data analysis, and produce a PDF report."
  idle_session_ttl_in_seconds = var.idle_session_ttl_seconds
  tags                        = var.tags
}

resource "aws_bedrockagent_agent_alias" "this" {
  agent_id         = aws_bedrockagent_agent.this.agent_id
  agent_alias_name = "${var.name}-default"
  tags             = var.tags
}
