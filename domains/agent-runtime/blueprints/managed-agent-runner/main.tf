# managed-agent-runner — AWS foundation for the agent-runner CLI (sibling repo).
# Provisions ONLY what the spec says lives in this repo: ECR repo for the runtime image,
# private artifact bucket, run-state DynamoDB table, and the scoped IRSA run-role.
# The CLI, harness adapters, Dockerfile, and Job template live in ../agent-runner.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags { tags = var.tags }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name = var.name
}

# ---------------------------------------------------------------------------
# ECR — runtime profile image (full-deploy@v1 → agent-runner-full-deploy:v1)
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "runtime" {
  name                 = "agent-runner-${var.runtime_profile}"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
  tags = var.tags
}

resource "aws_ecr_lifecycle_policy" "runtime" {
  repository = aws_ecr_repository.runtime.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# Artifact bucket — PRIVATE (R4: authenticated pull only, no public URLs)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-run-artifacts"
    status = "Enabled"
    filter { prefix = "runs/" }
    expiration { days = var.artifact_retention_days }
  }
}

# ---------------------------------------------------------------------------
# Run-state table (R2: survives pod recycle; status + last_heartbeat)
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "runs" {
  name         = "${local.name}-runs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"

  attribute {
    name = "run_id"
    type = "S"
  }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
  point_in_time_recovery { enabled = true }
  tags = var.tags
}

# ---------------------------------------------------------------------------
# IRSA run-role (R6: scoped per-run credential boundary)
# Trust: the EKS cluster OIDC provider, restricted to SAs named agent-runner-*
# in the runner namespace. Least-privilege: this bucket + this table + read ECR.
# Extend the inline policy per spec domain (e.g. terraform/eks for deploy specs).
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "run_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.oidc_provider_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
    # SA name pattern agent-runner-* in the runner namespace (per-run SAs).
    condition {
      test     = "StringLike"
      variable = "${var.oidc_provider_url}:sub"
      values   = ["system:serviceaccount:${var.runner_namespace}:agent-runner-*"]
    }
  }
}

resource "aws_iam_role" "run" {
  name               = "${local.name}-run-role"
  assume_role_policy = data.aws_iam_policy_document.run_trust.json
  tags               = var.tags
}

data "aws_iam_policy_document" "run_perms" {
  # Artifacts: write this run's prefix, read for status.
  statement {
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/runs/*"]
  }
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  # Run-state table.
  statement {
    effect    = "Allow"
    actions   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.runs.arn]
  }
  # Pull the runtime image.
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  # KMS for the SSE bucket.
  statement {
    effect    = "Allow"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "run" {
  name   = "base"
  role   = aws_iam_role.run.id
  policy = data.aws_iam_policy_document.run_perms.json
}

# Optional: attach an operator-supplied policy ARN for the spec's deploy domain
# (e.g. a terraform/eks deploy policy). Scoped, not admin.
resource "aws_iam_role_policy_attachment" "domain" {
  for_each   = toset(var.extra_policy_arns)
  role       = aws_iam_role.run.name
  policy_arn = each.value
}
