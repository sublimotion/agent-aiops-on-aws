# Artifact bucket for the Trinity evolution run.
# Private, SSE, versioned. CMA-ES state is tiny (<20K params ≈ 160KB/ckpt) and
# synced every iteration, so versioning + per-iter sync is the spot-reclaim
# insurance (cost-aware-routing lost a run to a mid-flight reclaim with nothing
# banked). Raw rollouts expire after N days; model_iter_*.npy + es_log.json are
# kept under a separate prefix that is NOT expired.
#
# If var.bench_bucket is set, we DO NOT create a bucket — the blueprint just
# writes under the existing shared bench bucket's trinity-coordinator/ prefix,
# and the IRSA policy in irsa-run-role.tf is scoped to that prefix. A dedicated
# bucket is created only when bench_bucket == "".

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

locals {
  create_bucket = var.bench_bucket == ""
  bucket_name   = local.create_bucket ? "${var.name}-${data.aws_caller_identity.current.account_id}" : var.bench_bucket
  # All keys live under this prefix (shared-bucket-safe).
  prefix     = "trinity-coordinator"
  bucket_arn = "arn:aws:s3:::${local.bucket_name}"
}

resource "aws_s3_bucket" "artifacts" {
  count  = local.create_bucket ? 1 : 0
  bucket = local.bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  count                   = local.create_bucket ? 1 : 0
  bucket                  = aws_s3_bucket.artifacts[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  count  = local.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  count  = local.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
    bucket_key_enabled = true
  }
}

# Expire only raw rollouts; keep checkpoints + es_log indefinitely.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  count  = local.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.artifacts[0].id
  rule {
    id     = "expire-raw-rollouts"
    status = "Enabled"
    filter { prefix = "${local.prefix}/rollouts/" }
    expiration { days = var.raw_rollout_retention_days }
  }
  rule {
    id     = "keep-checkpoints"
    status = "Enabled"
    filter { prefix = "${local.prefix}/" }
    # No expiration block here — checkpoints/es_log are retained. Clean up old
    # noncurrent versions to bound versioning cost.
    noncurrent_version_expiration { noncurrent_days = 90 }
  }
}
