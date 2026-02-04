terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "replica_region" {
  description = "AWS region for replica bucket"
  type        = string
  default     = "us-west-2"
}

variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
  default     = "secure-bucket-aiops-demo"
}

# KMS key for SNS encryption
resource "aws_kms_key" "sns_key" {
  description             = "KMS key for SNS topic encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name        = "${var.bucket_name}-sns-key"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_kms_alias" "sns_key_alias" {
  name          = "alias/${var.bucket_name}-sns-key"
  target_key_id = aws_kms_key.sns_key.key_id
}

# Logging bucket for access logs
#checkov:skip=CKV2_AWS_62:Logging bucket does not require event notifications
#checkov:skip=CKV_AWS_144:Logging bucket does not require cross-region replication to avoid circular dependencies
resource "aws_s3_bucket" "logging_bucket" {
  bucket = "${var.bucket_name}-logs"

  tags = {
    Name        = "${var.bucket_name}-logs"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "logging_bucket_versioning" {
  bucket = aws_s3_bucket.logging_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logging_bucket_encryption" {
  bucket = aws_s3_bucket.logging_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "logging_bucket_public_access_block" {
  bucket = aws_s3_bucket.logging_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#checkov:skip=CKV2_AWS_65:Logging bucket requires BucketOwnerPreferred for S3 log delivery
resource "aws_s3_bucket_ownership_controls" "logging_bucket_ownership" {
  bucket = aws_s3_bucket.logging_bucket.id

  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "logging_bucket_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.logging_bucket_ownership]
  bucket     = aws_s3_bucket.logging_bucket.id
  acl        = "log-delivery-write"
}

resource "aws_s3_bucket_lifecycle_configuration" "logging_bucket_lifecycle" {
  bucket = aws_s3_bucket.logging_bucket.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Main S3 Bucket
resource "aws_s3_bucket" "secure_bucket" {
  bucket = var.bucket_name

  tags = {
    Name        = var.bucket_name
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "secure_bucket_versioning" {
  bucket = aws_s3_bucket.secure_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption with KMS
resource "aws_s3_bucket_server_side_encryption_configuration" "secure_bucket_encryption" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "secure_bucket_public_access_block" {
  bucket = aws_s3_bucket.secure_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Disable ACLs (bucket owner enforced)
resource "aws_s3_bucket_ownership_controls" "secure_bucket_ownership" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Access logging
resource "aws_s3_bucket_logging" "secure_bucket_logging" {
  bucket = aws_s3_bucket.secure_bucket.id

  target_bucket = aws_s3_bucket.logging_bucket.id
  target_prefix = "access-logs/"
}

# Lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "secure_bucket_lifecycle" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"

    filter {}

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# SNS Topic for event notifications with encryption
resource "aws_sns_topic" "bucket_notifications" {
  name              = "${var.bucket_name}-notifications"
  kms_master_key_id = aws_kms_key.sns_key.id

  tags = {
    Name        = "${var.bucket_name}-notifications"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_sns_topic_policy" "bucket_notifications_policy" {
  arn = aws_sns_topic.bucket_notifications.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowS3ToPublish"
        Effect    = "Allow"
        Principal = { Service = "s3.amazonaws.com" }
        Action    = "sns:Publish"
        Resource  = aws_sns_topic.bucket_notifications.arn
        Condition = {
          ArnLike = {
            "aws:SourceArn" = aws_s3_bucket.secure_bucket.arn
          }
        }
      }
    ]
  })
}

# Event notifications
resource "aws_s3_bucket_notification" "secure_bucket_notification" {
  bucket = aws_s3_bucket.secure_bucket.id

  topic {
    topic_arn     = aws_sns_topic.bucket_notifications.arn
    events        = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
    filter_suffix = ""
  }

  depends_on = [aws_sns_topic_policy.bucket_notifications_policy]
}

# IAM role for replication
resource "aws_iam_role" "replication_role" {
  name = "${var.bucket_name}-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.bucket_name}-replication-role"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role_policy" "replication_policy" {
  name = "${var.bucket_name}-replication-policy"
  role = aws_iam_role.replication_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = aws_s3_bucket.secure_bucket.arn
      },
      {
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Effect   = "Allow"
        Resource = "${aws_s3_bucket.secure_bucket.arn}/*"
      },
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Effect   = "Allow"
        Resource = "${aws_s3_bucket.replica_bucket.arn}/*"
      }
    ]
  })
}

# Replica bucket in different region
provider "aws" {
  alias  = "replica"
  region = var.replica_region
}

# Logging bucket in replica region for replica bucket logs
#checkov:skip=CKV2_AWS_62:Replica logging bucket does not require event notifications
#checkov:skip=CKV_AWS_144:Replica logging bucket does not require cross-region replication
resource "aws_s3_bucket" "replica_logging_bucket" {
  provider = aws.replica
  bucket   = "${var.bucket_name}-replica-logs"

  tags = {
    Name        = "${var.bucket_name}-replica-logs"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "replica_logging_bucket_versioning" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_logging_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "replica_logging_bucket_encryption" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_logging_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "replica_logging_bucket_public_access_block" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_logging_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#checkov:skip=CKV2_AWS_65:Logging bucket requires BucketOwnerPreferred for S3 log delivery
resource "aws_s3_bucket_ownership_controls" "replica_logging_bucket_ownership" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_logging_bucket.id

  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "replica_logging_bucket_acl" {
  provider   = aws.replica
  depends_on = [aws_s3_bucket_ownership_controls.replica_logging_bucket_ownership]
  bucket     = aws_s3_bucket.replica_logging_bucket.id
  acl        = "log-delivery-write"
}

resource "aws_s3_bucket_lifecycle_configuration" "replica_logging_bucket_lifecycle" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_logging_bucket.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

#checkov:skip=CKV2_AWS_62:Replica bucket does not require event notifications
resource "aws_s3_bucket" "replica_bucket" {
  provider = aws.replica
  bucket   = "${var.bucket_name}-replica"

  tags = {
    Name        = "${var.bucket_name}-replica"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "replica_bucket_versioning" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "replica_bucket_encryption" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "replica_bucket_public_access_block" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "replica_bucket_ownership" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Access logging for replica bucket
resource "aws_s3_bucket_logging" "replica_bucket_logging" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  target_bucket = aws_s3_bucket.replica_logging_bucket.id
  target_prefix = "access-logs/"
}

resource "aws_s3_bucket_lifecycle_configuration" "replica_bucket_lifecycle" {
  provider = aws.replica
  bucket   = aws_s3_bucket.replica_bucket.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Replication configuration
resource "aws_s3_bucket_replication_configuration" "secure_bucket_replication" {
  depends_on = [
    aws_s3_bucket_versioning.secure_bucket_versioning,
    aws_s3_bucket_versioning.replica_bucket_versioning
  ]

  role   = aws_iam_role.replication_role.arn
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    id     = "replicate-all"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.replica_bucket.arn
      storage_class = "STANDARD"
    }
  }
}

# Output the bucket ARN and name
output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.secure_bucket.arn
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.secure_bucket.id
}

output "replica_bucket_arn" {
  description = "ARN of the replica S3 bucket"
  value       = aws_s3_bucket.replica_bucket.arn
}

output "logging_bucket_name" {
  description = "Name of the logging bucket"
  value       = aws_s3_bucket.logging_bucket.id
}
