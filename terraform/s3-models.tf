# S3 Bucket for ML Model Storage
# Models are downloaded here and streamed to vLLM via Run:ai Model Streamer

resource "aws_s3_bucket" "models" {
  bucket_prefix = "${var.project_name}-models-"

  tags = {
    Name = "${var.project_name}-models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.models.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Lifecycle policy for model versions
resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  # Rule for cleaning up old model versions
  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # Rule for aborting failed multipart uploads (applies to all objects)
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    filter {} # Empty filter applies to all objects

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# KMS key for model bucket encryption
resource "aws_kms_key" "models" {
  description             = "KMS key for ML model storage encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow vLLM role to decrypt"
        Effect = "Allow"
        Principal = {
          AWS = var.enable_vllm_deployment ? module.vllm_irsa[0].iam_role_arn : "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "Allow SageMaker role for model upload"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.sagemaker_execution_role.arn
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-models-kms"
  }
}

resource "aws_kms_alias" "models" {
  name          = "alias/${var.project_name}-models"
  target_key_id = aws_kms_key.models.key_id
}

# S3 VPC Endpoint policy to allow access to model bucket
resource "aws_s3_bucket_policy" "models" {
  bucket = aws_s3_bucket.models.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowVPCAccess"
        Effect = "Allow"
        Principal = {
          AWS = [
            var.enable_vllm_deployment ? module.vllm_irsa[0].iam_role_arn : "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root",
            aws_iam_role.sagemaker_execution_role.arn
          ]
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetObjectVersion"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
        Condition = {
          StringEquals = {
            "aws:SourceVpc" = module.vpc.vpc_id
          }
        }
      }
    ]
  })
}
