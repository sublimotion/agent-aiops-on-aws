# SageMaker Studio Module - Domain, User Profile, IAM, KMS

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# SageMaker full access policy
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# S3 access for SageMaker
resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# EBS access for SageMaker Code Editor
resource "aws_iam_role_policy" "sagemaker_ebs_policy" {
  name = "${var.project_name}-sagemaker-ebs-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EBSVolumeManagement"
        Effect = "Allow"
        Action = [
          "ec2:CreateVolume",
          "ec2:DeleteVolume",
          "ec2:AttachVolume",
          "ec2:DetachVolume",
          "ec2:DescribeVolumes",
          "ec2:DescribeVolumeStatus",
          "ec2:ModifyVolume",
          "ec2:DescribeVolumeAttribute"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = data.aws_region.current.name
          }
        }
      },
      {
        Sid    = "EBSTagging"
        Effect = "Allow"
        Action = [
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ]
        Resource = "arn:aws:ec2:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:volume/*"
      },
      {
        Sid    = "KMSForEBS"
        Effect = "Allow"
        Action = [
          "kms:CreateGrant",
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKeyWithoutPlaintext",
          "kms:ReEncrypt*"
        ]
        Resource = aws_kms_key.sagemaker.arn
      }
    ]
  })
}

# EKS access policy for SageMaker
resource "aws_iam_role_policy" "sagemaker_eks_policy" {
  count = var.eks_cluster_name != null ? 1 : 0

  name = "${var.project_name}-sagemaker-eks-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EKSDescribe"
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:AccessKubernetesApi"
        ]
        Resource = "arn:aws:eks:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:cluster/${var.eks_cluster_name}"
      },
      {
        Sid      = "EKSList"
        Effect   = "Allow"
        Action   = ["eks:ListClusters"]
        Resource = "*"
      },
      {
        Sid      = "ECRAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "arn:aws:ecr:${data.aws_region.current.name}:*:repository/*"
      }
    ]
  })
}

# KMS key for SageMaker encryption
resource "aws_kms_key" "sagemaker" {
  description             = "KMS key for SageMaker Studio encryption"
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
        Sid    = "Allow SageMaker"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.project_name}-sagemaker-kms"
  })
}

resource "aws_kms_alias" "sagemaker" {
  name          = "alias/${var.project_name}-sagemaker"
  target_key_id = aws_kms_key.sagemaker.key_id
}

# Security group for SageMaker Studio
resource "aws_security_group" "sagemaker_studio" {
  name        = "${var.project_name}-sagemaker-studio-sg"
  description = "Security group for SageMaker Studio"
  vpc_id      = var.vpc_id

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "NFS from VPC"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "HTTPS from VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-sagemaker-studio-sg"
  })
}

# SageMaker Studio Domain
resource "aws_sagemaker_domain" "studio" {
  domain_name = var.domain_name
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id
  subnet_ids  = var.subnet_ids
  kms_key_id  = aws_kms_key.sagemaker.arn

  default_user_settings {
    execution_role  = aws_iam_role.sagemaker_execution_role.arn
    security_groups = [aws_security_group.sagemaker_studio.id]

    code_editor_app_settings {
      default_resource_spec {
        instance_type = var.default_instance_type
      }
    }

    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type = var.default_instance_type
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = var.default_instance_type
      }
    }
  }

  domain_settings {
    security_group_ids = [aws_security_group.sagemaker_studio.id]
  }

  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = merge(var.tags, {
    Name = var.domain_name
  })
}

# SageMaker User Profile
resource "aws_sagemaker_user_profile" "default" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = var.user_profile_name

  user_settings {
    execution_role  = aws_iam_role.sagemaker_execution_role.arn
    security_groups = [aws_security_group.sagemaker_studio.id]

    code_editor_app_settings {
      default_resource_spec {
        instance_type = var.default_instance_type
      }
    }
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.user_profile_name}"
  })
}
