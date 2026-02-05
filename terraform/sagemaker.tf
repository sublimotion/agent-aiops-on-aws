# SageMaker Studio Domain with VSCode (Code Editor) support
# and EKS cluster connectivity

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

  tags = {
    Name = "${var.project_name}-sagemaker-execution-role"
  }
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

# EBS access for SageMaker Code Editor (required for workspace volumes)
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
            "aws:RequestedRegion" = var.aws_region
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
        Resource = "arn:aws:ec2:${var.aws_region}:${data.aws_caller_identity.current.account_id}:volume/*"
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
        Resource = var.enable_sagemaker_studio ? aws_kms_key.sagemaker[0].arn : "*"
      }
    ]
  })
}

# EKS access policy for SageMaker
#checkov:skip=CKV_AWS_355:EKS ListClusters requires * resource, cluster ARN used where possible
resource "aws_iam_role_policy" "sagemaker_eks_policy" {
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
        Resource = "arn:aws:eks:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster/${var.project_name}-eks-cluster"
      },
      {
        Sid    = "EKSList"
        Effect = "Allow"
        Action = [
          "eks:ListClusters"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
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
        Resource = "arn:aws:ecr:${var.aws_region}:*:repository/*"
      }
    ]
  })
}

# KMS key for SageMaker encryption
resource "aws_kms_key" "sagemaker" {
  count = var.enable_sagemaker_studio ? 1 : 0

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

  tags = {
    Name = "${var.project_name}-sagemaker-kms"
  }
}

resource "aws_kms_alias" "sagemaker" {
  count = var.enable_sagemaker_studio ? 1 : 0

  name          = "alias/${var.project_name}-sagemaker"
  target_key_id = aws_kms_key.sagemaker[0].key_id
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# Security group for SageMaker Studio
resource "aws_security_group" "sagemaker_studio" {
  count = var.enable_sagemaker_studio ? 1 : 0

  name        = "${var.project_name}-sagemaker-studio-sg"
  description = "Security group for SageMaker Studio"
  vpc_id      = module.vpc.vpc_id

  # Allow all outbound traffic
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow NFS for EFS (SageMaker uses EFS)
  ingress {
    description = "NFS from VPC"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # Allow HTTPS
  ingress {
    description = "HTTPS from VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "${var.project_name}-sagemaker-studio-sg"
  }
}

# SageMaker Studio Domain
resource "aws_sagemaker_domain" "studio" {
  count = var.enable_sagemaker_studio ? 1 : 0

  domain_name = var.sagemaker_domain_name
  auth_mode   = "IAM"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets
  kms_key_id  = aws_kms_key.sagemaker[0].arn

  default_user_settings {
    execution_role  = aws_iam_role.sagemaker_execution_role.arn
    security_groups = [aws_security_group.sagemaker_studio[0].id]

    # Enable Code Editor (VSCode) - uses built-in SageMaker Code Editor
    code_editor_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }

    # JupyterLab settings
    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }

    # Kernel Gateway for custom images
    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }

  # Domain settings
  domain_settings {
    security_group_ids = [aws_security_group.sagemaker_studio[0].id]
  }

  # Retention policy
  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = {
    Name = var.sagemaker_domain_name
  }
}

# SageMaker User Profile
resource "aws_sagemaker_user_profile" "default" {
  count = var.enable_sagemaker_studio ? 1 : 0

  domain_id         = aws_sagemaker_domain.studio[0].id
  user_profile_name = "default-user"

  user_settings {
    execution_role  = aws_iam_role.sagemaker_execution_role.arn
    security_groups = [aws_security_group.sagemaker_studio[0].id]

    code_editor_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }

  tags = {
    Name = "${var.project_name}-default-user"
  }
}

# Store EKS cluster info in SSM for easy access from SageMaker
resource "aws_ssm_parameter" "eks_cluster_name" {
  count = var.enable_sagemaker_studio ? 1 : 0

  name        = "/${var.project_name}/eks/cluster-name"
  description = "EKS cluster name for SageMaker connectivity"
  type        = "String"
  value       = module.eks.cluster_name

  tags = {
    Name = "${var.project_name}-eks-cluster-name"
  }
}

resource "aws_ssm_parameter" "eks_cluster_endpoint" {
  count = var.enable_sagemaker_studio ? 1 : 0

  name        = "/${var.project_name}/eks/cluster-endpoint"
  description = "EKS cluster endpoint for SageMaker connectivity"
  type        = "String"
  value       = module.eks.cluster_endpoint

  tags = {
    Name = "${var.project_name}-eks-cluster-endpoint"
  }
}
