# build.tf — CodeBuild project that bakes the full-deploy runtime image.
# Builds the agent-runner Dockerfile (arm64) and pushes to the ECR repo above,
# so cold start drops from ~3min (runtime install-deps) to an image pull.
# Managed (no instance lifecycle), native arm64, repeatable for future profiles.

resource "aws_iam_role" "build" {
  name = "${local.name}-build"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = var.tags
}

data "aws_iam_policy_document" "build" {
  # CloudWatch Logs for the build.
  statement {
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
  # ECR push to the runtime repo.
  statement {
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:PutImage",
      "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "build" {
  name   = "build"
  role   = aws_iam_role.build.id
  policy = data.aws_iam_policy_document.build.json
}

resource "aws_codebuild_project" "runtime" {
  name         = "${local.name}-runtime-build"
  description  = "Bakes the agent-runner full-deploy runtime image (arm64) to ECR"
  service_role = aws_iam_role.build.arn

  artifacts { type = "NO_ARTIFACTS" }

  environment {
    # arm64 (Graviton) build image — matches the Dockerfile's --platform linux/arm64.
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/amazonlinux2-aarch64-standard:3.0"
    type                        = "ARM_CONTAINER"
    privileged_mode             = true # required for docker buildx
    image_pull_credentials_type = "CODEBUILD"

    environment_variable {
      name  = "ECR_REPO"
      value = aws_ecr_repository.runtime.repository_url
    }
    environment_variable {
      name  = "AWS_REGION"
      value = var.aws_region
    }
    environment_variable {
      name  = "IMAGE_TAG"
      value = var.image_tag
    }
  }

  source {
    type            = "GITHUB"
    location        = var.agent_runner_repo_url
    git_clone_depth = 1
    buildspec       = "docker/buildspec.yml"
  }

  source_version = var.agent_runner_ref

  tags = var.tags
}

output "build_project" {
  description = "Start a build with: aws codebuild start-build --project-name <this>"
  value       = aws_codebuild_project.runtime.name
}
