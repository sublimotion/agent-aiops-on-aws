# IRSA run-role for the Trinity evolution Job.
# Trust: the EKS cluster OIDC provider, restricted to the runner SA in the runner
# namespace. Least-privilege: Bedrock invoke on exactly the 7 worker model/profile
# ARNs (across the throttle-rotation regions) + S3 RW on this blueprint's prefix.
#
# Carryover: Anthropic + DeepSeek + Nova workers are reached via cross-region
# inference profiles (us.*), which require BOTH the inference-profile ARN AND the
# underlying foundation-model ARNs in every region the profile may route to. We
# grant the us. profile ARNs and wildcard the underlying foundation-model ARNs
# per region. (Model-ID drift lesson, confirmed live 2026-06-24.)
#
# GPT-5.5 (optional ord-0 swap) is NOT covered here — it is invoked via the
# operator's bearer token over the OpenAI-compatible endpoint, a credential
# mounted as a K8s secret, NOT this IRSA role. This role intentionally has no
# grant for openai.gpt-5.5; that path bypasses SigV4 entirely.

data "aws_partition" "current" {}

locals {
  account = data.aws_caller_identity.current.account_id
  part    = data.aws_partition.current.partition

  # us. cross-region inference-profile IDs used by the pool (region-prefixed at
  # invoke time; the profile ARN itself is account+region scoped).
  inference_profile_ids = [
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-sonnet-4-6",
    "us.amazon.nova-pro-v1:0",
    "us.deepseek.r1-v1:0",
  ]

  # On-demand model IDs invoked directly (no inference profile).
  on_demand_model_ids = [
    "google.gemma-3-27b-it",
    "qwen.qwen3-32b-v1:0", # serves ord 5 (reasoning) AND ord 6 (direct)
  ]

  # Inference-profile ARNs (one per region).
  inference_profile_arns = flatten([
    for region in var.worker_regions : [
      for id in local.inference_profile_ids :
      "arn:${local.part}:bedrock:${region}:${local.account}:inference-profile/${id}"
    ]
  ])

  # Foundation-model ARNs the profiles + on-demand calls resolve to. Profiles can
  # route to any region, so grant foundation-model/* per region (resource-level,
  # still bedrock-only, not account-wide *). On-demand ids are a subset of these.
  foundation_model_arns = flatten([
    for region in var.worker_regions :
    ["arn:${local.part}:bedrock:${region}::foundation-model/*"]
  ])
}

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
    condition {
      test     = "StringLike"
      variable = "${var.oidc_provider_url}:sub"
      values   = ["system:serviceaccount:${var.runner_namespace}:${var.service_account_pattern}"]
    }
  }
}

resource "aws_iam_role" "run" {
  name                 = "${var.name}-run-role"
  assume_role_policy   = data.aws_iam_policy_document.run_trust.json
  max_session_duration = var.max_session_duration
  tags                 = var.tags
}

data "aws_iam_policy_document" "run_perms" {
  # Bedrock invoke on exactly the worker inference-profile ARNs + the foundation
  # models they (and the on-demand workers) resolve to. Converse uses InvokeModel.
  statement {
    sid    = "BedrockInvokeWorkers"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:Converse",
      "bedrock:ConverseStream",
    ]
    resources = concat(local.inference_profile_arns, local.foundation_model_arns)
  }

  # Artifacts: RW only under this blueprint's prefix.
  statement {
    sid       = "S3ReadWritePrefix"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
    resources = ["${local.bucket_arn}/${local.prefix}/*"]
  }
  statement {
    sid       = "S3ListPrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.prefix}/*"]
    }
  }

  # KMS for the SSE bucket.
  statement {
    sid       = "KmsForArtifacts"
    effect    = "Allow"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = ["*"]
  }

  # Caller identity check used by the Gate 0.0 step-0 smoke (sts:GetCallerIdentity
  # is implicitly allowed; no statement needed).
}

resource "aws_iam_role_policy" "run" {
  name   = "trinity-run-perms"
  role   = aws_iam_role.run.id
  policy = data.aws_iam_policy_document.run_perms.json
}
