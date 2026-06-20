variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "agent-runner"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "runtime_profile" {
  description = "Runtime profile name; ECR repo is agent-runner-<profile>"
  type        = string
  default     = "full-deploy"
}

variable "artifact_retention_days" {
  description = "Lifecycle expiry for run artifacts under runs/"
  type        = number
  default     = 30
}

variable "oidc_provider_arn" {
  description = "EKS cluster OIDC provider ARN (for IRSA trust)"
  type        = string
}

variable "oidc_provider_url" {
  description = "EKS cluster OIDC provider URL without https:// (e.g. oidc.eks.us-east-1.amazonaws.com/id/XXXX)"
  type        = string
}

variable "runner_namespace" {
  description = "Kubernetes namespace where agent-runner Jobs run"
  type        = string
  default     = "agent-runner"
}

variable "extra_policy_arns" {
  description = "Optional scoped policy ARNs to attach to the run role for the spec's deploy domain (NOT admin)"
  type        = list(string)
  default     = []
}

variable "max_session_duration" {
  description = "Max session duration (s) for the run role; vended harness sessions span up to this. 3600-43200."
  type        = number
  default     = 43200 # 12h — covers >8h interactive runs
}

variable "agent_runner_repo_url" {
  description = "GitHub URL of the agent-runner CLI repo (CodeBuild source for the runtime image)"
  type        = string
  default     = "https://github.com/sublimotion/agent-runner.git"
}

variable "agent_runner_ref" {
  description = "Git ref (branch/tag/SHA) of agent-runner to build the image from"
  type        = string
  default     = "main"
}

variable "image_tag" {
  description = "Tag for the baked runtime image (ECR repo is IMMUTABLE — bump to rebuild)"
  type        = string
  default     = "v2"
}

variable "tags" {
  type = map(string)
  default = {
    Project   = "agent-aiops-on-aws"
    Blueprint = "managed-agent-runner"
    ManagedBy = "terraform"
  }
}
