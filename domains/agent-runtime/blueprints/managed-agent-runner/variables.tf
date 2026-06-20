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

variable "tags" {
  type = map(string)
  default = {
    Project   = "agent-aiops-on-aws"
    Blueprint = "managed-agent-runner"
    ManagedBy = "terraform"
  }
}
