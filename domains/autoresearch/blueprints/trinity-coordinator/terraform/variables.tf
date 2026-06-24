variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "trinity-coordinator"
}

variable "aws_region" {
  description = "Primary region (bucket + role). Workers are invoked cross-region by the Job."
  type        = string
  default     = "us-east-1"
}

variable "bench_bucket" {
  description = "Existing shared bench bucket name. The blueprint writes under <bench_bucket>/trinity-coordinator/. If empty, a dedicated bucket named <name>-<account> is created."
  type        = string
  default     = ""
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
  description = "Kubernetes namespace where the agent-runner Job runs"
  type        = string
  default     = "agent-runner"
}

variable "service_account_pattern" {
  description = "ServiceAccount name (or pattern) the Job runs as, in runner_namespace"
  type        = string
  default     = "agent-runner-*"
}

variable "worker_regions" {
  description = "Regions the Job invokes Bedrock workers in (cross-region throttle rotation). Inference-profile model ARNs are granted in each."
  type        = list(string)
  default     = ["us-east-1", "us-west-2", "us-east-2"]
}

variable "raw_rollout_retention_days" {
  description = "Expire raw per-iteration rollouts after N days. model_iter_*.npy and es_log.json are kept (separate prefix, not expired)."
  type        = number
  default     = 30
}

variable "max_session_duration" {
  description = "Max session duration (s) for the run role. 3600-43200; 12h covers a full CMA-ES run."
  type        = number
  default     = 43200
}

variable "tags" {
  type = map(string)
  default = {
    Project   = "agent-aiops-on-aws"
    Blueprint = "trinity-coordinator"
    Domain    = "autoresearch"
    ManagedBy = "terraform"
  }
}
