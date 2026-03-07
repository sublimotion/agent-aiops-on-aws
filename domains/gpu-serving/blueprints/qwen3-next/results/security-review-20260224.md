# Security Review: Qwen3-Next Blueprint

**Date**: 2026-02-24
**Reviewer**: Automated (Claude Code)
**Blueprint**: `domains/gpu-serving/blueprints/qwen3-next/`
**Scope**: Terraform IaC, Dockerfiles, shell scripts, Kubernetes manifests

---

## Executive Summary

The qwen3-next blueprint deploys an EKS cluster with p5en.48xlarge GPU nodes for MoE model inference. The infrastructure has reasonable security foundations — KMS encryption on S3 buckets, versioning enabled, private subnets for compute, VPC endpoints for service access, and `HF_HUB_OFFLINE=1` to prevent runtime data exfiltration.

However, the review identified **5 HIGH**, **6 MEDIUM**, and **5 LOW** severity findings. The HIGH findings center on container privilege escalation, overly broad IAM policies, a hardcoded default password, and missing S3 public access blocks. These are acceptable for short-lived benchmark environments behind capacity blocks but must be remediated before any production deployment.

---

## Findings

### HIGH Severity

#### H1. Serving Container Runs as Root with Privilege Escalation

**File**: `main.tf:512-516`

```hcl
security_context {
  run_as_user                = 0
  run_as_non_root            = false
  allow_privilege_escalation = true
}
```

**Risk**: A container escape vulnerability in vLLM or its dependencies would grant root access on the host node. With `allow_privilege_escalation = true`, a compromised process can gain additional Linux capabilities beyond its initial set.

**Remediation**: Run as a non-root user. vLLM and SGLang do not require root for inference. Set `run_as_user = 1000`, `run_as_non_root = true`, `allow_privilege_escalation = false`. Note: GPU access via `/dev/nvidia*` requires the container user to be in the `video` group or have appropriate device permissions — test after changing.

**Mitigating factors**: The pod runs on a dedicated GPU node with no other tenant workloads. The capacity block instance is ephemeral (hours, not months).

---

#### H2. Dockerfiles Have No USER Directive

**Files**: `docker/Dockerfile.vllm-qwen3next`, `docker/Dockerfile.sglang-qwen3next`

Both Dockerfiles inherit root from their base images and never drop privileges:

```dockerfile
FROM vllm/vllm-openai:v0.15.0-cu130
# ... installs as root ...
# No USER directive
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
```

**Risk**: All processes inside the container run as UID 0. Combined with H1, this creates a root-on-root attack surface.

**Remediation**: Add a non-root user after dependency installation:

```dockerfile
RUN groupadd -r vllm && useradd -r -g vllm -d /workspace vllm
RUN chown -R vllm:vllm /workspace /mnt/fsx /mnt/nvme
USER vllm
```

---

#### H3. FSx CSI IAM Policy Uses Wildcard Resources

**File**: `main.tf:225-246`

```hcl
{
  Effect = "Allow"
  Action = [
    "fsx:DescribeFileSystems",
    "fsx:DescribeVolumes",
    "fsx:CreateVolume",
    "fsx:DeleteVolume",
    "fsx:TagResource"
  ]
  Resource = "*"
}
```

**Risk**: The FSx CSI driver role can describe, create, and delete volumes on any FSx filesystem in the account. A compromised pod with access to the service account token could manipulate unrelated filesystems.

**Remediation**: Scope to the specific FSx filesystem:

```hcl
Resource = [
  "arn:aws:fsx:${var.aws_region}:${data.aws_caller_identity.current.account_id}:file-system/${module.fsx_lustre[0].file_system_id}",
  "arn:aws:fsx:${var.aws_region}:${data.aws_caller_identity.current.account_id}:volume/*"
]
```

The `ec2:Describe*` actions legitimately require `Resource = "*"` (AWS does not support resource-level permissions for EC2 Describe calls).

---

#### H4. Hardcoded Default Grafana Admin Password

**File**: `variables.tf:237-242`

```hcl
variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}
```

**Risk**: If deployed without overriding this variable, Grafana is accessible with `admin/admin`. The NodePort security group (H5/M1) allows VPC-wide access, so any host in the VPC can log in.

**Remediation**: Remove the default value to force explicit configuration:

```hcl
variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
  # No default — must be set in terraform.tfvars or via -var
}
```

Alternatively, generate a random password and store it in AWS Secrets Manager.

---

#### H5. S3 Buckets Missing Public Access Block

**File**: `main.tf:1104-1108, 1128-1132`

Neither the `benchmark_results` nor `models` S3 bucket has an `aws_s3_bucket_public_access_block` resource. While bucket policies don't grant public access today, a misconfigured policy change could expose model weights or benchmark data.

**Risk**: Accidental public exposure of proprietary model weights or benchmark results containing infrastructure details.

**Remediation**: Add public access blocks for both buckets:

```hcl
resource "aws_s3_bucket_public_access_block" "benchmark_results" {
  bucket                  = aws_s3_bucket.benchmark_results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

---

### MEDIUM Severity

#### M1. NodePort Security Group Allows Full VPC CIDR

**File**: `main.tf:734-740`

```hcl
ingress {
  description = "Serving engine NodePort"
  from_port   = var.vllm_node_port
  to_port     = var.vllm_node_port
  protocol    = "tcp"
  cidr_blocks = [var.vpc_cidr]
}
```

**Risk**: Any resource in the VPC (system nodes, NAT gateway host, any future workload) can reach the serving endpoint. The vLLM/SGLang API has no authentication — any caller can submit inference requests or query `/metrics`.

**Remediation**: Restrict the source to the system node security group or a specific management subnet CIDR. For benchmark access, use `kubectl port-forward` instead of NodePort.

---

#### M2. No Kubernetes NetworkPolicies

**File**: `main.tf` (absent)

No `kubernetes_network_policy` resources are defined. All pods in all namespaces can communicate freely.

**Risk**: A compromised pod (e.g., a monitoring sidecar) could access the serving API, Prometheus metrics, or Grafana.

**Remediation**: Define NetworkPolicies for the `ml-inference` namespace to allow only:
- Ingress from system nodes on port 8000 (serving)
- Ingress from Prometheus on port 8000 (metrics scrape)
- Egress to FSx/NVMe mounts and DNS

---

#### M3. HuggingFace Token Secret Marked Optional

**File**: `main.tf:561-569`

```hcl
env {
  name = "HUGGING_FACE_HUB_TOKEN"
  value_from {
    secret_key_ref {
      name     = "hf-token"
      key      = "token"
      optional = true
    }
  }
}
```

**Risk**: If the secret exists but contains a valid token, and `HF_HUB_OFFLINE=1` is ever removed, the token could be used to exfiltrate data to HuggingFace Hub. The `optional = true` means the pod starts regardless of whether the secret exists, giving no signal that credentials are present.

**Remediation**: Remove this env block entirely. The blueprint runs in offline mode (`HF_HUB_OFFLINE=1`) and does not need a HuggingFace token. If a token is needed for future use, manage it via AWS Secrets Manager with explicit IAM scoping.

---

#### M4. Benchmark Results Bucket Has force_destroy = true

**File**: `main.tf:1106`

```hcl
resource "aws_s3_bucket" "benchmark_results" {
  bucket_prefix = "${var.project_name}-results-"
  force_destroy = true
  tags          = local.tags
}
```

**Risk**: `terraform destroy` will delete all benchmark results without confirmation. In a long-running deployment, this could destroy valuable data.

**Remediation**: Set `force_destroy = false` and require manual bucket emptying before destruction. The models bucket (`main.tf:1130`) already has `force_destroy = false`.

---

#### M5. Unpinned Container Image Tags

**File**: `variables.tf:149, 155`

```hcl
default = "vllm/vllm-openai:qwen3_5-x86_64-cu130"  # line 149
default = "lmsysorg/sglang:v0.5.2-cu130"             # line 155
```

**Risk**: Tag-based references can be mutated on the registry side. A compromised Docker Hub account could push a malicious image under the same tag.

**Remediation**: Pin to image digests:

```hcl
default = "vllm/vllm-openai:qwen3_5-x86_64-cu130@sha256:<digest>"
```

For air-gapped deployments using ECR, this is partially mitigated since images are pulled once during staging.

---

#### M6. Egress Security Group Allows All Outbound

**File**: `main.tf:742-747`

```hcl
egress {
  from_port   = 0
  to_port     = 0
  protocol    = "-1"
  cidr_blocks = ["0.0.0.0/0"]
}
```

**Risk**: A compromised GPU node can initiate outbound connections to any destination. Combined with the NAT gateway, this enables data exfiltration.

**Remediation**: Restrict egress to VPC CIDR and required VPC endpoints. The cluster uses VPC endpoints for AWS services and FSx; internet egress should not be needed from GPU nodes.

---

### LOW Severity

#### L1. No CPU/Memory Limits on Serving Container

**File**: `main.tf:577-586`

```hcl
resources {
  limits = {
    "nvidia.com/gpu" = var.vllm_gpu_count
  }
  requests = {
    "nvidia.com/gpu" = var.vllm_gpu_count
    cpu              = var.vllm_cpu_request
    memory           = local.kv.memory_request
  }
}
```

GPU limits are set, but CPU and memory have only requests, no limits. A memory leak in the serving process could consume all node memory and trigger OOM kills of other pods.

**Remediation**: Add explicit CPU and memory limits matching or slightly exceeding requests.

---

#### L2. World-Readable Mount Points in Dockerfiles

**Files**: `docker/Dockerfile.vllm-qwen3next:48`, `docker/Dockerfile.sglang-qwen3next:45`

```dockerfile
RUN mkdir -p /mnt/fsx /mnt/nvme /workspace/results
```

Directories created with default `755` permissions. If the container is shared or model weights are sensitive, other processes can read them.

**Remediation**: `RUN mkdir -p -m 750 /mnt/fsx /mnt/nvme /workspace/results`

---

#### L3. No Pod Security Standards Enforced

**File**: `main.tf` (absent)

No Pod Security Standards (PSS) labels are applied to the `ml-inference` namespace. Any pod spec is accepted, including privileged containers.

**Remediation**: Apply PSS labels to enforce at least `baseline` level:

```hcl
labels = {
  "pod-security.kubernetes.io/enforce" = "baseline"
}
```

Note: `restricted` level would conflict with GPU device access; `baseline` is the practical minimum.

---

#### L4. Unencrypted NodePort Service

**File**: `main.tf:704-726`

The NodePort service exposes plain HTTP on port 30080. Traffic between clients and the serving endpoint is unencrypted.

**Risk**: Inference requests and responses (potentially containing sensitive prompts) are transmitted in cleartext within the VPC.

**Remediation**: For benchmark use, this is acceptable (VPC-internal traffic). For production, add TLS termination via an ingress controller or service mesh.

---

#### L5. No Pod Disruption Budget

**File**: `main.tf` (absent)

No `PodDisruptionBudget` is defined. A node drain or cluster upgrade could terminate the serving pod without warning.

**Risk**: Loss of in-flight inference requests during maintenance events.

**Remediation**: Add a PDB with `minAvailable: 1` for the serving deployment.

---

## Positive Security Controls

The following security controls are already in place:

| Control | Implementation | Location |
|---------|---------------|----------|
| S3 encryption at rest | KMS (`aws:kms`) | `main.tf:1117-1125, 1141-1149` |
| S3 versioning | Enabled on both buckets | `main.tf:1110-1115, 1134-1139` |
| Private subnets for compute | GPU/system nodes in private subnets | Networking module |
| VPC endpoints | S3, ECR, FSx, STS, CloudWatch | Networking module |
| Air-gap enforcement | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | `main.tf:536-543` |
| Sensitive variable marking | `grafana_admin_password` marked sensitive | `variables.tf:241` |
| Default tags | All resources tagged with Project/Environment/ManagedBy | `main.tf:26-30` |
| Models bucket protected | `force_destroy = false` | `main.tf:1130` |
| IRSA | FSx CSI uses IAM Roles for Service Accounts | `main.tf:209-250` |

---

## Risk Matrix

| ID | Severity | CVSS Est. | Exploitability | Finding |
|----|----------|-----------|----------------|---------|
| H1 | HIGH | 7.8 | Medium | Container runs as root with privilege escalation |
| H2 | HIGH | 7.8 | Medium | Dockerfiles have no USER directive |
| H3 | HIGH | 6.5 | Low | FSx CSI IAM policy uses wildcard resources |
| H4 | HIGH | 6.2 | High | Hardcoded default Grafana password |
| H5 | HIGH | 6.0 | Low | S3 buckets missing public access block |
| M1 | MEDIUM | 5.3 | Medium | NodePort SG allows full VPC CIDR |
| M2 | MEDIUM | 5.0 | Medium | No Kubernetes NetworkPolicies |
| M3 | MEDIUM | 4.5 | Low | HF token secret marked optional |
| M4 | MEDIUM | 4.0 | Low | Benchmark S3 bucket force_destroy |
| M5 | MEDIUM | 3.7 | Low | Unpinned container image tags |
| M6 | MEDIUM | 5.0 | Medium | Unrestricted egress from GPU nodes |
| L1 | LOW | 3.0 | Low | No CPU/memory limits |
| L2 | LOW | 2.5 | Low | World-readable mount points |
| L3 | LOW | 2.5 | Low | No Pod Security Standards |
| L4 | LOW | 2.0 | Low | Unencrypted NodePort |
| L5 | LOW | 2.0 | Low | No PodDisruptionBudget |

---

## Recommendations by Priority

### Immediate (before next deployment)

1. Add `aws_s3_bucket_public_access_block` to both S3 buckets (H5)
2. Remove default Grafana password (H4)
3. Scope FSx CSI IAM policy to specific filesystem ARN (H3)

### Short-term (before production use)

4. Drop container to non-root user (H1, H2)
5. Add Kubernetes NetworkPolicies (M2)
6. Remove HF token env block (M3)
7. Restrict NodePort SG to management subnet (M1)
8. Restrict GPU node egress to VPC CIDR (M6)
9. Pin images to digests (M5)

### Long-term (production hardening)

10. Enforce Pod Security Standards (L3)
11. Add CPU/memory limits (L1)
12. Add TLS termination for serving endpoint (L4)
13. Add PodDisruptionBudget (L5)
14. Implement runtime security monitoring (Falco or equivalent)
15. Enable EKS control plane audit logging

---

## Methodology

This review examined the Terraform configuration (`main.tf`, `variables.tf`), container images (`docker/Dockerfile.*`), and operational scripts (`scripts/*.sh`) for the qwen3-next blueprint. Analysis focused on:

- AWS IAM least privilege (CIS AWS Benchmark 1.16)
- Container security (CIS Docker Benchmark 4.1, 5.3, 5.25)
- Kubernetes security (CIS Kubernetes Benchmark 5.2, 5.7)
- Network segmentation and encryption in transit
- Secret management
- OWASP infrastructure security top 10
- Shell script injection patterns

Tools: Manual code review. Automated scanning with `checkov` and `tfsec` recommended as a follow-up.
