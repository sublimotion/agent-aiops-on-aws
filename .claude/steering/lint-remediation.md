# Lint Remediation Guide

When a pre-commit check or linter fails, use this guide to fix the issue. Each entry maps a failure pattern to the agent-actionable fix.

## Checkov Failures

If Checkov flags a check that should be skipped, add an inline skip comment to the **specific resource block** (not globally in `.checkov.yaml`) unless the skip applies repo-wide.

### Inline skip format
```hcl
resource "aws_s3_bucket" "example" {
  #checkov:skip=CKV_AWS_144:Cross-region replication not required for logging bucket
  bucket = "my-logging-bucket"
}
```

### Common Checkov failures and fixes

| Check ID | What it flags | Fix |
|----------|--------------|-----|
| `CKV2_AWS_62` | S3 bucket missing event notifications | Skip — logging buckets don't need event monitoring. Add inline skip with reason. |
| `CKV2_AWS_65` | S3 bucket ownership controls | Set `object_ownership = "BucketOwnerPreferred"` — required for S3 log delivery. |
| `CKV_AWS_144` | S3 bucket missing cross-region replication | Skip — only primary data buckets need replication. Add inline skip. |
| `CKV2_AWS_64` | KMS key missing explicit policy | Skip — default AWS KMS policy is secure. Add inline skip. |
| `CKV_TF_1` | Module source not using commit hash | Skip — we use version tags for maintainability. Add inline skip. |
| `CKV_AWS_382` | Security group allows egress to 0.0.0.0/0 | Skip for NAT gateway SGs only. If not NAT-related, restrict egress to specific CIDRs. |
| `CKV_K8S_28` | Container has NET_RAW capability | Skip — vLLM containers require NET_RAW. Add inline skip referencing vLLM requirement. |
| `CKV_K8S_22` | Container filesystem not read-only | Skip — vLLM needs write access for model caching at `/root/.cache/`. Add inline skip. |
| `CKV_K8S_25` | Container has SYS_PTRACE capability | Skip — required for NVIDIA GPU profiling/debugging. Add inline skip. |
| `CKV_K8S_15` | Image pull policy not Always | Set `imagePullPolicy = "IfNotPresent"` with version-tagged images. This is intentional for caching. |
| `CKV_K8S_43` | Image not using digest | Skip — we use version tags, not digests. Add inline skip. |

### Checkov failure on a NEW check (not listed above)
1. Read the check description: `checkov --check <CHECK_ID> --list`
2. Determine if the check is relevant to this resource
3. If relevant: fix the underlying issue
4. If not relevant: add inline skip with justification AND add the check to `.checkov.yaml` with a comment explaining why

## Terraform Lint (tflint) Failures

| Rule | What it flags | Fix |
|------|--------------|-----|
| `terraform_naming_convention` | Resource/variable name not snake_case | Rename to snake_case: `myResource` → `my_resource` |
| `terraform_documented_variables` | Variable missing `description` | Add `description = "..."` to the variable block |
| `terraform_documented_outputs` | Output missing `description` | Add `description = "..."` to the output block |
| `terraform_unused_declarations` | Variable/local/output declared but not used | Remove the unused declaration. If it's intentionally reserved for future use, add a comment and suppress. |
| `terraform_standard_module_structure` | Module missing main.tf, variables.tf, or outputs.tf | Create the missing file(s). Even if empty, modules must have all three. |

## Terraform Format (terraform_fmt)

**Fix**: Run `terraform fmt -recursive` from the blueprint directory. This auto-fixes all formatting issues with no manual intervention needed.

## Terraform Validate (terraform_validate)

Common failures:

| Error pattern | Fix |
|--------------|-----|
| `Reference to undeclared resource` | Add the resource block or fix the reference name |
| `Missing required argument` | Check the module/resource docs for required fields |
| `Unsupported block type` | Check Terraform/provider version compatibility |
| `Cycle detected` | Break the dependency cycle — usually by using `depends_on` or refactoring the module boundary |
| `Invalid reference` | Check for typos in resource names or attribute paths |

## tfsec Failures

tfsec findings are informational unless they flag actual security issues. For GPU infrastructure:
- Network-related findings in private subnets with NAT are usually safe to skip
- S3 encryption findings should be fixed (use `aws_kms_key` + `server_side_encryption_configuration`)
- IAM findings should always be investigated — never skip without understanding the implication

## Trufflehog (Secrets Detection)

**If trufflehog flags a secret**: Do NOT commit. Remove the secret immediately.
1. Check if it's a real secret or a false positive (test keys, example values)
2. If real: rotate the secret, add the file to `.gitignore`, use AWS Secrets Manager instead
3. If false positive: add to `.trufflehog-ignore` with justification

## Pre-commit Hook Failure Workflow

When `pre-commit run -a` fails:
1. Read the failure output — identify which hook failed
2. Look up the hook in this guide
3. Apply the fix
4. Run `pre-commit run -a` again to verify
5. If the fix requires a skip/exception, always include a justification comment
