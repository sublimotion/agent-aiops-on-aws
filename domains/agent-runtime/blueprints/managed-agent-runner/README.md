# managed-agent-runner (blueprint)

AWS foundation for the [`agent-runner`](https://github.com/sublimotion/agent-runner) CLI.
This blueprint provisions **only** what the spec says lives in this repo; the CLI, harness
adapters, Dockerfile, and Job template live in the sibling `agent-runner` repo.

Spec: `domains/agent-runtime/specs/managed-agent-runner.md`

## What this provisions

| Resource | Purpose |
|----------|---------|
| ECR repo `agent-runner-full-deploy` | runtime profile image (`:v1`) |
| Private S3 bucket (`*-artifacts-<acct>`) | run reports + logs (R4: authenticated pull only) |
| DynamoDB table (`*-runs`) | run-state + `last_heartbeat` (R2) |
| IRSA run-role | scoped per-run credential boundary (R6); trust = cluster OIDC, SA `agent-runner-*` |

## Deploy

```bash
cd domains/agent-runtime/blueprints/managed-agent-runner
terraform init
terraform apply \
  -var oidc_provider_arn=arn:aws:iam::<acct>:oidc-provider/oidc.eks.<region>.amazonaws.com/id/<id> \
  -var oidc_provider_url=oidc.eks.<region>.amazonaws.com/id/<id>

# then export the CLI env and build the image
terraform output -raw cli_env | source /dev/stdin
( cd ../../../../../agent-runner && ./docker/build-and-push.sh full-deploy v1 )
```

The OIDC provider args come from the target EKS cluster
(`aws eks describe-cluster --name <c> --query cluster.identity.oidc.issuer`).

## Notes

- The run-role's base policy grants only this bucket + this table + ECR pull + KMS. For deploy
  specs that run `terraform apply`/`kubectl`, attach a **scoped** domain policy via
  `-var extra_policy_arns=[...]` — never admin.
- Node pinning: the Job template targets nodes labeled `agent-runner/pool=cpu` with toleration
  `agent-runner/dedicated`. Label a Bottlerocket CPU nodegroup accordingly before launching.
