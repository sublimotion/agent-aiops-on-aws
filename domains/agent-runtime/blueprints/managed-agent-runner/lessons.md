# Lessons — managed-agent-runner

Operational lessons captured during deployment of this blueprint and the `agent-runner` CLI.
(Compound-learner generates the YAML frontmatter on the first compound run.)

## Deployment lessons

- (none yet — blueprint scaffolded 2026-06-19; not yet deployed to a live cluster)

## Open items before first live run

- Build/push `agent-runner-full-deploy:v1` to ECR (needs Docker daemon running — was down at scaffold time)
- Create a Bottlerocket CPU nodegroup labeled `agent-runner/pool=cpu`, taint `agent-runner/dedicated`
- Create the `agent-runner-git` K8s secret (repo-url + deploy credential, repo-scoped)
- Wire the per-run ServiceAccount creation (annotate with `run_role_arn` output) into the launcher
