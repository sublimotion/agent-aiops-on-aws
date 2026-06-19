# Lessons — managed-agent-runner

Operational lessons captured during deployment of this blueprint and the `agent-runner` CLI.
(Compound-learner generates the YAML frontmatter on the first compound run.)

## Deployment lessons

First live e2e on `qwen3-next-bench-eks-cluster` (us-east-2), 2026-06-19 — **PASSED**.
Agent cloned the private repo, authed to Bedrock via IRSA, ran 49 turns / $3.66, used
Bash/Read/Edit/Write/Grep/Skill/Agent tools, wrote report+log to private S3.

- **`envsubst` eats the in-pod shell vars.** Rendering the Job template with bare `envsubst`
  blanked the initContainer's `$f`/`$rel`/`$(dirname)` (it substitutes *all* `$tokens`).
  Fix: pass an explicit SHELL-FORMAT allowlist (`envsubst '$AR_RUN_ID $AR_... '`) so only the
  template's own vars are replaced and runtime shell vars survive.
- **ConfigMap-as-code-transport works** but glob `for f in /cm/*` trips on the `..data`
  symlink dir — use `find -L /cm -type f` to enumerate real files.
- **Model auth = Bedrock via IRSA, no API key.** `CLAUDE_CODE_USE_BEDROCK=1` +
  `ANTHROPIC_MODEL=us.anthropic.claude-opus-4-8` + `bedrock:InvokeModel*` on the run role.
  Confirmed `apiKeySource=none` in the harness init event.
- **`is_error` overrides `subtype` in Claude Code's result event.** A run can be
  `subtype=success` yet `is_error=true` (e.g. "Not logged in"). `harness_complete` must check
  `is_error` first or it falsely reports SUCCEEDED.
- **DynamoDB state writes must MERGE (UpdateItem), not PutItem.** A heartbeat/status write with
  PutItem wipes harness/cluster/commit, which `stop`/`logs`/`status` then can't read.
- **run-id must be RFC1123** (lowercase, no `T`/`Z`) since it names the SA/Job/ConfigMap.
- **Shared bench nodes (m6i.xlarge) have only ~20Gi ephemeral-storage.** The original
  20Gi/40Gi requests were unschedulable; defaults are now 2Gi/8Gi and env-configurable.
- **Node pinning made optional** (`AGENT_RUNNER_NODE_POOL`) — empty schedules on existing
  untainted CPU nodes; set it only when a dedicated Bottlerocket pool exists.

## Still open

- Image is stock `python:3.12-slim` + install-deps at startup (~2-4 min cold start). Bake
  `agent-runner-full-deploy:v1` later for faster starts (Dockerfile + build-and-push.sh ready;
  Docker daemon was down at scaffold time).
- `security:` config block (harness allowlist + opt-in gVisor runtimeClass + FQDN egress) —
  next iteration, see spec discussion.
- Stale-heartbeat warning fires on Completed runs (cosmetic; status should suppress when terminal).
