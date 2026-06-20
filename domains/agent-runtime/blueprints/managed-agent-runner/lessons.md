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

## Interactive mode (added 2026-06-19)

`launch --interactive` runs the harness in a **tmux session**; `attach` = `kubectl exec -it
… tmux attach`. Verified live on us-east-2: typed an instruction into the remote REPL, the
agent (Opus 4.8 via Bedrock/IRSA) used the Write tool, file created. Detach (Ctrl-b d) keeps
the session running; reattach from any laptop; `stop` kills it (verified).

- **Claude Code is a TUI with cascading first-run dialogs** that stall a headless-interactive
  pod: theme picker → folder-trust → MCP-server approval (the repo's `.mcp.json` `gpu-infra`).
  Fix: pre-seed `/root/.claude.json` with `hasCompletedOnboarding`, the worktree pre-trusted,
  and MCP approval suppressed. Do NOT rely on `--dangerously-skip-permissions` — **it's refused
  when running as root** (the container is root).
- **Interactive pods stay alive** until the REPL exits or `stop` — `while tmux has-session`.
  So the `activeDeadlineSeconds` ceiling and `stop` are the lifecycle controls; an idle
  interactive pod burns node resources until stopped.
- **Report/verdict are degraded in interactive mode** — tmux `pipe-pane` captures raw text, not
  structured stream-json, so the verdict logic doesn't apply (status goes INTERACTIVE→ENDED, not
  SUCCEEDED/FAILED). Batch mode keeps full fidelity. This is the accepted tmux tradeoff.
- **tmux send-keys races shell readiness** — when scripting the pane, the opening prompt can
  interleave with prompt-dismissal keystrokes. For real operator use this is moot (you type into
  the live `❯`); only matters for automated pane-driving.

## Credential vending — harness/compute separation (added 2026-06-20, verified live)

The driver holds the pod's live IRSA identity; it vends a **frozen, time-boxed STS session**
to the harness and runs the harness with only those keys. Verified live on us-east-2: run.log
showed `vend: harness session harness-<run-id> valid until <+12h>`, harness ran under it.

- **Use `assume-role-with-web-identity`, NOT role-chaining.** Chaining caps at 1h and would kill
  the harness's AWS access mid-run. Web-identity assume (re-presenting the projected SA token)
  honors the role's `MaxSessionDuration` — set it to 43200 (12h) to cover >8h runs.
- **Unset `AWS_WEB_IDENTITY_TOKEN_FILE` / `AWS_ROLE_ARN` in the harness env** so a prompt-injected
  harness can't refresh the session or fall back to the driver's live identity.
- **Driver keeps its own identity**: batch runs the harness in a `( . creds; harness_invoke )`
  subshell; interactive sources the vended env inside the tmux session before the REPL. The
  driver's push/state/S3 calls still use the live IRSA identity.
- **Distinct session name `harness-<run-id>`** → CloudTrail attributes harness API calls
  separately from the driver. Free audit boundary.
- Graceful fallback: if no web-identity token is present (e.g. laptop driver), the harness
  inherits the single identity — logged, not fatal.
- **Deferred** (per operator, trusted single-operator runtime): per-project/per-run role
  scoping (shared role for now), and revoke-on-stop (pod deletion + STS expiry suffices).

## Baked image — CodeBuild, not EBS snapshot (added 2026-06-20, verified live)

Cold start dropped **~3min → 42s** by baking the toolbelt into the image instead of
installing at startup. Container build, not EBS snapshot: we cache ~250MB of *static CPU
binaries* (terraform/kubectl/aws/claude/codex), which is exactly what image layers are for.
EBS snapshots earn their place for GPU work (100s of GB of model weights), not here.

- **Build mechanism: CodeBuild (arm64 project), not a local laptop build** — managed, no
  instance lifecycle, repeatable for future profiles. Private repo → one-time
  `aws codebuild import-source-credentials` with a `gh auth token` PAT.
- **Build MULTI-ARCH (arm64+amd64), not arm64-only.** First build was arm64-only and the
  cluster's CPU nodes are amd64 (m6i) → `ImagePullBackOff: no match for platform in manifest`.
  Drop hardcoded `FROM --platform=`, use buildx `TARGETARCH`, map it to aarch64/x86_64 for
  awscli, `--platform linux/arm64,linux/amd64` + binfmt/qemu.
- **ECR repo is IMMUTABLE — bump the tag, don't overwrite.** Re-pushing `v1` was rejected
  ("tag is immutable and cannot be overwritten"). Made `image_tag` a var; immutability is a
  feature (a deployed tag never silently changes) — bump to `v2` to rebuild.
- **install-deps stays in the image as a no-op** (idempotent skip-if-present) — so the stock
  base path still works, and the ConfigMap still ships `lib/` for fast CLI/adapter iteration
  without rebuilding the image. Only the heavy toolbelt is baked.
- Set `AGENT_RUNNER_BASE_IMAGE=<ecr>/agent-runner-full-deploy:v2` to use the baked image; the
  Stage 0 `pf_image_tag` check then validates the ECR tag exists before scheduling.

## Still open

- Image is stock `python:3.12-slim` + install-deps at startup (~2-4 min cold start). Bake
  `agent-runner-full-deploy:v1` later for faster starts (Dockerfile + build-and-push.sh ready;
  Docker daemon was down at scaffold time).
- `security:` config block (harness allowlist + opt-in gVisor runtimeClass + FQDN egress) —
  next iteration, see spec discussion.
- Stale-heartbeat warning fires on Completed runs (cosmetic; status should suppress when terminal).
