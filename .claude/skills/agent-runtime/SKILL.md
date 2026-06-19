---
name: agent-runtime
description: Launch and manage detached agent runs on EKS via the agent-runner CLI. Use when the user says "launch an agent run", "agent runtime", "agent-runner", "run this spec on the cluster", "check run status", "agent run logs", or "stop a run". Do NOT use for writing the agent-runner code itself (edit the sibling ../agent-runner repo), Terraform of the foundation (use terraform-automation), or GPU serving deployments (use infra-deployer).
---

# Agent Runtime (agent-runner CLI)

Drives the `agent-runner` CLI — a harness-agnostic, detached agent runtime on EKS. The operator
commits a spec, launches a Kubernetes Job that runs a headless coding-harness loop (Claude Code
or Codex) against that commit in its own git worktree, can close the laptop, and reconnects later
for a visual status report pulled from private S3.

**Command reference is `agent-runner help`** — run it rather than memorizing flags (they live in
the sibling repo and may drift). This skill encodes the *workflow*, the *env setup*, and the
*gotchas* that `--help` can't carry.

- CLI lives in the sibling repo: `../agent-runner` (github.com/sublimotion/agent-runner)
- Spec: `domains/agent-runtime/specs/managed-agent-runner.md`
- Foundation (ECR, private S3, run-state table, IRSA role): blueprint
  `domains/agent-runtime/blueprints/managed-agent-runner/`

## The contract (read before launching)

**git commit is the handoff.** The run executes exactly the committed SHA on its own branch
(`agent-run/<run-id>`); the agent's commits come back as a reviewable diff. Therefore:

1. The spec must be **committed and pushed** before launch (Stage 0 blocks otherwise).
2. **Run the CLI from inside the target repo** — the spec path is relative to that repo, and the
   commit is resolved from its `git HEAD`.

## Env setup (required once per shell)

The CLI is configured by `AGENT_RUNNER_*` env vars. Get them from the blueprint's Terraform output:

```bash
cd domains/agent-runtime/blueprints/managed-agent-runner
terraform output -raw cli_env | source /dev/stdin   # exports REGISTRY/BUCKET/TABLE/REGION
export AGENT_RUNNER_RUN_ROLE_ARN="$(terraform output -raw run_role_arn)"
export AGENT_RUNNER_DEFAULT_CLUSTER=arn:aws:eks:us-east-2:615299764834:cluster/qwen3-next-bench-eks-cluster
export AGENT_RUNNER_NAMESPACE=agent-runner
```

`--cluster` is a launch-time arg (defaults to `AGENT_RUNNER_DEFAULT_CLUSTER`), deliberately not in
the spec — the same spec runs on any cluster.

## Workflow

### Launch
```bash
cd <target-repo>                      # e.g. the aiops repo
git add <spec> && git commit && git push
../agent-runner/bin/agent-runner launch <spec-path> [--harness claude-code|codex] [--cluster <ctx>] [--deadline <sec>]
# → prints a run-id; detach freely
```
Stage 0 preflight runs first (commit pushed, node SG, base image, bucket, table, adapter). It
**blocks** on any failure — fix and relaunch rather than forcing.

### Reconnect
```bash
../agent-runner/bin/agent-runner status <run-id>   # pulls report.html from private S3, opens locally
../agent-runner/bin/agent-runner logs <run-id> [--follow]
../agent-runner/bin/agent-runner ls                # recent runs from the run-state table
../agent-runner/bin/agent-runner stop <run-id>     # hard kill: delete Job + node cleanup
```

`status` does an **authenticated** `aws s3 cp` (no public URL) and opens the report. The report
shows status, the agent's latest output, a tool-usage histogram, and the full chronological
activity log.

## Gotchas (paid for in the first live run — don't relearn)

- **Model auth is Bedrock via IRSA, not an API key.** The run role carries `bedrock:InvokeModel*`;
  the Job sets `CLAUDE_CODE_USE_BEDROCK=1` + `ANTHROPIC_MODEL`. If a run logs "Not logged in",
  the SA→role annotation or Bedrock perms are the cause, not a missing key.
- **A run can report `subtype=success` yet `is_error=true`** (e.g. auth failure). Trust the
  `verdict`/`is_error`, not the bare subtype, when judging a run.
- **Private-repo clone needs a token** in the `agent-runner-git` secret (`repo-url` + `token`),
  not just the URL.
- **Shared CPU nodes are storage-constrained** (~20Gi ephemeral). Defaults are sized for that;
  raise via `AGENT_RUNNER_ESS_REQ/LIM` only on bigger nodes.
- **Node pinning is opt-in** (`AGENT_RUNNER_NODE_POOL`). Empty = schedule on any node. Set it only
  when a dedicated Bottlerocket pool labeled `agent-runner/pool=<x>` exists.
- **The wrapper pushes the agent's branch unconditionally** after the harness finishes — the
  agent's own narrative ("I didn't push") may be wrong; check the branch on the remote.
- Cold start is ~2-4 min (stock `python:3.12-slim` + install-deps). `/work/run.log` doesn't exist
  until install completes — a pod can be Running with no log yet; that's normal.

## Scope

This skill is operator-facing (it helps drive the CLI). It does **not** propagate into the
agent-runner pod — that runs `claude -p` headless with its own skill set. To change in-pod behavior,
edit the runner image / harness adapters in `../agent-runner`.
