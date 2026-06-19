# Agent Runtime Spec: Managed Agent Runner

## Overview

A self-hosted "managed agent" runtime that gives the Claude Code / Codex detached-agent
experience for this repo's own work: author a spec locally, commit it, launch a detached
agent run on EKS, close the laptop, and reconnect later for a visual status report.

The agent inside the run is a **headless coding-harness loop** (the same RALPH / deployer
flow this repo already runs interactively) executing against a checked-out git commit in its
own worktree. It survives >8h, holds no live connection to the operator, and persists all
state and results to AWS so the operator can detach and reattach freely.

**Why self-hosted instead of Anthropic Managed Agents / AgentCore Runtime:**
- The agent runs `terraform apply` / `kubectl` / `aws` against this AWS account and must sit
  inside the operator's VPC with **IRSA-scoped** credentials — a managed plane cannot hold
  those creds or reach private clusters.
- Runs routinely exceed AgentCore Runtime's **8h microVM session ceiling** (a long RALPH
  deployment loop). A Kubernetes Job has no such cap.
- The runtime must be **harness-agnostic** (Claude Code, Codex, others) — not Claude-shaped
  by definition.

This is "Managed Agents' developer experience, self-hosted for AWS-privileged, harness-agnostic work."

**Packaging — sibling CLI repo (like `mdc` / `gpu-infra` / verifier):** the runtime is a
standalone, reusable CLI + container-profile catalog in its own repo (`agent-runner`),
versioned independently and wrapped by this repo's `fe` CLI. This spec *references* it as an
external tool; the launcher, harness adapters, container profiles, and Job templating live in
`agent-runner`, not in this blueprint. A blueprint here holds only the spec, the IRSA run-role
Terraform, and operational artifacts (lessons/results).

**Consumption (follow the `mdc` / `gpu-infra` precedent):** this repo pins a released
`agent-runner` version (e.g. `scripts/install-tools.sh` fetches `agent-runner@v0.1.0`), and
`fe agent …` is a **thin wrapper** that passes `--cluster` / IAM context through — it adds no
launch logic of its own (no duplication, mirrors how `fe card`/`fe learn` wrap mdc/gpu-infra).
The `agent-runner` repo owns its own release/versioning the way `mdc get` loads versioned cards.

## Core UX (the contract)

```
1. Author spec locally, iterate, `git commit` (+ push)
2. fe agent launch <spec-path> --cluster <name> [--harness claude-code|codex]
       └─ creates an EKS Job that checks out the committed SHA into its own git worktree/branch
       └─ runs the chosen harness headless against that spec
3. Close laptop. No held connection.
4. fe agent status <run-id>   → downloads the latest visual report from private S3, opens locally
   fe agent logs   <run-id>   → streams/pulls structured logs
   fe agent attach <run-id>   → reattach to live tail (optional)
   fe agent stop   <run-id>   → hard kill switch (deletes the Job, revokes the run role)
```

**Git commit is the contract.** The agent runs exactly the committed SHA, on its own
branch/worktree, so everything it produces (lessons.md updates, results, the compound step)
returns as a reviewable diff / PR — never a mutation of the operator's working tree.

## Requirements

### R1 — Harness-agnostic (first-class, multi-implementation at v1)

The runtime MUST NOT hardcode a single agent harness. The harness is the container
**entrypoint**, selected per run via a `--harness` arg / `harness:` config field. A small
**harness adapter** per harness maps three things to a common interface:

| Adapter responsibility | Claude Code | Codex |
|------------------------|-------------|-------|
| **Invoke** (pass the task headless) | `claude -p "<task>" --output-format json --max-turns N --allowedTools Bash,Read,Write,Edit,Glob,Grep` | `codex exec --json "<task>"` (Responses API path) |
| **Stream** (structured JSON events → run log) | parse `--output-format json` stream | parse `--json` event stream |
| **Detect completion** (success / failure / turn-exhausted) | final `result` event | final event / exit code |

> **Critical (Claude Code):** `--allowedTools` is mandatory. Omitting it is a *silent* failure —
> the harness runs, burns turns, and produces nothing because it can never call a tool. The
> adapter smoke test MUST assert the first turn invokes a tool, not just that the process exited 0.

**v1 ships TWO working adapters — `claude-code` and `codex` — not one with a documented seam.**
The abstraction is a named deliverable; an adapter interface with a single implementation does
not satisfy this requirement (this repo has been burned by single-impl abstractions before).
Adding a third harness (e.g. OpenCode) MUST require only a new adapter, no change to the
launcher, Job spec, status, or guardrail layers.

### R2 — Detached, long-running

- No held operator connection. Launch returns a `run-id` and exits.
- Runs MUST survive >8h (rules out AgentCore Runtime). Implemented as a Kubernetes **Job**
  (not a Deployment) with no active deadline below the operator-set ceiling (see R6).
- Run state (status, started/updated timestamps, harness, commit SHA, cluster, exit reason)
  persists to the `agent-memory` DynamoDB table so status survives pod recycling.

### R3 — Git worktree isolation

- The Job checks out the launched commit SHA into a **dedicated worktree on a per-run branch**
  (`agent-run/<run-id>`).
- All agent commits land on that branch and are pushed; the operator reviews via diff/PR.
- Concurrent runs never collide (separate worktrees, separate branches).

### R4 — Visual status via private S3 (no public URL)

- The agent regenerates a **single-file HTML progress report every N steps** (reuse the
  `visual-explainer` skill output shape; mirror the `results-vault` dashboard conventions) and
  writes it to a **private** S3 bucket under `s3://<bucket>/runs/<run-id>/report.html`.
- `fe agent status <run-id>` does an **authenticated** `aws s3 cp` with the operator's own IAM
  creds and opens the report locally. **No presigned URLs, no public bucket policy** — a
  presigned URL is a bearer token and a public surface; this design avoids both.
- The bucket is private, SSE-encrypted, and lifecycle-expires run artifacts (e.g. 30d).
- **Liveness, not just progress:** the run-state writer updates a DynamoDB `last_heartbeat`
  every 60s. `fe agent status` warns if `last_heartbeat` age >5 min (possible wedge / OOM /
  spot-reclaim), and the HTML report shows heartbeat age. A stale report timestamp alone is not
  a liveness signal — a wedged agent stops emitting reports too.

### R5 — Cluster is a launch-time arg, not a spec field

- The operator runs several EKS clusters. The target cluster is `--cluster <name>`
  (defaults to a configured value), kept **out of the spec** so the same spec runs anywhere.
- Agent Jobs run as **CPU pods** on the chosen cluster's nodes (Bottlerocket EC2 node pool
  preferred — containerd/`nerdctl`-consistent, immutable, scale-to-zero between runs).

### R6 — Guardrails (first-class, not an afterthought)

An autonomous, >8h, detached agent running `terraform apply` with no human watching is
high-blast-radius. The runtime MUST enforce:

- **IRSA per run** — each Job's ServiceAccount assumes a **scoped run role**, not a broad
  admin role. The role is the credential boundary (least privilege for the spec's domain).
- **Tool allowlist** — the harness is launched with an explicit allowlist (e.g. Claude Code
  `--allowedTools`); no unrestricted tool access.
- **Hard kill switch** — `fe agent stop <run-id>` deletes the Job AND revokes/detaches the run
  role so a wedged agent loses credentials immediately. **Graceful-then-force with cleanup:**
  delete Job → wait ≤30s for pod termination → if still running, force-delete → reclaim orphaned
  processes on the node (force-deleted pods leak processes/handles that the next run can't
  reclaim — same failure class as the GPU-memory leak in tech-stack.md).
- **Time + spend ceilings** — operator-set max wall-clock (Job `activeDeadlineSeconds`) and a
  documented cost guard; exceeding either terminates the run.
- **Full audit** — every agent action path logs to CloudWatch; the run role's API calls are
  CloudTrail-visible.

### R7 — Container runtime env (one image now, profile-ready)

The runner container is, in effect, **a containerized copy of the operator's deploy
environment** — the agent must be able to run the same toolchain a RALPH loop uses locally.

- **v1 ships ONE profile, `full-deploy`**, bundling: the harness CLIs (`claude`, `codex` + their
  node/python runtimes), the deploy toolbelt (`terraform`, `kubectl`, `aws`, `git` with worktree
  support), and the report generator. Arch: ARM64/Graviton (matches the bench fleet); document
  if x86 is needed for a given harness.
- **Profile-ready indirection:** the spec/launch carries a `runtime: <profile>@<version>` field
  (e.g. `full-deploy@v1`). v1 has only `full-deploy`, but the field MUST exist so adding
  `benchmark-lite`, `verifier`, etc. later is **config, not redesign** — mirrors the
  harness-adapter discipline (R1) applied to runtimes.
- **Sub-agent runtimes (future, not v1):** a workflow spec may later assign different profiles to
  different roles (lead = `full-deploy`, a verifier sub-agent = `verifier-lite`). v1 is
  single-runtime per the operator's "simplify for now"; the profile field is the seam that makes
  multi-runtime a later config change. Do not build sub-agent runtime dispatch in v1.
- **Profiles are versioned in the `agent-runner` repo**, not here — the spec only names one.
- **Build the image with the slim-image discipline from `domains/ai-infra/shared/images/`**
  (this toolbelt is heavy — claude+codex runtimes + terraform+kubectl+aws). Multi-stage:
  stage 1 builds/installs tools on a `-devel` base; stage 2 copies binaries onto a stripped
  `-runtime` base (e.g. `ubuntu:24.04`). Strip test artifacts, `__pycache__`/`*.pyc`, pip caches
  (`--no-cache-dir`). Layer order for cache efficiency: system pkgs → static toolbelt →
  harness CLIs (most volatile last). Document a compressed size target (≤2 GB, CPU-only).

## Components to build

| Component | What | Where |
|-----------|------|-------|
| **Launcher CLI** | `agent-runner {launch,status,logs,attach,stop}`, wrapped by `fe agent …` | **`agent-runner` sibling repo** |
| **Harness adapters** | claude-code + codex (v1); common invoke/stream/complete interface | `agent-runner` repo |
| **Runner image(s)** | `full-deploy@v1` profile; entrypoint dispatches on `--harness`; bundles toolbelt, worktree logic, report generator. **Output capture: redirect harness stdout/stderr to files, never `subprocess.PIPE`** (64KB pipe buffer deadlocks long runs — see Carryover) | `agent-runner` repo |
| **Job spec (templated)** | K8s Job: SA (IRSA), commit SHA, harness, runtime profile, cluster, deadline, allowlist, **explicit Bottlerocket AMI / nodeSelector** | `agent-runner` repo |
| **Report generator** | periodic single-file HTML → private S3 (visual-explainer shape) | in-image step |
| **Run-state writer** | status + `last_heartbeat` (60s) → `agent-memory` DynamoDB | in-image step |
| **IRSA run-role + artifact bucket** | scoped per-spec-domain role, private S3 bucket | **this blueprint's Terraform** |

## Memory Backend

- **Run state**: DynamoDB (`agent-memory` module) — `run_id` hash key; status, harness, commit
  SHA, cluster, branch, timestamps, exit reason. TTL e.g. 30d.
- **Artifacts/results**: private S3 under `runs/<run-id>/` (report.html, logs, any produced files).
- **Conversation/loop state**: held in-process by the harness for the life of the Job; not a
  managed cross-session memory (each run is self-contained against one commit).

## Auth / Identity

| Component | Configuration |
|-----------|---------------|
| Agent → AWS | **IRSA** — per-run ServiceAccount assumes a scoped run role (R6). No static keys. |
| Operator → status | Operator's own IAM creds via `aws s3 cp` (R4). No separate auth layer. |
| Git push | Deploy key / token mounted as a K8s secret, scoped to this repo only. |
| Multi-tenancy | Single-operator prototype; per-run branch + per-run role gives run isolation. |

## Infrastructure Modules

| Module | Source | Notes |
|--------|--------|-------|
| Run-state table | `domains/agent-runtime/modules/agent-memory` | reuse — DynamoDB run state |
| Artifact bucket | blueprint-level `aws_s3_bucket` | private, SSE, lifecycle 30d |
| Run IAM role + IRSA | blueprint-level | scoped per spec domain; assumable by Job SA |
| EKS Job + SA | blueprint `k8s/` | templated per run; targets `--cluster` |

**Not used (deliberate divergence from the agent-runtime template):** AgentCore Runtime
(8h ceiling), `cognito-app-auth` (no end-user web auth — operator-only via IAM),
`websocket-proxy`/ALB (no live connection held — detached by design).

## Carryover (prior lessons this spec must honor)

From the carryover audit against prior blueprints — each is a lesson already paid for elsewhere:

| # | Prior lesson (source) | How this spec honors it |
|---|------------------------|--------------------------|
| P0 | **Subprocess pipe deadlock** — `subprocess.PIPE` buffer fills at ~64KB and blocks the loop (ThunderAgent research, MEMORY) | Runner image redirects harness stdout/stderr to **files**, never PIPE; logs stream to S3 incrementally (R7 / Components) |
| P1 | **EKS spot/CPU nodegroups can launch missing the node SG** → IRSA pods can't reach `sts`/in-cluster DNS, silent failure (tech-stack.md L43-51) | **Stage 0 pre-launch check**: verify the target node has both cluster SG + node SG before scheduling the Job; attach node SG if missing |
| P1 | **Force-deleted pods leak processes/handles** (tech-stack.md L220-222) | `stop` flow is graceful-then-force **with node process cleanup** (R6) |
| P1 | **Claude Code headless needs `--allowedTools` or silently has no tools** (VP SWE-bench, MEMORY) | Mandatory flag + smoke test asserts a tool is called (R1) |
| P2 | **Monitor long runs / set up background polling** (feedback memory) | DynamoDB `last_heartbeat` every 60s; `status` warns at >5 min stale (R4) |
| P2 | **AL2023 uses nodeadm, not bootstrap.sh** (tech-stack.md L40-42) | Job spec pins **Bottlerocket AMI / nodeSelector explicitly**; if AL2023 is ever targeted, use nodeadm user-data (Components / R5) |
| P1 | **Slim multi-stage image discipline** — devel→runtime, strip artifacts, layer order, size target (`ai-infra/shared/images/`) | R7 mandates multi-stage build + strip + size target for the heavy toolbelt image |
| P2 | **Verify image tag exists before deploy** — pulling a missing tag wastes time (tech-stack.md) | Stage 0 `docker manifest inspect` on the profile image tag before scheduling |
| P2 | **Sibling-CLI consumption pattern** (`mdc`/`gpu-infra` via `fe`) | Pinned `agent-runner` release; `fe agent` is a thin wrapper, no logic duplication (Packaging) |

Confirmed **not** applicable (deliberate divergence): AgentCore Runtime lessons from
`research-agent/lessons.md` (8h idle timeout, OTEL, `--environment-variables`) — this spec runs
K8s Jobs, not AgentCore microVMs. Mistral chat-template / context-compaction lessons from
`agent-harness` — harness-internal, not this runtime's concern.

## Stage 0 — Pre-launch checks (gate before scheduling a Job)

- Target `--cluster` reachable; node pool has ≥1 schedulable node with **both cluster SG + node SG**
- IRSA OIDC provider present; run-role trust policy admits the Job's ServiceAccount
- Artifact bucket exists, private, SSE on; DynamoDB run-state table reachable
- Git deploy credential (secret) present and scoped to this repo
- Requested `runtime:` profile + `--harness` adapter exist in the pinned `agent-runner` version
- The `runtime: <profile>@<version>` image **tag actually exists in the registry**
  (`docker manifest inspect` or equivalent) before scheduling — fail fast, don't discover a
  missing tag after the Job is admitted (tech-stack.md: verify image tags before deployment)

## Success Criteria

- [ ] `fe agent launch <spec> --cluster <c> --harness claude-code` creates a Job that checks
      out the committed SHA into a per-run worktree/branch and starts the loop
- [ ] Same launch with `--harness codex` runs the identical flow via the Codex adapter
      (proves harness-agnosticism — R1)
- [ ] Operator can close the laptop; run continues; `fe agent status <run-id>` later downloads
      and opens the latest private-S3 HTML report (no public URL)
- [ ] A run exceeding 8h completes (or hits the operator-set deadline) without a session ceiling
- [ ] `fe agent stop <run-id>` deletes the Job and revokes the run role within seconds
- [ ] Agent's commits land on `agent-run/<run-id>` branch and are reviewable as a diff/PR
- [ ] Run state in DynamoDB survives a pod recycle (status still queryable)
- [ ] `last_heartbeat` advances every ≤60s during a run; `fe agent status` warns when stale >5 min
- [ ] A run producing >64KB of harness output over hours does not deadlock (output-to-file, not PIPE)
- [ ] Stage 0 fails fast and clearly when the target node is missing the node SG (no silent IRSA hang)
- [ ] All readiness audit checks PASS or have documented PENDING rationale

## Non-Requirements

- No live file-sync / hot-reload dev loop — the handoff boundary is **git commit**, not live sync
- No held operator connection / interactive streaming as the primary path (attach is optional)
- No public status URL / presigned links (security: private S3 + authenticated pull only)
- No AgentCore Runtime, Cognito end-user auth, or ALB/websocket proxy
- No multi-region / HA (single cluster per run; `--cluster` selects which)
- No managed cross-session agent memory (each run is self-contained against one commit)

## Known Limitations

- (none yet — update during development)

## Spec history

| Date | Change |
|------|--------|
| 2026-06-19 | Initial spec created — harness-agnostic (claude-code + codex), EKS Job, git-commit contract, private-S3 status |
| 2026-06-19 | Added `agent-runner` sibling-repo packaging, R7 profile-ready container env, carryover fixes (pipe deadlock, node-SG pre-launch, force-delete cleanup, allowedTools, heartbeat, Bottlerocket AMI pin), Stage 0 gate |
| 2026-06-19 | Re-audit pass: added slim multi-stage image discipline (R7), Stage 0 image-tag verification, and `mdc`/`gpu-infra`-style sibling-CLI consumption (pinned release + thin `fe agent` wrapper) |
