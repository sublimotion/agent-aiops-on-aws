# Progress Tracking Format

A **progress file** is the single-source-of-truth for where a blueprint stands in its lifecycle. It lives at `<blueprint>/results/progress.md` and is updated by deployer agents during execution or reconstructed from artifacts by `scripts/progress.sh`.

## Schema

```yaml
---
# Progress Schema v1
blueprint: ""           # e.g. ray-serve-ft, kimi-k2.5
domain: ""              # gpu-serving | agent-runtime | autoresearch
spec: ""                # path to spec file
status: ""              # not_started | in_progress | blocked | complete
last_updated: ""        # ISO 8601 timestamp
last_stage: ""          # most recent stage or phase completed

stages: []              # ordered list of stage objects (see below)
# - id: "stage-1"
#   name: "Foundation"
#   status: "complete"      # not_started | in_progress | blocked | complete | skipped
#   started_at: ""          # ISO 8601 or null
#   completed_at: ""        # ISO 8601 or null
#   notes: ""               # optional one-liner

phases: []              # spec-defined phases/tests (optional, blueprint-specific)
# - id: "T1"
#   name: "Replica Crash Recovery"
#   status: "complete"
#   notes: ""

artifacts:
  lessons: false        # lessons.md exists
  readiness_audit: []   # list of audit dates
  deployment_log: []    # list of log dates
  compound: []          # list of compound dates
  benchmark_report: false
---
```

## Stage IDs by Domain

### gpu-serving (infra-deployer)

| ID | Name |
|----|------|
| `stage-0` | Deployment card lookup |
| `stage-1` | Foundation |
| `stage-2` | Build machine |
| `stage-3` | Storage and model staging |
| `stage-4` | Capacity reservation and GPU node |
| `stage-4a` | GPU health validation |
| `stage-5` | Serving stack deployment |
| `stage-6` | Pre-benchmark validation |
| `stage-7` | Readiness audit |
| `stage-8` | Compound |

### agent-runtime (agentcore-deployer)

| ID | Name |
|----|------|
| `stage-1` | Foundation (Terraform) |
| `stage-2` | Container Build |
| `stage-3` | AgentCore Runtime |
| `stage-4` | Auth Wiring (Cognito) |
| `stage-5` | WebSocket Proxy |
| `stage-6` | Integration Test |
| `stage-7` | Readiness Audit |
| `stage-8` | Compound |

### autoresearch (autoresearch-runner)

| ID | Name |
|----|------|
| `stage-1` | Read Spec |
| `stage-2` | Validate Environment |
| `stage-3` | Setup Codebase |
| `stage-4` | Configure Loop |
| `stage-5` | Run Baseline |
| `stage-6` | Execute Loop |
| `stage-7` | Analyze Results |
| `stage-8` | Capture Lessons |

### Spec-defined phases

Phases are extracted from the spec's `## Test Scenarios` or `## Experiment Protocol` sections. They are blueprint-specific and appended after the deployer stages.

## Reconstruction

Run `scripts/progress.sh <blueprint-path>` to rebuild progress.md from existing artifacts:

- Deployment logs → stage completion timestamps
- Readiness audits → stage-7 status + verdict
- Compound summaries → stage-8 status
- Lessons.md existence → artifact tracking
- Spec test scenarios → phase definitions
- Results files (ft_summary.md, benchmark-report.md) → phase completion

## Agent Updates

Deployer agents append a stage entry when transitioning stages. Format:

```markdown
## Stage N: Name — STATUS

**Started**: YYYY-MM-DD HH:MM UTC
**Completed**: YYYY-MM-DD HH:MM UTC (or "in progress")
**Notes**: One-line summary of what happened
```
