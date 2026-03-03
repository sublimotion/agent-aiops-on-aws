# Blueprint Artifact Templates

Every deployment must produce these operational artifacts. Deployer agents (infra-deployer, agentcore-deployer) are responsible for creating them.

## Required Artifacts

### 1. Readiness Audit — `results/readiness-audit-<YYYYMMDD>.md`

Created by deployer agents at Stage 7 (Readiness Audit). One per deployment session.

```markdown
# Readiness Audit — <blueprint-name> — <YYYY-MM-DD>

## EKS Cluster
| Check | Status | Details |
|-------|--------|---------|
| Cluster status ACTIVE | PASS/FAIL | ... |
| API endpoint reachable | PASS/FAIL | ... |
| System nodes Ready | PASS/FAIL | ... |
| CoreDNS running | PASS/FAIL | ... |

## Storage
| Check | Status | Details |
|-------|--------|---------|
| FSx lifecycle AVAILABLE | PASS/FAIL | ... |
| PV/PVC bound | PASS/FAIL | ... |
| NVMe RAID mounted | PASS/FAIL/PENDING | ... |

## Container Images (ECR)
| Check | Status | Details |
|-------|--------|---------|
| <image> in ECR | PASS/FAIL | ... |

## GPU / Accelerator Plugins
| Check | Status | Details |
|-------|--------|---------|
| NVIDIA device plugin | PASS/PENDING | ... |
| EFA device plugin | PASS/PENDING | ... |

## Monitoring
| Check | Status | Details |
|-------|--------|---------|
| Prometheus running | PASS/FAIL | ... |
| Grafana running | PASS/FAIL | ... |

## Serving Layer
| Check | Status | Details |
|-------|--------|---------|
| Deployment exists | PASS/FAIL | ... |
| Health endpoint 200 | PASS/FAIL | ... |

## Config Scripts
| Check | Status | Details |
|-------|--------|---------|
| All configs pass bash -n | PASS/FAIL | ... |

## Action Items
| # | Priority | Action | Owner |
|---|----------|--------|-------|

## Overall Verdict
**PASS** / **CONDITIONAL PASS** / **FAIL**
```

### 2. Deployment Log — `results/deployment-log-<YYYYMMDD>.md`

Created by deployer agents during deployment. Timestamped entries.

```markdown
# Deployment Log — <blueprint-name> — <YYYY-MM-DD>

## Session Info
- **Operator**: <name or agent>
- **Blueprint**: <path>
- **Capacity block**: <reservation-id or N/A>
- **Start time**: <HH:MM TZ>

## Log

### <HH:MM> — Stage 1: Foundation
- Ran `terraform init` and `terraform apply`
- Outputs: VPC=<id>, EKS=<name>, FSx=<dns>
- **Status**: PASS

### <HH:MM> — Stage 2: Build Machine
- Launched m6i.xlarge in <subnet>
- Built Docker images, pushed to ECR
- **Status**: PASS

### <HH:MM> — <action>
- <details>
- **FAILED**: <error message>
- **Fix**: <what was done>
- **Lesson #N**: <lesson text>
- **Status**: PASS (after fix)

## Summary
- Total stages completed: N/8
- Issues encountered: N
- Lessons captured: N
- End time: <HH:MM TZ>
```

### 3. Lessons Learned — `lessons.md`

Append-only file in the blueprint root. Created on first deployment, appended after each session.

```markdown
# Lessons — <blueprint-name>

## Lesson #1 — <short title> — <YYYY-MM-DD>

**Context**: <what was being attempted>
**Observation**: <what happened>
**Rule**: <imperative statement of what to do>
**Why**: <rationale if non-obvious>
```

### 4. Compound Summary — `results/compound-<YYYYMMDD>.md`

Created by compound-learner agent after each deployment. See `.claude/agents/compound-learner.md` for the full template.

### 5. Benchmark Report — `results/benchmark-report.md` (if applicable)

Created by benchmark-analyst agent after benchmark runs. See `.claude/agents/benchmark-analyst.md` for the full template.
