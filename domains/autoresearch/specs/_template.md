# Autoresearch Spec: [Name]

## Status: DRAFT | IN_PROGRESS | COMPLETED

## Overview
Brief description of the autoresearch experiment.

## Components

### 1. Compute
- **Platform**: Bare metal GPU / Colab / Cloud instance
- **Instance Type**: (specify)
- **GPUs**: Count and type

### 2. Codebase
- **Source**: Repository URL or local path
- **Fixed files**: Files the agent must NOT edit (define the metric)
- **Agent-editable files**: Files the agent iterates on
- **Agent instructions**: Path to program.md

### 3. Experiment Protocol
- **Metric**: What to optimize (lower/higher is better)
- **Time budget**: Per-experiment wall-clock limit
- **Loop structure**: How the agent iterates
- **Termination**: When to stop
- **Logging**: How results are recorded

### 4. Networking
- **Access**: How to reach the compute

### 5. Storage
- **Data**: Where training/eval data lives
- **Results**: Where experiment logs are stored

## Success Criteria
Concrete, testable outcomes.

## Non-Requirements
What's explicitly out of scope.

## Known Limitations
Constraints to be aware of.

## Carryover Audit (spec-design gate)
Before running this experiment, confirm no lesson from a prior blueprint was left behind:
- [ ] Ran the `carryover-auditor` agent on this spec (or equivalent self-check): scanned every `domains/**/lessons.md` whose stack (`model`/`engine`/`gpu_arch`/`hardware`/`failure_categories`) overlaps this experiment.
- [ ] Every applicable prior lesson — especially `outcome: failure`/`partial` — is reflected here as a protocol step, environment check, or success criterion, OR noted as not applicable, citing its source (`<blueprint>/lessons.md` #N).

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
