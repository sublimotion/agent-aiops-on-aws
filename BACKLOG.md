# Harness Engineering Improvement Backlog

Improvements inspired by OpenAI's harness engineering patterns, ghost library concepts, and verification cascade thinking. Prioritized by impact and effort.

> Source analysis: [OpenAI Harness Engineering](../obsidian-notes/03_Resources/Agent-Infrastructure/OpenAI-Harness-Engineering-Codex-Zero-Human-Code.md) | [Ghost Libraries](../obsidian-notes/03_Resources/Agent-Infrastructure/Ghost-Libraries-Spec-Driven-Software-Distribution.md)

## Completed

### ✅ Lint Remediation Guide
**What**: Agent-readable guide mapping lint/checkov/tflint failures to specific fixes.
**Artifact**: `.claude/steering/lint-remediation.md`
**Pattern**: OpenAI's linter-first verification tier — agents fix lint issues autonomously instead of asking humans.

### ✅ Verification Criteria in Spec Template
**What**: Concrete, mechanically checkable conditions for each deployment stage (GPU health, serving stack, benchmark, readiness).
**Artifact**: `domains/gpu-serving/specs/_template.md` — `## Verification Criteria` section
**Pattern**: OpenAI's structural tests tier — every spec defines its own pass/fail criteria.

### ✅ Blueprint Reviewer as Pre-Deployment Gate
**What**: Blueprint-reviewer runs at Stage 0 (before infra changes) and post-deployment. Added verification criteria and lint readiness checks.
**Artifact**: `.claude/agents/blueprint-reviewer.md` — checks 6-7, "When to run" section
**Pattern**: OpenAI's agent peer review tier — automated coherence checking before deployment.

## Backlog

### 1. Quality Grades for AI-Generated Artifacts
**Effort**: Medium (2-3 hours) | **Impact**: High
**What**: Assign quality grades (A/B/C/D) to every generated artifact — configs, scripts, Terraform modules, lessons. Grade informs whether the artifact needs human review before use.
**Why**: OpenAI grades every AI-generated file and uses grades to route review effort. Without grades, all artifacts get equal (minimal) review, which means high-risk configs get the same scrutiny as boilerplate.
**How to implement**:
1. Add a `quality_grade` field to `lessons.md` YAML frontmatter and `results/*.md` files
2. Define grading rubric in `.claude/steering/quality-grades.md` (e.g., A=production-ready, B=needs spot-check, C=needs human review, D=draft/experimental)
3. Update compound-learner to assign grades when elevating lessons to steering rules
4. Update blueprint-reviewer to flag ungraded artifacts

**References**:
- [OpenAI Harness Engineering — quality_score.md pattern](../obsidian-notes/03_Resources/Agent-Infrastructure/OpenAI-Harness-Engineering-Codex-Zero-Human-Code.md)
- `.claude/agents/compound-learner.md` — where grades get assigned
- `.claude/agents/blueprint-reviewer.md` — where grades get checked

---

### 2. Doc-Gardener Background Agent
**Effort**: Medium (3-4 hours) | **Impact**: High
**What**: Background agent that runs weekly to detect and fix documentation drift — stale lessons, broken cross-references, outdated steering rules, specs that diverge from deployed state.
**Why**: OpenAI's "garbage collection" pattern: 20% of agent time is cleanup. Our steering files, lessons, and specs accumulate entropy. The compound-learner only fires after deployments, so drift between deployments goes undetected. Version refresh protocol exists but is manual.
**How to implement**:
1. Create `.claude/agents/doc-gardener.md` with these checks:
   - Steering rules older than 90 days without version refresh → flag as stale
   - Lessons in blueprint `lessons.md` not elevated after 3+ deployments → flag for compound-learner
   - Specs with verification criteria containing blank `_____` placeholders after first deployment → flag
   - Cross-references between CLAUDE.md routing table and actual spec files → detect orphans
2. Add a cron schedule or `/doc-gardener` command
3. Output a drift report to `results/doc-gardener-<YYYYMMDD>.md`

**References**:
- [OpenAI garbage collection pattern](../obsidian-notes/03_Resources/Agent-Infrastructure/OpenAI-Harness-Engineering-Codex-Zero-Human-Code.md)
- `.claude/steering/tech-stack.md` — version refresh protocol (already exists, needs automation)
- CLAUDE.md "Version Refresh" section — the manual process this would automate

---

### 3. Ghost Library Modules for Shared Infrastructure
**Effort**: High (1-2 days) | **Impact**: Medium
**What**: Convert shared Terraform modules (VPC, EKS, S3/FSx) into spec-driven ghost modules. Instead of importing a module, each blueprint gets a spec and the deployer agent generates a tailored implementation.
**Why**: Ghost library pattern — specs are durable, code is regenerated. Currently, shared modules create tight coupling: a change to the VPC module affects all blueprints. With ghost modules, each blueprint gets infrastructure tailored to its needs while conforming to the spec's constraints. Divergence between blueprints is intentional, not accidental.
**How to implement**:
1. Write `domains/gpu-serving/specs/infra-vpc.md` — VPC spec with CIDR ranges, AZ strategy, endpoint requirements
2. Write `domains/gpu-serving/specs/infra-eks.md` — EKS spec with version, add-ons, access mode
3. Update infra-deployer Stage 1 to read infra specs and generate Terraform from them (instead of importing shared modules)
4. Add spec-fidelity check to blueprint-reviewer: does the generated Terraform match the infra spec?

**Threshold check** (from ghost library analysis): Infrastructure modules are medium-complexity with clear API contracts. They pass the "value is in API design" heuristic. However, security-sensitive resources (IAM, KMS) should remain as traditional shared modules.

**References**:
- [Ghost Libraries — threshold question](../obsidian-notes/03_Resources/Agent-Infrastructure/Ghost-Libraries-Spec-Driven-Software-Distribution.md)
- [Ghost Libraries — verification implications](../obsidian-notes/03_Resources/Agent-Infrastructure/Ghost-Libraries-Spec-Driven-Software-Distribution.md)
- `domains/gpu-serving/specs/_template.md` — existing spec format to extend

---

### 4. Behavioral Telemetry for Deployment Agents
**Effort**: Medium (3-4 hours) | **Impact**: Medium
**What**: Instrument deployer agents with behavioral telemetry — track cost, tokens_per_edit, loop_count, stage transitions, and failure patterns. Use the Learned Verifier's behavioral features to predict deployment success/failure.
**Why**: The Learned Verifier proved that 4 behavioral features predict coding agent patch correctness (AUC=0.727). The same principle applies to infrastructure agents: a deployer that loops excessively at Stage 4a probably has a hardware issue; one that consumes high tokens at Stage 5 might be fighting a config problem. Early detection = less wasted GPU time.
**How to implement**:
1. Add telemetry hooks to infra-deployer: log token count, cost, and wall-clock time per stage
2. Define behavioral thresholds per stage (from historical deployment data in `results/`)
3. Create a lightweight "deployment health" check that runs between stages — flag if behavioral metrics exceed 2σ from baseline
4. Feed deployment telemetry into the Learned Verifier's adapter system (new adapter: `deployment_trace.py`)

**References**:
- [Learned Verifier — 4-feature RF](../obsidian-notes/01_Projects/Learned-Verifier-Experiment/Learned-Verifier-Experiment.md)
- [Behavioral Verification Convergence](../obsidian-notes/01_Projects/Learned-Verifier-Experiment/Synthesis-Behavioral-Verification-Convergence-April-2026.md)
- `/Users/phi/Documents/workbench/learned-verifier/src/learned_verifier/telemetry.py` — feature extraction to reuse
- `.claude/agents/infra-deployer.md` — where telemetry hooks would go

---

### 5. Structural Enforcement Script
**Effort**: Low (1-2 hours) | **Impact**: Medium
**What**: A pre-commit or CI script that validates repository structure — every blueprint has a matching spec, every spec is in the CLAUDE.md routing table, every blueprint directory has the required artifact files.
**Why**: The blueprint-reviewer already checks this, but only when explicitly invoked. A structural enforcement script runs automatically on every commit, catching drift before it compounds. OpenAI's approach: structural checks are the cheapest verification tier, so run them first and run them always.
**How to implement**:
1. Create `scripts/check-structure.sh` that:
   - For each `domains/*/blueprints/*/`, verify a matching spec exists in `domains/*/specs/`
   - For each spec, verify it appears in CLAUDE.md's routing table
   - For each blueprint, verify `lessons.md` exists (or is in `.gitignore` if brand-new)
   - Verify `.claude/steering/project-structure.md` lists all blueprints and specs
2. Add to `.pre-commit-config.yaml` as a local hook
3. Wire into blueprint-reviewer check #4 (steering file accuracy) so it runs the same script

**References**:
- [OpenAI Harness Engineering — linters as first verification tier](../obsidian-notes/03_Resources/Agent-Infrastructure/OpenAI-Harness-Engineering-Codex-Zero-Human-Code.md)
- `.claude/agents/blueprint-reviewer.md` — check #4 (steering file accuracy)
- `.claude/steering/project-structure.md` — the structure file this enforces

---

### 6. Entropic Objective + PUCT Archive for RL Blueprints (TTT-Discover)
**Effort**: Medium (1-2 days for entropic-objective swap; 2-3 days for PUCT archive harness) | **Impact**: Medium-High for RL work, low otherwise
**What**: Lift two specific techniques from TTT-Discover (Stanford/NVIDIA/Together, May 2026) into our RL-flavored blueprints — `kernel-optimization-agent`, `rl-conductor`, `verifier-reward` follow-ups, and any future RLVR work.
**Why**: The paper's headline framing ("RL at test time") is mostly marketing. The actual levers are (a) the **entropic objective** `J_β = log E[exp(β·r)]` with β tuned per-step to a fixed KL budget — beats expected-reward RL by ~40% on TriMul (1985 → 1203 µs at matched compute); (b) a **PUCT archive of (code, thinking-tokens) pairs** as warm-starts — without it, the same method collapses to worse-than-best-of-N (5274 µs). Both are recipe-level changes, not new algorithms, and both ablate cleanly. We don't need to reproduce the paper to benefit.
**How to implement**:
1. Document both techniques as a steering note (`.claude/steering/rl-recipes.md` or extend `tech-stack.md`) so future RL blueprints reach for them by default.
2. For `kernel-optimization-agent`: lift the reward shape (`1 / geomean(runtime)`, hard-0 on incorrect/timeout) and PUCT archive design from `examples/circle_packing/` and the TriMul reward in https://github.com/test-time-training/discover.
3. For `verifier-reward` Phase 2 / RLVR follow-up: try the entropic objective with adaptive β (KL budget γ = ln 2) as the default loss instead of GRPO/PPO. LoRA rank 32, lr 4e-5, single grad step per 512-rollout batch are the reference hyperparameters.
4. Skip running TTT-Discover's pipeline end-to-end — it's gated on Tinker credits ($500/problem) and gpt-oss-120b. We have our own GPU access; the value is the technique, not the reproduction.

**Threshold check**: Only worth pulling in once we have an active RL blueprint with verifiable reward (kernel runtime, gold test pass, etc.). For pure inference/serving work, no leverage.

**References**:
- Paper: https://test-time-training.github.io/discover.pdf
- Repo: https://github.com/test-time-training/discover
- Key ablation: Table 8 (TriMul) — entropic vs expected-reward, with/without PUCT reuse
- Adjacent blueprint: `domains/autoresearch/blueprints/kernel-optimization-agent/`
- Adjacent blueprint: `domains/autoresearch/blueprints/verifier-reward/` — recall ceiling root cause is semantic mismatch, but entropic objective could help if we move to RLVR
- Negative result the authors own: failed to improve second autocorrelation inequality (0.959 vs prior 0.961) — entropic + PUCT isn't magic.

---

## Priority Order

| # | Item | Effort | Impact | Dependencies |
|---|------|--------|--------|-------------|
| 5 | Structural enforcement script | Low | Medium | None |
| 1 | Quality grades | Medium | High | compound-learner, blueprint-reviewer |
| 2 | Doc-gardener agent | Medium | High | None |
| 4 | Deployment telemetry | Medium | Medium | Learned Verifier adapter |
| 3 | Ghost library modules | High | Medium | infra specs, blueprint-reviewer |
| 6 | TTT-Discover techniques (entropic obj + PUCT) | Medium | Medium-High (RL only) | Active RL blueprint with verifiable reward |

Recommended sequence: Start with #5 (quick win, enables #2), then #1 and #2 in parallel, then #4 (connects to Learned Verifier research), then #3 (largest scope, evaluate after other improvements are in place). #6 fires when the next RL blueprint kicks off.
