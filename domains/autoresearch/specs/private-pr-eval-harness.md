# Autoresearch Spec: Private-PR Eval Harness

## Status: IN_PROGRESS (pilot, Phase 0–2 — target repo: `pydantic/pydantic`)

## Overview

Build a contamination-free coding-agent benchmark from **recent merged PRs of a target
codebase** — the "your merged PRs are a private benchmark no model trained on" thesis from
Databricks' internal eval (blog, 2026-07-08). Each PR becomes one task: summarize the intent
into a prompt, strip the solution and its rationale, **seal the repo's git history** so the
agent can't recover the fix from `git log`, hold out the PR's test files, and score by whether
the agent's patch makes those held-out tests pass.

**Why this exists**: our SWE-bench-Lite production number (175/300 = 58.3%, VP-SWE-bench
blueprint) is on a years-old dataset now almost certainly in-distribution for frontier models.
It can no longer tell us which model/harness is actually better — only which memorized Lite.
A private-PR harness regenerates a clean number on demand from any repo with merged PRs and a
test suite, and is re-runnable as models improve.

**What it reuses**: ~80% of the VP-SWE-bench pipeline — held-out-test verification (no LLM
judge), Docker-isolated test execution, prediction-JSONL format, and the 8-harness ensemble
(OpenCode, SERA, Claude Code, Codex, Droid) from the agent-harness / agent-swarm blueprints.

**What is new** (three gates, all from the Databricks blog, none we do today):
1. **Git-history seal** — the working copy is cut off from `.git` history during the run.
2. **Solution-leak strip** — the task prompt carries PR *intent*, not the fix or why it's correct.
3. **Complexity-stratified sampling** — tasks sampled to a low/medium/high mix that matches the
   target org's real agent traffic, so the headline number is representative, not accidental.

## Design

**Unit of work**: one merged PR → one task. A task is `{prompt, base_commit, held_out_tests,
touched_paths, complexity_tier}`.

**Cells**: matrix of `{model} × {harness}` from the existing ensemble, one pass rate per cell,
plus the union (ensemble) pass rate. Same shape as agent-swarm's fix-rate matrix.

**Ground truth**: the PR's own test files (or manually repaired variants), executed in Docker.
**No LLM judge** — an LLM judge "rewards sounding right over being right" (Databricks), which is
exactly our verifier-reward semantic-mismatch ceiling (recall 0.14, FM-001 CLOSED). Held-out
tests are the only accepted verifier.

### Gate 1 — Git-history seal (fail-closed)

**Invariant**: during an agent run, `git log`, `git show`, `git diff <base>..HEAD`, and the
packed refs of the fixing commit MUST be unreachable from the workspace. An agent that runs
`git log -p` must not be able to read the solution diff.

**Acceptance check** (run before scoring any cell): a probe task where the agent is instructed
to `git log -p` and dump output — the fixing commit's diff MUST NOT appear. If it appears, the
seal has failed and results for that run are void.

Implementation is free (detached working tree, shallow single-commit re-init, `.git` removed and
replaced with a base-commit-only stub, filesystem overlay, etc.) — the spec enforces the
*invariant and its probe*, not the mechanism.

### Gate 2 — Solution-leak strip (fail-closed)

**Invariant**: the task prompt derived from a PR MUST NOT contain (a) the diff/patch, (b) the PR
description's statement of *why* the fix is correct, or (c) the names of the held-out test
functions. It MAY contain the observable problem (bug report, feature request, failing behavior).

**Acceptance check**: a sample of ≥20 generated prompts is hand-reviewed; any prompt from which a
reviewer can reconstruct the intended code change is rejected and the strip rule is tightened.
Log the reject rate. (Databricks: "removed why a fix was correct to avoid making it too easy.")

### Gate 3 — Complexity-stratified sampling

**Invariant**: the evaluated task set matches a declared complexity distribution (default, from
Databricks' observed mix: ~25% low / ~60% medium / ~15% high), NOT the accidental difficulty of
whatever PRs were pulled first.

**Tagging**: each candidate PR is assigned a tier by a cheap, auditable rule (default: Haiku
classifier over `{files_touched, net_lines, #test_targets, cross-module?}`; a pure heuristic on
those same features is an acceptable fallback). Tags are stored per-PR with their features so the
distribution is inspectable.

**Note on the Databricks quote**: that ¼-low/~60%-medium figure came from mining *Unity AI
Gateway* logs of live agent traffic — a proprietary Databricks feature. We do NOT need it. The
distribution is a *sampling prior*; we obtain it either (a) as the declared default above, or
(b) by mining our own LiteLLM proxy logs IF/when real agent traffic flows through it (currently
`litellm-glm52.yaml` has no `success_callback` — no traffic logged today). Log-mining is a
future enhancement, not a dependency.

### Test-repair allowance

Some held-out tests over-constrain the implementation (assert a specific approach, not the
behavior). Per Databricks, a sampled subset is hand-reviewed and such tests rewritten to accept
alternative correct implementations. Log every repaired test with a one-line reason. This is the
only sanctioned human edit to ground truth.

## Phases

### Phase 0: Target selection + carryover audit
- **Pinned target (pilot): `pydantic/pydantic`.** Rationale: active repo with frequent
  human-authored merged PRs, a strong dockerizable pytest suite, and — critically — **NOT in the
  SWE-bench repo set** (django/sympy/flask/requests/pytest/sphinx/astropy/xarray/pylint/seaborn/
  matplotlib/scikit-learn), so it is genuinely contamination-fresh. Single-language (Python) first
  cut; design generalizes to polyglot later.
  - Watch-item: pydantic-core is Rust; filter to PRs whose tests run in the pure-Python `pydantic`
    package unless a Rust toolchain is added to the eval image.
- Run the Carryover Audit (below) before writing any harness code. **Done** — two passes
  (SWE-bench/harness/verifier + agent-runtime), findings folded into Phases 2–4 and the Execution
  topology section.

### Phase 1: Task-mining pipeline
- `mine_prs.py`: pull merged PRs in a date window; filter (human-authored, has tests, self-contained).
- `synthesize_task.py`: PR → `{prompt (leak-stripped), base_commit, held_out_tests, touched_paths}`.
- `tag_complexity.py`: assign tier + store features.
- **Exit**: ≥50 tasks generated; Gate-2 hand-review passes on 20; complexity histogram printed.

### Phase 2: Sealed-run + eval harness
- Adapt VP-SWE-bench orchestrator: create workspace at `base_commit`, **apply Gate-1 seal**, run
  the agent (per cell), extract `git diff` as the candidate patch, patch in held-out tests, run in
  Docker, record pass/fail.
- **Docker build-error protocol**: the first eval pass will likely hit build errors on certain
  repos (VP-SWE-bench v1: 99/300 errors, mostly sympy/matplotlib/seaborn). Fix = pull prebuilt
  images with `--namespace swebench`, then re-run only the error instances (v2 → 0 errors). Wrap
  this in a `retry_docker_errors.py` since it recurs.
- **Context/tool-calling traps for self-hosted models** (from agent-harness/agent-swarm):
  - Qwen-family models emit **bare JSON** tool calls (`{"name":...,"arguments":...}`), not
    OpenAI-format `tool_calls`. vLLM's hermes parser returns empty → OpenCode/Claude Code/Droid
    score 0%. Only SERA's `_BARE_JSON_TOOL_RE` fallback works. The matrix is therefore **jagged**:
    Qwen cells are SERA-only unless every adapter is patched with the bare-JSON extractor.
  - Use **≥32K context** for Qwen/Mistral on vLLM (`--max-model-len 32768 --enforce-eager
    --gpu-memory-utilization 0.95`); 16K broken-pipes by ~turn 5.
  - `ClientOSError: [Errno 32] Broken pipe` = token-limit overflow (vLLM rejected the request),
    NOT a network fault → fix with context trimming + tool-output truncation, not retries.
  - Preserve Mistral chat-template role order during any context compaction (`tool` must follow
    `assistant`+`tool_calls`, no `user` in between) or requests 400.
- **Exit**: Gate-1 probe passes (no solution leak via git); end-to-end works on 10 tasks across 2 cells.

### Execution topology (generation on agent-runtime; eval on a Docker box)

Generation and Docker-eval MUST run on **separate machines** — one runtime Job cannot do both
without breaking the runtime's security posture. Split them, hand off via S3:

| Stage | Runs on | Rationale |
|-------|---------|-----------|
| Phase 1 task-mining (mine/strip/tag) | local, or a short agent-runtime Job | cheap, no Docker |
| **Phase 3 generation** (agent → patch, per cell) | **agent-runtime Job** (`fe agent launch`) | long/detached, survives >8h (no AgentCore 8h cap); per-run git worktree = Gate-1 seal *for free*; writes `predictions/<cell>.jsonl` to private S3 |
| **Phase 3 Docker eval** (held-out tests in containers) | **separate CPU box** — the `m7i.4xlarge` from VP-SWE-bench | needs a Docker daemon; the run role is **S3/DynamoDB/ECR/Bedrock/KMS only** (no pod/nodegroup perms), and DinD-in-Job needs privilege that fights scoped IRSA |

- **Gate-1 seal via the runtime, not a bespoke step**: the runtime already checks out the committed
  SHA into an isolated per-run worktree/branch. Extend that checkout to strip `.git` history to the
  base commit — the seal rides on machinery the runtime already has. Run the Gate-1 probe as the
  first task of every generation Job.
- **Model routing per cell**: closed-model cells hit **Bedrock** (run role already scoped for it);
  open-model cells (GLM-5.2, Qwen3.5) point the harness at a self-hosted vLLM/SGLang endpoint from
  the serving blueprints. No infra-scaling perms needed in the run role either way.
- **Handoff**: generation Job emits `predictions/<cell>.jsonl` → S3; the Docker box polls S3, patches
  in held-out tests, scores, writes `eval/<cell>/` back to S3. Same two-machine pattern VP-SWE-bench
  already validated.
- **Runtime carryover** (mostly handled by `fe agent launch`; cite so the blueprint doesn't rediscover):
  - redirect harness stdout/stderr to **files, never `subprocess.PIPE`** (64KB pipe-buffer deadlock,
    ThunderAgent lesson);
  - Stage-0 check the target node has both cluster SG + node SG before scheduling (IRSA pods silently
    can't reach `sts` otherwise);
  - pre-seed `/root/.claude.json` (`hasCompletedOnboarding`, pre-trust worktree, suppress MCP approval)
    — Claude Code's first-run TUI dialogs stall headless runs (managed-agent-runner L39-43);
  - DynamoDB state/heartbeat writes use **`UpdateItem` (merge), not `PutItem`** — PutItem wipes
    harness/cluster/commit fields `status`/`stop` need (managed-agent-runner L24-25);
  - set the run role's **`MaxSessionDuration=43200` (12h)** so the vended STS session actually covers
    a >8h run (managed-agent-runner L60-62);
  - run-id must be **RFC1123** (lowercase, no `T`/`Z`) — it names the Job/SA/ConfigMap (L26); raise
    Job ephemeral-storage above the 2Gi/8Gi default for large repos/harness outputs (L27-28).

### Phase 3: Full matrix run
- Run the `{model} × {harness}` matrix on the stratified task set; compute per-cell + union pass
  rates, per-tier breakdown, and **cost per completed task** (not per token — Databricks' key
  point: per-token price mispredicts task cost; cf. our Codex 140× token blowup).
- **Report both all-tasks and patch-only pass rates** (VP-SWE-bench: 58.3% vs 62.1%) and log the
  **empty-patch rate per cell** — agents that never edit produce 0 diffs (agent-harness edit rates
  ranged Aider 0% → OpenCode 88%). An empty patch is a fail, not a missing datapoint.
- **Exit**: pass-rate matrix + $/task Pareto frontier, per-tier.

### Phase 4: Analysis
- **Lead with the ensemble union pass rate** — in every prior run the union beat the best single
  cell by 37–57% (agent-harness: SERA 16% + LangGraph 14% → union 22%; agent-swarm 8-harness →
  36%), and each harness contributed 1–2 *unique* passes from different repo/issue combos. Report
  which tasks are unique to each cell, not just per-cell rates.
- Pareto frontier (quality vs $/task), per-tier win-by-tier, harness-driven cost spread. Compare
  to our stale SWE-bench-Lite 58.3% to quantify the contamination gap.
- **Distribution sanity check**: confirm the sampled task set matches the declared complexity
  prior within ±10pp per tier; if the target repo's PRs are wildly skewed, the headline won't
  generalize — say so.
- **Exit**: which model/harness wins per complexity tier, with $/task and CIs.

## Components

### Compute
- **EC2**: `m7i.4xlarge` (16 vCPU, 64 GB, ≥200GB gp3) — Docker test eval is CPU-bound (VP-SWE-bench proven).
- **API/serving**: hosted (Bedrock/Anthropic) for closed models; self-hosted vLLM/SGLang on g7e/B300
  for open models (GLM-5.2, Qwen3.5) via the existing serving blueprints.
- **Verification cost**: negligible vs generation (VP-SWE-bench: $3.20 for 300 issues). Docker test
  execution is CPU-bound, not a GPU workload — cost lives in the agent generation calls.

### Codebase
```
scripts/
├── mine_prs.py            # merged-PR fetch + filter
├── synthesize_task.py     # PR → leak-stripped task (Gate 2)
├── tag_complexity.py      # tier + features (Gate 3)
├── seal_workspace.py      # Gate 1 seal + probe
├── run_matrix.py          # {model}×{harness} orchestrator (adapts VP-SWE-bench)
└── analyze_results.py     # Pareto, per-tier, $/task, CIs
```

### Storage
- **Tasks**: `results/tasks.jsonl` (+ per-task features/tier)
- **Predictions**: `results/predictions/<cell>.jsonl`
- **Docker eval**: `results/eval/<cell>/`
- **Repaired tests / reject log**: `results/ground_truth_edits.md`

## Success Criteria
- **Minimum**: ≥50 stratified tasks; all 3 gates pass their acceptance checks; a pass-rate number
  for ≥3 cells with Docker eval.
- **Target**: full `{model}×{harness}` matrix with per-tier pass rates and $/task Pareto frontier;
  a defensible statement of "which agent wins per complexity tier on a clean codebase."
- **Stretch**: reproduce the Databricks qualitative findings on our own data — (a) an open model
  (GLM-5.2) statistically tied with the strongest closed model on quality, (b) harness/context
  discipline drives >2× cost spread at equal quality.
- **Negative result**: the private-PR number tracks SWE-bench-Lite rankings closely → contamination
  wasn't distorting our conclusions and Lite is still usable. Still valuable.

## Non-Requirements
- **NOT** rebuilding Unity AI Gateway. Log-mining for the complexity prior is optional/future and
  depends on real proxy traffic we don't have yet. Default to the declared distribution.
- **NOT** an LLM judge, ever (Gate: held-out tests only).
- **NOT** polyglot on the first cut — start dockerizable-Python, generalize later.

## Known Limitations
- Ground-truth quality is capped by the target repo's test suite; low-coverage PRs are silently
  un-scorable and must be filtered, not guessed.
- "Statistically tied" claims require reporting n + CIs (the Databricks blog omits them) — do not
  call two cells tied without a test. (Carryover from verifier-reward: small gaps on a few hundred
  tasks are often noise.)
- Complexity tier is a heuristic proxy; publish the tagging rule so the distribution is auditable.
- **Pass rate is a lower bound**: repos with version-specific dependency conflicts can't be verified
  without Docker (agent-harness could only gold-eval Django/pytest/sympy locally). Full Docker eval
  is mandatory for a complete number; any unevaluable task must be reported as such, not silently
  scored as a fail.

## Carryover Audit (spec-design gate)
Before running this experiment, confirm no lesson from a prior blueprint was left behind:
- [x] Ran `carryover-auditor` over `domains/**/lessons.md` overlapping {SWE-bench eval,
  coding-agent harnesses, verifiers} — 6 blueprints scanned, 0 contradictions, 0 P0 gaps; six P1/P2
  carryovers folded into Phases 2–4 + Known Limitations (Docker error protocol, bare-JSON/Qwen
  jagged matrix, ≥32K context + broken-pipe, empty-patch + all-vs-patch-only reporting, ensemble
  union as headline, verification cost, lower-bound caveat, distribution sanity check).
- [x] Reflected here or marked N/A with citation:
  - VP-SWE-bench: Docker eval needs `--namespace swebench` to pull prebuilt images (v1 had 99
    build errors → 0). Python 3.12 for SWE-bench repos (3.14 breaks distutils). Claude Code
    headless MUST pass `--allowedTools` or no tools available. → fold into `run_matrix.py`.
  - verifier-reward: LLM-judge verification has a hard recall ceiling from semantic mismatch
    (FM-001 CLOSED) → Gate: held-out tests only, no judge. Small pass-rate gaps are noise → require CIs.
  - agent-harness / agent-swarm: token cost varies ~140× by harness (Codex vs OpenCode);
    context-per-turn discipline dominates cost → report $/task, not $/token. Harness spread is
    50pp for weak models, 16pp for strong → run the full matrix, don't fix one harness.
  - infra: gold eval Python compat — SWE-bench repos need Python 3.12 (3.14 removed distutils and
    breaks old pytest); install setuptools in the eval venv for the distutils shim. Carryover from
    verification-primitives.

---

> **Note**: Operational artifacts (lessons, results, analysis) belong in the blueprint directory,
> not in this spec.
