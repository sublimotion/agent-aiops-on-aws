# Autoresearch Spec: Verification Primitives — SWE-bench Production Eval

## Status: COMPLETE

## Overview

Run Claude Code with verification primitives + two-stage checkpoint guidance on SWE-bench Lite (300 issues), using the official Docker-based evaluation harness. Compare against the published Claude Code baseline.

**Why Lite, not Verified**: SWE-bench Verified uses a standardized harness (mini-SWE-agent) that we cannot modify. We need control over the generation harness to inject verification primitives. SWE-bench Lite allows any generation pipeline — we control how patches are produced, Docker eval handles the standardized testing.

**Prior result**: Two-stage checkpoint (Edit@40% + Verify@55%) on Sonnet 4.6, custom harness, n=300 SWE-bench Lite: 95% fix rate, 24.3% gold pass (p=0.0009 vs control). But this was local gold eval (lower bound, no Docker isolation).

**Goal**: What is the true Docker-evaluated pass rate of Claude Code + two-stage verification primitives on SWE-bench Lite?

## Design

**Single cell**: Claude Code + Sonnet 4.6 + verification CLI scripts + two-stage CLAUDE.md guidance, SWE-bench Lite (300 issues).

**Baseline**: Published Claude Code SWE-bench Lite number (no re-run needed).

### Injection Mechanism

Verification primitives are **CLI scripts** in the workspace. Claude Code calls them via its built-in bash tool — the most natural tool invocation pattern for LLMs. The two-stage checkpoint is **CLAUDE.md instructions**.

**Mechanism parity note**: The original experiment registered tools as Python functions in a custom agent loop. Claude Code's equivalent is bash — both are first-class tool invocation in their respective harnesses. The CLAUDE.md workflow guidance maps to the original experiment's system message checkpoint injection. The self-logging telemetry in each script maps to the original experiment's tool invocation tracking. Same signals, native mechanism for each harness.

```
workspace/
├── CLAUDE.md                    # Two-stage guidance + tool descriptions
└── verify/
    ├── generate_tests.py        # LLM call → adversarial test file
    ├── run_tests.py             # Execute tests in workspace
    ├── adversarial_review.py    # v009 rubric, single LLM call
    └── telemetry.jsonl          # Auto-appended by each script on invocation
```

**CLAUDE.md guidance** (injected into each issue workspace):
```markdown
## Workflow

You are fixing a GitHub issue. Follow this workflow:

1. Read the issue and explore the codebase to understand the problem
2. Make your edit early — don't over-explore. Aim to have your fix written
   within the first 40% of your effort.
3. After editing, verify your fix:
   - Run `python3 verify/generate_tests.py --diff <your_diff>` to generate
     adversarial tests designed to break your patch
   - Run `python3 verify/run_tests.py --test-file <generated_tests>` to
     execute them
   - Run `python3 verify/adversarial_review.py --diff <your_diff>` for a
     final adversarial code review
4. If verification fails, iterate on your fix.

## Tools

- `verify/generate_tests.py` — Generates edge-case tests targeting your patch.
  Catches issues before submission.
- `verify/run_tests.py` — Runs generated tests in the workspace.
- `verify/adversarial_review.py` — Adversarial code review using a 5-axis rubric.
```

### Verification Scripts

Reuse implementations from verification-primitives experiment, wrapped as CLI with **built-in telemetry**:

- **generate_tests.py**: Takes `--diff` and `--problem` args, calls Haiku via Bedrock with adversarial prompt, outputs test file. ~$0.008/call.
- **run_tests.py**: Takes `--test-file`, runs pytest in workspace, returns pass/fail per test. Compute only.
- **adversarial_review.py**: Takes `--diff` and `--problem`, calls Haiku with v009 rubric, returns structured verdict. ~$0.003/call.

All scripts are self-contained (no imports from the experiment codebase). They need `boto3` for Bedrock calls.

### Self-Logging Telemetry

Every verification script appends a JSON line to `verify/telemetry.jsonl` on each invocation:

```json
{"ts": "2026-04-01T12:34:56Z", "tool": "generate_tests", "inputs": {"diff_len": 1234, "problem_len": 500}, "outputs": {"test_count": 5, "success": true}, "elapsed_s": 8.2, "cost_usd": 0.008}
```

The orchestrator collects `verify/telemetry.jsonl` from each workspace after the run.

**Composition pattern classification** reuses the same taxonomy from the original experiment (ignore, generate_run, generate_run_iterate, full_pipeline, full_pipeline_iterate), derived directly from the telemetry log.

## SWE-bench Evaluation Details

### Prediction Format (JSONL)

```json
{"instance_id": "django__django-11630", "model_name_or_path": "claude-code-verify", "model_patch": "diff --git a/..."}
```

Each line: `instance_id` (from dataset), `model_name_or_path` (label), `model_patch` (the diff string).

### Docker Evaluation Command

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path results/predictions_lite.jsonl \
    --max_workers 8 \
    --run_id claude-code-verify-lite \
    --cache_level env
```


### Docker Infrastructure

- **3-layer images**: base → env (~60 images) → instance
- **Storage**: 120GB+ free disk required (`--cache_level env` keeps env images, ~100GB)
- **Compute**: x86_64, max_workers < 0.75 * cpu_count
- **Output**: `evaluation_results/<run_id>/results.json` + `instance_results.jsonl`
- **ARM (M-series Mac)**: Add `--namespace ''` to build locally instead of pulling from DockerHub

## Phases

### Phase 1: Setup (2-3 days)

1. **Package verification scripts as standalone CLI tools**
   - Port from experiment codebase, remove dependencies on harness_eval
   - Add `--diff`, `--problem`, `--workspace` CLI args
   - Each script self-logs to `verify/telemetry.jsonl`
   - Test on 5 known-outcome issues locally

2. **Set up SWE-bench Docker eval on EC2**
   - Instance: `m7i.4xlarge` (16 vCPU, 64 GB, **200GB gp3**)
   - Install: `pip install swebench`, Docker, Claude Code (`npm install -g @anthropic-ai/claude-code`)
   - Validate: `python -m swebench.harness.run_evaluation --predictions_path gold --max_workers 1 --instance_ids sympy__sympy-20590 --run_id validate-gold`
   - Pre-warm: run `--cache_level env` on 5 instances to pull Docker env images

3. **Build run orchestrator** (`scripts/swebench_claude_code.py`)
   - For each SWE-bench issue:
     1. Create workspace (git clone at base_commit)
     2. Copy `verify/` scripts + `CLAUDE.md` into workspace
     3. Run Claude Code headless: `claude -p "Fix this issue: {problem_statement}" --allowedTools Bash,Read,Write,Edit --output-format json`
     4. Extract diff from workspace (`git diff`)
     5. Format as SWE-bench prediction JSONL line
     6. Collect `verify/telemetry.jsonl` from workspace
   - Concurrency: 4-8 parallel (rate limit dependent)

**Exit criteria**: End-to-end pipeline works on 10 issues, Docker eval produces correct pass/fail results.

### Phase 2: Run (2-3 days)

1. **Run Claude Code + verification on SWE-bench Lite (300 issues)**
2. **Run Docker evaluation**: `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite --predictions_path results/predictions_lite.jsonl --max_workers 8 --run_id claude-code-verify-lite`
3. **Compare to**: (a) published Claude Code Lite baseline, (b) our prior local gold eval (24.3%)

**Exit criteria**: Pass rate number on 300 issues with Docker eval.

### Phase 3: Analysis (1-2 days)

1. **Compare to published Claude Code Lite baseline** (Fisher exact test)
2. **Local vs Docker eval**: Compare to our prior 24.3% local gold eval — quantifies the Docker uplift
3. **Tool adoption**: % of issues where Claude Code called verification scripts (from telemetry)
4. **Composition patterns**: Classify from telemetry (same taxonomy as prior experiment)
5. **Cost analysis**: Verification overhead as % of total
6. **Visual report**

**Exit criteria**: Clear answer with statistical test.

## Components

### Compute
- **EC2**: `m7i.4xlarge` (16 vCPU, 64 GB, 200GB gp3) — Docker eval is CPU-bound
- **API**: Sonnet 4.6 via Anthropic API (Claude Code) + Haiku via Bedrock (verification tools)

### Codebase
```
scripts/
├── verify/
│   ├── generate_tests.py          # Standalone CLI, Bedrock Haiku, self-logging
│   ├── run_tests.py               # Standalone CLI, subprocess pytest, self-logging
│   └── adversarial_review.py      # Standalone CLI, Bedrock Haiku, self-logging
├── swebench_claude_code.py        # Orchestrator: issue → Claude Code → prediction
├── claude_code_workspace_template/
│   └── CLAUDE.md                  # Two-stage guidance template
└── analyze_results.py             # Comparison + composition pattern analysis
```

### Storage
- **Results**: `results/predictions_lite.jsonl`
- **Docker eval**: `results/eval_lite/`
- **Telemetry**: `results/telemetry/<instance_id>.jsonl` (collected from workspaces)

## Cost Estimate

| Item | Cost |
|------|------|
| Claude Code (Sonnet 4.6) x 300 Lite issues | ~$300-600 |
| Verification tools (Haiku) x ~600 calls | ~$10 |
| EC2 m7i.4xlarge x 5 days | ~$40 |
| **Total** | **~$350-650** |

## Success Criteria

- **Minimum**: Pass rate established on Lite with Docker eval, tool adoption measured
- **Target**: Pass rate beats published Claude Code Lite baseline by >3pp (Fisher p<0.05)
- **Stretch**: >50% on SWE-bench Lite
- **Negative result**: Claude Code already subsumes the two-stage checkpoint — verification primitives only help simpler harnesses

## Risk Register

- **Claude Code ignores verification scripts** (MEDIUM): May not follow CLAUDE.md guidance. Mitigation: test on 20 issues, iterate on wording. If adoption <20%, try stronger prompting.
- **Docker eval infrastructure** (MEDIUM): 120GB+ Docker images, build failures. Mitigation: `--cache_level env`, pre-warm on 5 instances.
- **Claude Code headless mode** (LOW): Some features may not work in `-p` mode. Mitigation: test early, check `--output-format json` output structure.
- **Rate limits** (LOW): 4-8 concurrent Sonnet calls should be fine with standard tier.
- **ARM vs x86** (LOW): If running on Mac, need `--namespace ''` for Docker. EC2 avoids this.

## Relationship to Other Specs

- **verification-primitives** (COMPLETE): Source of two-stage design, tool implementations, composition pattern taxonomy, all prior results
- **agent-harness** (COMPLETE): Claude Code integration tested there (Phase 2b)
- **verifier-reward** (COMPLETE): v009 adversarial rubric used in `adversarial_review` tool

---

> **Note**: Operational artifacts belong in the blueprint directory, not in this spec.
