# Private-PR Eval Harness — Pilot Blueprint

Spec: `domains/autoresearch/specs/private-pr-eval-harness.md`

Contamination-free coding-agent benchmark built from a target repo's recent merged PRs
(Databricks-style). Each PR → one task: strip the solution, **seal git history**, hold out
the PR's test files, score by whether the agent's patch makes those tests pass. **No LLM judge.**

**Pilot target:** `pydantic/pydantic` (not in the SWE-bench repo set → contamination-fresh;
all mined PRs merged 2025-08→2026-07). **Pilot scope:** Phase 0–2 (harness + gates validated),
STOP before the live `{model}×{harness}` matrix (separate budgeted launch).

## Pipeline (scripts/)

| Script | Stage | What it does |
|--------|-------|--------------|
| `mine_prs.py` | 1 | Fetch merged PRs, filter to self-contained bug-fixes with tests. `--merged-after` recency floor. |
| `tag_complexity.py` | 1 (Gate 3) | Heuristic low/med/high tier + features; prints histogram vs the 25/60/15 prior. |
| `synthesize_task.py` | 1 (Gate 2) | PR → leak-stripped task prompt (symptom only; no diff/why/test-names). Prefers linked-issue body. |
| `seal_workspace.py` | 2 (Gate 1) | Clone at base_commit, destroy history, re-init as single sealed commit. `--probe` = acceptance check. |
| `run_eval.py` | 2 | Apply candidate patch, inject held-out tests from the fix commit, run pytest, score. `venv` (pilot) / `docker` (prod) modes. |
| `validate_batch.py` | 2 | Seal→gold-eval differential across N tasks/tiers to prove generalization. |
| `plan_generation.py` | 2 exit | Plan the 2 generation cells; `fe agent launch --dry-run` proves the Jobs render (no pods). |

## Pilot results

- **All 3 gates PASS** + eval scoring validated (differential base→gold on #13363).
- **Batch generalization: 9/10 gold_pass** across tiers. The 1 fail (#12636) was a mining artifact
  (14-file V1/CI PR) the eval correctly refused — → filter tightened to reject non-self-contained PRs.
- **50 clean tasks**, distribution 26/58/16 (within 2pp of the prior).
- **Both generation cells render valid EKS Jobs** (claude-code/Bedrock + opencode/vLLM), dry-run.

See `lessons.md` for the full findings and gotchas. `results/` holds tasks, batch verdicts, and the
generation plan.

## Execution topology (see spec)

Generation (agent → patch) runs on the **agent-runtime EKS Job** (one Job per harness cell, in-image
loop over sealed tasks). Docker-based test eval runs on a **separate CPU box** (run role lacks pod
perms; DinD fights scoped IRSA). S3 is the handoff.

## Next (separate budgeted launch — NOT the pilot)

1. Build the cell container entrypoint: loop tasks → seal → run harness headless → predictions to S3.
2. Stand up the Docker-eval CPU box; precompile pydantic-core in the base image (per efficiency finding).
3. Launch 2 generation Jobs, score predictions, compute per-tier + union pass rates and $/task.
