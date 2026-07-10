# Private-PR Eval Harness — Lessons (pilot, pydantic/pydantic)

Pilot scope: Phase 0–2. Target `pydantic/pydantic`. Goal = validate the
mine→strip→tag→seal→eval pipeline + 3 gates on ~10 tasks × 2 cells, then STOP.

## Phase 0 — target selection
- **pydantic is contamination-safe on recency, not on library familiarity.** All 62 mined
  candidates merged 2025-08 → 2026-07; ~24 are post-2026-02 (past Opus 4.8's Jan-2026 cutoff).
  The library *code* is in every model's training set, but the specific issue→fix pairs are
  recent enough to be unmemorized. Recency floor is the real defense; `--merged-after` added to
  the miner (all 62 already clear 2025-07-01, so it's a no-op here but matters for older repos).
- pydantic-core is Rust — filtered implicitly by requiring a Python test file; watch if extending.

## Phase 1 — mining + Gate 2 (leak-strip)
- **Yield ~14%**: 450 PRs scanned → 62 kept. Rejections: no_test=252 (56%!), bot=82, no_source=54.
  The dominant filter is "PR has no test file" — most merged PRs are docs/CI/refactor. Budget
  ~7 PRs scanned per usable task.
- **Natural complexity mix ≈ the Databricks prior.** Heuristic tags (net_lines/n_files) gave
  21% low / 55% med / 24% high vs the 25/60/15 prior — all within 10pp, no skew warning. pydantic's
  PR mix is representative out of the box; no resampling needed for the pilot.
- **Gate-2 bug #1 (over-strip):** first `strip_leak` dropped any line containing common verbs
  (fix/add/change/because) → prompts reduced to useless fragments. Fix: only redact lines that
  *explicitly announce the solution* ("the fix is", "I fixed by"), drop code fences, keep symptom
  prose intact. Issues describe symptoms — that's why they're the preferred source.
- **Gate-2 bug #2 (wrong issue linked):** pydantic PR bodies close issues via **full URL**
  (`Fixes https://github.com/pydantic/pydantic/issues/13369`), not `Fixes #N`. The old `#N` regex
  missed the URL and false-matched a stray `#123` inside the HTML-comment PR template → every PR
  got the SAME bogus issue body. Fix: strip `<!-- -->` comments first, then match both `#N` and
  the issues-URL form. Reclassified 10 PRs issue→title_only (was 54/8, now 44/18 issue/title).
- **Gate-2 acceptance PASS:** systematic scan of all 62 prompts for `diff --git` / `def test_` /
  held-out test filenames → 0 real leaks. One false positive (#13206) is an inline ` ```code``` `
  span in prose, benign. Prompts carry the observable problem only.
- **base_commit = merge_commit's first parent** (`gh api commits/<sha> -q .parents[0].sha`) — the
  tree the PR branched from. Correct checkout point for the agent.

## Phase 2 — git-seal (Gate 1) + eval harness
- **Gate-1 seal validated.** Mechanism: clone at base, `rm -rf .git`, re-init, single "base
  snapshot" commit. Adversarial probe is conclusive — `git show <fix-commit>` returns
  `fatal: bad object` (fix genuinely unreachable), `git log` = 1 commit, no remotes,
  `rev-list --all --count == 1`. Agent keeps a working git for its own diffs but cannot time-travel.
- **Eval scoring model validated (differential test on #13363):**
  - BASE tree, held-out tests injected → **1 failed / 63 passed** (test detects the bug) → passed=false ✓
  - GOLD source-patch applied → **66 passed** (fix resolves + adds cases) → passed=true ✓
  - The scorer discriminates: injecting the post-fix test files from the merge commit as ground
    truth, then requiring all-pass, correctly separates fixed from unfixed. No LLM judge.
- **Gold candidate = PR diff restricted to SOURCE files.** Split the PR diff on `diff --git`
  headers; the tests half is ground truth (injected), the source half is the "known-good" patch
  used to validate scoring. An agent's candidate patch takes the source half's place.
- **Eval infra gotchas (pilot, local uv-venv mode):**
  - `uv pip install` refuses without a venv (PEP 668) → create `.eval-venv` per workspace with
    `uv venv --python 3.12` (matches carryover: SWE-bench repos want 3.12, not 3.14).
  - pydantic's `pyproject.toml` addopts inject `pytest-benchmark` flags → neutralize with
    `pytest -o addopts= -p no:cacheprovider` so held-out tests run standalone.
  - Test deps: install the repo's **declared PEP-735 `dependency-groups`** (`--group dev
    --group testing-extra`) instead of hand-guessing packages — directly dodges the
    "version-specific deps" lower-bound trap (agent-harness L158). Fallback to a common-extras
    list for repos without dependency-groups.
- **Docker mode is the production path** (separate CPU box per spec); venv mode is the pilot
  mechanism check and is faster (no container build). Both share the same inject→run→score logic.

## Phase 2 — efficiency finding (matters for the production Docker path)
- **Per-task editable install rebuilds pydantic-core (Rust) from scratch (~2-4 min/task).** A
  10-task batch is ~30-40 min, almost all Rust compilation. For the full matrix this dominates
  wall-clock. Mitigations for the Docker/CPU-box path: (a) build a base image with pydantic-core
  precompiled at the pinned commit and only re-apply the source patch, or (b) `uv` shared build
  cache / prebuilt wheel, or (c) reuse one venv across tasks that share a base_commit. The pilot
  eats the cost (correctness > speed); production must not.

## Phase 2 — batch validation (generalization) PASS
- **9/10 gold_pass across all 3 tiers** (low/medium/high) on the seal→inject→score pipeline.
  Confirms the mechanism isn't overfit to one hand-picked task.
- **The 1 failure (#12636) was a MINING ARTIFACT the eval correctly refused to green-light** —
  a valuable negative. It touched 14 files including `pydantic/generics.py` (a **Pydantic V1** file
  absent in V2), `pyproject.toml`, `.github/workflows/ci.yml`, `requirements-testing.txt`, and
  `tests/mypy/test_mypy.py`. It is NOT a self-contained behavior change. The harness surfacing this
  as gold_pass=False (instead of rubber-stamping) is exactly the honest-failure behavior we want.
- **Filter tightened as a result** (mine_prs.py): reject PRs touching build/CI/env plumbing
  (`pyproject.toml`, `.github/`, `requirements*`, `uv.lock`, `setup*`, `tox.ini`, `Makefile`,
  `Dockerfile`) or stale V1 files (`pydantic/generics.py`); drop tasks whose only tests are
  harness-unsupported (mypy-plugin tests need a mypy run, not pytest). This is the spec's
  "self-contained" Gate operationalized — discovered empirically, not guessed.

## Phase 2 — clean re-mine + generation wiring (PASS)
- **Tightened filter → 50 clean tasks**, #12636 excluded, `not_self_contained=77` now rejected.
  Distribution improved to **26% low / 58% med / 16% high** vs the 25/60/15 prior (all within 2pp) —
  removing plumbing PRs pulled the mix right onto the Databricks prior. 37 issue-sourced, 13 title-only.
- **Generation topology DECIDED (option A, in-image loop).** `fe agent launch --dry-run` proved the
  launcher runs ONE agent against ONE commit — and that commit is THIS repo's HEAD (9fcbc6f), not the
  target repo's base. So "one launch per task" (option B) is wrong: it checks out the wrong repo.
  Correct design = **one Job per harness cell whose container entrypoint loops the sealed task set**
  (seal → run harness → predictions/<cell>/<pr>.diff to S3). 2 harnesses = 2 Jobs, not 2×N.
- **Gotcha:** `fe agent launch --dry-run` prints its banner to **stderr**, the rendered Job YAML to
  **stdout**. Detect success on `kind: Job` in stdout, not "DRY RUN" in stdout.
- **Both pilot cells render valid Jobs (dry-run, no pods):** claude-code (Bedrock/Opus) +
  opencode (self-hosted vLLM/GLM-5.2) — the two best single harnesses from prior blueprints.

## Phase 3 (pilot) — FIRST LIVE OOD DATAPOINT (end-to-end works)
- **Opus 4.8 (Bedrock, Claude Code headless) solved a real OOD task end-to-end.** Task #13363
  (pydantic, post-cutoff, git-sealed): given only the symptom, Opus edited
  `pydantic/experimental/pipeline.py` (+12/-2, 31 turns, $1.37) with a fix that DIFFERS from the
  human gold patch but is functionally correct → held-out tests **66 passed → PASSED=True**.
  Full chain proven: mine→strip→seal→generate(live)→inject held-out tests→score. No LLM judge.
- **Decision confirmed: run generation LOCALLY, not on the agent-runtime image.** The box is
  already Bedrock-configured (`CLAUDE_CODE_USE_BEDROCK=1`, Opus ARN). Runtime would require baking
  a custom ECR image with a fan-out entrypoint (runtime runs THIS repo's specs vs one SHA, not N
  sealed pydantic workspaces) — not worth it for a 10-task pilot; reserve for the 50×N matrix.
- **Cells switched to 2× closed-model (Opus + Sonnet, both Bedrock).** GLM-5.2/opencode cell
  dropped for the pilot: no GLM endpoint live, no GPU nodes up. The contamination-gap question
  needs only one model through both benchmarks; two closed cells add model-spread for free, $0 GPU.
- **run_generation.py gotcha:** `claude -p` exits 0 even at `max_turns` → judge success by
  **non-empty candidate patch**, not returncode. Empty diff = agent explored but never edited (real fail).
- **Cost signal:** ~$1.37/task for Opus at 30 turns. 10 tasks × 2 models ≈ $25-30 generation.

## Phase 3 (pilot) RESULT — contamination gap ≈ ZERO (n=10, 2 closed cells)

| cell | pydantic OOD (sealed, post-cutoff) | SWE-bench-Lite | gap | empty patches |
|------|-----|-----|-----|-----|
| Opus 4.8   | 6/10 = 60% | 58.3% | +2pp | 3 |
| Sonnet 4.6 | 6/10 = 60% | 58.3% | +2pp | 1 |

**Answer to "does this eval prep yield a better/harder OOD eval?": NO measurable contamination
penalty at n=10.** Both models score ~the same on fresh git-sealed pydantic tasks as on Lite.
This is the spec's pre-registered **negative-result branch**: contamination is NOT visibly
distorting Lite rankings for these models. A *valuable* null — Lite still looks usable here.

**The confound is real and asymmetric (the actual interesting finding):**
- Opus produced **3 empty patches** (explored, never edited within 30 turns) vs Sonnet's 1.
- **Opus conditional on producing a patch: 6/7 = 86%.** So Opus is *more capable per attempt* but
  *less reliable at committing an edit* — its aggregate 60% is dragged down by the harness (turn
  budget / Parkinson's-Law under-editing), NOT by OOD difficulty. This matches memory
  [[project_single_stream_decode_levers]]-adjacent harness findings: edit-rate is a harness
  property, and stronger models can under-edit without turn pressure.
- Implication: "which model is better" on this eval is currently a **harness artifact**, not a
  capability signal. Must inject edit-checkpoint pressure (verification-primitives two-stage) before
  the aggregate pass rate means anything cross-model.

**Caveats (do NOT over-read):** n=10 → CI ≈ ±30pp, +2pp is pure noise. pydantic ≠ Lite repos, so
"OOD" conflates contamination with codebase difficulty. Result validates the INSTRUMENT, not a
publishable contamination number. Both cells share the exact per-tier split (low 4/5, medium 2/5)
— medium tier is where both models drop, a mild difficulty signal worth watching at larger n.

**Cost:** ~$1.37/task Opus, cheaper Sonnet; full pilot (20 gen + 20 score) ≈ $25-30, ~1.5h wall.

## Pilot (Phase 0–2) — STATUS: harness complete & validated
- Gate 1 (git-seal): PASS — fix commit `fatal: bad object` from sealed ws.
- Gate 2 (leak-strip): PASS — 0 real leaks across prompts; symptom-only.
- Gate 3 (complexity): PASS — heuristic tags match the prior within 2pp.
- Eval scoring: PASS — differential base(fail)→gold(pass) on #13363; 9/10 gold_pass across tiers.
- Generation cells: PASS — 2 cells render valid EKS Jobs (dry-run).
- **Deliberately STOPPED before spending on the live matrix** — that is the separate budgeted launch.
- **NEXT budgeted launch:** build the cell container entrypoint (loop tasks, seal, run harness,
  push predictions to S3) + the separate Docker-eval CPU box that scores predictions. Precompile
  pydantic-core in the eval base image (per efficiency finding) before scaling to the full 50×2.
