# Trinity Coordinator — lessons (Stage 0 setup + $0 gates)

Run context: authoring/setup environment on branch
`agent-run/trinity-coordinator-68fd2c3-20260624-210954`. The actual evolution run
executes as a detached EKS GPU Job (agent-runtime), not here. This pass completed
every $0 gate that does not require GPU/torch/boto3, built + validated the Bedrock
adaptation layer and the blueprint Terraform, and surfaced two launch blockers.

## Environment facts (verified 2026-06-24)

- **No local GPU / no torch / no boto3** in the authoring env. Expected: the run
  is a detached EKS Job (g6e-class) per the spec; the coordinator forward pass +
  CMA-ES driver run there. Pure-Python adaptation modules self-test here; the
  fugu train/eval path is exercised only inside the Job.
- **`bedrock:ListFoundationModels` is IAM-denied** for the run-role
  (`agent-runner-run-role`), but **`bedrock-runtime:Converse` works** (the harness
  itself runs on Bedrock). So Gate 0.0 was done by **actual Converse pings**, not
  catalog listing — a stronger liveness check anyway.
- AWS identity: `assumed-role/agent-runner-run-role/...`, account `615299764834`,
  default region `us-east-2` (workers invoked cross-region by the Job).

## Gate 0.0 — worker liveness via Converse (model-ID drift caught live)

Probed every worker with a real Converse call. Drift found and fixed:

| Ord | Worker | Result | Fix |
|----|--------|--------|-----|
| 0 | claude-opus-4-8 | ✅ `us.anthropic.claude-opus-4-8` | anchor (replaces GPT-5) |
| 1 | claude-sonnet-4-6 | ✅ `us.anthropic.claude-sonnet-4-6` | — |
| 2 | **Nova Premier** | ❌ provider-marked **Legacy / access-denied** on-demand | **replaced with `us.amazon.nova-pro-v1:0`** (Amazon distinct-provider slot preserved) |
| 3 | gemma-3-27b | ✅ `google.gemma-3-27b-it` | — |
| 4 | deepseek-r1 | ❌ bare `deepseek.r1-v1:0` rejects on-demand → ✅ `us.deepseek.r1-v1:0` | **us. inference profile required** |
| 5/6 | qwen3-32b | ✅ `qwen.qwen3-32b-v1:0` | dual-mode confirmed (below) |

**Spec correction**: the spec's pool table lists Nova Premier (`amazon.nova-premier-v1:0`)
as "✅ confirmed" at ord 2 and DeepSeek-R1 as bare `deepseek.r1-v1:0`. Both are
WRONG for this IAM principal as of 2026-06-24 — Nova Premier is Legacy/denied and
DeepSeek-R1 needs the `us.` profile. The blueprint pool
(`scripts/worker_pool_bedrock.py`) carries the corrected IDs; the spec's "Live?"
column should be updated. This is exactly the model-ID-drift carryover lesson,
reconfirmed live.

**Dual-mode Qwen3-32B (BLOCKING probe) — RESOLVED, one ID, two modes:**
- Reasoning (ord 5): `additionalModelRequestFields={"reasoning_effort":"high"}`
  → response contains a `reasoningContent` block. **Confirmed.**
- Direct (ord 6): no flag → `text`-only block, no reasoning. **Confirmed.**
- Bedrock's Qwen3 uses **OpenAI-style `reasoning_effort`** (high|medium|low|
  minimal|none), **NOT** the Anthropic `reasoning_config` schema — passing
  `{"reasoning_config":{"type":"enabled"}}` raises a `validation_error`
  ("unknown variant `type`"). 21 role-cells (not 42) is therefore correct: both
  modes share role templates, differing only by the reasoning_effort flag.

Other live frontier swap candidates probed: ✅ `minimax.minimax-m2.5`,
✅ `deepseek.v3.2`, ✅ `us.amazon.nova-pro-v1:0`, ✅ `mistral.mistral-large-3-675b-instruct`,
✅ `us.meta.llama4-maverick-17b-instruct-v1:0`, ✅ `openai.gpt-oss-120b-1:0`.
❌ `us.anthropic.claude-opus-4-7` (validation error), ❌ `zai.glm-5`,
❌ `moonshot.kimi-k2-thinking` (silent/denied).

## Gate 0.1 — static setup audit (BLOCKING $0 check) — PASS

By static read of the vendored fugu (no training compute spent):
- Coordinator base model = `Qwen/Qwen3-0.6B` (es_log `configs.model_name`). ✅
- Hidden-state layer = **26** (`opt_layer_indices: [26]`), extracted at
  **position -2 / penultimate token** (`fugu/hidden_state_utils.py`
  `extract_hidden_state_at_position(position=-2)`). ✅ NOT a default/last layer.
- Worker prompts: chat templating is applied server-side by Bedrock Converse
  (we pass role/text blocks); upstream's `apply_chat_template` was for local vLLM.
  The role-specific prompt construction in `fugu/core.py` is unchanged. ✅

## Gate 0.2 / 0.2b — adaptation + parser audit

- **Dispatch seam identified**: every worker call funnels through
  `fugu/utils.py:_query_llm`, which routes by model-name substring to
  per-provider clients. `scripts/bedrock_clients.py:install()` rebinds those
  provider functions to one Converse path — the spec's "fixed algorithm, adapt
  only the clients" contract is honored (no edits to es.py/core.py/head_modules).
- **Gate 0.2b script written** (`scripts/gate_0_2b_role_audit.py`): the 21-cell
  (7 workers × 3 roles) output-parser audit, reusing the EXACT parsers from
  `core.py` (`_parse_thinker_response`, `_parse_verification_response`). Must run
  inside the Job (needs boto3+creds) before any training spend. NOT yet executed
  (no boto3 here) — this is the first thing the Job runs.

## Adaptation layer (built + self-tested here)

- `scripts/worker_pool_bedrock.py` — 7-worker pool, head-order = es_log order,
  corrected Bedrock IDs, reasoning knobs. Self-test ✅.
- `scripts/bedrock_clients.py` — Converse dispatch: per-(worker,region) semaphore,
  cross-region rotation `[us-east-1, us-west-2, us-east-2]` on ThrottlingException,
  expo backoff+jitter (cap 20s, 8 attempts), dropped-episode telemetry, thinking-
  block stripping, optional GPT-5.5 `openai_compat` bearer-token branch. Compiles ✅.
- `scripts/cost_bedrock.py` — verified Bedrock pricing, drop-in for fugu.cost;
  usage-based + estimator paths; run-start snapshot. Self-test ✅.
- `scripts/run_trinity_agent.py` — in-Job entry point: installs monkeypatches,
  strips ports (forces cloud path), per-iter S3 sync **incl iter 0**, hard cost
  cap, Phase-0.5 gate checks (worker/role non-degeneracy, throttle, entropy>1.5).
  Compiles ✅.
- All five modules `py_compile` clean.

## Terraform (blueprint-local, per agent-runner contract) — validates

- `terraform/irsa-run-role.tf` — OIDC trust + Bedrock invoke scoped to the 7
  worker inference-profile + foundation-model ARNs across the 3 rotation regions,
  S3 RW scoped to the `trinity-coordinator/` prefix only. GPT-5.5 deliberately NOT
  granted (bearer-token path bypasses IRSA).
- `terraform/artifact-bucket.tf` — private, SSE-KMS, versioned; raw rollouts
  expire after N days, checkpoints/es_log retained. Shared-bucket-safe (writes
  under a prefix) or creates a dedicated bucket.
- `terraform fmt` clean, `terraform validate` ✅. (checkov not installed here —
  run in pre-commit/CI before apply.)

## BLOCKERS (surfaced to operator)

1. **Phase 0 eval target `model_iter_60.npy` is ABSENT.** The vendored copy ships
   only `es_log.json` (config block, **no training history**) + the data split —
   no `logs/ckpt/models/` dir, no checkpoint, never committed. README claims the
   checkpoint is present; it is not. **Phase 0 (eval-only baseline) cannot run
   without it.** Options: (a) fetch the checkpoint from the OpenReview
   supplementary and add it to the blueprint; (b) skip Phase 0 and start at Phase
   0.5 (train from scratch — does NOT need the checkpoint); (c) treat Phase 0.5's
   first eval as the baseline. OQ1 ("is the bundled ckpt the headline model?") is
   moot until the file exists.
2. **Launch requires inputs not available in this env**: the `agent-runner` CLI is
   not on PATH; `terraform apply` needs real EKS OIDC provider ARN/URL +
   namespace; the run-role + bucket must be applied (Gate 0.0 step 0). These are
   operator-provided. The blueprint is launch-ready *pending* those inputs.

## Pre-registered before any full run (per spec)

- Pool composition: **Opus 4.8 at ord 0** (GPT-5.5 OFF — not visible to this IAM
  principal; optional bearer-token swap documented, gated on Gate 0.0 conditional).
- Hard cost cap enforced in `run_trinity_agent.py` (`--cost-cap-usd`, halts at cap).
- Best-static-worker is the bar Trinity must beat by ≥3pp (the bar GRPO failed).
- `cost_bonus_weight=0.0` reproduces upstream; the sweep is the OQ3 extension.
