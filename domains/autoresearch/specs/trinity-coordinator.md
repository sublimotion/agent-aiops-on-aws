# Autoresearch Spec: Trinity Coordinator — Bedrock + Agent-Runtime Reproduction

## Status: DRAFT (2026-06-24)

## Overview

Reproduce **Trinity** (Xu et al., *"TRINITY: An Evolved LLM Coordinator"*, ICLR 2026, arXiv:2512.04695) — Sakana AI's evolved multi-agent coordinator — on an **all-Bedrock worker pool** (closed + open models) with the multi-turn orchestration loop executed inside this repo's **agent runtime** (EKS detached run via `agent-runner`).

Trinity is the published, code-available core of the orchestration approach behind Sakana's **Fugu** product. We have the authors' code submission (vendored at `vendor/trinity-upstream/`, from the OpenReview supplementary). This spec adapts it from its original mixed-provider setup (OpenAI/Anthropic/Gemini direct APIs + locally-vLLM-served open models) to a single-account Bedrock pool, and runs the evolution + evaluation as a managed agent-runtime job.

### Why this experiment

This directly closes the loop on our **GRPO router negative result** (`domains/autoresearch/specs/grpo-router-negative-result.md`). That writeup proved a single-policy GRPO LLM router collapses on a multi-modal cost-aware reward and is strictly worse than best-static. Trinity is the architecture our own writeup named as the alternative: **gradient-free evolution (CMA-ES) over a tiny parameter set, not RL over a 7B policy.** The Trinity paper's own algorithm bake-off corroborates our finding — on LiveCodeBench, `sep-CMA-ES 0.615 > SFT 0.592 > RS 0.374 > REINFORCE 0.253`; REINFORCE (≈ our GRPO) is dead last at 0.25, matching our observed +0.24 collapse.

Reproducing Trinity gives us:
1. **A working orchestrator** where our GRPO attempt failed — same problem class, different optimizer.
2. **A clean head-to-head**: Trinity (CMA-ES, hidden-state head, multi-turn) vs. our Conductor/GRPO repro (NL routing, single-policy RL). We already have the GRPO half; this completes the comparison.
3. **A Bedrock-native, agent-runtime-native** implementation, reusable for cost-aware routing follow-ups.

### What Trinity is (architecture)

| Component | Detail |
|-----------|--------|
| **Coordinator SLM** | Qwen3-0.6B. Reads its own **penultimate-token hidden state** — does NOT generate routing text. Generated tokens are discarded. |
| **Router head** | Single linear layer: `hidden_size → (L + 3)`. L logits select a worker; 3 logits select a **role** (Thinker / Worker / Verifier). ~10K params. |
| **SVD fine-tuning** | Decompose selected base-model weight matrices via SVD; **only the singular-value scales are learned**, U/V kept fixed. Tunes one layer (paper: layer 26 / "second-to-last"). |
| **Total trainable params** | < 20K (head + SVD scales). |
| **Optimizer** | **CMA-ES** (gradient-free; `cma` library). Population λ=32, mCMA=16 replications, σ₀=0.03. Optimizes the flattened (SVD scales + head weights) vector. |
| **Roles** | **Thinker** (plans/decomposes/critiques), **Worker** (executes — derivation/code/result), **Verifier** (ACCEPT/REVISE; ACCEPT halts the loop). |
| **Loop** | Each turn: coordinator reads hidden state → picks (worker, role) → builds role-specific prompt → queries worker → appends output. Halts on Verifier ACCEPT or turn budget (5). |

### Why CMA-ES dodges the GRPO failure

Our negative result's formal argument: GRPO's within-batch advantage normalization `(r − mean) / std` mixes easy/hard questions whose optimal worker differs, producing a multi-modal landscape with no monotone gradient direction — the gradient pulls toward the cheap tier. **CMA-ES computes no gradients.** It's a black-box population optimizer over a ~10K-dim vector; multi-modal landscapes are exactly its competence. Trinity also conditions on the question (via hidden state) where our shared-policy simulator had no question features — but the headline lever is the optimizer.

## Key adaptations from upstream

### 1. Worker pool → all-Bedrock

Upstream pool (7 workers): GPT-5, Gemini-2.5-pro, Claude-4-Sonnet (closed, direct APIs) + Gemma-3-27B, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B (reasoning), Qwen3-32B (direct) (open, locally vLLM-served).

**Bedrock replacement pool** — selected against the **live us-east-1 catalog (verified 2026-06-24)**. Upstream had 3 closed-frontier (GPT-5, Gemini-2.5-pro, Claude-4-Sonnet) + 4 open. We preserve the role-class spread (3 closed-frontier, 1 mid-open, 2 reasoning-open, 1 direct-open) with today's strongest verified Bedrock options.

| Ord | Role-class | Upstream | Bedrock replacement | Bedrock model ID | Live? (2026-06-24) |
|-----|-----------|----------|---------------------|------------------|--------------------|
| 0 | closed-frontier | GPT-5 | **Claude Opus 4.8** (newest frontier) | `anthropic.claude-opus-4-8` (`us.` profile) | ✅ confirmed (`us.`+`global.`) |
| 1 | closed-frontier | Claude-4-Sonnet | **Claude Sonnet 4.6** | `anthropic.claude-sonnet-4-6` | ✅ confirmed |
| 2 | closed-frontier | Gemini-2.5-pro | **Nova Premier** (Amazon frontier; distinct provider) | `amazon.nova-premier-v1:0` | ✅ confirmed |
| 3 | open-mid | Gemma-3-27B | **Gemma 3 27B** | `google.gemma-3-27b-it` | ✅ confirmed |
| 4 | open-reasoning | DeepSeek-R1-32B | **DeepSeek-R1** | `deepseek.r1-v1:0` | ✅ confirmed |
| 5 | open-reasoning | Qwen3-32B (reasoning) | **Qwen3-32B** (reasoning mode) | `qwen.qwen3-32b-v1:0` | ✅ confirmed |
| 6 | open-direct | Qwen3-32B (direct) | **Qwen3-32B** (direct mode) | `qwen.qwen3-32b-v1:0` (`direct` payload flag) | ✅ confirmed |

**On GPT-5.5 (user-requested) and Opus 4.8**: live verification 2026-06-24 —
- **Opus 4.8 confirmed Bedrock-native** (`anthropic.claude-opus-4-8`, `us.`+`global.` inference profiles, IAM-invokable via Converse; set at ord 0).
- **GPT-5.5 (`openai.gpt-5.5`) is account-scoped, reached via a Bedrock API key — not visible to this experiment's IAM principal.** Findings (2026-06-24): `aws bedrock list-foundation-models` returns **0 `gpt-5` models** for IAM user `aiops` (account `615299764834`) in us-east-1/2, us-west-2, eu-central-1, ap-northeast-1 — only OpenAI `gpt-oss` open-weights. Direct `converse(modelId="openai.gpt-5.5")` returns `ValidationException: invalid model identifier` under both `aiops` SigV4 creds AND the operator's `AWS_BEARER_TOKEN_BEDROCK`. The operator's **codex** reaches `openai.gpt-5.5` natively (codex confirms it does *not* route through a LiteLLM proxy — earlier "LiteLLM" guess was wrong).
- **Root cause**: a long-term Bedrock API key is backed by its own **auto-created IAM user** with its own model-access grants; **GPT-5.5's marketplace/model-access subscription is enabled for the account/principal that minted the operator's bearer token, NOT for `aiops`.** Model *visibility* (marketplace enablement) is account+region scoped; the two identities resolve to different enablement. Same laptop, two identities → two Bedrock views. (Operator to confirm with `source ~/.codex/.env && aws bedrock list-foundation-models --region us-east-2 --query "modelSummaries[?contains(modelId,'gpt-5')].modelId"`.)
- **Implication**: Trinity's agent-runtime Job authenticates via **IRSA SigV4** as a role in account `615299764834` — which cannot see or invoke GPT-5.5. To use GPT-5.5 the Job must carry the **operator's bearer token** (the credential that holds the grant), mounted as a K8s secret, and call Bedrock's **OpenAI-compatible endpoint** (`https://bedrock-runtime.<region>.amazonaws.com/openai/v1/chat/completions` with `Authorization: Bearer <token>`) — NOT the SigV4 Converse path.
- **Decision**: keep **Opus 4.8 at ord 0** as the verified, IAM-invokable frontier anchor (the default, fully self-contained path). GPT-5.5 is an **OPTIONAL ord-0 swap** documented in `bedrock_clients.py` (see "GPT-5.5 optional path" below). Do NOT hardcode `openai.gpt-5.5` as a SigV4 Converse `modelId` — it fails for the Job's principal (cost-aware-routing model-ID-drift lesson, confirmed live here).

**Other live frontier/strong options** (swap candidates per access/pricing/cost-cap): `anthropic.claude-opus-4-7`, `zai.glm-5`, `minimax.minimax-m2.5`, `moonshot.kimi-k2-thinking`, `nvidia.nemotron-super-3-120b`, `qwen.qwen3-next-80b-a3b`, `mistral.mistral-large-3-675b-instruct`, `deepseek.v3.2`.

All workers reachable via **Bedrock Converse API**, single-account billing. No self-hosted GPU serving for workers — the major infra simplification over upstream (4 GPUs serving open models on ports 8326-8329). Anthropic frontier models (Opus 4.8, Sonnet 4.6) require the `us.` cross-region inference-profile prefix.

The coordinator SLM (Qwen3-0.6B) still runs locally on GPU — it's tiny (0.6B, ~1.5GB) and its hidden states must be read directly, so it cannot be a Bedrock endpoint.

### 2. LLM clients → Bedrock Converse

`vendor/trinity-upstream/fugu/llm_clients.py` already has `query_anthropic()` using Bedrock Converse. Adaptation:
- **Keep** `query_anthropic` (already Bedrock).
- **Replace** `query_oai` (hardcoded GPT-5) and `query_gemini` (Google GenAI SDK) with Bedrock Converse calls — same pattern as `query_anthropic`, different model IDs.
- **Replace** `query_deepseek` (Together AI) and `query_locally_hosted_model` (vLLM HTTP) with Bedrock Converse for the open workers.
- **Net result**: one `query_bedrock(model_id, messages, ...)` function via Converse; the per-provider functions become thin wrappers mapping friendly names → Bedrock IDs. Reuse the throttle/backoff loop already in `query_anthropic`.
- Reasoning vs direct mode for Qwen3-32B: pass `additionalModelRequestFields` / payload flags through Converse (reasoning workers get thinking enabled).

New file: `scripts/bedrock_clients.py` — a Converse-based drop-in for `fugu/llm_clients.py`'s dispatch. Reuse the cost-aware-routing blueprint's `worker_proxy.py` Converse patterns (thinking-block filtering, usage-token accounting) which we already debugged against all these model families.

**Throttle handling at CMA-ES population scale (concrete, not "reuse the pattern")**: each CMA-ES iteration evaluates λ=32 × mCMA=16 = **512 candidate episodes**, each a multi-turn (≤5) Bedrock conversation — a burst far above default per-model regional rate limits (frontier models often 100-200 req/min). `bedrock_clients.py` MUST implement, mirroring cost-aware-routing `worker_proxy.py`:
- **Per-worker asyncio semaphore** (start at concurrency=10 per worker; tune per model's TPM headroom) so one model can't saturate.
- **Round-robin across regions** `['us-east-1', 'us-west-2', 'us-east-2']` on `ThrottlingException`, advancing region per retry.
- **Exponential backoff with jitter** (`base 0.5s × 2^attempt + rand(0,0.5)`, cap 20s), max ~8 attempts, then count the episode as a dropped rollout (reward 0, logged) rather than crashing the iteration.
- **Throttle-rate telemetry** per iter; if dropped-episode rate >2%, the run is rate-limited not compute-bound — raise concurrency caps or add a region before the next iter.
Phase 0.5 Exit Criterion 3 validates this end-to-end at the real 512-candidate burst before the full run.

**GPT-5.5 optional path (ord-0 swap; off by default)**. The default pool's ord 0 is Opus 4.8 (SigV4 Converse, fully self-contained). To swap in GPT-5.5 — which is reachable ONLY via the operator's account-scoped Bedrock API key (see Worker-pool §) — `bedrock_clients.py` carries a **second client branch** gated on a `worker.transport` field:
- `transport: "converse"` (default, all 6 other workers + Opus 4.8) → SigV4 `bedrock-runtime.converse`, IRSA creds.
- `transport: "openai_compat"` (ord 0 iff GPT-5.5 enabled) → HTTPS POST to `https://bedrock-runtime.<region>.amazonaws.com/openai/v1/chat/completions`, `Authorization: Bearer ${BEDROCK_BEARER_TOKEN}`, OpenAI chat schema (`model: "openai.gpt-5.5"`, `max_completion_tokens`). This reuses the OpenAI-compatible request shape already in `fugu/llm_clients.py:query_oai` / `query_locally_hosted_model` — only the base URL + auth header change.
- **Credential delivery**: the bearer token is mounted as a **K8s secret** (`bedrock-bearer-token`) into the Job, exposed as env `BEDROCK_BEARER_TOKEN`. It is NOT the IRSA role — it is the operator's grant-carrying credential. The IRSA role still provides Converse access for the other 6 workers + S3. Two credentials coexist in the Job: IRSA (SigV4, 6 workers + S3) and the bearer token (GPT-5.5 only).
- **Gate 0.0 conditional**: only enable this branch if, with the bearer token, `list-foundation-models` shows `openai.gpt-5.5` AND a 1-token OpenAI-compat ping returns cleanly. Otherwise ord 0 stays Opus 4.8. Pre-register which pool composition the run used (the eval number depends on it).
- **Cost + throttle**: GPT-5.5's per-token price and rate limits attach to the bearer-token account, not ours — track its spend separately in `cost_bedrock.py` and give it its own semaphore (its limits differ from our IAM principal's).

### 3. Cost model → verified Bedrock pricing

`fugu/cost.py` has hardcoded per-provider pricing (GPT/DeepSeek/Anthropic/Gemini/OpenSource tiers). Replace with the verified Bedrock pricing table (per 1M in/out tokens) snapshotted at run-start, the same way the cost-aware-routing blueprint does it (`aws pricing get-products`). Cost feeds the CMA-ES `cost_bonus_weight` reward term.

### 4. Execution → agent runtime

The evolution loop is long-running (paper regime: 1.5k–40k evaluations; our budget below) and makes thousands of Bedrock calls. Run it as a **detached agent-runtime job** (`agent-runner` on EKS, per `domains/agent-runtime/specs/managed-agent-runner.md`):
- Author this spec, commit, launch a detached run on EKS via `fe agent`.
- The Job holds **IRSA-scoped Bedrock invoke + S3 RW** credentials (no long-lived keys).
- Coordinator SLM runs on a single GPU node (g6e.xlarge-class is enough for 0.6B); workers are Bedrock API calls from inside the Job.
- State (CMA-ES solver pickle, best model `.npy`, `es_log.json`, rollouts) persists to S3 every iteration — survives the run exceeding any session ceiling.
- Operator detaches and reattaches for a visual status report.

This replaces upstream's `server.sh` (local vLLM serving) + bare `python evaluate_trinity_livecodebench.py` with a managed, credential-scoped, resumable run.

## Phases

### Phase 0: Eval-only validation with bundled checkpoint (Days 1-2)

**Goal**: Confirm the vendored trained coordinator (`vendor/trinity-upstream/logs/ckpt/models/model_iter_60.npy`, ~156KB, consistent with <20K params) reproduces the paper's LiveCodeBench result — *without any training* — on the Bedrock pool.

1. **Gate 0.0** — verify all 7 Bedrock worker model IDs + pricing live (`aws bedrock list-foundation-models`, `aws pricing get-products`). Swap any unavailable worker, preserving role-class coverage. **Carryover from cost-aware-routing Gate 0.0** (model-ID drift: GLM-5 was `zai.glm-5` not `zai.glm5`; Llama 4 needed `us.` inference-profile prefix; Nova needs Converse not raw `invoke_model`; Anthropic 4.5+ absent from Pricing API). **Dual-mode Qwen3-32B probe (BLOCKING)**: the pool treats `qwen.qwen3-32b-v1:0` as TWO distinct workers — reasoning (ord 5) and direct (ord 6). Confirm Bedrock Converse expresses both via payload flags: (a) reasoning-mode call → verify thinking blocks appear; (b) direct-mode call → verify none. If both modes don't work via the same model ID, decide now (find a reasoning inference-profile variant, OR collapse to 6 workers and remap ordinals, OR substitute a different model). This is a **$0 check that blocks a $1K-5K run** — do not defer to "Phase 1 will find out."
2. **Gate 0.1** — decompose Qwen3-0.6B (`decompose_model.py`), confirm SVD weights regenerate (we excluded the 4MB `svd_weights.pt` from the vendored copy; it rebuilds deterministically). **Setup audit (carryover from `feedback_paper_repro_setup_audit`, BLOCKING $0 check)**: by static read of `fugu/core.py` + `fugu/hidden_state_utils.py` (and one single-forward smoke), confirm BEFORE spending any training compute: (a) coordinator base-model variant is Qwen3-0.6B as upstream specifies; (b) the worker prompts go through `apply_chat_template`; (c) the hidden state is extracted from **layer 26** (`opt_layer_indices: [26]` in `es_log.json`), NOT a default/last layer. A wrong layer extracts noise and the head trains on garbage — it would only surface as the Phase 0.5 entropy≈0 criterion after ~$150-250 is spent. Catch it here for $0.
3. **Gate 0.2** — port `fugu/llm_clients.py` dispatch to Bedrock Converse; smoke each of the 7 workers with a 1-token Converse ping (liveness only).
4. **Gate 0.2b — per-(worker × role) output audit (BLOCKING; carryover from cost-aware-routing)**. A 1-token ping does NOT catch output-format failures, and **multi-turn episodes compound format drift across the 3 roles**. cost-aware-routing's Gate 0.2b caught **24/33 cell failures** (Kimi K2 Thinking intermediate `\boxed`, Opus extended-thinking content blocks, GLM-5 nested `<think>` tags, mid-reasoning truncation) — and the PUNCHLIST flags this as "CRITICAL — the rl-conductor reward bugs hid for weeks because this gate didn't exist." Probe each of the **7 workers × 3 roles = 21 cells** with a multi-token query: Thinker prompt → confirm `<suggestion>`/`<suggested_role>` parse and any thinking tags are stripped; Worker prompt → confirm answer/code extraction; Verifier prompt → confirm ACCEPT/REJECT parses. (Reasoning and direct Qwen3-32B modes share the same role-specific prompt templates, so 21 cells is correct; if the dual-mode probe in Gate 0.0 reveals they need distinct templates, expand to 14 × 3 = 42 cells.) Reuse cost-aware-routing's Converse `worker_proxy.py` patterns (thinking-block filtering, usage-token accounting). Fail any cell → fix extraction (don't lower the bar). Pass criterion: every cell parses cleanly ≥90% of the time.
5. Run `evaluate_trinity_livecodebench.py` with `model_iter_60.npy` on the bundled LiveCodeBench v6 split (`.data_splits/livecodebenchv6_42_v0.2_t0.2.json`).
6. **Exit criterion**: pass@1 within ±0.05 of the paper's constrained LiveCodeBench number (paper reports 0.61 vs MoA 0.39 in-distribution). If the bundled ckpt is a smaller demo run, document its actual config (from `es_log.json`: 60 iters, layer 26, 7-worker pool) and report its number as the baseline.

**Estimated cost**: ~$50-150 Bedrock (eval set × 7 workers × multi-turn). No training compute. Coordinator on 1 small GPU node for the eval duration.

### Phase 0.5: Tiny-scale CMA-ES smoke (Day 3, BLOCKING for Phase 1)

**Goal**: Before committing the $1K-5K full run, validate the *training* path at 2-5% of budget. Carryover from cost-aware-routing, whose hardest lesson was a $200 spot-reclaim loss and whose iter-0 diagnostic gate ("brand bias > 25% → halt") cost ~$5 but prevented wasting $1,500. Phase 0 only tests the *eval* path (bundled checkpoint); this tests the *training* path.

Run **3 CMA-ES iters** at the full config (λ=32, mCMA=16, same worker pool + reward) — i.e. ~3 × 32 × 16 ≈ 1,500 candidate-evaluations. Budget ~$150-250.

**Exit criteria (all must pass before Phase 1):**
1. **Worker non-degeneracy**: all 7 workers selected ≥once across the 3 iters (not collapsed to one — the exact failure mode our GRPO router hit).
2. **Role non-degeneracy**: Thinker, Worker, AND Verifier all used; verifier triggers at least one early halt.
3. **Throttle survival**: λ=32 × mCMA=16 = 512 candidates/iter is a bursty Bedrock load — confirm the Converse client's cross-region rotation + backoff handles it with zero dropped episodes.
4. **Question-conditioning signal**: worker-selection entropy > 1.5 nats AND different question types yield different worker distributions (the diagnostic our shared-policy GRPO collapse failed). If entropy ≈ 0, the linear head isn't learning — halt and run the separability probe (OQ2) before spending more.

If any criterion fails, halt before the full run. Cheap insurance against the expensive phase.

### Phase 1: Full CMA-ES reproduction (Week 1-2, conditional on Phase 0.5)

**Goal**: Train a Trinity coordinator from scratch with CMA-ES on the Bedrock pool, reproduce the evolution curve.

**Prerequisites**: Phase 0 eval runs clean, the bundled checkpoint's number is understood, AND Phase 0.5 smoke passes all 4 non-degeneracy/throttle/conditioning criteria.

**Config** (from `es_log.json`, adapted):
- Coordinator: Qwen3-0.6B, SVD layer 26, linear head, output dim = 7 workers + 3 roles = 10.
- CMA-ES: σ₀=0.03, λ=32, mCMA=16 replications, 60 iters (paper's full regime is larger; start at 60 to match the bundled run, extend if the curve hasn't converged).
- Reward bonuses: diversity 0.15, turn 0.1, cost (set per a cost-aware sweep — this is our extension point; upstream ran cost_bonus_weight=0.0).
- Task: `mix_m_m_r_l` (MATH500 + MMLU + RLPR + LiveCodeBench) per the bundled config, OR LiveCodeBench-only for a focused first run.
- max_turns=5, temperature=0.1, max_tokens=4096.

**Compute**: coordinator + CMA-ES driver on 1 GPU node (the 0.6B model + population eval is light; the bottleneck is Bedrock worker latency, not local GPU). Bedrock spend dominates: 60 iters × λ=32 candidates × 16 replications × (up to 5 turns × worker calls) — this is **thousands of multi-turn episodes**. Budget carefully; see cost note.

**Exit criteria**:
1. CMA-ES validation score (held-out) improves monotonically and beats best-static-worker by ≥3pp (the bar our GRPO router failed to clear — it went *below* best-static).
2. Role usage is non-degenerate (Thinker/Worker/Verifier all used; verifier actually triggers early halts).
3. Worker selection is question-conditioned (entropy > 0; different question types → different worker distributions) — the diagnostic our GRPO collapse failed.

**Estimated cost**: This is the expensive phase. Rough order: 60 × 32 × 16 ≈ 30K candidate-evaluations, each a multi-turn episode of ~2-5 Bedrock calls. At ~$0.01-0.05/episode (depends on worker mix; frontier workers are pricey) → **$1K-5K Bedrock**. Pre-register a hard cost cap and a cheaper-pool fast-follow (drop GPT-class workers for the first full run; add them only for the final eval). Mirror cost-aware-routing's lesson: **smoke at tiny scale (2-3 CMA-ES iters) before committing the full run.**

### Phase 2: Head-to-head vs GRPO negative result (Week 3, optional)

**Goal**: Clean comparison paper — Trinity (CMA-ES) vs Conductor/GRPO (our negative result), same Bedrock pool, same eval.

- Run both routers on the identical worker pool + eval set.
- Reuse the cost-aware-routing GRPO artifacts (`results/runs/grpo_sim_*.json`, smoke runs) for the GRPO arm.
- Headline: "On a heterogeneous Bedrock pool, evolved coordination beats RL coordination, which collapses below best-static. Here's the clean comparison and the formal reason."
- Target: workshop paper extending our `grpo-router-negative-result.md` writeup.

## Components

### Codebase

- **Vendored upstream**: `domains/autoresearch/blueprints/trinity-coordinator/vendor/trinity-upstream/` (the OpenReview code submission, verbatim except large regenerable weights). Do NOT edit in place — treat as upstream reference.
- **Adaptation layer** (new, in blueprint root):
  - `scripts/bedrock_clients.py` — Converse-based replacement for `fugu/llm_clients.py` dispatch.
  - `scripts/worker_pool_bedrock.py` — the 7-worker Bedrock pool config + role-class mapping.
  - `scripts/cost_bedrock.py` — verified Bedrock pricing, replacing `fugu/cost.py` tables.
  - `scripts/run_trinity_agent.py` — agent-runtime entry point (wraps `fugu` train/eval, wires IRSA creds, S3 checkpointing).
- **Fixed (agent must not edit)**: vendored `fugu/algorithms/es.py`, `head_modules.py`, `decompose_model.py`, `core.py` routing loop. We adapt the *clients* and *cost*, not the algorithm.

### Infrastructure (blueprint-local Terraform — required by the agent-runtime contract)

Per `managed-agent-runner.md`, the **IRSA run-role and the artifact bucket are defined in this blueprint**, not in the `agent-runner` CLI repo (which only templates the Job). Provision before Phase 0 (new **Gate 0.0 step 0**):

- `terraform/irsa-run-role.tf` — an IAM role with a trust policy admitting the EKS cluster's OIDC provider + the run's ServiceAccount, scoped to:
  - `bedrock:InvokeModel` + `bedrock:Converse` on exactly the 7 worker model ARNs (+ `us.`/`global.` inference-profile ARNs for Anthropic).
  - `s3:PutObject` / `s3:GetObject` / `s3:ListBucket` on the artifact bucket only.
- `terraform/artifact-bucket.tf` — the private S3 bucket `s3://<bench-bucket>/trinity-coordinator/` with SSE, versioning, and a lifecycle rule (expire raw rollouts after N days, keep `model_iter_*.npy` + `es_log.json`).
- **Gate 0.0 step 0**: `terraform apply` the role + bucket; confirm the EKS OIDC trust resolves and a Job-launched pod can `aws sts get-caller-identity` into the run-role and write a smoke object to the bucket. `run_trinity_agent.py` assumes these exist — it does not create them.

### Agent-runtime execution (the `agent-runner` contract)

The evolution run launches as a detached EKS Job via the `fe agent` thin wrapper over `agent-runner` (per `managed-agent-runner.md`): author + commit this spec, `fe agent run domains/autoresearch/specs/trinity-coordinator.md`, detach, reattach for a visual status report. `run_trinity_agent.py` is the in-Job entry point — it consumes the IRSA-mounted creds (no long-lived keys), pins the git commit, runs `fugu` train/eval, and streams checkpoints+rollouts to S3 every iter. The Job has no microVM session ceiling (a full CMA-ES run can exceed 8h). Coordinator GPU node is requested by the Job's container profile (g6e-class, see Compute).

### Compute

- **Coordinator + CMA-ES driver**: 1 GPU node (g6e.xlarge / g6e.2xlarge — 0.6B model is tiny; CPU-heavy CMA-ES + light GPU forward). Runs inside the agent-runtime EKS Job.
- **Workers**: Bedrock API (no GPU on our side).
- **Bedrock regions**: us-east-1 primary (best model availability); cross-region rotation for throttle headroom (reuse cost-aware-routing's pattern).

### Storage

- `s3://<bench-bucket>/trinity-coordinator/` — CMA-ES solver pickle, `model_iter_*.npy`, `es_log.json`, per-iteration rollouts, verified-prices snapshot.
- Checkpoint every CMA-ES iteration (small: <20K params = ~160KB/ckpt; sync every iter — the cost-aware-routing spot-reclaim lesson: cheap state, sync often). **Always checkpoint at iter 0** (cost-aware-routing skipped iter-0 ckpt and lost the entire first run to a mid-flight reclaim with nothing banked).
- **Rollouts sync to S3 after EVERY CMA-ES iteration**, not buffered to end-of-run. Per-iteration rollouts (worker picks, role usage, hidden states, per-question routing) are the **primary data for post-hoc analysis** (worker entropy, role usage, question-conditioning diagnostics) — losing them to a spot reclaim or NVMe wipe wastes the run even if the final model survives.
- **Artifact durability before teardown (steering `tech-stack.md` §artifact-durability, 2026-06-24)**: before ANY node scale-to-0 / terminate, enumerate the full results dir, tar + pull + verify locally, and confirm S3 + git have the keep-set. The agent-runtime Job + per-iter S3 sync makes this mostly automatic, but the final exfiltration check is mandatory.

## Known risks / lessons carried over

- **Spot reclaim** (`feedback_grpo_resume_state.md`, and the cost-aware-routing 2026-05-26 reclaim that cost ~$200): checkpoint every iteration, sync to S3 every iteration, always ckpt at iter 0. CMA-ES state is tiny so this is free. The agent-runtime Job + S3 persistence makes resume clean.
- **GPU spot capacity scarcity** (cost-aware-routing, 2026-05-26): p5/p4d spot was globally unavailable for hours. Trinity needs only a *small* GPU (0.6B coordinator) — g6e-class, which had ample spot capacity even during the p5 squeeze ($1.3-2.2/hr). This is a major de-risking vs the cost-aware-routing setup.
- **Bedrock model-ID drift** (cost-aware-routing Gate 0.0): GLM-5 was `zai.glm-5` not `zai.glm5`; Llama 4 needed `us.` inference-profile prefix; Nova needs Converse not raw invoke_model. Re-verify every worker ID with a 1-token Converse ping before any run.
- **Bedrock cost is the dominant expense** (not GPU). Phase 1's 30K candidate-evaluations × multi-turn × frontier workers can run to thousands of dollars. Smoke tiny, pre-register a cost cap, and consider a cheap-pool-first run.
- **The reward landscape lesson stands**: our negative result is *why* this experiment uses CMA-ES. If Trinity also fails to beat best-static on the Bedrock pool, that's a stronger negative result (the architecture, not just the optimizer, doesn't transfer). Pre-register best-static as the bar.

## Non-requirements

- **Reproducing Fugu the product** — out of scope and impossible (closed, unreleased, no architecture spec). We reproduce Trinity, the published core.
- **Reproducing Conductor** — already effectively done (our GRPO negative result). Phase 2 reuses those artifacts for comparison; we don't re-run Conductor from scratch.
- **Self-hosting the open workers on GPU** — deliberately replaced with Bedrock. The whole point of this adaptation is single-account Bedrock + agent-runtime, not a GPU serving fleet.
- **Beating the paper's full-power 0.862 LiveCodeBench number** — that used GPT-5 + Gemini-2.5-pro + Claude-4-Sonnet with constraints removed. Our Bedrock pool substitutes models (Opus 4.8 / Sonnet 4.6 / Nova Premier frontier tier; GPT-5.5 only if access-granted). We target reproducing the *method and relative gains* (coordinator beats best-static and MoA), not the exact SOTA number — a different frontier mix will shift the absolute pass@1.

## Open questions

1. **Is the bundled `model_iter_60.npy` the paper's headline model or a demo run?** The 60-iter config suggests a real-but-modest run. Phase 0 establishes its actual number as our baseline.
2. **Does Qwen3-0.6B's hidden state separate our Bedrock-pool question types** as cleanly as the paper's SVM analysis showed for theirs? Run the separability probe (linear SVM on penultimate-token hidden states) before training — cheap signal on whether the linear head can work.
3. **Cost-aware extension**: upstream ran `cost_bonus_weight=0.0`. Our cost-aware-routing program is *about* the cost axis. After reproducing the baseline, sweep `cost_bonus_weight` — this is the novel contribution that connects Trinity to our cost-aware-routing line.
4. **Reasoning vs direct Qwen3-32B as two pool slots**: upstream treats them as distinct workers (same weights, different inference mode). Confirm Bedrock Converse can express both modes for `qwen.qwen3-32b-v1:0` via payload flags.

## Relationship to other specs

- **`grpo-router-negative-result.md`**: the negative result this experiment answers. Trinity is the recommended alternative made concrete. Phase 2 is the head-to-head.
- **`cost-aware-routing` blueprint**: source of reusable Bedrock Converse client patterns (`worker_proxy.py`), verified-pricing tooling, Gate 0.0/0.2 procedures, and spot/checkpoint lessons. The cost-aware extension (OQ3) merges the two lines.
- **`managed-agent-runner.md`**: the execution substrate. Trinity's long evolution run is the kind of detached, credential-scoped, resumable job that runtime exists for.
