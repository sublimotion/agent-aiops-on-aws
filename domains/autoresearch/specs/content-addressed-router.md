# Autoresearch Spec: Content-Addressed Router — a pool-agnostic learned router

## Status: DRAFT (2026-06-29)

## Overview

Trinity's CMA-ES head is **positional**: its output layer is `hidden → (L agents + 3 roles)`, and output neuron *i* is bound to whatever model sits at `llm_names[i]`. It transfers across *tasks* (it reads a frozen-backbone hidden state, not model IDs) but **not across pools** — add or swap a model and the head geometry changes, forcing a retrain (Trinity OQ5, lessons #29). That positional binding is the single thing standing between "a router for this pool" and "a router for any pool."

This experiment tests the fix: a **content-addressed router** that scores each model by its *properties* instead of a bare output position. Then adding a model = adding its description (no architecture change), and one trained router can score *any* pool by reading model cards. Price lives in the description, so cost-aware routing is native. The architecture is a **two-tower / dual-encoder** (§2): `score(request, model) = ⟨f(hidden_state), g(model)⟩`, with `g` a learned model-embedding (capturing latent domain competence) over grounded descriptor anchors. The hand-descriptor head and the semantic-router intent classifier are each a degenerate one-frozen-tower case of this — see §2.

**This is the only "train once, apply broadly" router bet on the board.** Per `docs/verifier-router-mechanism-selection.md`, a router is a **case-3** tool (headroom > 0 AND verify-expensive/upfront/high-volume); this spec does NOT claim routers are always worth it — it claims that *when* a router is warranted, a content-addressed one is the reusable form, dissolving the pool-lock that makes the positional head a per-pool rebuild.

### Why this experiment, why now

- We have the full prior arc: positional Trinity head (pool-locked), the clean cascade result (qwen3-235b ties Opus at 1/20th cost; routing lifts +7.5pt but cost-loses on a frontier verifier), the learned-RF non-generalization (AUC 0.363 cross-model), and the mechanism-selection framework. The open lever is whether *learned* routing can be made pool-portable at all.
- It directly answers Trinity OQ6 and the user question "train a router and apply it everywhere" — the honest answer is "only a content-addressed one could, and only where headroom justifies it."

## Findings being solidified (carried in)

- **Routing headroom = oracle − best-static** is the gate; measure with the solo pass-matrix before any training (`differentiation_probe.py`). On LiveCodeBench/8-pool it was +15pt theoretical, but the strong tier was compressed → modest realizable headroom.
- **Positional heads are pool-locked** (Trinity #29); **discriminative learned verifiers/routers don't transfer** (RF AUC 0.363→0.801 per-model). The hypothesis here is that *descriptor-conditioning* breaks that lock because the head learns model *properties*, not positions.
- **Grading must be model-fair** (Trinity #31): the core grader scored non-`<answer>` models 0.0 and biased training. Any training here uses the fixed extraction / the clean standalone harness, never core.py's grading.

## Research Questions

1. **Pool transfer (the headline):** does a two-tower router trained on pool A route a *held-out* pool B (models unseen in training, only their descriptors given → embedding via the descriptor encoder) **above best-static-on-B AND above the semantic-router intent-classifier baseline**, where a positional head would be undefined? Target: ≥ best-static-B, beat the one-tower intent classifier, ideally approach the per-pool-trained ceiling. (Beating best-static proves routing transfers; beating the intent classifier proves the *second learned tower* earns its cost over the gateway's one-tower form.)
1b. **Does the learned model-embedding capture latent domain competence?** Inspect the embedding space (§2a collapse gate): do domain-specialist models separate into distinct regions, and does a model strong-but-cheap at a domain get routed there? This is the user's core intuition — "encode domain-specific model for routing" — as a measurable claim, not just an architecture.
2. **Add-a-model AND swap-a-model for free:** (a) *append* a new model via its descriptor (no retrain) — does the router use it when its descriptor dominates for a problem type? (b) *swap* a model at an existing slot for a different one (new descriptor) — does routing recalibrate correctly? Trinity #29's nuance: a positional head's swap leaves logits transferred but the policy *miscalibrated*; the content-addressed head should fix the swap case too (it reads properties, not positions). Both compared to warm-start-retrain.
3. **Cost-awareness via descriptor:** with price in the descriptor, does sweeping a budget term shift routing toward cheaper-adequate models *without retraining* (the cost knob is read, not learned per-pool)?
4. **Does it beat the reactive cascade's ROI?** Even if it transfers, is the trained content-addressed router worth more than a no-training verifier-gated cascade with a cheap (PRM/small-judge) verifier? (The framework's case-2-vs-case-3 question, made concrete.)

## Components

### 1. Compute
- 1 small GPU (g6e/L40S class) — the router backbone is Qwen3-0.6B-scale; bottleneck is Bedrock worker latency, not local GPU (same as Trinity). Workers via Bedrock Converse.
- Budget: training is CMA-ES over a small vector (gradient-free, tiny params); dominant cost is Bedrock episodes. Pre-register a hard cost cap; smoke before full run (Trinity lessons).

### 2. Architecture — a TWO-TOWER (dual-encoder) router

**Unifying frame:** routing is a **retrieval problem over models**. Embed the request and each model into one shared space; route to the best-scored model:

```
score(request, model) = ⟨ f(h_request) , g(model) ⟩          (dual-encoder / two-tower)
```

This **generalizes** the whole landscape — semantic router and the hand-descriptor head are each a *degenerate case where one tower is frozen*:

| design | request tower `f` | model tower `g` | limitation |
|---|---|---|---|
| semantic router | learned intent classifier | **frozen** intent→model lookup table | model side doesn't learn; per-pool/per-domain retrain of the table |
| hand-descriptor head (this spec, v1) | frozen hidden state `h` | **fixed** descriptor `d_m` (hand features) | model side can't capture *latent* domain competence beyond the chosen features |
| **two-tower (this spec, v2)** | `f(h)` (learned projection of hidden state) | `g(model)` **learned** model embedding (+ grounded descriptor anchors) | the generalization — both sides learned in one space |

- **Backbone:** reuse Trinity's frozen Qwen3-0.6B + layer-26 hidden state `h`; `f` is a small learned projection of `h` into the shared space.
- **Model tower `g(model)` = learned model-embedding ⊕ grounded descriptor.** The learned embedding is what captures **latent domain competence** — a model secretly strong at SQL gets pulled toward the SQL region of request-space *through training*, even if its price/size descriptor looks unremarkable (this is the user's "encode domain-specific model for routing" intuition, made architectural). It is **concatenated with the absolute grounded descriptor** below so cost/context stay anchored to real, transferable values rather than free-floating.
- **Grounded descriptor (the anchor half of `g`):** `d_m = [norm($/Mtok_in), norm($/Mtok_out), capability_emb(m), norm(ctx_window), reasoning_flag]`. **All descriptor values MUST be absolute and externally grounded — never pool-relative ranks** (pool-relative features silently recreate the learned-verifier AUC-0.363 "thresholds don't transfer" failure).
  - **Price** = a committed, dated, manually-**verified price snapshot** (mirror cost-aware-routing's `results/verified-prices-*.json` → frozen `configs/pool.yaml`; read from the file, NEVER a live Pricing API). The live catalog is untrustworthy (Trinity Gate 0.0: `ListFoundationModels` IAM-denied, Nova Premier Legacy, DeepSeek-R1 needs `us.` profile). A Gate-0.0 liveness+price probe runs before any run; a price drift between snapshot and run is a fail-closed condition, not a silent test-time shift.
  - **`capability_emb` options to ablate:** (a) per-model pass-rate + avg cost measured on a **FROZEN, model-independent REFERENCE benchmark committed before training** — identical problems for every model in both pool A and pool B (NOT pool A's train split; a pass-rate measured on A's distribution is pool-relative and won't transfer — this is the option-(a) trap); (b) a learned embedding indexed by model family; (c) a text-encoder embedding of the model card. Start with (a) **bound to the frozen reference set**.
  - **Reference-set disjointness (anti-leakage, BLOCKING):** the frozen reference benchmark MUST be **disjoint** from the pool-B transfer eval set, with its own committed problem-id list and stated size/n. If a pool-B model's `capability_emb` were measured on problems later used in the B eval, the descriptor would encode B's answers and inflate the transfer result. Reference set and every arm's eval set are committed, dated, non-overlapping id lists.
- **Scoring:** `score(m | h) = ⟨f(h), g(m)⟩` — inner product (or small bilinear form) of the two towers in the shared space. `f` (request projection) and `g`'s learned-embedding component are **shared across all models** (this is what makes it pool-agnostic). Routing = softmax over `{score(m|h) : m ∈ pool}`. Role logits handled as in Trinity (3 extra outputs, fixed).
- **Trainable params:** `f` + the per-model learned embedding (a `|train_pool| × dim` table for seen models; the `descriptor→embedding` map for unseen ones — see below) + the role head. The **unseen-model path is the load-bearing design choice:** a per-model embedding row only exists for models in the train pool, so for held-out pool B, `g` must be computable **from the descriptor alone** — i.e. train a small `descriptor → embedding` encoder so any model (seen or unseen) gets an embedding from its grounded features. Without this, the learned embedding is just a positional table by another name (back to pool-lock). The learned-embedding table for seen models is a *training-time* refinement; the deployable `g` is the descriptor-encoder.
- **Optimizer = CMA-ES (gradient-free), primary.** Although `f`/`g` are differentiable, **the reward is NOT**: routing is a discrete argmax/softmax-sample over models, then Bedrock grading → pass@1. There is no differentiable loss w.r.t. pass-rate (same reason Trinity/`fugu` is gradient-free end-to-end). A differentiable *score* ≠ a differentiable *objective*. The GRPO-router negative result (a single-policy RL router collapsed *below* best-static on exactly this reward shape) is the cautionary precedent for reaching for gradients here. If a gradient variant is ablated, it MUST specify the estimator (REINFORCE / Gumbel-softmax relaxation) and be pre-registered against the GRPO-collapse risk — not hand-waved as "the path is differentiable."

### 2a. PRECONDITION — the two-tower router only pays off on a DOMAIN-SPECIALIZED pool

This is the load-bearing condition, and our own data is the warning. The shared embedding space is only worth learning when **models specialize by request region** — model A wins SQL, B wins algorithms, C wins long-context — so the model-embedding spreads out and `⟨f(h), g(m)⟩` has real structure to exploit. **On a capability-tied pool the space collapses:** on LiveCodeBench our strong tier *tied* at 0.825 (deepseek-v3 / qwen3-235b / Opus), so every strong model would map to nearly the same point — a low-rank, degenerate embedding the expensive two-tower router learns nothing a "pick the cheapest of the tied cluster" rule doesn't already give. Capability-variance ≠ domain-specialization: a ladder of "same skill, different strength" is the wrong regime; a set of "different skills" is the right one.

- **Embedding-collapse gate (BLOCKING, quantified):** after training, measure the learned model embeddings' **effective rank** (participation ratio of the singular-value spectrum) and **mean pairwise cosine**. Gate: **effective rank ≥ 0.6 · #models AND mean pairwise cosine < 0.5**. Fail either → the pool isn't domain-specialized, the embedding is degenerate → fall back to the framework's case-1/2 answer (cheapest-adequate model or reactive cascade); do NOT ship the two-tower router. (Thresholds are the pre-registered starting point; refine once, before the full run, on the smoke — not post-hoc.) This is the architectural analog of the headroom diagnostic: it says the *expensive* router was unnecessary *before* you ship it.
- **Encoder-extrapolation check (BLOCKING — the AUC-0.363 failure, one level up):** the `descriptor → embedding` encoder is trained on only a handful of pool-A models, so it can **overfit pool A's descriptor distribution** even with absolute grounded features — the same non-transfer that sank the learned-RF, now in the encoder. Mitigations, all required: (a) **leave-one-model-out CV on pool A** — hold out each A model, predict its embedding from descriptor, check routing quality survives; (b) a **minimum train-pool size** justified by that LOO curve (don't train a descriptor-encoder on 3 models and claim it extrapolates); (c) report whether pool-B descriptors fall **inside** pool-A's descriptor convex hull — extrapolation beyond it is flagged, not silently trusted. If the encoder fails LOO on A, it cannot transfer to B; stop before the B eval.
- **Pool requirement:** the train + transfer pools MUST be assembled to be **domain-specialized** (a code-specialist, a math-specialist, a long-context model, a cheap generalist, etc.), evaluated on a **multi-domain task mix** — not a single-domain ladder. A single-domain easy task (our LiveCodeBench setup) is the explicitly-wrong regime for this experiment and would reproduce the saturation/collapse finding.

### 3. Protocol
- **Pool A (train):** a **domain-specialized** pool (§2a) — distinct *skills*, not a capability ladder: e.g. a code-specialist, a math-specialist, a long-context model, a cheap generalist, a reasoning model. Must also span cost (so the cost anchor is learnable), but specialization-by-domain is the primary requirement. A multi-domain **task mix** (code + math + long-context + general) is mandatory — a single-domain task makes the model-embedding collapse (our LiveCodeBench finding).
- **Pool B (held-out transfer):** ≥2 *domain-specialist* models NOT in A, only descriptors provided at test. The transfer test.
- **Grading:** clean standalone harness (`clean_router_eval.py` lineage), model-fair extraction (fenced-block fallback, never core.py's tag-only grader — Trinity #31). The descriptor signal is only as honest as the grader; a tag-biased grader would favor `<answer>`-emitting models and corrupt every descriptor.
- **Shared-split discipline (REQUIRED for the A→B headline, not just asserted):** ALL arms — the pool-A-trained transfer head, the pool-B-ceiling head (per-pool-trained upper bound), the semantic-router intent-classifier baseline, best-static-B, the cascade, random, oracle — are evaluated on the **identical fixed pool-B problem set, fixed seed, identical model-fair extraction**, and report **n**. Pin the seed; commit the B problem-id list. Without this the transfer claim is uninterpretable (carryover: cost-aware-routing shared-split + Trinity #33 n=40/seed-42).
- **Baselines:** best-static-on-B (single-solve), **semantic-router-style intent classifier**, verifier-gated cascade on B (cheap verifier — PRM or small-judge), random routing, oracle. pass@1 + measured \$/prob each. The two-tower head must beat the intent-classifier baseline to justify the second learned tower — otherwise the gateway's cheaper one-tower router is sufficient.
  - **Intent-classifier arm spec (it is *learned*, so it gets the same discipline):** a small classifier (embedding + linear head, or a fine-tuned small encoder) maps a request → one of a **pre-committed intent taxonomy** (the domain labels of the task mix: code / math / long-context / general / …). The intent→model table is fit on the **pool-A train split only** (frozen at test, mirroring how a gateway is configured per pool); for pool B the table is re-fit on a held-out A-disjoint calibration slice (a gateway *does* retrain its table per pool — that per-pool retrain dependency is exactly the limitation the two-tower head claims to remove, so the baseline must be given its fair per-pool fit). Commit the taxonomy, the classifier train ids, and the table-fit ids — all disjoint from the shared pool-B eval set.
- **Diagnostic first:** run the solo pass-matrix on both pools to record headroom; if pool B headroom ≈ 0, the transfer question is moot (single model wins) — document and stop (the framework's case-1 exit).
- **Run infrastructure (inherit Trinity's hard-won, by name — these are NOT optional):** per-PID cost-aggregation sink (`CAR_TRINITY_TELEMETRY_DIR`) or the cost cap is **blind** under spawn isolation (#20/#22: `spend=$0.00` while workers spent real money); `sitecustomize` monkeypatch install so spawned Pool workers resolve the Bedrock dispatch AND the new descriptor-featurizer/head import path (#8: a `__main__`-only patch never reaches spawn workers → silent 0%-GPU stall); iter-0 S3 durability assertion + per-iter resume checkpoint (spot-reclaim safety); `max_tokens=8192` + empty-reasoning-response fallback (#19); per-worker home-region routing (#57); kill-main-then-spawn-workers on stop (#30).

### 4. Storage
- S3 per-iter checkpoint + es_state (Trinity resume pattern), descriptors + pass-matrices committed, raw rollouts retained. Artifact-durability gate before teardown.

## Success Criteria
1. **Transfer:** two-tower router on held-out pool B beats best-static-B by a margin a positional head *cannot even attempt* (it's undefined on B), AND beats the semantic-router one-tower intent classifier (justifying the second learned tower). Even matching best-static-B while a positional head requires a full retrain is a win for the *mechanism*; failing to beat the one-tower classifier means the gateway's cheaper form suffices (a valid negative result).
2. **Add-a-model:** routing to a descriptor-only new model is within ε of warm-start-retrain routing.
3. **Cost knob reads, not retrains:** budget sweep shifts \$/prob monotonically on a *fixed* trained head across pools.
4. **Honest ROI verdict:** explicit comparison vs the no-training cheap-verifier cascade — state which wins on accuracy AND cost, per regime. A negative result ("cascade still wins, content-addressing doesn't pay") is a valid, publishable outcome.

## Non-Requirements
- Not chasing SOTA pass@1 — the point is *pool-portability of a learned router*, measured against the cheaper alternatives.
- Not a positional-head reproduction (that's Trinity; this supersedes its OQ6).
- Not claiming routers are universally worth it — the mechanism-selection framework explicitly bounds this to case-3 regimes.

## Known Limitations
- **Descriptor quality is the experiment's ceiling.** If `capability_emb` doesn't capture what makes a model good *for a problem type*, the head can't route by properties. The pass-matrix-scalar start is grounded but coarse; the text-card embedding is richer but noisier. This is the main risk.
- **Headroom dependence stands.** Content-addressing makes a router *portable*, not *worthwhile* — it still only pays where routing headroom exists. On compressed pools / easy tasks it inherits Trinity's saturation finding.
- **Cross-model capability_emb may itself not transfer** — the RF lesson (thresholds don't transfer) could recur if the descriptor scalars are pool-relative rather than absolute. Use absolute, externally-grounded descriptor values (published price, measured solo pass-rate on a *fixed reference set*), not pool-relative ranks.

## Carryover Audit (spec-design gate)
- **Trinity #29 (positional lock)** — this spec exists to fix it; the success criterion is explicitly "does something a positional head cannot."
- **Trinity #31 (model-fair grading)** — mandated: clean harness only, never core.py grading. A biased grader would silently favor tag-emitting models and corrupt the descriptor signal.
- **Learned-verifier AUC 0.363 (thresholds don't transfer)** — the failure mode to beat; mitigation is absolute/externally-grounded descriptors, and the transfer test IS the check.
- **Mechanism-selection framework** — the diagnostic-first protocol + the case-3 bounding + the cascade-ROI baseline are all required by `docs/verifier-router-mechanism-selection.md`; this spec must not claim a universal router.
- **Cost-aware-routing Gate 0.0 / spot / shared-split lessons** — model-ID drift probe, resume, fixed shared eval split across all arms (carryover-auditor flagged this for any multi-arm comparison).

## Relationship to other specs
- **`domains/autoresearch/specs/trinity-coordinator.md`** — direct parent; this is OQ6 made a spec. Reuses its backbone, harness, lessons.
- **`docs/verifier-router-mechanism-selection.md`** — the decision framework that bounds this; this spec is the case-3 "reusable router" investigation it points to.
- **`domains/autoresearch/blueprints/learned-verifier/`** — the non-transfer prior; the descriptor approach is the hypothesis for breaking it.
