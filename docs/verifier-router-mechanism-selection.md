# Verifier / Router Mechanism Selection — a cost-function decision framework

**Status**: reference · **Date**: 2026-06-29
**Audience**: any spec that involves a verifier, a model router, a cascade, or test-time selection.
**One-line**: choosing *whether and how* to verify/route is an operations-research problem (model cascade / learning-to-defer); this doc gives the factors and the decision tree, grounded in our own experiments + the RLVR verifier catalog.

> **Specs should @-reference this** before designing a verifier or router. It tells you which mechanism (if any) to invest in for a given (pool, task, budget) — so you don't train a CMA-ES router for a problem a single cheap model already covers, or bolt on a frontier verifier that costs more than it saves.

---

## The reframe: it's a cost-minimization problem, not a modeling problem

The instinct is to reach for a mechanism ("train a router", "build a verifier"). The right first move is to **measure the regime** and let the regime pick the mechanism. Formally this is **learning-to-defer / model cascades** — the published instance is **FrugalGPT** (Chen et al., 2023): minimize expected \$ subject to an accuracy floor, over a deferral policy. Viola-Jones detector cascades are the classic ancestor. You are almost never inventing; you are locating yourself on a known frontier.

## The factors (the cost function's coefficients)

Four factors decide the mechanism. The first two collapse into one measurable quantity.

1. **Routing headroom** = `oracle − best_static`, where `oracle` = "solved if *any* pool model solves it" and `best_static` = the single best model. This is the *interaction* of two things people list separately:
   - **problem capability-variance** (do problems differentiate models?) and
   - **pool capability-spread** (do models differ on which they solve?).
   Either at zero ⇒ headroom ≈ 0 ⇒ **routing cannot help**, regardless of the other. Don't track them separately — measure the product directly as headroom.
2. **Cost-per-intelligence spread** = the \$ range across models at a given accuracy. Large spread ⇒ big prize for *downshifting* to the cheapest adequate model (no routing needed).
3. **Verification cost** = \$ to judge one candidate. This is the factor that most often kills a cascade: if verify-cost > downshift-saving, the cascade loses on cost even when it wins on accuracy.
4. **Verifier portability** = does a *learned* verifier transfer across models/domains, or must it be retrained continuously? Varies sharply by verifier *type* (see catalog below) and sets the amortization term.

**The cheap diagnostic that measures 1–3 at once:** a per-model **solo pass matrix** on a fixed problem set (each model answers each problem; grade; record cost). ~\$5–10. It yields headroom, cost-spread, and per-model cost-per-intelligence *without building any router*. Run it first. (Reference impl: `domains/autoresearch/blueprints/trinity-coordinator/scripts/differentiation_probe.py`.)

## The decision tree

```
run the solo pass matrix (cheap)
│
├─ headroom ≈ 0  (cheap model already covers ~90% of cases)
│     → SHIP the best cost-per-intelligence SINGLE model. No router, no verifier.
│       (This is the most common real-world outcome.)
│
├─ headroom > 0  AND  verify_cost < downshift_saving
│     → VERIFIER-GATED CASCADE (reactive, no training):
│       cheap solve → verifier ACCEPT/REJECT → escalate on reject.
│       The verifier is the difficulty oracle; it discovers hard problems
│       reactively, so you don't need a predictive router.
│       LEVER: the verifier must be CHEAP or this collapses (see factor 3).
│
└─ headroom > 0  AND  (verify expensive  OR  must route upfront / latency-bound
                       OR  volume high enough to amortize training)
      → TRAIN A PREDICTIVE ROUTER (e.g. CMA-ES head reading hidden state):
        route before spending, skipping the try-then-verify tax.
        Only worth it when the reactive cascade's verify-tax exceeds training+serving cost.
```

---

## Grounding: our three experiments are a worked example

**Trinity cost-aware routing** (LiveCodeBench, 8-model Bedrock pool, n=40 clean harness):
- headroom present but modest: oracle 0.975, best-static 0.825 → **+15pt theoretical**.
- cost-spread **huge**: deepseek-v3 / qwen3-235b / Opus **all tie at 0.825**, but qwen3-235b costs **\$0.00063/prob vs Opus \$0.01278 — 20× cheaper, same accuracy**.
- verifier-gated cascade reached **0.900 (+7.5pt)** — escalation genuinely rescues failures — **but cost \$0.0178/prob, *more* than always-Opus (\$0.0128)** because the verifier was Opus judging every solve. **Routing lifted accuracy but lost on cost; not Pareto-dominant with a frontier verifier.**
- **Decision-tree output: case 1/2.** With a compressed strong tier, the honest answer was "ship qwen3-235b" (case 1); the cascade is only worth it with a *cheaper* verifier (case 2 lever). A trained CMA-ES head (case 3) was **not** warranted — and an attempt to train one was void anyway (grading bug). The framework predicted this.

**Learned-verifier RF** (`domains/autoresearch/blueprints/learned-verifier`): the cheap verifier ($0.004/call, 3 behavioral features) — but **cross-model AUC 0.363 (worse than random)**, both directions. "Features transfer, thresholds don't." → factor 4 in action: a *discriminative learned* verifier is cheap but **non-portable**; it needs continuous per-policy retraining. The retraining cost is a real term in the cost function.

**GRPO router negative result** (`grpo-router-negative-result.md`): a single-policy RL router *collapsed below* best-static on the multi-modal cost-aware reward — the cautionary tale for reaching for case 3 when the regime doesn't justify it.

---

## Verifier portability by TYPE (from the RLVR Verifier Catalog)

Factor 4 is not one number — it depends on *which kind* of verifier. RLVR training runs are "verifier factories" (the ORM, PPO critic, PRM, anti-hack detector are byproducts you can lift to inference). Ordered **most → least portable** (so: most reusable across models/domains → most needs continuous retraining):

| Verifier type | Portability | Cost | Notes for mechanism selection |
|---|---|---|---|
| **Deterministic checker / unit test** (ORM) | perfect (policy-agnostic) | ~0 if a checker exists | Use whenever a checker exists. Most hackable, narrowest scope. |
| **Symbolic** (Z3/Isabelle/FOL) | best OOD | solver cost | Narrow applicability (needs formalizable claims). |
| **PRM** (process reward model) | transfers across **domains** (math→code +4%) | 1 classifier call | More portable than a critic; a candidate reusable cheap verifier. |
| **Generative** (LLM-as-judge / GenRM) | OOD-better than scalar, but skews code/math | 1+ LLM call (CoT) | What our Trinity cascade used (Opus). Portable but the **cost** is the problem. |
| **Anti-hack / legitimacy** (behavioral) | environment-bound, no cross-env evidence | rule-filter→judge cascade | Verifies *process legitimacy* not correctness. Strong same-env. |
| **Critic / value model** (PPO) | **policy-bound — not portable at all** | ~2S× decode | Highest-leverage *and* least-shared artifact; value-guided MCTS (PPO-MCTS). |

**Two load-bearing reads:**
- **Reuse difficulty tracks policy-specificity.** The most *valuable* artifacts (critic, anti-hack) are the *least* portable — which is exactly why "continuous retraining against the live policy is the moat, not the artifact." This *is* our RF finding (AUC 0.363→0.801 per-model) generalized.
- **PRMs are the surprising portable option**: math-trained PRMs match/beat code-trained PRMs on code, latching onto shared reasoning patterns (self-correction). If you need a cheap *and* somewhat-portable verifier for a cascade, a PRM beats a discriminative behavioral RF on the portability axis.

Full catalog (mechanism, train-time origin, test-time reuse recipe, calibration gotchas, decay caveats per type): obsidian `03_Resources/LLM-Optimization/Reinforcement-Learning/RLVR-Verifier-Catalog-Test-Time-Reuse.md`. Taxonomy backbone: *Trust but Verify* (arXiv:2508.16665).

---

## Where this sits vs the AI-gateway / semantic-router landscape

AI gateways are converging from **heuristic governance rules → learned routing**. The exemplar is **vLLM Semantic Router** (an Envoy `ext_proc` control plane): a small fine-tuned classifier + embeddings map a request's **intent / domain / complexity** to a model tier or reasoning-on/off, with policy plugins (PII, jailbreak, semantic cache, hallucination detection) and a "token economy layer." It is a real, shipped, latency-conscious deployment substrate with governance baked in — the operational layer this research does **not** build.

The distinction is **layer, not rivalry** — they are complementary:

- **Gateways provide the *mechanism* (and the substrate); this framework provides the *decision*.** Semantic Router gives you an intent classifier + policy rules and a place to run them. It does not tell you *whether routing pays off for your (pool, task, budget)* — which is exactly what the solo-pass-matrix diagnostic + decision tree above answer. Most regimes land in case 1/2 (ship the cheapest-adequate model, or a reactive cascade), where an upfront intent router is more machinery than the regime needs.

- **Intent-routing vs capability-routing.** Gateways route on *what the request is about* (intent/domain → "code goes to the code model") — a **proxy for difficulty**, computed upfront. Our experiments measured the thing the proxy stands in for: *which model actually solves it* (measured pass-rate; reactive verifier-gate or hidden-state head). When cheap models match frontier (our result: qwen3-235b ties Opus at 1/20th cost), the proxy is unnecessary — you don't classify intent to route, you just use the cheap model. Intent-routing is solving a *harder* problem (predict difficulty upfront) than most regimes require.

- **Upfront vs reactive — and when each wins.** An intent classifier is **predictive/upfront** (cheaper at inference, no try-then-verify tax) = the framework's **case 3** tool (latency-bound / high-volume). A verifier-gated cascade is **reactive** (discovers difficulty instead of predicting it) = **case 2**, no training. The gateway's strength is precisely case 3; this framework's contribution is telling you when you're *not* in case 3.

- **The shared blind spot: portability.** A gateway's intent classifier is **per-pool / per-domain** — retrain it when the model lineup or task distribution shifts (the same pool-lock as a positional CMA-ES head; the same non-transfer as the learned-RF AUC 0.363). Semantic Router's docs don't cover adding a model or transferring across pools. The **content-addressed router** (`domains/autoresearch/specs/content-addressed-router.md`) is the research answer to *their* unstated gap too: route by model *descriptors* (price, capability, context) so a new model is a descriptor row, not a retrain.

**Net:** the contribution here is not "build another router" — gateways already ship good ones. It is the **decision theory** for what (if anything) to put in the gateway: a cheap diagnostic that says which regime you're in, the honest finding that *cost-per-intelligence is collapsing the need for routing at all*, and the *portable* router form for the case-3 minority where routing is warranted. Deploy the diagnostic everywhere; deploy a router only where it earns its keep; make it content-addressed so it isn't pool-locked.

---

## Checklist for a verifier/router spec

1. **Measure first.** Run the solo pass matrix → headroom, cost-spread, per-model cost. Don't design before this.
2. **Take the decision-tree branch.** headroom≈0 → single model; cheap-verify → cascade; expensive/upfront/high-volume → predictive router.
3. **If a cascade: budget the verifier explicitly.** verify_cost vs downshift_saving decides Pareto-dominance. Prefer the cheapest verifier *type* that clears the accuracy bar (deterministic > PRM > small-LLM-judge > frontier-judge).
4. **If a learned verifier/router: price the retraining.** Discriminative learned verifiers (behavioral RF, critic) don't transfer — continuous per-policy retraining is a cost-function term, not a one-time build.
5. **Reuse RL byproducts where they exist.** If an RLVR/PPO run is in the pipeline, its critic/PRM/anti-hack detector are calibrated verifiers you already paid for — lift them (best-of-N rerank, value-guided decode, runtime gate) instead of training fresh.

## Related
- `domains/autoresearch/blueprints/learned-verifier/VERIFICATION_FRAMEWORK.md` — the *primitives + implementation playbook* (complements this doc's *mechanism-selection*).
- `domains/autoresearch/blueprints/learned-verifier/VERIFIER_ECONOMICS.md` — verify-method cost table (test-exec vs SVG vs learned vs judge).
- `domains/autoresearch/blueprints/trinity-coordinator/` — the worked cost-aware-routing example (probe, clean cascade, lessons #31–33).
- `domains/autoresearch/specs/content-addressed-router.md` — the portable (descriptor-conditioned) router bet; the case-3 "reusable router" form.
- vLLM Semantic Router (`vllm-semantic-router.com`) — exemplar AI-gateway / intent-router; the deployment substrate this framework's decision theory sits above (see landscape section).
- `docs/verification-program-retrospective.md` — cross-program verification audit.
- obsidian `RLVR-Verifier-Catalog-Test-Time-Reuse.md` — verifier types as RL byproducts + reuse recipes.
