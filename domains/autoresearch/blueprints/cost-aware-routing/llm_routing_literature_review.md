# Literature Review: Cost-Aware LLM Routing & Cascading

## Executive Summary

LLM routing/cascading is a maturing field with production deployments at AWS, OpenRouter, and LiteLLM. The research splits into two camps: **supervised routers** (trained on human preferences, simpler) and **RL-based routers** (optimizing compound reward functions, newer). Cost-aware reward formulation remains under-explored—most work optimizes quality alone, then benchmarks cost reduction as a side effect. **Sakana's Conductor is novel** for explicitly training an RL agent to orchestrate workflows, but its transferability to cost-quality Pareto frontiers is unvalidated. The opportunity: cost-first framing with decomposition.

---

## 1. State-of-the-Art in Supervised LLM Routing

### FrugalGPT (Chen et al., 2023)
**Paper**: Proposed cascading + prompt adaptation for 98% cost reduction vs GPT-4 parity or ~4% quality gain at equivalent cost.

**Mechanism**: Learns which combinations of LLMs to use per query. Three cost levers: prompt adaptation (shorter prompts to cheaper models), LLM approximation (fallback hierarchies), and cascading (simple→cheap, escalate if confidence low).

**Key result**: Up to 98% cost reduction maintaining GPT-4 performance on MMLU/GSM8K.

**Limitation**: Doesn't optimize quality-cost jointly; cost savings are **empirical outcome**, not the training objective.

---

### RouteLLM (LMSYS/Anyscale, 2024)
**Repository**: github.com/lm-sys/RouteLLM (4,935 stars, Apache 2.0)

**Mechanism**: Trains a router classifier on human preference data. Router learns a **threshold-based decision** ("send to GPT-4 if confidence > τ, else use Claude 3.5 Sonnet"). Augments preference data with synthetic examples for better coverage.

**Results**: >2× cost savings on MMLU/MTBench while maintaining quality. Router is a small classification model (fasttext-like), not an LLM.

**Key insight**: Preference-trained routers outperform rule-based heuristics; small model suffices for routing decision.

**Limitation**: Threshold τ is hand-tuned post-training, not jointly optimized with cost. Single binary choice per query (no decomposition).

---

### AutoMix (Madaan et al., NeurIPS 2024)
**Core idea**: Categorizes tasks into {Simple, Complex, Unsolvable} → routes to {cheap LLM, expensive LLM, SLM} accordingly. More nuanced than binary routing.

**Citation count**: 62–70 citations (conference + arXiv versions).

**Limitation**: Still task-category classification, not true cost-quality Pareto optimization.

---

## 2. Uncertainty-Aware & Confidence-Based Routing (2024–2026)

### CP-Router
**Mechanism**: Routes between small LLM and larger LLM based on **estimated prediction uncertainty** (conformal prediction). Measures model confidence for each query.

**Advantage**: Uncertainty signal is more principled than threshold heuristics; reflects genuine model doubt, not just token probabilities.

---

### STEER
**Approach**: Uses model confidence to route between smaller and larger reasoning models. Cost-aware in spirit—saves tokens on simple queries.

---

### UCCI (2026, Production Result)
**Finding**: On real production workload (75,000 queries), calibrated uncertainty routing achieved **31% cost reduction** while maintaining quality. Uses isotonic regression for confidence calibration.

**Critical detail**: Calibration is essential—"most deployed routers use uncalibrated confidence," leading to poor thresholds.

---

## 3. RL-Trained Routers: The Conductor Paper & Beyond

### Sakana's Conductor (Nielsen et al., ICLR 2026, arXiv:2512.04388)
**Your starting point**: RL-trained 7B Qwen agent that orchestrates multi-step workflows, routing queries to a worker pool (Haiku, Sonnet, Opus, GPT-4, etc.).

**Training method**: **GRPO** (Group Relative Policy Optimization), optimizing `reward = correctness` on MATH500 + reasoning benchmarks.

**Result**: ~95% MATH500 (paper claim), beating best single LLM (Opus ~80%).

**Novelty**: Structured orchestration (decompose → route → aggregate) vs black-box classifier.

**Your observation**: Achieves Opus-level correctness but **doesn't optimize for cost**. Conductor uses GPT-4 liberally when it helps, ignoring API expense.

**Limitation**: Worker pool is Sakana-specific; not transferable. No Pareto frontier.

---

### GraphRAG-Router (2025–2026, Emerging)
**Mechanism**: RL-trained hierarchical router coordinating multiple GraphRAG systems + generator LLMs. **Two-stage training**: (1) supervised fine-tuning, (2) **curriculum-based RL with cost-aware reward**.

**Reward formulation**: NOT disclosed in abstracts, but described as "difficulty-aware and economical generator allocation."

**Result**: Reduces LLM overuse by ~30% while maintaining generalization across 6 QA benchmarks.

**Significance**: **First explicitly cost-aware RL-trained router we found**. Mechanism is closest to your proposed pivot.

---

## 4. Token-Level & Speculative Routing (2025–2026)

### TIDE (2026)
**Innovation**: Per-token early exit—100% of prefill tokens, 99% of decode tokens exit before final layer on small models (8B).

**Cost implication**: Some tokens can exit at layer 10 (cheap), others need all 32 layers (expensive). Orthogonal to model selection but compatible.

---

### RelayLLM
**Approach**: Small model generates tokens normally; on token divergence (per-token confidence below threshold), escalates to large model. Only ~1.07% of tokens require large-model attention.

**Result**: 42–94% cost reduction vs full large-model responses on math/coding.

---

### SpecRouter
**Mechanism**: Adaptive routing within speculative decoding. Routes at token level during draft phase—different drafts get different routing paths.

---

## 5. Production Deployments

### AWS Bedrock Intelligent Prompt Routing
**Availability**: GA service with Anthropic (Haiku/Sonnet pairs) and Meta Llama families.

**Mechanism**: "Advanced prompt matching and model understanding"—likely supervised classifier, similar to RouteLLM.

**Cost savings**: Up to 30% without accuracy loss.

**Simplicity**: Users define two models and criteria; no complex workflow orchestration.

---

### OpenRouter (80 trillion monthly tokens)
**Model**: Unified API across 400+ models from 60+ providers. **Routing is implicit**—user picks model, OpenRouter handles provider failover. No smart routing exposed.

---

### LiteLLM (1 billion requests processed)
**Cost tracking**: Per-key, per-user, per-team attribution. Fallback/load balancing but **no intelligent routing**.

---

## 6. Critical Assessment: When Does Routing Fail?

### RouteLLM's Threshold Problem
- Threshold τ learned from one task distribution may not transfer. Requires re-tuning per domain.
- Binary routing is rigid—can't express "use Sonnet 70% of the time, Haiku 30%."

### Conductor's Non-Transferability
- 7B model trained on Sakana's specific worker pool. New pool (e.g., add Grok, remove GPT-4) → retrain.
- No mechanism to reweight workers by cost without full retraining.

### Supervised Routers' Data Dependency
- RouteLLM, UCCI need human preference annotations. Expensive to collect; may not align with production cost structure.

### Uncertainty Routing's Calibration Ceiling
- If the model has systematic blind spots (e.g., always overconfident on code), calibration alone can't fix it.

---

## 7. Open Questions & Research Gaps

1. **Cost-Quality Pareto Frontier**: No published work explicitly optimizes a multi-objective function like `R = α·quality + β·cost` with learned α, β. FrugalGPT and GraphRAG hint at it but don't expose the frontier.

2. **Compositional Cost Models**: How to handle complex costs—token count, latency SLA, cache hits, multi-turn context carryover? Current work assumes flat per-token cost.

3. **RL Reward Formulation**: GraphRAG mentions "curriculum cost-aware reward" but doesn't detail it. Is it Lagrangian? Constrained optimization? Pareto scalarization?

4. **Generalization Across Pools**: Can a single router trained on {Opus, Sonnet, Haiku} generalize to {Grok, Nemotron, Mistral Large} without retraining?

5. **Multi-Step Decomposition vs. Single Routing**: Is Conductor's workflow orchestration necessary, or is RouteLLM's simple classification sufficient? When does decomposition help?

6. **Closed-Loop Cost Optimization**: Should routers observe actual costs and adjust? E.g., if Sonnet API prices drop, does the router adapt?

---

## 8. Where Conductor Diverges from Existing Work

| Aspect | FrugalGPT | RouteLLM | GraphRAG-Router | Conductor |
|--------|-----------|----------|-----------------|-----------|
| **Training method** | Learned cascades (not RL) | Supervised (preference data) | RL + curriculum | RL (GRPO) |
| **Cost-aware objective** | Implicit (post-hoc evaluation) | No | Yes (curriculum stage) | No—optimizes quality only |
| **Workflow type** | Simple cascade | Binary choice | RAG-specific routing | Structured decomposition + routing |
| **Transferability** | Per-domain tuning needed | Per-domain | Unknown | Worker-pool-specific |
| **Publicly available** | Paper only | Code (GitHub) | Paper only | Paper only |

---

## 9. Novel Opportunity: Cost-First RL Router with Decomposition

**Your pivot idea**: Train GRPO agent that optimizes `reward = quality / cost` or `reward = quality - λ·log(cost)` on decomposable tasks.

**Why it's novel**:
- GraphRAG-Router uses cost in curriculum but not primary objective.
- Conductor optimizes quality first, cost is emergent.
- No published work uses cost-first framing with task decomposition.

**Hypothesis**: A 7B router trained on cost-aware reward will learn **different decomposition patterns** than Conductor.
- Conductor decomposes for correctness: "break into sub-problems, use strong models."
- Cost-first router decomposes for efficiency: "solve locally if possible, escalate only hard parts."

**Validation approach**:
1. Train on Sakana's MATH500 task with cost-aware reward (`R = accuracy - λ·tokens_used`).
2. Compare decompositions to Conductor's (shorter workflows? fewer escalations?).
3. Benchmark on cost-quality frontier: can you hit Sonnet-level accuracy at Haiku-level cost?
4. Test transfer: does the router work with a different worker pool (e.g., Claude 4.5 Opus, Claude 4.6 Sonnet, GPT-4.5 Turbo)?

---

## 10. References & URLs

**Foundational**:
- FrugalGPT: https://arxiv.org/abs/2305.05176
- RouteLLM: github.com/lm-sys/RouteLLM; MMLU/MTBench results in paper

**Cost-Aware RL**:
- GraphRAG-Router: Mentioned in recent arXiv; search "hierarchical routing strategy reinforcement learning"

**Token-Level Routing**:
- TIDE: arXiv 2026, token-level early exit
- RelayLLM: 42–94% cost reduction via per-token escalation

**Uncertainty & Production**:
- UCCI (2026): 31% cost reduction with calibrated uncertainty on 75k queries
- AWS Bedrock Intelligent Prompt Routing: aws.amazon.com/bedrock/intelligent-prompt-routing/

**Conductor (Your Reference)**:
- arxiv 2512.04388 (ICLR 2026), "Learning to Orchestrate Agents"

**Open-Source Frameworks**:
- RouteLLM: github.com/lm-sys/RouteLLM (serves + evaluates routers)
- LiteLLM: litellm.ai (cost tracking, not routing logic)
- OpenRouter: openrouter.ai (provider abstraction, no smart routing)

---

## Conclusion

**Current state**: Supervised routers (RouteLLM, UCCI) are production-ready and prove 30–98% cost reduction is achievable. RL-based routers (GraphRAG, Conductor) are emerging but not cost-first.

**Your opportunity**: Explicitly train a GRPO agent on `correctness × cost-efficiency` as the primary objective, with task decomposition as the search space. This inverts Conductor's quality-first framing and is orthogonal to (and complementary with) token-level routing and uncertainty-based fallback mechanisms.

**Key advantage**: If successful, a cost-aware router would be **more transferable** than Conductor (cost logic is model-agnostic) and **more efficient** than simple supervised routers (RL discovers decomposition strategies supervised methods miss).

