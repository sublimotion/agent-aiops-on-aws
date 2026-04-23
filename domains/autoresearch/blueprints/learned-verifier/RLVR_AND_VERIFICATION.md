# RLVR, RL Environments, and the Verification-to-Reward Bridge

**Date**: 2026-03-22
**Context**: Deep research on how RL post-training reward signals connect to our Learned Verifier Experiment. Complements VERIFICATION_FRAMEWORK.md (primitives and methodology) and VERIFIER_ECONOMICS.md (cost analysis). This note covers the research landscape, the emerging RL environment ecosystem, and the practical bridge from verification to RL reward signals.

---

## Part 1: RLVR — Reinforcement Learning with Verifiable Rewards

### What RLVR Is

RLVR is an RL post-training paradigm where the reward signal comes from an objectively checkable function rather than a learned reward model. Popularized by DeepSeek-R1 (arXiv:2501.12948) and Allen AI Tulu 3, it has become the dominant framing for RL post-training in 2025-2026.

**RLVR vs RLHF**:
- **RLHF**: Trains a reward model on human preference pairs (Bradley-Terry), then optimizes policy against that proxy. Subject to reward hacking, drift, and distribution shift.
- **RLVR**: Replaces the learned reward model with a verification function `f(response, problem) -> {0, 1}`. No reward model training. No preference data. Deterministic ground truth where available.

**Why it matters for our work**: The verification spectrum (Hard → Strong → Skill → Consensus → Behavioral → Learned → None) maps directly onto the RLVR reward signal taxonomy. Each tier of our verification framework can serve as a reward signal — the question is quality, cost, and calibration.

### GRPO: The Dominant Algorithm

GRPO (Group Relative Policy Optimization, arXiv:2402.03300) eliminates the critic/value network from PPO:
1. Sample K completions per prompt (typically 8-64)
2. Score all K with the verification function
3. Compute advantages relative to the group (normalized within-group)
4. Update policy with clipped surrogate objective

Key insight: "GRPO is Secretly a Process Reward Model" (arXiv:2509.21154) — GRPO with outcome rewards is mathematically equivalent to a PRM-aware RL objective with Monte-Carlo-based process rewards. It implicitly does token-level credit assignment even with outcome-only rewards.

**DAPO** (arXiv:2503.14476): Open-source improvement on GRPO, achieved 50 on AIME 2024 with Qwen2.5-32B. Built on veRL.

### The Reward Signal Taxonomy

Ordered from most to least verifiable:

#### Tier 1: Formally Verifiable (Ground Truth)

| Signal Type | Mechanism | Latency | Domain |
|------------|-----------|---------|--------|
| Formal proof verification | Lean/Coq/Isabelle proof checker | Seconds | Math theorems |
| Test execution (code) | Run against test suite in Docker | Seconds-minutes | Software engineering |
| Exact answer matching | Compare to known solution | Microseconds | Math, factual QA |

#### Tier 2: Programmatically Verifiable (Rule-Based)

| Signal Type | Mechanism | Latency | Domain |
|------------|-----------|---------|--------|
| Constraint checking | Structural property verification | Milliseconds | Instruction following |
| Similarity scoring | Edit distance / BLEU to reference | Milliseconds | Code evolution (SWE-RL) |
| Regex/pattern matching | Output format verification | Microseconds | Structured output |

#### Tier 3: Model-Based Verification

| Signal Type | Mechanism | Latency | Domain |
|------------|-----------|---------|--------|
| LLM-as-judge | Strong LLM evaluates correctness | Seconds | Open-ended tasks |
| Learned reward model | Trained on preference/outcome data | Milliseconds | General (RLHF fallback) |
| Process reward model (PRM) | Step-level evaluation | Milliseconds | Reasoning chains |
| SVG consensus | Two independent generations agree | Seconds (2 calls) | Code, generalizable |
| Agentic rubrics | Context-grounded checklists | Seconds | Code, knowledge work |

#### Tier 4: Derived/Composite Signals

| Signal Type | Mechanism | Latency | Domain |
|------------|-----------|---------|--------|
| Consensus/majority vote | Multiple rollouts agree | N × generation time | Any |
| Adversarial/evolved verification | Test cases co-evolve with policy | Variable | Code (EvolveCoder) |
| Implicit process rewards (PRIME) | Token-level log-prob ratios | Near-zero (same model) | Any |
| Auto-generated environments (ReSyn) | Synthetic tasks with built-in verifiers | Pre-computed | Reasoning |

### Key Systems

| System | Reward Type | Result | Cost | Paper |
|--------|------------|--------|------|-------|
| DeepSeek-R1 | Exact match (math) + tests (code) via GRPO | Frontier reasoning | Massive (2048+ H800s) | 2501.12948 |
| SWE-RL | Similarity scoring (rule-based) | 41% SWE-bench Verified | Moderate | 2502.18449 |
| R4P | Learned patch reasoner (GRPO-trained) | 72.2% verification accuracy | 50x faster than tests | 2510.22775 |
| SWE-RM | MoE reward model (30B/3B active) | +10.4pp on SWE-bench | Execution-free | 2512.21919 |
| SERA-32B | SVG (SFT, not RL) | 49.5% SWE-bench | $2K (40 GPU-days) | 2601.20789 |
| Agentic Rubrics | Context-grounded checklists | 54.2% (+3.5pp) | No test execution | 2601.04171 |
| PRIME | Implicit PRM from outcomes | 2.5x sample efficiency | Same model, no annotation | 2502.01456 |
| EvolveCoder | Adversarial test evolution | +4.2pp on benchmarks | Co-evolving tests | 2603.12698 |
| Mini-SE (R4P) | Trained entirely with learned rewards | 32.8% Pass@1 | No test execution | 2510.22775 |

---

## Part 2: The RL Environment Ecosystem

### The Landscape (35+ Companies)

SemiAnalysis reports 35+ companies building RL environments across domains. The market is bifurcating into:

**Environment Platform Providers**:
- **HUD** (hud.ai) — Environment SDK with Docker isolation + MCP tool integration. Fresh environment per evaluation. Gateway across Claude/GPT/Gemini.
- **OpenEnv** (Meta + HuggingFace, Oct 2025) — OSS standardized `step()/reset()/close()` API. Hub-based sharing. Compatible with TRL, veRL, OpenRLHF, NeMo-RL. First serious attempt to standardize the environment layer.
- **Prime Intellect** — "Environments Hub" for centralized RL environment discovery. Ran INTELLECT-2 (first globally distributed RL of 32B model) and INTELLECT-3 (100B+ MoE).

**Contractor Networks / Data Providers**:
- **Surge** (~$1B ARR) — Largest player. Operates EnterpriseBench, Hemingway-bench. Active with Western and Chinese labs.
- **Mercor** — AI-powered talent marketplace, $60-$120+/hr roles. Publishes APEX-Agents Leaderboard.
- **Handshake**, **Aboda.ai** — Professional domain expert hiring.

**Physical Science RL**:
- **Medra** ($52M Series A) — Robotic biology labs. CRISPR, cell culture, NGS. Closed-loop RL for drug discovery.
- **Periodic Labs** — Physical lab systems for science RL.

**Task/Dataset Generators**:
- **SWE-smith** — Turns any GitHub repo into a "SWE-gym." 52K task instances, 26K trajectories, 250+ Docker environments.
- **SWE-Dev** (ACL'25) — Scalable test case construction from PyPI/GitHub. 7B: 23.4%, 32B: 36.6% on SWE-bench.
- **SWE-rebench** — 450K PRs → 21,336 valid tasks (95% rejection rate). LLM generates installation recipes.
- **ReSyn** (arXiv:2602.20117) — Autonomously generates diverse reasoning environments with built-in verifiers.

### The RL Environment Stack

```
Layer 5: Weight Update Integration
  veRL, OpenRLHF, TRL, NeMo-RL — training loop frameworks
  In-flight weight updates yield ~2x iteration improvement for long rollouts

Layer 4: Trajectory Collection & Storage
  HuggingFace datasets, JSONL logs, environment traces
  SWE-smith: 26K trajectories on HF Hub

Layer 3: Reward Computation  ← THIS IS WHERE VERIFICATION LIVES
  Test execution (Docker), formal verification, rubrics,
  learned verifiers, SVG consensus, LLM-as-judge

Layer 2: Environment Instantiation
  Docker containers (dominant), OpenEnv step/reset/close API,
  physical labs (Medra), browser sandboxes (HUD + MCP)

Layer 1: Task Generation / Curation
  GitHub mining (SWE-rebench: 450K→21K), synthetic bugs (SWE-smith),
  expert-written (GDPval: 1000+ tasks), UI mockups ($20K/site)
```

**Key insight**: Our verification framework operates at Layer 3. The verification spectrum maps directly onto the reward computation layer. The rest of the stack (environment instantiation, trajectory storage, weight updates) is infrastructure we can leverage from existing OSS frameworks.

### Cost and Scale Data

| Data Point | Source |
|-----------|--------|
| DeepSWE: 512 Docker containers, 64 H100s, 6 days, ~$100K+ | SemiAnalysis |
| Kimi: 10,000+ parallel container instances | SemiAnalysis |
| SWE-rebench: 450K PRs → 21,336 valid tasks (95% rejection) | SemiAnalysis |
| SWE-smith: 52K tasks, 26K trajectories, 250+ Docker envs | GitHub |
| UI gym environments: ~$20K per website mockup | SemiAnalysis |
| DeepSeek V3.2: 24,667 GitHub-extracted tasks | SemiAnalysis |
| Surge: ~$1B ARR | SemiAnalysis |
| Mercor contractor rates: $60-$120+/hr | mercor.com |
| Medra: $52M Series A for lab automation | medra.ai |
| SERA-32B SVG training: $2K total (40 GPU-days) | Allen AI |
| Google DeepMind: <5% compute on post-training for Gemini 2.5 Pro | SemiAnalysis |
| R4P verification: 50x faster than test execution | arXiv:2510.22775 |
| SWE-RM: 400K+ trajectories for training | arXiv:2512.21919 |

### Lab Strategies

| Lab | Approach | Scale |
|-----|----------|-------|
| **Anthropic** | 12+ vendors, exclusive startup contracts, sandbox standardization | Broad domain expansion: coding → computer use → biology |
| **OpenAI** | Smaller vendor pool, building in-house teams, GDPval (1000+ tasks, 44 occupations) | "Hundreds" of $20K website mockups for UI agents |
| **Google DeepMind** | Decentralized by research team, leverages owned platforms (Sheets, Docs) | <5% post-training compute (scaling for Gemini 3) |
| **Kimi/Moonshot** | Massive parallel environments | 10K+ simultaneous instances |

---

## Part 3: The Verification → RL Reward Bridge

This is the critical section for our experiment: how does a good verifier become a good RL reward signal?

### The Calibration Gate

**The most important finding from this research**: Verification quality for ranking (best-of-N selection) has different requirements than verification quality for RL training. The differentiator is **calibration (ECE)**.

From SWE-RM (arXiv:2512.21919):
- Two verifiers with near-identical ranking performance (TTS +4.7% vs +4.5%)
- AUC differed by 0.095 (0.805 vs 0.710)
- ECE differed by **7x** (0.481 vs 0.069)
- RL training with the poorly-calibrated verifier **collapsed with significant instability**
- Mechanism: miscalibration "couples multiplicatively with the policy gradient, injecting additional variance into the gradient estimator"

**Implications for our SVG consensus (AUC 0.981, precision 1.0)**:
- SVG is excellent for ranking and best-of-N selection (our current use case)
- For RL use, we need to measure ECE. Binary SVG scores (accept/reject at threshold=0.8) may need calibration
- The precision=1.0 finding is encouraging (no false positives), but RL needs well-calibrated continuous scores, not just binary decisions

### Noisy Rewards Are Destructive

arXiv:2603.16140 ("Noisy Data is Destructive to RLVR") proved:
- Prior claims of noise tolerance were based on contaminated experiments
- With genuinely wrong annotations, models underperform clean baselines by 8-10%
- **Algorithmic improvements (GRPO variants) cannot compensate for noisy rewards**
- Reward signal quality is more important than algorithm choice

**Connection to our framework**: This validates precision-first verification design. SVG consensus at precision=1.0 means zero noise in the positive labels. The risk is in false negatives (recall=0.528) — but false negatives just reduce data volume, they don't inject noise.

### Overoptimization Scaling Laws

Gao et al. (arXiv:2210.10760) established:
- **RL**: Gold reward follows `R_gold = α√(d_KL) − β·d_KL` — initial improvement then decline as KL increases
- **Best-of-N**: Follows a log functional form — less susceptible to overoptimization
- α and β scale with reward model parameters — larger RMs overoptimize later but still eventually decline

**Key implication**: Best-of-N selection with a learned verifier (our proposed architecture) is **inherently safer** than using the same verifier as an RL reward signal. This is a strong argument for the phased approach in our framework.

### PRM vs ORM in Code

| Approach | Type | Mechanism | Paper |
|----------|------|-----------|-------|
| Test execution | ORM (hard) | Binary pass/fail on final output | Standard |
| SWE-RM | ORM (soft) | Trajectory-level score | 2512.21919 |
| R4P group-wise | ORM (comparative) | Group-level patch comparison | 2510.22775 |
| PRIME implicit | PRM (implicit) | Token-level log-prob ratios from outcome labels | 2502.01456 |
| Agentic Rubrics | PRM-like | Step-by-step rubric scoring | 2601.04171 |
| Behavioral features (ours) | PRM-like | Action distribution, context growth, Parkinson's ratio | Our Phase 0 |

**PRIME** (arXiv:2502.01456) is the most relevant for our pipeline:
- Derives process rewards implicitly: `r(y_t) = β·log[π(y_t|y<t) / π_ref(y_t|y<t)]`
- No explicit step-level annotation — single model serves as policy, PRM, and reference
- Online PRM updates prevent reward hacking (offline PRMs degrade)
- 2.5x sample efficiency vs outcome-only RL

**Our behavioral features are essentially a PRM for coding agents.** Action distribution, context growth rate, Parkinson's ratio, loop detection — these are process signals. They don't require test execution and they transfer across codebases (Critic Rubrics confirmed this). Combined with SVG consensus (outcome signal), we have a natural PRM+ORM ensemble.

### The SVG-as-Reward Gap

**No one has published RL training with SVG-derived rewards.** This is an open research opportunity.

SERA (Allen AI) explicitly chose SFT over RL for cost reasons ($2K vs 26x more). Their finding that "all verification thresholds perform similarly" for SFT does NOT mean thresholds don't matter for RL — SWE-RM proved exactly the opposite.

However, **Agentic Rubrics** (arXiv:2601.04171) is close:
- Context-grounded rubric checklists without test execution
- Expert agent explores repo to generate verification criteria
- Explicitly designed as RL reward signals
- 54.2% on SWE-bench (+3.5pp over baselines)

### Practical Pipeline: Verifier → RL Reward

```
Step 1: Validate RL-readiness
  ├─ Measure AUC (we have: 0.981 ✓)
  ├─ Measure ECE calibration (UNKNOWN — critical gap)
  └─ If ECE > 0.1 → calibrate before RL use

Step 2: Choose integration strategy
  ├─ Best-of-N selection (CURRENT — safe, log overoptimization)
  ├─ Rejection sampling for SFT (safe, no RL infra needed)
  ├─ Outcome reward for GRPO (requires low ECE)
  ├─ Hybrid reward: SVG + execution where available (SWE-RM pattern)
  └─ Implicit PRM: PRIME-style token-level (highest leverage, most complex)

Step 3: Infrastructure
  ├─ Best-of-N: Generator + Verifier only (inference, no training)
  ├─ RL: GRPO framework (veRL/TRL/OpenRLHF) + rollout engine (vLLM)
  ├─ Reward serving: Colocated if small (SWE-RM 3B active MoE)
  └─ KL penalty tuning + periodic recalibration

Step 4: Monitor failure modes
  ├─ Reward hacking (pattern exploitation)
  ├─ Calibration drift (policy improves, verifier distribution shifts)
  ├─ Collapse under high ECE
  ├─ Context truncation → null rewards (need 256K support)
  └─ Goodhart plateau (proxy vs gold divergence)
```

### Requirements for a Verifier as RL Reward (from SWE-RM)

1. High AUC across full distribution (not just ranking)
2. Low ECE (< 0.1) — the gate between "good for best-of-N" and "good for RL"
3. 20K+ training examples for generalization
4. 2:1 positive-to-negative ratio optimal
5. 256K context support (avoid truncation-induced null rewards)
6. Mix-policy training data (on-policy + off-policy) for robustness
7. Combine execution-free with execution-based feedback in hybrid reward

---

## Part 4: Bridging Research to Applied AI

### What the Stack Looks Like Today

For an applied team wanting to use RL post-training with verification:

```
┌─────────────────────────────────────────────────────┐
│                    Training Loop                      │
│  veRL / TRL / OpenRLHF + GRPO                        │
│  Weight sync: colocated (resharding) or disaggregated │
├─────────────────────────────────────────────────────┤
│                   Rollout Engine                      │
│  vLLM (sleep mode, RLHF APIs) / SGLang               │
│  K=8-64 completions per prompt                        │
├─────────────────────────────────────────────────────┤
│               Reward Computation  ← OUR LAYER         │
│  Tier 1: Test execution (Docker) where available      │
│  Tier 2: SVG consensus (2 inference calls)            │
│  Tier 3: Learned verifier (1 inference call)          │
│  Tier 4: Behavioral/process signals (free telemetry)  │
│  Hybrid: combine tiers for best signal                │
├─────────────────────────────────────────────────────┤
│               Environment Layer                       │
│  OpenEnv (step/reset/close API)                       │
│  Docker containers / sandboxes                        │
│  Task curation: SWE-smith / SWE-rebench pipelines     │
├─────────────────────────────────────────────────────┤
│              Task Generation Layer                    │
│  GitHub mining (SWE-rebench: 450K→21K)               │
│  Synthetic bugs (SWE-smith: 52K tasks)               │
│  Expert-written (GDPval: 1000+ tasks)                │
│  LLM-generated tests from PRD/specs (emerging)       │
└─────────────────────────────────────────────────────┘
```

### The Phased Adoption Path

| Phase | What You Deploy | Reward Signal | RL Required? | Cost |
|-------|----------------|---------------|-------------|------|
| **0: Baseline** | Agent + blind submit | None | No | ~20% pass rate |
| **1: Best-of-N + SVG** | Agent + SVG consensus + selection | SVG score | No | ~53% pass, $67/week verify |
| **2: Best-of-N + learned** | Agent + trained verifier + selection | Learned score | No | ~72% pass, $6/week verify |
| **3: Rejection sampling SFT** | Fine-tune agent on verified-good outputs | SVG/learned labels | No (SFT only) | Agent improves base rate |
| **4: GRPO with learned reward** | RL training loop with verifier as reward | Calibrated verifier | Yes (veRL/TRL) | Agent learns to self-verify |
| **5: Implicit PRM** | PRIME-style token-level rewards | Outcome + process | Yes (advanced) | Dense credit assignment |

**The critical transition is Phase 2 → Phase 3.** Best-of-N is safe (log overoptimization). SFT on verified outputs is safe (just selecting good data). RL training (Phase 4) requires calibration guarantees that best-of-N and SFT don't need.

### What Our Verification Framework Provides to the RL Stack

| Framework Component | RL Stack Role |
|--------------------|---------------|
| **Verification spectrum** | Maps directly to reward signal taxonomy (Tier 1-4 above) |
| **SVG consensus (precision=1.0)** | Clean positive labels for rejection sampling / SFT (Phase 3) |
| **Behavioral features** | Process reward signals (PRM-equivalent, free from telemetry) |
| **Skill verifier** | Bootstrap labels for domains without tests (cold-start for Phases 2-3) |
| **Tiered deployment** | Progressive adoption path from no-RL (Phase 1) to full RL (Phase 5) |
| **ECE measurement** | Gate between best-of-N (safe) and RL reward (requires calibration) |

### Open Research Questions

1. **What is the ECE of SVG consensus scores?** This is the gate for RL use. We have AUC=0.981 but ECE is unknown. Measuring this on our n=300 dataset is the next step.

2. **Can SVG consensus serve as an RL reward signal?** Nobody has published this. SERA used SVG for SFT filtering only. The combination of SVG's precision=1.0 (clean signal) with GRPO's group normalization (relative advantage) might work — but the calibration question must be answered first.

3. **Does the behavioral PRM + SVG ORM ensemble outperform either alone as an RL reward?** PRIME showed implicit PRMs give 2.5x sample efficiency. Our behavioral features (action distribution, context growth, Parkinson's ratio) are process signals. Combined with SVG outcome signal, this could be a powerful hybrid reward.

4. **How does adversarial test evolution (EvolveCoder) interact with learned verifiers?** EvolveCoder co-evolves tests with the policy, making verification harder over training. Could a learned verifier do the same — getting better as the policy improves?

5. **Is the "verification threshold doesn't matter for SFT" finding (SERA) a fundamental property or an artifact?** If thresholds genuinely don't matter for SFT, that simplifies our Phase 3 significantly. But it might not hold for harder codebases or smaller models.

6. **Can LLM-generated tests from PRD/specs serve as RL reward environments?** (From our earlier discussion.) The infrastructure exists (Docker, OpenEnv). The question is test quality — generated tests with false positives inject noise, and noise is destructive to RLVR.

---

## Key Citations

| Paper | arXiv | Key Contribution |
|-------|-------|-----------------|
| DeepSeek-R1 | 2501.12948 | Foundational RLVR; GRPO; emergent reasoning |
| DeepSeek-Math | 2402.03300 | Introduced GRPO |
| GRPO is Secretly a PRM | 2509.21154 | GRPO = implicit PRM (mathematical proof) |
| R4P | 2510.22775 | Learned patch verifier, 72.2% accuracy, 50x faster |
| SWE-RM | 2512.21919 | MoE reward model, ECE calibration critical for RL |
| SERA | 2601.20789 | SVG training, $2K cost, 49.5% SWE-bench |
| Agentic Rubrics | 2601.04171 | Context-grounded checklists as RL reward |
| PRIME | 2502.01456 | Implicit process rewards, 2.5x sample efficiency |
| EvolveCoder | 2603.12698 | Adversarial test co-evolution |
| Noisy Data Destructive | 2603.16140 | Reward quality > algorithm quality |
| DAPO | 2503.14476 | Open-source improved GRPO on veRL |
| Gao et al. | 2210.10760 | Overoptimization scaling laws (RL vs best-of-N) |
| SWE-RL | 2502.18449 | RL on GitHub evolution, similarity scoring |
| M2RL | 2602.12566 | Multi-domain RLVR, minimal cross-domain interference |
| Open-Reasoner-Zero | 2503.24290 | R1-Zero in 1/10 steps |
| SWE-smith | (GitHub) | 52K tasks from any repo |
| SWE-Dev | 2505.16975 | Scalable test construction |
| ReSyn | 2602.20117 | Auto-generated verification environments |
| Kimi K2 | 2507.20534 | Joint RL real+synthetic, 1T-param MoE |
| SemiAnalysis | (newsletter) | 35+ RL environment companies, ecosystem mapping |

---

*This note is research documentation. It complements VERIFICATION_FRAMEWORK.md (the methodology) and VERIFIER_ECONOMICS.md (the business case). Together they form the complete picture: why verification matters (economics), how to implement it (framework), and where the field is heading (this note).*
