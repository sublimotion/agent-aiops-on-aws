# SOTA Reasoning Models: Training Techniques Analysis

A technical analysis of how state-of-the-art reasoning models are trained, based on published research and available documentation.

---

## Table of Contents

1. [Overview of Reasoning Models](#overview-of-reasoning-models)
2. [Core Training Paradigms](#core-training-paradigms)
3. [DeepSeek R1 Deep Dive](#deepseek-r1-deep-dive)
4. [OpenAI o1/o3 (What We Know)](#openai-o1o3-what-we-know)
5. [Process vs Outcome Reward Models](#process-vs-outcome-reward-models)
6. [Key Algorithms](#key-algorithms)
7. [Emerging Techniques](#emerging-techniques)
8. [Test-Time Compute Scaling](#test-time-compute-scaling)
9. [Open Questions](#open-questions)

---

## Overview of Reasoning Models

| Model | Organization | Open Weights | Key Innovation |
|-------|--------------|--------------|----------------|
| o1, o3 | OpenAI | No | Hidden chain-of-thought + RL |
| DeepSeek R1 | DeepSeek | Yes | GRPO, pure RL without SFT |
| Claude (Extended Thinking) | Anthropic | No | Extended thinking tokens |
| QwQ-32B | Alibaba/Qwen | Yes | Long-form reasoning |
| Gemini 2.0 Flash Thinking | Google | No | Thinking mode |

---

## Core Training Paradigms

### The Evolution of Reasoning Training

```
Generation 1: Supervised Fine-Tuning (SFT)
├── Train on human-written chain-of-thought examples
├── Limited by quality/quantity of human demonstrations
└── Model learns to imitate, not reason

Generation 2: RLHF with Outcome Rewards
├── Reward model scores final answers
├── PPO optimizes policy toward correct outputs
└── Model may learn shortcuts, not genuine reasoning

Generation 3: Process Reward Models (PRM)
├── Reward each reasoning step, not just final answer
├── Forces model to develop sound intermediate logic
└── Requires step-level annotations (expensive)

Generation 4: Pure RL (DeepSeek R1 approach)
├── RL directly on base model without SFT warmup
├── Reasoning emerges from reward signal alone
└── GRPO eliminates need for value function
```

### The Key Insight

**Reasoning capabilities can emerge purely from reinforcement learning** without requiring:
- Human-annotated reasoning traces
- Supervised fine-tuning as a prerequisite
- Explicit chain-of-thought demonstrations

This was demonstrated by DeepSeek R1-Zero achieving strong reasoning through RL alone.

---

## DeepSeek R1 Deep Dive

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepSeek R1 Training                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Base Model  │───▶│   Cold Start │───▶│     GRPO     │   │
│  │ (DeepSeek V3)│    │     Data     │    │      RL      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                 │            │
│                                                 ▼            │
│                                          ┌──────────────┐   │
│                                          │  DeepSeek R1 │   │
│                                          └──────────────┘   │
│                                                 │            │
│                              ┌──────────────────┼───────┐   │
│                              ▼                  ▼       ▼   │
│                         ┌────────┐         ┌───────┐       │
│                         │Distill │         │Distill│  ...  │
│                         │ Qwen   │         │ Llama │       │
│                         └────────┘         └───────┘       │
└─────────────────────────────────────────────────────────────┘
```

### R1-Zero vs R1

| Aspect | R1-Zero | R1 |
|--------|---------|-----|
| SFT before RL | No | Yes (cold start) |
| Readability | Poor (language mixing) | Clean |
| Reasoning | Strong but chaotic | Strong and structured |
| Purpose | Research proof-of-concept | Production model |

### GRPO: Group Relative Policy Optimization

**The core innovation** that makes R1 training efficient:

```
Standard PPO:
┌─────────────────────────────────────────────┐
│ Requires separate value network V(s)        │
│ Actor-Critic architecture                   │
│ Value function estimates expected returns   │
│ High memory/compute overhead                │
└─────────────────────────────────────────────┘

GRPO:
┌─────────────────────────────────────────────┐
│ No value network needed                     │
│ Sample multiple outputs per prompt          │
│ Compute advantages relative to group mean   │
│ More sample-efficient, simpler architecture │
└─────────────────────────────────────────────┘
```

**GRPO Objective Function:**

```
L_GRPO = E[ min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â) ] - β·D_KL[π_θ || π_ref]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  [probability ratio]
- Â = reward - mean(group_rewards)  [relative advantage]
- β·D_KL = KL penalty against reference policy
```

**Why it works:**
- Groups multiple outputs for the same question
- Computes relative advantage within each group
- No need to estimate absolute value of states
- KL penalty prevents policy from diverging too far

### Reward Signals in R1

```
┌─────────────────────────────────────────────────────────┐
│                    Reward Structure                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Process Rewards (per step):                            │
│  ├── Is this reasoning step logically valid?            │
│  ├── Does it follow from previous steps?                │
│  └── Is it on track toward the solution?                │
│                                                          │
│  Outcome Rewards (final):                               │
│  ├── Is the final answer correct?                       │
│  ├── For math: exact match                              │
│  └── For code: passes test cases                        │
│                                                          │
│  Combined signal guides both HOW and WHAT               │
└─────────────────────────────────────────────────────────┘
```

### Emergent Behaviors

Without explicit training, R1-Zero developed:
- **Self-reflection**: "Wait, let me reconsider..."
- **Verification**: Checking intermediate results
- **Backtracking**: Abandoning failed approaches
- **Strategy switching**: Trying alternative methods

These emerged purely from the RL reward signal.

---

## OpenAI o1/o3 (What We Know)

OpenAI has disclosed limited details. Based on public information:

### Confirmed/Likely Techniques

| Technique | Evidence |
|-----------|----------|
| Hidden chain-of-thought | Visible "thinking" indicator, summarized reasoning |
| Reinforcement learning | Mentioned in blog posts |
| Process reward models | Referenced in prior OpenAI research |
| Test-time compute scaling | Performance improves with more "thinking time" |

### OpenAI's Process Reward Model Research

From "Let's Verify Step by Step" (2023):

```
Outcome Supervision:
  Input: Problem + Full solution
  Label: Correct/Incorrect (binary)

Process Supervision:
  Input: Problem + Each reasoning step
  Label: Good/Bad/Neutral per step

Result: Process supervision significantly outperforms
        outcome supervision for training reliable reasoners
```

**PRM800K Dataset**: 800,000 step-level human feedback labels released by OpenAI for training process reward models.

### Speculated Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 o1/o3 Inference Flow                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Query                                              │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────┐                                    │
│  │ Generate hidden │◀──┐                                │
│  │ reasoning tokens│   │                                │
│  └────────┬────────┘   │                                │
│           │            │                                │
│           ▼            │                                │
│  ┌─────────────────┐   │                                │
│  │  Process Reward │   │ Continue if                   │
│  │     Model       │───┘ not confident                  │
│  └────────┬────────┘                                    │
│           │                                              │
│           ▼ (when confident)                            │
│  ┌─────────────────┐                                    │
│  │ Generate visible│                                    │
│  │    response     │                                    │
│  └─────────────────┘                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Process vs Outcome Reward Models

### Comparison

| Aspect | Outcome RM | Process RM |
|--------|------------|------------|
| **Feedback granularity** | Final answer only | Each reasoning step |
| **Annotation cost** | Low | High |
| **Reward signal** | Sparse | Dense |
| **Gaming risk** | High (shortcuts) | Low |
| **Performance** | Lower | Higher |
| **Debugging** | Hard (black box) | Easier (step visibility) |

### Why Process Rewards Win

```
Outcome Reward Problem:

  Correct answer via wrong reasoning → Rewarded ✓
  Wrong answer via correct reasoning → Penalized ✗

  Model learns: "Just get the right answer somehow"
  Result: Brittle, unreliable reasoning

Process Reward Solution:

  Each step evaluated independently
  Correct reasoning → Rewarded regardless of final answer

  Model learns: "Develop sound reasoning process"
  Result: Robust, generalizable reasoning
```

### Math-Shepherd: Automatic Process Supervision

Key innovation: Generate process rewards **without human annotation**

```
Method:
1. Sample multiple solution paths from model
2. For each step, check if remaining steps can reach correct answer
3. Steps that lead to dead ends → negative reward
4. Steps that enable correct solutions → positive reward

Result:
- Mistral-7B: 77.9% → 84.1% on GSM8K
- Further gains to 89.1% with verification
```

---

## Key Algorithms

### 1. PPO (Proximal Policy Optimization)

Standard RL algorithm for LLM fine-tuning:

```python
# Simplified PPO objective
L_PPO = E[min(
    r(θ) * A,                           # Unclipped
    clip(r(θ), 1-ε, 1+ε) * A           # Clipped
)]

# Where r(θ) = π_new(a|s) / π_old(a|s)
# Clipping prevents too-large policy updates
```

**Drawback**: Requires training a separate value network.

### 2. GRPO (Group Relative Policy Optimization)

DeepSeek's innovation:

```python
# Key difference: No value network
# Instead, compute advantage relative to group

for prompt in batch:
    outputs = [sample(model, prompt) for _ in range(G)]
    rewards = [reward_model(o) for o in outputs]
    baseline = mean(rewards)
    advantages = [r - baseline for r in rewards]

    # Update policy using relative advantages
    loss = grpo_loss(outputs, advantages)
```

### 3. DPO (Direct Preference Optimization)

Simpler alternative to PPO:

```python
# No reward model needed
# Directly optimize on preference pairs

L_DPO = -E[log σ(β(
    log π(y_w|x) - log π_ref(y_w|x) -
    log π(y_l|x) + log π_ref(y_l|x)
))]

# y_w = preferred response
# y_l = dispreferred response
```

### 4. STaR (Self-Taught Reasoner)

Bootstrap reasoning through self-improvement:

```
Loop:
1. Generate solutions with rationales
2. Keep only those with correct final answers
3. Fine-tune on successful rationales
4. Repeat with harder problems
```

---

## Emerging Techniques

### Quiet-STaR: Implicit Reasoning

Train models to generate internal "thoughts" at every token:

```
Standard generation:
  "The answer is" → "42"

Quiet-STaR generation:
  "The answer is" → [internal: "sum the numbers..."] → "42"

The internal thoughts are generated but hidden,
improving next-token prediction.
```

**Results**: Zero-shot improvements on GSM8K (5.9%→10.9%) from continued pretraining alone.

### Self-Play for Reasoning

Models generate their own training data through:
1. Solving problems multiple ways
2. Verifying solutions against each other
3. Selecting highest-confidence reasoning paths
4. Training on self-generated solutions

### Verification-Guided Search

At inference time:
```
1. Generate multiple solution candidates
2. Score each with process reward model
3. Select highest-scoring solution
4. Optionally: beam search over reasoning steps
```

---

## Test-Time Compute Scaling

### Key Finding

From "Scaling LLM Test-Time Compute Optimally":

> A smaller base model with optimal test-time compute allocation
> can outperform a 14x larger model on equivalent compute budgets.

### Compute-Optimal Strategies

```
Easy Problem:
  └── Direct generation (minimal thinking)

Medium Problem:
  └── Standard chain-of-thought

Hard Problem:
  └── Extended search + verification
      ├── Generate many candidates
      ├── Score with reward model
      └── Select best or continue searching
```

### Practical Implication

**Training compute vs inference compute tradeoff:**
- Larger model = more training compute, faster inference
- Smaller model + more thinking = less training, slower inference
- For hard problems, inference scaling can be more efficient

---

## Open Questions

### 1. Optimal Reward Signal Design
- How to balance process vs outcome rewards?
- Can we automate process reward annotation reliably?
- What's the right granularity for step-level feedback?

### 2. Reasoning Generalization
- Do reasoning skills transfer across domains?
- Is there a "general reasoning" capability or domain-specific?
- How to avoid reasoning shortcuts that don't generalize?

### 3. Efficiency
- Can we get reasoning benefits without 10-100x inference cost?
- Distillation effectiveness: How much reasoning transfers to smaller models?
- Sparse activation / early exit for easy problems?

### 4. Verification and Reliability
- How to know when the model's reasoning is trustworthy?
- Can models learn to recognize their own uncertainty?
- Role of formal verification in reasoning pipelines?

---

## Key Papers & Resources

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | 2023 | Process reward models + PRM800K |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | GRPO, pure RL reasoning |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | 2023 | Automatic process supervision |
| [Quiet-STaR](https://arxiv.org/abs/2403.09629) | 2024 | Implicit reasoning training |
| [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) | 2024 | Compute-optimal inference |
| [STaR](https://arxiv.org/abs/2203.14465) | 2022 | Self-taught reasoning |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | Foundation for reasoning research |

---

## Summary

The state-of-the-art in reasoning model training has evolved from simple supervised fine-tuning to sophisticated RL-based approaches:

1. **Pure RL works**: DeepSeek R1-Zero proved reasoning can emerge without human demonstrations
2. **Process rewards > outcome rewards**: Step-level feedback produces more reliable reasoners
3. **GRPO simplifies training**: No value network needed, group-relative advantages suffice
4. **Test-time compute matters**: Optimal inference allocation can substitute for model scale
5. **Distillation preserves reasoning**: Smaller models can inherit reasoning from larger ones

The field is moving toward models that genuinely reason rather than pattern-match, with the training signal (process rewards) being as important as the training algorithm (GRPO/PPO).
