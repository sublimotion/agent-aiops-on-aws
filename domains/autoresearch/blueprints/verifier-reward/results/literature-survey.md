# Verifier Reward — Literature Survey

Date: 2026-03-24

## Related Work

### Tier 1: Patch Verification & Selection

| Paper | Date | Key Idea | Relation |
|-------|------|----------|----------|
| [Patch Reasoner (R4P)](https://arxiv.org/abs/2510.22775) | Oct 2025 | Trains patch verification model with group-wise RL. 72.2% accuracy, 50x faster than test execution. | Closest match — reasoning-based (no execution) like ours, but trains a dedicated model vs prompt engineering. Higher recall, lower precision. |
| [CodeMonkeys](https://arxiv.org/abs/2501.14723) | Jan 2025 | Best-of-N with test-script voting. 57.4% SWE-bench Verified (~$2,300). | Test-execution analog. Our approach is 5 orders of magnitude cheaper and execution-free. Complementary as fast pre-filter. |
| [Large Language Monkeys](https://arxiv.org/abs/2407.21787) | Jul 2024 | Coverage scales log-linearly with samples (15.9% to 56% at 250 samples), but selection plateaus. | Explicitly identifies the verification bottleneck we address. Precision=1.00 verifier would unlock their scaling. |
| [SWE-Search (MCTS)](https://arxiv.org/abs/2410.20285) | Oct 2024 | MCTS with Value Agent + Discriminator Agent for SWE-bench. 23% relative improvement. | Their Discriminator uses confirmatory framing — adversarial framing could improve it. |
| [SWT-Bench](https://arxiv.org/abs/2406.12952) | Jun 2024 | LLM-generated test cases validate patches. Doubles SWE-Agent precision. | Test-generation complement to our LLM-as-judge. No execution needed for ours. |
| [SWE-bench+](https://arxiv.org/abs/2410.06992) | Oct 2024 | 32.67% of "successful" patches involve solution leakage, 31.08% pass due to weak tests. | Validates why our verifier is needed — ground-truth tests allow plausible-but-wrong patches through. |
| [Agentless](https://arxiv.org/abs/2407.01489) | Jul 2024 | Three-phase (localize, repair, validate) without agents. 32% SWE-bench Lite at $0.70/issue. | Their validation phase uses basic heuristics — our adversarial judge could improve selection. |

### Tier 2: RLVR & Reward Models for Code

| Paper | Date | Key Idea | Relation |
|-------|------|----------|----------|
| [Rate or Fate? (RLVR theory)](https://arxiv.org/abs/2601.04411) | Jan 2026 | Youden's J = TPR - FPR determines RLVR success. J > 0 guarantees learning. | **Theoretical validation.** Our J = 0.33 - 0 = 0.33 > 0. Low recall slows learning but doesn't prevent it. Proves precision-first design is correct. |
| [Aletheia (RLVR for code)](https://arxiv.org/abs/2601.12186) | Jan 2026 | Controlled study of RLVR for code verifiers. Negative samples stabilize training at scale. | Directly studies how to train with verifiers like ours. Negative samples align with adversarial framing. |
| [SWE-RL (Meta)](https://arxiv.org/abs/2502.18449) | Feb 2025 | Similarity-score reward for RL. Llama3-SWE-RL-70B at 41.0% SWE-bench Verified. | Our precision=1.00 binary reward is cleaner than their similarity score. |
| [CodeScaler](https://arxiv.org/abs/2602.17684) | Feb 2026 | Execution-free reward model via preference data. 10x lower latency than test-based. | Same spirit (reward without execution), but they train a model; we use prompt engineering. |
| [Process-Supervised RL for Code](https://arxiv.org/abs/2502.01715) | Feb 2025 | Line-by-line process supervision surpasses outcome supervision. | Suggests combining our outcome-level verifier with step-level feedback. |
| [Rubrics as Rewards](https://arxiv.org/abs/2507.17746) | Jul 2025 | Structured rubric-based RL feedback. +31% on medical benchmarks. | Our adversarial rubric is a specialized instance. Validates rubric-based rewards over simple judge ratings. |

### Tier 3: Adversarial & Rubric-Based Verification

| Paper | Date | Key Idea | Relation |
|-------|------|----------|----------|
| [EvolveCoder](https://arxiv.org/abs/2603.12698) | Mar 2026 | Adversarial test refinement — iteratively strengthens tests to break solutions. | Shares adversarial philosophy. They make tests adversarial; we make the rubric adversarial. |
| [Data-Driven Reasoning Rubrics](https://arxiv.org/abs/2602.06795) | Feb 2026 | Auto-generates error taxonomies for LLM verification. 45% improvement over baseline. | Could automate rubric discovery — potentially finding better adversarial rubrics or solving FM-001. |
| [Rubric-Supervised Critic](https://arxiv.org/abs/2603.03800) | Mar 2026 | Critic from 24 behavioral features + semi-supervised learning. Best-of-8 reranking +16% on SWE-bench. | Validates rubric-based approach. Uses behavioral features we could incorporate. |
| [CVeDRL](https://arxiv.org/abs/2601.22803) | Jan 2026 | 0.6B code verifier trained with difficulty-aware RL. 28.97% higher pass rates than GPT-3.5. | Small dedicated verifier as alternative to our LLM-as-judge approach. |

### Tier 4: Process Reward Models

| Paper | Date | Key Idea | Relation |
|-------|------|----------|----------|
| [VPRMs](https://arxiv.org/abs/2601.17223) | Jan 2026 | Rule-based intermediate step verification. +20% F1 over outcome-only. | Suggests decomposing patch verification into steps could improve recall without sacrificing precision. |
| [DAJ (Data-Reweighted Judge)](https://arxiv.org/abs/2601.22230) | Jan 2026 | Reasoning-based LLM judge with bi-level data reweighting for best-of-N. SOTA on LiveCodeBench. | Relevant if we train a dedicated verifier model from our rubric scores. |
| [PRM Survey](https://arxiv.org/abs/2510.08049) | Oct 2025 | Comprehensive PRM survey across code, math, agents. | Reference for PRM landscape. |
| [Thinking Longer, Not Larger](https://arxiv.org/abs/2503.23803) | Mar 2025 | Test-time compute scaling with reward models. 32B beats 671B on SWE-bench. | Our verifier could serve as the reward model in their search framework. |

---

## Open Questions

### Q1: Does adversarial framing transfer to trained verifiers?
Our adversarial rubric works via prompt engineering on a general model (Haiku). Would the same adversarial framing improve a *trained* verifier (like R4P or CVeDRL)? Or does training on verification data already implicitly learn adversarial reasoning?

### Q2: Can automated rubric discovery find better adversarial rubrics?
Data-Driven Reasoning Rubrics (arxiv 2602.06795) auto-generates error taxonomies. Could this discover adversarial rubrics that handle FM-001 (reformatting noise) — our main unsolved failure mode?

### Q3: What's the optimal precision-recall tradeoff for RLVR?
"Rate or Fate?" proves J > 0 is sufficient, but what's the learning speed curve? At J=0.33 (our verifier), how many RL steps are needed vs a hypothetical J=0.80 verifier with some FPs? Is there a sweet spot?

### Q4: Does process-level adversarial verification improve recall?
Our verifier is outcome-level (judges the final patch). VPRMs show process verification adds +20% F1. Could we decompose into adversarial sub-questions — "is the bug correctly localized?", "is the fix semantically complete?", "are there side effects?" — and improve recall from 0.33 without losing precision?

### Q5: How does the verifier scale with N candidates?
Large Language Monkeys show coverage scales log-linearly. With precision=1.00 and recall=0.33, what's the expected yield curve as N grows? At what N does our verifier reliably find a correct patch if one exists?

### Q6: Adversarial framing during generation vs post-hoc?
Phase 2b of our spec tests self-critique during generation. EvolveCoder does adversarial test refinement during training. Is adversarial reasoning more effective as a generation-time prompt, a post-hoc filter, or a training signal?

---

## Next Steps from Literature

### Near-term (no new infrastructure)

1. **Validate RLVR theory**: Compute expected learning curves using "Rate or Fate?" framework with our J=0.33. Determine minimum RL training budget needed for convergence.

2. **Scale N for best-of-N**: Run our verifier on larger candidate pools (N=8 from existing multi-harness data, N=50+ via repeated sampling). Measure yield curve — does precision hold as N grows?

3. **Automated rubric search**: Apply data-driven rubric generation (arxiv 2602.06795) to our failure cases. Target FM-001 (reformatting noise) which accounts for all remaining FNs.

4. **Process decomposition**: Split v009 adversarial rubric into sub-questions (localization, fix semantics, side effects, completeness). Test whether decomposed adversarial verification recovers recall on the 4/6 missed patches.

### Medium-term (requires training infrastructure)

5. **Train a dedicated adversarial verifier**: Use R4P's group-wise RL approach but with adversarial framing in the training signal. Could combine our rubric scores as training labels with their RL objective.

6. **RLVR experiment**: Use our v001+v009 ensemble as reward model for GRPO training on a coding agent (e.g., Qwen 3-32B). Compare to SWE-RL's similarity-score reward and SERA's SVG-filtered SFT.

7. **Behavioral feature ensemble**: Combine our rubric scores with the 24 behavioral features from Rubric-Supervised Critic (arxiv 2603.03800) for a hybrid verifier.

### Longer-term (research directions)

8. **Adversarial framing as a general principle**: Test whether confirmatory x adversarial ensemble improves LLM-as-judge in other domains (math proofs, legal reasoning, medical diagnosis). If it generalizes, this is a contribution beyond coding.

9. **Execution-grounded adversarial verification**: Combine our "find the bug" framing with EGCA's execution trace divergence (arxiv 2603.16158) — adversarial reasoning guided by actual execution differences.

10. **Scaling law for verification**: Map the relationship between verifier quality (precision, recall), candidate pool size (N), and downstream pass rate. Derive the optimal allocation between generation compute and verification compute.
