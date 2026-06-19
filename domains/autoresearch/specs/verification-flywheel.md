# Autoresearch Spec: Verification Flywheel as Harness Evaluation Layer

## Status: COMPLETE

## Overview
The learned-verifier cascade is a **harness augmentation** — an evaluation overlay that replaces expensive test execution with cheap, fast patch quality signals. In the autoresearch loop, it plays the role of `prepare.py`: the fixed evaluation metric that the agent optimizes against.

```
training-recipes autoresearch:
  prepare.py (fixed)  →  defines val_bpb        →  $0, seconds
  train.py (editable) →  agent edits recipe      →  5-min GPU experiment

coding-agent autoresearch:
  learned-verifier (fixed)  →  defines patch_quality  →  $0.029, seconds
  agent config (editable)   →  agent edits harness     →  N patches per experiment
```

Instead of running Docker containers to check if patches pass tests ($0.50+/instance, minutes), the cascade verifier gives quality signal for $0.029/instance in seconds. This makes the Karpathy autoresearch loop viable for coding agent optimization — fast iteration, cheap evaluation, autonomous improvement.

Based on [Shopify's fine-tuned Flow agent](https://shopify.engineering/fine-tuning-agent-shopify-flow) weekly retrain flywheel and Karpathy's autoresearch-colab.

## The Evaluation Stack

The learned-verifier provides three tiers of evaluation, each trading cost for accuracy:

| Tier | Verifier | Cost | Latency | AUC | Role in autoresearch |
|------|----------|------|---------|-----|---------------------|
| 0 | RF classifier (behavioral) | $0.000 | <1ms | 0.696 | Fast filter — reject obvious failures, accept obvious passes |
| 1 | Multiprompt (2-prompt Sonnet) | $0.029 | ~10s | 0.806 | Primary evaluation — scores RF-uncertain patches |
| 2 | Debate (advocate+challenger+judge) | $0.094 | ~30s | 0.765 | Tiebreaker — resolves remaining uncertain cases |

**Full cascade**: RF → multiprompt → debate resolves 96.8% of instances at 70.2% accuracy, ~$0.029/instance average.

The cascade bootstraps itself for new models via the **flywheel**: content verifier labels train the RF, RF handles more over iterations, content verifier gets called less. After ~200 labeled traces, the RF is production-ready.

## Components

### 1. Compute
- **Harness host**: Any machine running the coding agent (OpenHands, SERA, OpenCode, etc.)
- **Verifier overlay**: `pip install learned-verifier` — adds evaluation to existing harness
- **GPU**: Not required for verification — RF is CPU (<1ms), LLM calls are API
- **API**: Bedrock or Anthropic credentials for Tier 1/2 (content verifiers)
- **RF-only mode**: $0, no API — Tier 0 alone for rapid iteration during harness development

### 2. Codebase

The learned-verifier is installed as a library overlay on the coding agent harness:

```
coding-agent-harness/          ← the thing being optimized
├── agent_config.yaml          ← AGENT-EDITABLE: prompts, tools, budget, strategy
├── run_benchmark.py           ← launches agent on SWE-bench instances
└── ...

learned-verifier/              ← the evaluation layer (like prepare.py)
├── src/learned_verifier/
│   ├── cascade.py             ← Cascade.default(provider="bedrock") — the evaluator
│   ├── verifier.py            ← Verifier ABC, BehavioralVerifier, ContentVerifier
│   ├── classifiers/rf_verifier.py  ← Tier 0: RF ($0, <1ms)
│   ├── rubrics/multiprompt.py      ← Tier 1: multiprompt ($0.029)
│   ├── rubrics/debate.py           ← Tier 2: debate ($0.094)
│   ├── adapters/                    ← Trace format → TraceInput converters
│   │   ├── openhands.py            ← OpenHands event stream
│   │   ├── claude_otel.py          ← Claude Code OTel spans
│   │   └── generic.py              ← Any JSON with field aliases
│   └── metrics.py             ← FIXED: AUC, ECE, precision, evaluate()
├── cli.py                     ← lv-verify command
└── data/
    └── models/                ← Per-model RF weights (bootstrapped by flywheel)
```

**Fixed files** (agent must NOT edit):
- `src/learned_verifier/metrics.py` — evaluation protocol
- `src/learned_verifier/rubrics/` — content verifier prompts (validated at AUC=0.812)
- Adapter implementations — trace format converters
- Gold labels — ground truth for calibration

**Agent-editable files** (the harness, NOT the verifier):
- Agent system prompts, tool definitions, scaffolding
- Budget allocation (tokens, turns, cost caps)
- Tool use strategy (when to read, edit, test, verify)
- Checkpoint/verification insertion points
- Harness-level configuration (parallelism, retry logic, context management)

### 3. Experiment Protocol

The autoresearch loop optimizes the **coding agent harness**, using the learned-verifier as the evaluation function:

- **Primary metric**: `pass_rate` — fraction of patches the cascade accepts with high confidence
- **Secondary metrics**: `cascade_precision` (are accepts actually correct?), `cost_per_patch` (agent + verification cost), `wall_time` (throughput)
- **Calibration check**: Gold-test spot checks every 5 experiments OR when RF resolution rate shifts >5pp from baseline. Shopify's Flow agent showed a 35% activation rate drop at 1% traffic despite benchmark parity — OOD drift from novel harness configs is the primary risk to cascade accuracy.
- **RF retraining trigger**: Retrain RF when the autoresearch loop changes agent behavior significantly (new tool strategy, prompt rewrite, scaffold redesign). As harness configs evolve, behavioral feature distributions shift and the RF's decision boundaries may become stale. This is expected — the flywheel handles it — but each major harness change should be treated as a partial cold start for Tier 0.
- **Batch size for delta detection**: N>=50 instances per experiment to detect a 5pp improvement at 80% power given 70.2% cascade accuracy. Smaller batches risk masking real improvements in evaluation noise.
- **Time budget**: Configurable per experiment — run agent on N instances, evaluate all patches via cascade
- **Loop structure**: Run agent batch → cascade evaluates patches → log metrics → hypothesize harness improvement → edit config → repeat

### 4. The Autoresearch Loop

```
SETUP (once):
  1. Install learned-verifier as overlay on coding agent harness
  2. Configure adapter for trace format (openhands, claude_otel, generic)
  3. Bootstrap RF for this model family:
     - If no RF exists → cold start, Tier 1 evaluates everything ($0.052/patch)
     - After 200 traces → flywheel trains model-specific RF → Tier 0 handles 70%+
  4. Run baseline: agent on N instances with default config → cascade evaluates → record pass_rate

LOOP FOREVER:
  1. READ current harness config and experiment log
  2. HYPOTHESIZE a harness improvement:
     - Prompt engineering (system prompt, tool descriptions, examples)
     - Budget allocation (more turns for hard problems, fewer for easy)
     - Tool strategy (read-before-edit ratio, test frequency, verify checkpoints)
     - Scaffold design (planning phase, reflection, self-verification)
     - Context management (what to include/exclude from agent context)
  3. EDIT the harness config — one hypothesis per experiment
  4. RUN the agent on a batch of SWE-bench instances
  5. EVALUATE all patches through the cascade:
     - Tier 0 (RF): instant, free — resolves ~70% of patches
     - Tier 1 (multiprompt): ~10s, $0.029 — resolves ~20% more
     - Tier 2 (debate): ~30s, $0.094 — resolves remaining ~7%
  6. LOG in structured format:
     === EXPERIMENT N ===
     Hypothesis: <one-line description of harness change>
     Change: <what was modified in agent config>
     Patches evaluated: <count>
     Pass rate: <value> (baseline: <baseline>, delta: <+/- change>)
     Cascade cost: $<total> ($<per_patch>/patch)
     Agent cost: $<total> ($<per_patch>/patch)
     RF resolved: <pct>% | Multiprompt resolved: <pct>% | Debate resolved: <pct>%
     Status: IMPROVEMENT | NO_CHANGE | REGRESSION
     ===
  7. DECIDE: If improvement, keep the change. If regression, revert config.
  8. REPEAT from step 1.
```

### 5. RF Flywheel (Self-Bootstrapping Evaluation)

The RF classifier bootstraps itself for each new model family. This is the Shopify Flow parallel — Shopify fine-tuned Qwen3-32B on merchant conversations scored by an LLM judge, retraining weekly on 2x H200 nodes (12 hrs). The evaluation layer gets cheaper as it learns.

```
FLYWHEEL (runs in background alongside autoresearch loop):

  Iteration 0: No RF → all patches through Tier 1 → $0.052/patch
  Iteration 1: 50 silver labels → train RF → RF handles 78% → $0.011/patch
  Iteration 2: 100 silver labels → retrain RF → RF handles 77% → $0.012/patch
  ...
  Iteration 4: 200 silver labels → RF AUC=0.711 (above gold ceiling) → production-ready
  Steady state: RF handles 70%, multiprompt handles 23%, debate handles 4% → $0.029/patch
```

Each autoresearch experiment generates traces that feed back into the RF. The evaluation layer improves as the autoresearch loop runs — fewer API calls per experiment over time.

| Shopify Flow Agent | Verification Flywheel | Match |
|---|---|---|
| Production merchant conversations (free byproduct) | Behavioral features from agent traces ($0 byproduct) | Exact |
| LLM judge scores 4 dimensions (expensive) | Multiprompt content verifier ($0.052/instance) | Exact |
| Fine-tuned Qwen3-32B (cheap at inference) | RF classifier (free, <1ms, trained on silver labels) | Exact |
| Weekly retrain on 2x H200, 12 hrs | Retrain RF after each experiment batch (~200 traces) | Close — different cadence triggers |
| LLM judge routes high-quality → training, quarantine low | Content verifier labels → RF silver labels, uncertain → debate | Exact |
| Tagged slice analysis for gap identification | Per-model RF required (features transfer, thresholds don't) | Close — both segment-aware |
| Workflow activation rate as production guardrail | Gold-test spot checks every 5 experiments for calibration | Exact |
| Python DSL reformulation (+22pp syntactic correctness) | Adapter normalization (consistent TraceInput across harnesses) | Analogous |
| **35% activation drop at 1% traffic despite benchmark parity** | **OOD drift risk: cascade accuracy may degrade on novel harness configs** | **Key shared risk** |

## Why This Matters for Autoresearch

The bottleneck in coding agent autoresearch isn't compute — it's **evaluation cost and latency**.

| Evaluation method | Cost/patch | Latency | Patches/hour | Autoresearch viable? |
|-------------------|-----------|---------|-------------|---------------------|
| Gold tests (Docker) | $0.50+ | 2-5 min | ~20 | Barely — too slow for rapid iteration |
| Human review | $5-50 | hours | ~5 | No — doesn't scale |
| **Cascade verifier** | **$0.029** | **<15s** | **~240** | **Yes — 12x faster, 17x cheaper than gold** |
| RF-only (Tier 0) | **$0.000** | **<1ms** | **~unlimited** | **Yes — free, instant, good for screening** |

With the cascade as evaluation layer, the agent can:
- Run 240 patch evaluations per hour (vs 20 with Docker tests)
- Iterate on harness design with 30-second feedback loops (RF-only screening)
- Reserve expensive gold tests for periodic calibration, not every experiment
- Bootstrap evaluation for new model families in ~200 traces

This is the same insight as training-recipes: `prepare.py` made GPT-2 experiments viable by defining a cheap, fast metric. The cascade makes coding agent experiments viable by replacing Docker test execution with learned verification.

## Demo Dataset: CoderForge-Preview

SWE-bench Lite has known representativeness issues: 32.67% solution leakage, 31.08% weak test cases, and our own T10 results showed precision 1.00→0.78 and base rate 12%→51% when moving from Lite to Verified. Instead, we use **CoderForge-Preview** (`togethercomputer/CoderForge-Preview`) — 155K Docker-verified trajectories across 1,655 repos — as the primary dataset for demonstrating and validating the flywheel.

### Why CoderForge

| Property | SWE-bench Lite | CoderForge-Preview |
|----------|---------------|-------------------|
| Size | 300 instances | 155K verified trajectories (51K tasks, 1,655 repos) |
| Languages | Python only (12 repos) | Python (multi-repo, diverse) |
| Gold labels | Weak (31% suspect) | Docker-verified pass/fail per trajectory |
| Behavioral features | Must generate traces ourselves | Pre-existing OpenHands traces (turn count, edits, tokens) |
| Distribution diversity | Narrow (12 OSS repos) | Broad (R2E-Gym + SWE-Smith + SWE-Rebench) |
| Flywheel bootstrap cycles | 1 (300 traces) | 775 (155K / 200 traces per cycle) |

CoderForge's scale and diversity mirror the Shopify parallel: Shopify's flywheel works because production merchant conversations are diverse and continuously generated. CoderForge's 1,655 repos across 51K tasks is the closest open analog to "production diversity" for coding agents.

### Flywheel Demo Protocol

```
PHASE 1 — Cold Start Bootstrap (200 traces):
  1. Sample 200 random CoderForge trajectories (stratified by repo)
  2. Extract behavioral features via OpenHands adapter
  3. Run full cascade (Tier 0 skipped — no RF yet)
  4. Multiprompt labels become silver training data
  5. Train initial RF → measure AUC vs Docker gold labels
  6. Record: cost/patch ($0.052), RF AUC, resolution rate

PHASE 2 — Flywheel Iteration (5 cycles × 200 traces):
  7. For each cycle:
     a. Sample next 200 traces (non-overlapping)
     b. Run cascade WITH RF (Tier 0 → Tier 1 → Tier 2)
     c. Measure: RF resolution rate, cost/patch, cascade accuracy vs Docker labels
     d. Retrain RF on accumulated silver labels
  8. Track convergence: cost/patch should drop from $0.052 → ~$0.029 as RF handles more
  9. Record per-cycle: RF AUC, resolution rate, cost, accuracy

PHASE 3 — OOD Generalization (cross-distribution):
  10. Train RF on CoderForge traces only
  11. Evaluate cascade on SWE-bench Verified (500 instances) — different distribution
  12. Measure accuracy drop (expected: features transfer, thresholds don't)
  13. Retrain RF on 200 SWE-bench traces → measure recovery
  14. This validates the "partial cold start" protocol for new distributions

PHASE 4 — Calibration Experiment (closes the loop):
  15. Simulate 2 harness configs: default OpenHands vs modified (e.g., +checkpoint injection)
  16. Run both on 50 CoderForge tasks each
  17. Cascade ranks configs by pass_rate → pick "winner"
  18. Docker gold tests on same tasks → confirm cascade's ranking matches gold
  19. If cascade and gold agree: flywheel is validated as autoresearch eval layer
  20. If they disagree: characterize the failure mode (FP-heavy? FN-heavy? OOD?)
```

**Estimated cost**: ~$60 total (1,200 cascade evaluations × $0.029 avg + RF training is free). Compare to Docker evaluation of same: ~$600 (1,200 × $0.50).

### CoderForge Adapter

CoderForge trajectories use OpenHands format (str_replace_editor + bash). The existing `openhands` adapter handles this directly:

```python
from learned_verifier.adapters.openhands import from_openhands
from datasets import load_dataset

ds = load_dataset("togethercomputer/CoderForge-Preview", split="train")
for row in ds:
    trace = from_openhands(row["trajectory"])
    gold_label = row["resolved"]  # Docker-verified pass/fail
    result = cascade.verify(trace=trace, problem=row["problem_statement"], diff=row["patch"])
    # Compare result.verdict to gold_label for calibration
```

## Integration with Existing Harnesses

### OpenHands
```python
from learned_verifier.adapters.openhands import from_openhands
from learned_verifier.cascade import Cascade

cascade = Cascade.default(provider="bedrock")
for trajectory in openhands_results:
    trace = from_openhands(trajectory)
    result = cascade.verify(trace=trace, problem=problem_text, diff=patch_diff)
    # result.verdict: ACCEPT | REJECT | UNCERTAIN
    # result.confidence: float [0, 1]
    # result.cost_usd: float
```

### Claude Code (OTel)
```python
from learned_verifier.adapters.claude_otel import from_claude_otel
trace = from_claude_otel(otel_spans)
result = cascade.verify(trace=trace, problem=problem_text, diff=patch_diff)
```

### Any harness (generic adapter)
```python
from learned_verifier.adapters.generic import from_dict
trace = from_dict({
    "total_cost_usd": 0.42,
    "tokens_per_edit": 1847,
    "loop_count": 12,
})
result = cascade.verify(trace=trace, problem=problem_text, diff=patch_diff)
```

### CLI (harness-agnostic)
```bash
# RF-only screening (free, instant)
lv-verify --trace trajectory.json --rf-only

# Full cascade
lv-verify --trace trajectory.json --problem problem.txt --diff patch.diff --provider bedrock

# Batch evaluation for autoresearch
lv-verify --batch results/*.json --problem-dir problems/ --diff-dir patches/ --output metrics.jsonl
```

## Success Criteria

1. **Harness overlay works**: `pip install learned-verifier` + 5 lines of code adds cascade evaluation to any harness
2. **Autoresearch viable**: Agent iterates on harness config with cascade as eval, achieving measurable pass_rate improvements
3. **Flywheel bootstraps**: New model family goes from 0 RF to production-ready RF in 200 CoderForge traces (~$10)
4. **Cost reduction**: Cascade evaluation costs <$0.03/patch vs >$0.50 for gold tests (17x cheaper), validated on CoderForge Docker labels
5. **Evaluation quality**: Cascade accuracy within 5pp of Docker gold labels on CoderForge spot-check calibration
6. **OOD transfer**: RF trained on CoderForge recovers to >90% of in-distribution accuracy after 200-trace partial cold start on SWE-bench Verified
7. **Calibration experiment**: Cascade correctly ranks 2 harness configs on CoderForge tasks, confirmed by Docker gold tests — validates that cascade signal is actionable, not just cheap

## Existing Validation

| Finding | Evidence | Implication for autoresearch |
|---------|----------|------------------------------|
| Cascade resolves 96.8% at 70.2% accuracy | E11 (500 instances) | Reliable eval signal on SWE-bench distribution; OOD accuracy unvalidated |
| RF handles 70% at $0 | E9, E_flywheel | Majority of eval is free after bootstrap |
| Multiprompt AUC=0.812 | E10 (160 instances) | Strong content-level signal on RF-uncertain |
| Silver labels beat gold labels | E_flywheel (AUC 0.779 vs 0.696) | Flywheel produces better RF than test outcomes |
| 200 traces to production RF | E_flywheel convergence | Fast bootstrap for new model families |
| Error tolerant to 1.5x | E_flywheel ablation | Robust even with imperfect content verifier |
| Features transfer, thresholds don't | E6, E7 (69K instances) | Per-model RF required, but same 3 features work everywhere |
| Behavioral RF is exploit-resistant | E_new6 (100% exploit detection) | Safe against BenchJack-style gaming |

## Known Risks

### OOD Drift (Shopify's Lesson)

Shopify's fine-tuned Flow agent scored at benchmark parity but showed a **35% lower workflow activation rate** at 1% production traffic. Real merchants asked out-of-distribution queries the model hadn't seen in training. The same risk applies here: as the autoresearch loop produces novel harness configs, the patches those configs generate may shift out of the cascade's training distribution.

**Mitigations**:
- Gold-test calibration every 5 experiments (detect drift early)
- RF retraining after major harness changes (adapt to new behavioral distributions)
- Track RF resolution rate as a leading indicator (a drop signals distribution shift before accuracy degrades)

### Cascade Accuracy as Eval Signal

70.2% cascade accuracy is sufficient for ranking harness configs, but evaluation noise requires adequate batch sizes. At N=50, a real 5pp improvement is detectable at 80% power. At N=20, the noise floor masks most improvements. The autoresearch loop must resist the temptation to run small batches for speed — false negatives from undersized experiments waste more time than the larger batches cost.

### Adapter Normalization

The RF's 3 behavioral features (cost, tokens_per_edit, loop_count) must be consistently extracted across harness formats. Shopify found that tool naming, response field ordering, and system prompt alignment each influenced fine-tuned model accuracy. Similarly, inconsistent `TraceInput` representations across adapters (openhands, claude_otel, generic) will introduce noise into RF features. Adapter quality is a prerequisite, not an afterthought.

## Non-Requirements
- Replacing gold tests entirely — cascade is for rapid iteration, gold tests for calibration
- Universal model support — RF needs per-model bootstrap (200 traces)
- Perfect accuracy — 70% cascade accuracy enables autoresearch; periodic gold checks catch drift
- Production deployment — this is the evaluation layer for research loops
- SWE-bench Lite as primary benchmark — Lite has known representativeness issues (solution leakage, weak tests); CoderForge-Preview provides broader, higher-quality evaluation data

---

> **Note**: The learned-verifier library and its validation experiments live in
> `/Users/phi/Documents/workbench/learned-verifier/`. This spec defines how to
> use it as a harness evaluation overlay for coding agent autoresearch.
> Demo dataset: `togethercomputer/CoderForge-Preview` (155K trajectories, Docker-verified).
> Related spec: `domains/autoresearch/specs/coderforge-eval.md`.
> Operational artifacts belong in the blueprint directory.
