---
name: verification-cascade
description: Evaluate coding agent patches using the learned-verifier cascade (RF + multiprompt + debate). Use when user says "verify patch", "evaluate patch quality", "run cascade", "check if patch is correct", "batch evaluate traces", "bootstrap RF", "train verifier", or "flywheel". Do NOT use for writing or editing the verification rubrics themselves, running gold Docker tests, or modifying the cascade architecture. This skill is for USING the verifier as an evaluation overlay, not developing it.
---

# Verification Cascade Skill

Harness evaluation overlay that replaces expensive Docker test execution ($0.50+/instance, minutes) with cheap, fast patch quality signals ($0.029/instance, seconds). The learned-verifier is the `prepare.py` equivalent for coding agent autoresearch -- the fixed evaluation metric that the agent optimizes against.

## When to Use This Skill

- Evaluating whether a coding agent's patch is correct (without running tests)
- Batch-evaluating patches from a benchmark run (SWE-bench, CoderForge)
- Bootstrapping an RF classifier for a new model family (flywheel)
- Using patch quality as a reward signal for RL or autoresearch loops
- Screening patches during harness development (RF-only, free, instant)
- Integrating verification into an existing coding agent harness

## The Cascade

Three tiers, each trading cost for accuracy. Each tier fires only if the previous returns UNCERTAIN.

| Tier | Verifier | Cost | Latency | AUC | When it fires |
|------|----------|------|---------|-----|---------------|
| 0 | RF classifier (behavioral) | $0.000 | <1ms | 0.696 | Always (if trace available) |
| 1 | Multiprompt (2-prompt Sonnet) | $0.029 | ~10s | 0.806 | RF uncertain |
| 2 | Debate (advocate+challenger+judge) | $0.094 | ~30s | 0.631 | Multiprompt uncertain |

Full cascade resolves 96.8% of instances at 70.2% accuracy, ~$0.029/instance average.

## Codebase

The learned-verifier library lives at `/Users/phi/Documents/workbench/learned-verifier/`.

```
src/learned_verifier/
  cascade.py          # Cascade.default(provider, model) -- the main entry point
  verifier.py         # Verifier ABC, BehavioralVerifier, ContentVerifier
  providers.py        # make_call_fn("bedrock"/"anthropic", model) -- LLM call factories
  cli.py              # lv-verify command
  schemas.py          # TraceInput, VerifierOutput, Verdict (ACCEPT/REJECT/UNCERTAIN)
  classifiers/
    rf_verifier.py    # Tier 0: 4-feature RF
    models/rf_4feature.pkl  # Pre-trained on 500 CoderForge instances
  rubrics/
    multiprompt.py    # Tier 1: diverse multi-prompt framing
    debate.py         # Tier 2: advocate + challenger + judge
    v009_adversarial.py  # Single adversarial rubric (used within multiprompt)
  adapters/
    generic.py        # Dict/JSON with field aliases
    openhands.py      # OpenHands event stream
    claude_otel.py    # Claude Code OTel spans
    sharegpt.py       # ShareGPT / Hermes Agent trajectories
```

## CLI Usage

The `lv-verify` command is the primary interface. Requires `pip install -e .` from the learned-verifier repo.

### Single patch verification

```bash
# Content-only (no trace, uses Bedrock Sonnet by default)
lv-verify --problem problem.txt --diff patch.diff

# With RF gate (cheaper -- RF resolves ~70% for free)
lv-verify --problem problem.txt --diff patch.diff --trace trace.json

# Anthropic API instead of Bedrock
lv-verify --problem problem.txt --diff patch.diff --provider anthropic

# Cheaper model tier
lv-verify --problem problem.txt --diff patch.diff --model haiku
```

### RF-only mode (free, no LLM calls)

```bash
# Binary verdict
lv-verify --trace trace.json --rf-only

# Raw probability (for RL reward signal)
lv-verify --trace trace.json --rf-only --reward-only
```

### Batch mode

```bash
# Evaluate N traces, output JSONL
lv-verify --batch traces/*.json --problem-dir problems/ --diff-dir diffs/ --output results.jsonl

# RF-only batch (free, instant)
lv-verify --batch traces/*.json --rf-only --output results.jsonl
```

### Describe cascade

```bash
lv-verify --describe
lv-verify --describe --rf-only
```

### Output format

Compact (default):
```json
{"verdict": "ACCEPT", "confidence": 0.8234, "tier": "rf", "cost": "$0.0000"}
```

Full JSON:
```bash
lv-verify --trace trace.json --rf-only --format json
```

## Python API

### Full cascade

```python
from learned_verifier.cascade import Cascade

cascade = Cascade.default(provider="bedrock", model="sonnet")
result = cascade.verify(trace=trace, problem=problem_text, diff=patch_diff)
# result.verdict: Verdict.ACCEPT | Verdict.REJECT | Verdict.UNCERTAIN
# result.confidence: float [0, 1]
# result.cost_usd: float
# result.tier_name: str ("rf", "multiprompt", "debate")
```

### With adapter (OpenHands example)

```python
from learned_verifier.adapters.openhands import from_openhands
from learned_verifier.cascade import Cascade

cascade = Cascade.default(provider="bedrock")
for trajectory in openhands_results:
    trace = from_openhands(trajectory)
    result = cascade.verify(trace=trace, problem=problem_text, diff=patch_diff)
```

### RF-only (free)

```python
from learned_verifier.classifiers.rf_verifier import RFVerifier
from learned_verifier.adapters.generic import from_dict

rf = RFVerifier()  # loads pre-trained model
trace = from_dict({"total_cost_usd": 0.42, "tokens_per_edit": 1847, "loop_count": 12})
result = rf.verify(trace=trace)
prob = rf.predict_proba(trace)  # raw float for RL reward
```

### Generic adapter (any harness)

```python
from learned_verifier.adapters.generic import from_dict

trace = from_dict({
    "total_cost_usd": 0.42,
    "tokens_per_edit": 1847,
    "loop_count": 12,
    "svg_accepted": False,
})
```

## Adapters

Each adapter converts a harness-specific trace format into `TraceInput` (flat feature vector).

| Adapter | Harness | Import |
|---------|---------|--------|
| `generic` | Any JSON | `from learned_verifier.adapters.generic import from_dict` |
| `openhands` | OpenHands | `from learned_verifier.adapters.openhands import from_openhands` |
| `claude_otel` | Claude Code | `from learned_verifier.adapters.claude_otel import from_claude_otel` |
| `sharegpt` | ShareGPT/Hermes | `from learned_verifier.adapters.sharegpt import from_sharegpt` |

**RF features** (the 4 that matter): `total_cost_usd`, `tokens_per_edit`, `loop_count`, `svg_accepted`. The adapter must populate at least these. Other fields are used by content verifiers but not required for RF-only mode.

## Flywheel Bootstrap (New Model Family)

When you have a new model family (e.g., switching from Claude to Qwen), the RF needs retraining. The flywheel bootstraps this automatically:

### Cold start (no RF)

All patches go through Tier 1 (multiprompt). Cost: $0.052/patch. After collecting ~50 traces with content verifier verdicts, train initial RF.

### Bootstrap loop

```
Iteration 0: No RF         → all through Tier 1    → $0.052/patch
Iteration 1: 50 labels     → RF handles 78%        → $0.011/patch
Iteration 2: 100 labels    → RF handles 77%        → $0.012/patch
...
Iteration 4: 200 labels    → RF AUC=0.711          → production-ready
Steady state: RF handles 70%, multiprompt 23%, debate 4% → $0.029/patch
```

### Training the RF

```python
from learned_verifier.classifiers.rf_verifier import RFVerifier

rf = RFVerifier()
# traces: list of TraceInput, labels: list of 0/1 (from content verifier verdicts)
rf.train(traces, labels)
rf.save("path/to/model.pkl")
```

Or via the experiment script:

```bash
cd /Users/phi/Documents/workbench/learned-verifier
python experiments/train_rf.py --features data/features/your_model.csv --output models/your_model.pkl
```

### Key finding

Silver-label RF (trained on content verifier verdicts) achieves AUC=0.779, **exceeding** the gold-label ceiling of 0.696. Content verifier labels are better training signal than noisy test outcomes.

## Integration with Autoresearch

The learned-verifier is the **evaluation layer** in the Karpathy autoresearch loop:

```
coding-agent autoresearch:
  learned-verifier (fixed)  → defines patch_quality  → $0.029, seconds
  agent config (editable)   → agent edits harness     → N patches per experiment
```

### In the loop

```python
# After agent runs on a batch of instances:
for trace_file in batch_results:
    trace = from_openhands(load_json(trace_file))
    result = cascade.verify(trace=trace, problem=problems[trace.id], diff=diffs[trace.id])
    metrics.append(result)

pass_rate = sum(1 for r in metrics if r.verdict == Verdict.ACCEPT) / len(metrics)
cost = sum(r.cost_usd for r in metrics)
```

### What NOT to edit

The learned-verifier is the fixed evaluation metric. The autoresearch agent must NOT edit:
- Rubric prompts (`rubrics/`)
- Metrics definitions (`metrics.py`)
- Cascade logic (`cascade.py`)
- Adapter implementations (`adapters/`)

The agent edits the **harness config** (system prompts, tool strategy, budget allocation, scaffolding).

## Troubleshooting

### `NotFittedError` or missing RF model

The pre-trained RF model ships at `src/learned_verifier/classifiers/models/rf_4feature.pkl`. If missing:

```bash
cd /Users/phi/Documents/workbench/learned-verifier
python experiments/train_rf_e9.py  # trains on 500 CoderForge instances
```

### `ModuleNotFoundError: learned_verifier`

Install in development mode:

```bash
cd /Users/phi/Documents/workbench/learned-verifier
pip install -e ".[dev]"
```

### Bedrock credential errors

Ensure AWS credentials are configured:

```bash
aws sts get-caller-identity  # should return your account
```

The default provider is `bedrock`. Switch to `anthropic` if you don't have Bedrock access:

```bash
lv-verify --provider anthropic --problem problem.txt --diff patch.diff
```

### RF returns UNCERTAIN for everything

The RF thresholds are `accept >= 0.65`, `reject < 0.35`. If your model family has different behavioral distributions, the RF needs retraining via the flywheel. Check the RF probability distribution:

```bash
lv-verify --batch traces/*.json --rf-only --reward-only --output probs.jsonl
# Inspect the distribution -- if clustered around 0.5, features don't discriminate
```

### Batch mode file matching

Batch mode matches trace files to problem/diff files by filename stem. If `traces/django__django-12345.json` exists, it looks for `problems/django__django-12345.txt` (tries extensions: .txt, .md, .diff, .patch, .json).

### Content verifier returns low confidence

Multiprompt AUC=0.812 is the ceiling. If consistently low confidence:
1. Check that `--problem` includes the full problem statement (not just title)
2. Check that `--diff` is a valid unified diff (not raw file content)
3. Try `--preset full` (4 prompts instead of 2, costs 2x but AUC=0.812 vs 0.806)

## Success Criteria

When using the cascade as an evaluation overlay:

- [ ] RF-only mode returns verdicts without any API calls
- [ ] Batch mode processes N traces and outputs valid JSONL
- [ ] Full cascade with Bedrock returns verdict + confidence + cost
- [ ] Cost per patch < $0.035 on average (with RF gate)
- [ ] Flywheel bootstraps new model RF to AUC > 0.65 within 200 traces
- [ ] Integration adds < 10 lines of code to existing harness
