# E_safe1: Structured Verification Feedback Loop

**Status:** Proposed
**Cost:** ~$5-10
**Priority:** Medium-high
**Source:** SAFEdit (arxiv:2604.25737) — 17.4pp from iterative refinement via Failure Abstraction Layer
**Vault note:** `01_Projects/Learned-Verifier-Experiment/Experiment-Backlog.md` (E_safe1 section)

## Hypothesis

When v009 rejects a patch with **structured diagnostic feedback** (not just "reject"), feeding that feedback to the agent for a targeted retry achieves higher pass rate than best-of-N at equivalent cost.

## Motivation

Our verification cascade is currently a **gate**: accept or reject. SAFEdit showed that converting the verifier into a **feedback loop** — where rejection includes structured diagnostics that feed back to the editor — contributes 17.4pp to task success rate. That's more impact than the architectural decomposition itself (+3.8pp).

The key insight: best-of-N generates N independent attempts (no learning between attempts). Retry with structured feedback gives the agent *specific information* about what went wrong.

## Data Requirements

**Input:**
- n=50 patches from SWE-bench Verified that v009 currently rejects
  - Mix of true negatives (actually bad patches) and false negatives (good patches v009 misses)
- Source: `results/gold_swebench_verified_sonnet.jsonl` filtered to v009 reject verdicts

**Output:**
- Per-patch: {original_verdict, diagnostic_feedback, retry_patch, retry_verdict, gold_label}
- Aggregate: retry_pass_rate, cost_per_conversion, comparison_to_bon_n3

## Design

### Step 1: Modify v009 Output Format

Current v009 output: `{verdict: "REJECT", confidence: 0.8}`

New v009-feedback output:
```json
{
  "verdict": "REJECT",
  "confidence": 0.8,
  "diagnostics": {
    "failed_criteria": ["completeness", "edge_case_handling"],
    "evidence": [
      "Patch modifies `parse_args()` but does not handle the empty-input case described in the issue",
      "No test for the boundary condition in line 42"
    ],
    "suggested_direction": "Add handling for empty input to parse_args() and verify the boundary condition"
  }
}
```

This mirrors SAFEdit's Failure Abstraction Layer (FAL): raw failure signals compressed into structured, actionable diagnostics.

### Step 2: Generate Targeted Retries

For each rejected patch, construct a retry prompt:
```
You previously attempted to fix this issue and produced the patch below.
A verification system rejected your patch for these specific reasons:

[diagnostics.failed_criteria]
[diagnostics.evidence]
[diagnostics.suggested_direction]

Please produce an improved patch that addresses these specific issues.
Your previous patch (for reference):
[original_patch]

Original issue:
[issue_description]
```

Key: the agent sees its PREVIOUS work + SPECIFIC failure reasons. It's not starting fresh.

### Step 3: Re-verify and Compare

1. Run v009 on retry patches
2. Run gold eval on retry patches (where Docker infra available)
3. Compare:

| Strategy | Cost per attempt | Expected pass rate | Notes |
|----------|------------------|--------------------|-------|
| Single shot | 1 gen ($0.02) + 1 verify ($0.03) = $0.05 | Baseline | Current approach |
| Retry with feedback | 2 gen + 2 verify = $0.10 | Baseline + X pp | This experiment |
| Best-of-3 | 3 gen + 3 verify = $0.15 | Baseline + Y pp | Standard comparison |
| Best-of-3 + retry on top-1 | 3 gen + 3 verify + 1 retry + 1 verify = $0.18 | Max? | Combined |

### Step 4: Behavioral Feature Analysis

Run Phase 3 behavioral RF on both original and retry patches:
- Do retry patches show different behavioral profiles?
- Does the RF score retries higher than originals (even before gold eval)?
- If yes → behavioral features capture "informed revision" as a signal

## Script Outline

```python
# scripts/run_structured_feedback.py

def generate_v009_diagnostics(patch, issue, rubric_version="v009"):
    """Run v009 with enhanced output format that includes failure diagnostics."""
    # Modify v009 prompt to request structured feedback on rejection
    pass

def generate_retry(agent, issue, original_patch, diagnostics):
    """Generate a targeted retry using original patch + structured diagnostics."""
    pass

def run_experiment(n=50):
    """
    1. Load n rejected patches from gold eval data
    2. Generate v009 diagnostics for each
    3. Generate retry patches
    4. Re-evaluate with v009 + gold eval
    5. Compare to best-of-N baseline
    """
    pass
```

## Success Criteria

- **Retry pass rate > best-of-3 pass rate** at lower cost → feedback loop beats blind sampling
- **Retry pass rate > single-shot + 10pp** → structured feedback has real value
- **Cost per converted patch < best-of-3 cost per converted patch** → economically viable

## Failure Modes

- Agent ignores diagnostics and generates fresh attempt → no advantage over best-of-N
- v009 diagnostics are too vague to be actionable → FAL quality matters
- Retry patches overfit to v009 criteria but fail gold tests → gaming the verifier

## Connection to Other Experiments

- **E5 (Constraint-Guided Verification)**: Pre-hoc constraints (E5) + post-hoc diagnostics (E_safe1) = full-loop verification
- **E4 (Segmental Rewards)**: FAL-style structured failure categorization enriches segment labels
- **E_new5 (Tier 0 Linters)**: Tier 0 can also provide structured feedback, not just gate
- **E_cond3 (Cascade Router)**: Router could learn when to retry vs when to re-route to different agent
