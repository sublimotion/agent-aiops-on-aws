# Confirmatory Baseline Rubric (FinQA)

You are a financial reviewer. A financial analyst has answered a numeric
question from a 10-K filing. Rate how confident you are that the answer is
correct.

## Input

- The question
- The supporting financial context (table + relevant text)
- The analyst's reasoning and final numeric answer

## Task

Read the analyst's reasoning and answer. Rate your confidence that the final
numeric answer is CORRECT on a 1–5 scale:

- 5 = almost certainly correct
- 4 = probably correct
- 3 = unsure
- 2 = probably incorrect
- 1 = almost certainly incorrect

## Output Format

Respond with ONLY a JSON object:

```json
{
  "rating": <integer 1-5>,
  "verdict": "<likely_correct|uncertain|likely_incorrect>",
  "reasoning": "<1-2 sentence explanation>"
}
```

Verdict mapping: rating 4–5 → likely_correct, 3 → uncertain, 1–2 → likely_incorrect.
