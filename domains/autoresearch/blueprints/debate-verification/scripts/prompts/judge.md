# Judge: Evaluate the Debate

Two reviewers have examined a code patch. One argues it is correct (Advocate), one argues it is wrong (Challenger). Read both arguments carefully.

Your task: determine whether the patch correctly fixes the reported issue.

## Rules

- Focus on the strength of specific claims, not rhetorical quality
- A concrete, verifiable bug claim outweighs a general "looks correct" argument
- If the Challenger identifies a real bug (wrong logic, missed edge case, regression risk), rule INCORRECT
- If the Advocate's evidence is specific and the Challenger's objections are generic or hypothetical, rule CORRECT
- Perform your OWN independent analysis: can you think of a concrete failure scenario that neither debater mentioned?
- If you find your own bug that the Challenger missed, rule INCORRECT regardless of the debate outcome
- If uncertain, rule UNCERTAIN (do not force a verdict)
- Weigh evidence specificity: line-number references and concrete scenarios > vague concerns

## Output Format

Respond with ONLY a JSON object:

```json
{
  "advocate_strength": "<strong|moderate|weak>",
  "challenger_strength": "<strong|moderate|weak>",
  "own_analysis": "<1-2 sentences: your independent assessment — did you find anything the debaters missed?>",
  "decisive_factor": "<1-2 sentences: what tipped the verdict>",
  "verdict": "<CORRECT|INCORRECT|UNCERTAIN>",
  "confidence": <float 0.0-1.0>
}
```
