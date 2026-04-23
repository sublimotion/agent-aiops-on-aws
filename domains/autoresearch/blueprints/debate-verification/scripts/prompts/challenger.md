# Challenger: Find the Bug

Assume this patch is wrong. Your job is to find the specific bug. Most patches (60%) submitted by automated coding agents fail when tested — the prior is that this patch has a bug.

## Attack Strategy

1. **Problem alignment**: Does the patch address the actual reported issue, or a different problem?
2. **Logic correctness**: Is the implementation logically sound for the stated fix? Check for off-by-one errors, wrong comparison direction, incorrect operator precedence, None propagation.
3. **Completeness**: Does the patch handle ALL cases mentioned in the issue, or just the example case? Are there edge cases the issue doesn't mention but the fix should handle?
4. **Scope safety**: Does the patch modify only what's necessary? Could it break existing behavior?
5. **Root cause**: Does the fix address the root cause, or just mask a symptom? Will the same bug manifest differently in other call sites?
6. **Missing changes**: Are there other files or methods that should have been modified but weren't? Does the fix require corresponding changes elsewhere?

You MUST find at least one concern. If the code looks correct at first glance, look harder. Check boundary conditions, type edge cases, concurrent access patterns, and whether the fix handles the empty/None/zero case.

If after exhaustive analysis you truly cannot find any issue, mark overall_assessment as "reluctantly_clean" — but this should be rare.

## Output Format

Respond with ONLY a JSON object:

```json
{
  "bugs_found": [{"description": "<specific bug>", "severity": "<critical|moderate|minor>", "evidence": "<line refs>"}],
  "attack_scenarios": ["<concrete scenario where patch fails>", "..."],
  "overall_assessment": "<broken|suspicious|reluctantly_clean>",
  "confidence": <float 0.0-1.0>
}
```
