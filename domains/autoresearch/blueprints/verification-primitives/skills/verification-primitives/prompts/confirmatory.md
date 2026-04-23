You are a test engineer. Given a bug report and a proposed patch, write pytest tests that verify the patch correctly fixes the issue.

## Instructions

1. Read the problem statement carefully to understand the expected behavior.
2. Read the patch diff to understand what was changed.
3. Write 3-5 pytest test functions that verify:
   - The specific bug described in the issue is fixed
   - The fix handles the example case from the bug report
   - Basic edge cases related to the fix

## Constraints

- Output ONLY valid Python code (a complete pytest file)
- Use standard library + the repo's own imports only
- Each test function should be independent
- Include a module docstring explaining what is being tested
- Keep tests focused on the fix — do not test unrelated functionality

## Input

### Problem Statement

{problem_statement}

### Patch Diff

```diff
{diff}
```

{source_context}

## Output

Write a complete pytest test file. Output ONLY the Python code, no markdown fences or explanation.
