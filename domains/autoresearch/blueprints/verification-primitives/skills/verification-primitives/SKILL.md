# Verification Primitives

Three tools for verifying your patch before submission.

## Available Tools

### generate_tests

Generate test cases for your patch. Two modes:

- **confirmatory**: "Write tests that verify this patch correctly fixes the issue"
- **adversarial**: "Write tests designed to break this patch — target edge cases, boundary conditions, and assumptions the patch might have missed"

Call this after writing your patch to get test coverage.

### run_tests

Execute a test file against the current workspace. Returns pass/fail per test with stdout/stderr.

Call this after generating tests to see if they pass.

### adversarial_review

Request an adversarial code review of your patch. An expert reviewer will try to find bugs in your patch. Returns a structured verdict: likely_correct, uncertain, or likely_incorrect.

Call this before submitting to get a second opinion.

## Best Practice

Before submitting your final patch, consider using these tools to validate your work. Patches that pass adversarial tests and adversarial review are more likely to be correct.
