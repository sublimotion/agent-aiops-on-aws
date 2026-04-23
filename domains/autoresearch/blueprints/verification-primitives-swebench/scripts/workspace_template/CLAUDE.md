# Bug Fix Workflow

You are fixing a GitHub issue in this repository. Follow this workflow:

1. **Understand**: Read the issue description and explore relevant source files to understand the bug.
2. **Edit early**: Make your fix within the first 40% of your effort. Don't over-explore — once you understand the problem, write the fix.
3. **Verify**: After editing, verify your fix using the tools in the `verify/` directory:
   - Generate adversarial tests: `python3 verify/generate_tests.py --diff <diff_file> --problem <problem_file> --output test_patch.py`
   - Run those tests: `python3 verify/run_tests.py --test-file test_patch.py --workspace .`
   - Get an adversarial code review: `python3 verify/adversarial_review.py --diff <diff_file> --problem <problem_file>`
4. **Iterate**: If tests fail or the review finds issues, fix your patch and re-verify.

## Verification Tools

Three tools are available in the `verify/` directory:

### `verify/generate_tests.py`
Generates adversarial test cases designed to break your patch. Catches edge cases and regressions before submission.
```
python3 verify/generate_tests.py --diff my_fix.diff --problem problem.txt --output test_patch.py
```

### `verify/run_tests.py`
Runs a test file against the current workspace. Returns pass/fail per test with details.
```
python3 verify/run_tests.py --test-file test_patch.py --workspace .
```

### `verify/adversarial_review.py`
Adversarial code review using a 5-axis rubric (problem alignment, logic, completeness, scope, safety). Returns a structured verdict.
```
python3 verify/adversarial_review.py --diff my_fix.diff --problem problem.txt
```

## Tips

- To create a diff file: `git diff > my_fix.diff`
- The problem description is in `problem.txt` in this workspace
- You can run the repo's existing tests too: `python3 -m pytest <test_file> -x`
- If generate_tests or adversarial_review fails, it may be a transient API error — retry once
