# Skill Testing Framework

Tests for validating Claude Code skills following Anthropic's testing guidelines.

## Test Categories

### 1. Trigger Tests (`test-triggers.sh`)
Validates that skills activate for the right prompts and don't activate for wrong ones.

### 2. Functional Tests (`test-functional.sh`)
Validates that skills produce correct outputs given sample inputs.

### 3. Baseline Comparison (`test-baseline.sh`)
Compares token usage and tool call count with vs. without skills enabled.

## Running Tests

```bash
# Run all tests
cd .claude/skills/tests
./run-all.sh

# Run specific category
./test-triggers.sh
./test-functional.sh
```

## Adding Tests

Each skill should have entries in all three test files. Follow the pattern in existing tests.
