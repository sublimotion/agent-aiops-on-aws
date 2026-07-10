"""Unit tests for extractors.

Each test reflects a documented per-worker quirk from spec Gate 0.2b. When a
new worker exhibits a new failure mode in production, add a test here FIRST,
then fix the extractor — never the other way around.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extractors import (
    extract_math, extract_mcq, extract_code, strip_reasoning,
)


# ---------------------------------------------------------------------------
# strip_reasoning
# ---------------------------------------------------------------------------

def test_strip_thinking_block():
    raw = "<thinking>let me think</thinking>\nThe answer is \\boxed{42}."
    assert "thinking" not in strip_reasoning(raw).lower()
    assert "\\boxed{42}" in strip_reasoning(raw)


def test_strip_think_block_glm():
    raw = "<think>step 1</think>\nFinal: \\boxed{x}"
    assert "<think" not in strip_reasoning(raw)


def test_strip_minimax_answer_unwrap():
    raw = "<minimax_thinking>blah</minimax_thinking>\n<minimax_answer>The answer is C</minimax_answer>"
    out = strip_reasoning(raw)
    assert "minimax_answer" not in out
    assert "minimax_thinking" not in out
    assert "answer is C" in out


def test_strip_unclosed_thinking_returns_empty():
    """Truncated mid-thinking response is untrustworthy."""
    raw = "<thinking>this never closes and the answer is \\boxed{99}"
    assert strip_reasoning(raw) == ""


def test_strip_nested_or_repeated_blocks():
    raw = "<thinking>a</thinking>more<thinking>b</thinking>final"
    assert "<thinking>" not in strip_reasoning(raw)


# ---------------------------------------------------------------------------
# Math extraction
# ---------------------------------------------------------------------------

def test_math_takes_last_boxed_not_first():
    """K2 Thinking emits intermediate \\boxed{} in reasoning. Take LAST."""
    raw = ("<think>maybe \\boxed{12} but let me retry. "
           "actually \\boxed{17} no wait.</think>\n"
           "Final answer: \\boxed{42}.")
    ans, method = extract_math(raw)
    # After strip, the only boxed expression remaining is 42
    assert ans == "42"
    assert method == "boxed"


def test_math_boxed_inside_dollar_delimiter():
    raw = "Therefore the answer is $\\boxed{7}$."
    ans, method = extract_math(raw)
    assert ans == "7"


def test_math_dollar_delim_fallback():
    raw = "After computation, $x = 3.14$"
    ans, method = extract_math(raw)
    assert ans == "x = 3.14"
    assert method == "dollar_delim"


def test_math_final_number_fallback():
    raw = "Working through the steps, we get 256."
    ans, method = extract_math(raw)
    assert ans == "256"
    assert method == "final_number"


def test_math_empty_after_strip():
    raw = "<thinking>unclosed"
    ans, method = extract_math(raw)
    assert ans == ""
    assert method == "empty_after_strip"


# ---------------------------------------------------------------------------
# MCQ extraction
# ---------------------------------------------------------------------------

def test_mcq_letter_only_final_line():
    raw = "Looking at the options:\n\nC"
    ans, method = extract_mcq(raw)
    assert ans == "C"
    assert method == "letter_only"


def test_mcq_wrapped_letter():
    raw = "After analysis:\n(B)"
    ans, _ = extract_mcq(raw)
    assert ans == "B"


def test_mcq_answer_is_pattern():
    raw = "The correct answer is D because of the gradient."
    ans, method = extract_mcq(raw)
    assert ans == "D"
    assert method == "answer_is_X"


def test_mcq_strips_reasoning():
    raw = "<thinking>maybe A or B</thinking>\nThe answer is C."
    ans, _ = extract_mcq(raw)
    assert ans == "C"


def test_mcq_takes_last_letter_in_prose():
    raw = "We rejected A and B, then chose D."
    ans, _ = extract_mcq(raw)
    assert ans == "D"


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def test_code_takes_last_python_block():
    """GLM 5 / DeepSeek emit reasoning code blocks before the real answer."""
    raw = (
        "Let me think:\n```python\n# scratch work\nx=1\n```\n"
        "Now the real solution:\n```python\ndef solve(n):\n    return n*2\n```"
    )
    ans, method = extract_code(raw)
    assert "def solve" in ans
    assert "scratch" not in ans
    assert method == "python_block_last"


def test_code_untagged_def_fallback():
    raw = "Here is the function:\n\ndef foo(x):\n    return x + 1\n"
    ans, method = extract_code(raw)
    assert "def foo" in ans
    assert method == "untagged_def"


def test_code_strips_thinking_first():
    raw = "<thinking>def scratch(): pass</thinking>\n```python\ndef real():\n    pass\n```"
    ans, method = extract_code(raw)
    assert "def real" in ans
    assert "scratch" not in ans
