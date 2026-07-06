"""Tests for the new dataset extractors added in Phase 1a/1b dataset diversification.

GSM8K, TriviaQA/NQ short-answer, BFCL function-calling, MMLU-Pro 10-options.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extractors import (
    extract_gsm8k, extract_short_answer, extract_bfcl, extract_mcq, extract,
)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def test_gsm8k_canonical_hash_format():
    raw = "Let me work through it.\n\n3 + 4 = 7\n\n#### 7"
    ans, method = extract_gsm8k(raw)
    assert ans == "7"
    assert method == "gsm8k_hash"


def test_gsm8k_with_thousand_separator():
    raw = "Total: 1,234 dollars.\n#### 1,234"
    ans, _ = extract_gsm8k(raw)
    assert ans == "1234"


def test_gsm8k_falls_back_to_boxed():
    raw = "Reasoning...\n\nFinal: \\boxed{42}"
    ans, method = extract_gsm8k(raw)
    assert ans == "42"
    assert method == "boxed"


def test_gsm8k_falls_back_to_trailing_int():
    raw = "After computing, the answer is 100."
    ans, method = extract_gsm8k(raw)
    assert ans == "100"
    assert method == "trailing_int"


def test_gsm8k_takes_last_hash():
    """Reasoning with intermediate '#### N' should still take the last."""
    raw = "Step 1 yields #### 5\nStep 2 yields #### 12\nFinal: #### 17"
    ans, _ = extract_gsm8k(raw)
    assert ans == "17"


def test_gsm8k_strips_thinking():
    raw = "<think>maybe #### 99</think>\nFinal: #### 42"
    ans, _ = extract_gsm8k(raw)
    assert ans == "42"


# ---------------------------------------------------------------------------
# Short answer (TriviaQA / NQ)
# ---------------------------------------------------------------------------

def test_short_answer_strips_prefix():
    raw = "After research:\nThe answer is George Washington."
    ans, method = extract_short_answer(raw)
    assert ans == "George Washington"
    assert method == "final_line"


def test_short_answer_no_prefix():
    raw = "<thinking>checking facts</thinking>\nParis"
    ans, _ = extract_short_answer(raw)
    assert ans == "Paris"


def test_short_answer_handles_explicit_answer_label():
    raw = "Answer: 1969"
    ans, _ = extract_short_answer(raw)
    assert ans == "1969"


def test_short_answer_empty_after_strip():
    ans, method = extract_short_answer("<thinking>incomplete")
    assert ans == ""
    assert method == "empty_after_strip"


# ---------------------------------------------------------------------------
# BFCL
# ---------------------------------------------------------------------------

def test_bfcl_fenced_json_block():
    raw = (
        "I will call the function:\n"
        "```json\n"
        '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}\n'
        "```"
    )
    ans, method = extract_bfcl(raw)
    assert "get_weather" in ans
    assert method == "json_block"


def test_bfcl_takes_last_json_block():
    """Workers may emit reasoning JSON before the final tool call."""
    raw = (
        "Considering options:\n```json\n{\"name\": \"DRAFT\", \"arguments\": {}}\n```\n"
        "Final call:\n```json\n{\"name\": \"book_flight\", \"arguments\": {\"date\": \"2026-06-01\"}}\n```"
    )
    ans, _ = extract_bfcl(raw)
    assert "book_flight" in ans
    assert "DRAFT" not in ans


def test_bfcl_inline_json_fallback():
    raw = "I'll invoke {\"name\": \"calc\", \"arguments\": {\"x\": 5}}"
    ans, method = extract_bfcl(raw)
    assert "calc" in ans
    assert method == "inline_json"


def test_bfcl_no_match():
    raw = "I'm not sure what to call here."
    ans, method = extract_bfcl(raw)
    assert ans == ""
    assert method == "no_match"


# ---------------------------------------------------------------------------
# MCQ extended to 10 options (MMLU-Pro)
# ---------------------------------------------------------------------------

def test_mmlu_pro_letter_g():
    raw = "Reviewing all 10 options...\n\nG"
    ans, _ = extract_mcq(raw)
    assert ans == "G"


def test_mmlu_pro_letter_j():
    raw = "The answer is J because it's the most precise option."
    ans, _ = extract_mcq(raw)
    assert ans == "J"


def test_mcq_letter_e_still_works():
    raw = "Final answer: E"
    ans, _ = extract_mcq(raw)
    assert ans == "E"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_dispatch_gsm8k():
    ans, _ = extract("#### 7", "gsm8k")
    assert ans == "7"


def test_dispatch_triviaqa():
    ans, _ = extract("Answer: Paris", "triviaqa")
    assert ans == "Paris"


def test_dispatch_bfcl():
    raw = '```json\n{"name": "f", "arguments": {}}\n```'
    ans, _ = extract(raw, "bfcl")
    assert "name" in ans


def test_dispatch_mmlu_pro():
    ans, _ = extract("\nH", "mmlu-pro")
    assert ans == "H"


def test_dispatch_mtbench_passthrough():
    raw = "<thinking>thinking</thinking>\nThis is a chat response."
    ans, method = extract(raw, "mtbench")
    assert "chat response" in ans
    assert method == "passthrough_for_judge"


def test_dispatch_unknown_dataset_raises():
    import pytest
    with pytest.raises(ValueError):
        extract("x", "unknown_benchmark_xyz")
