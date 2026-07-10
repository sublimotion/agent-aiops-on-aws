"""Tests for graders added in Phase 1a/1b dataset diversification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.graders import (
    grade_gsm8k, grade_short_answer, grade_bfcl, grade_mcq,
    grade_mtbench, grade,
)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def test_gsm8k_int_match():
    assert grade_gsm8k("42", "42")


def test_gsm8k_int_match_with_gold_hash():
    """Gold often arrives as 'reasoning... #### 42' from the dataset."""
    assert grade_gsm8k("42", "Step by step blah blah\n#### 42")


def test_gsm8k_float_int_compat():
    assert grade_gsm8k("42.0", "42")


def test_gsm8k_comma_strip():
    assert grade_gsm8k("1234", "1,234")


def test_gsm8k_wrong():
    assert not grade_gsm8k("17", "42")


def test_gsm8k_empty():
    assert not grade_gsm8k("", "42")
    assert not grade_gsm8k("42", "")


# ---------------------------------------------------------------------------
# Short answer (TriviaQA / NQ)
# ---------------------------------------------------------------------------

def test_short_answer_canonical_match():
    assert grade_short_answer("Paris", "Paris")


def test_short_answer_case_insensitive():
    assert grade_short_answer("paris", "Paris")


def test_short_answer_strips_articles():
    assert grade_short_answer("the United States", "United States")


def test_short_answer_strips_punctuation():
    assert grade_short_answer("George Washington.", "George Washington")


def test_short_answer_alias_list():
    """TriviaQA-style: gold is a list of acceptable aliases."""
    aliases = ["George Washington", "Washington", "G. Washington"]
    assert grade_short_answer("Washington", aliases)
    assert grade_short_answer("george washington", aliases)
    assert not grade_short_answer("Lincoln", aliases)


def test_short_answer_dict_form():
    gold = {"canonical": "Earth", "aliases": ["the Earth", "Planet Earth"]}
    assert grade_short_answer("planet earth", gold)
    assert grade_short_answer("Earth", gold)


def test_short_answer_partial_match():
    """'President George Washington' should match 'Washington' (substring)."""
    assert grade_short_answer("President George Washington", "Washington")


# ---------------------------------------------------------------------------
# BFCL
# ---------------------------------------------------------------------------

def test_bfcl_strict_match():
    pred = '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}'
    gold = '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}'
    assert grade_bfcl(pred, gold)


def test_bfcl_wrong_function_name():
    pred = '{"name": "get_temp", "arguments": {"city": "Paris"}}'
    gold = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    assert not grade_bfcl(pred, gold)


def test_bfcl_missing_required_arg():
    pred = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    gold = '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}'
    assert not grade_bfcl(pred, gold)


def test_bfcl_strict_rejects_extras():
    pred = '{"name": "f", "arguments": {"a": 1, "b": 2}}'
    gold = '{"name": "f", "arguments": {"a": 1}}'
    assert not grade_bfcl(pred, gold, strict=True)
    # Loose mode allows extras
    assert grade_bfcl(pred, gold, strict=False)


def test_bfcl_value_int_vs_str():
    """Some workers emit '5' (str), gold has 5 (int). BFCL allows."""
    pred = '{"name": "f", "arguments": {"x": "5"}}'
    gold = '{"name": "f", "arguments": {"x": 5}}'
    assert grade_bfcl(pred, gold)


def test_bfcl_dict_inputs():
    """Both predicted and gold can arrive as already-parsed dicts."""
    pred = {"name": "f", "arguments": {"x": 1}}
    gold = {"name": "f", "arguments": {"x": 1}}
    assert grade_bfcl(pred, gold)


def test_bfcl_function_synonym():
    """BFCL grader accepts 'function' as alias for 'name'."""
    pred = '{"function": "f", "arguments": {"x": 1}}'
    gold = '{"name": "f", "arguments": {"x": 1}}'
    assert grade_bfcl(pred, gold)


def test_bfcl_unparseable():
    assert not grade_bfcl("{not json", "{}")


# ---------------------------------------------------------------------------
# MCQ extended to A-J
# ---------------------------------------------------------------------------

def test_mcq_letter_f_through_j():
    for letter in "FGHIJ":
        assert grade_mcq(letter, letter)


def test_mcq_invalid_letter_outside_aj():
    """K, Z etc. should not match even if they're equal."""
    assert not grade_mcq("K", "K")
    assert not grade_mcq("Z", "Z")


# ---------------------------------------------------------------------------
# MTBench stub
# ---------------------------------------------------------------------------

def test_mtbench_always_false():
    """MTBench has no parseable gold — grader is a stub; reward layer must
    fall back to judge_fn."""
    assert not grade_mtbench("anything", "anything")
    assert not grade_mtbench("", "")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_dispatch_gsm8k():
    assert grade("42", "42", "gsm8k")


def test_dispatch_triviaqa():
    assert grade("Paris", ["Paris", "City of Light"], "triviaqa")


def test_dispatch_bfcl():
    pred = '{"name": "f", "arguments": {}}'
    gold = '{"name": "f", "arguments": {}}'
    assert grade(pred, gold, "bfcl")


def test_dispatch_mmlu_pro():
    assert grade("H", "H", "mmlu-pro")


def test_dispatch_mgsm_routes_to_math():
    """MGSM is multilingual GSM8K. Grader receives the EXTRACTED answer
    (extractor handles \\boxed); grader does math equality."""
    assert grade("42", "42", "mgsm")


def test_dispatch_mtbench_stub():
    assert not grade("anything", "anything", "mtbench")


def test_dispatch_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        grade("x", "y", "unknown_xyz")
