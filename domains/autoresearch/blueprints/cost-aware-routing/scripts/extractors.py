"""Per-dataset answer extractors with reasoning-block stripping.

Adapted from rl-conductor/reward.py. Key changes from the rl-conductor original:
  - strip_reasoning() handles every reasoning-tag flavor we care about,
    including Anthropic structured `thinking` content blocks (handled in
    the worker proxy, not here)
  - extract_math returns the LAST \\boxed{} (not first) — thinking blocks
    contain intermediate boxed expressions on Kimi K2 Thinking / GLM 5
  - extract_code returns the LAST ```python``` block for the same reason
  - All extractors return (answer, method) so we can audit fallback rates
    per-worker per-dataset (Gate 0.2b)
"""
from __future__ import annotations

import re
from typing import Tuple

# Reasoning-block patterns. Order matters — strip outer wrappers first,
# then any inner think tags. All non-greedy, DOTALL.
_REASONING_PATTERNS = [
    re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<minimax_thinking>.*?</minimax_thinking>\s*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<minimax_answer>(.*?)</minimax_answer>", re.DOTALL | re.IGNORECASE),
]

# Heuristic: if any of these tags appear UNCLOSED, the response was truncated
# mid-reasoning and we cannot trust any extracted answer.
_UNCLOSED_PATTERNS = [
    (re.compile(r"<thinking>", re.IGNORECASE), re.compile(r"</thinking>", re.IGNORECASE)),
    (re.compile(r"<think>", re.IGNORECASE),    re.compile(r"</think>", re.IGNORECASE)),
    (re.compile(r"<reasoning>", re.IGNORECASE),re.compile(r"</reasoning>", re.IGNORECASE)),
]


def strip_reasoning(raw: str) -> str:
    """Remove reasoning blocks. Return empty string if any tag is unclosed
    (signals truncation and untrustworthy output)."""
    if not isinstance(raw, str):
        return ""
    for open_re, close_re in _UNCLOSED_PATTERNS:
        opens = len(open_re.findall(raw))
        closes = len(close_re.findall(raw))
        if opens > closes:
            return ""
    out = raw
    # The minimax_answer pattern *captures* the inner content; replace with group 1.
    out = _REASONING_PATTERNS[4].sub(lambda m: m.group(1), out)
    for pat in _REASONING_PATTERNS[:4]:
        out = pat.sub("", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Math (MATH500, AIME, GPQA-numeric)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_DOLLAR_RE = re.compile(r"\$([^$]+)\$")
_FINAL_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:/\-?\d+(?:\.\d+)?)?)\s*[.,!?]?\s*$")


def extract_math(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")

    # Stage 1: LAST \\boxed{...} (intermediate boxed in thinking ignored)
    boxed_matches = list(_BOXED_RE.finditer(text))
    if boxed_matches:
        return (boxed_matches[-1].group(1).strip(), "boxed")

    # Stage 2: LAST $...$ delimited expression on the final non-empty line
    final_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if final_lines:
        dollars = list(_DOLLAR_RE.finditer(final_lines[-1]))
        if dollars:
            return (dollars[-1].group(1).strip(), "dollar_delim")

    # Stage 3: trailing number on final line
    if final_lines:
        m = _FINAL_NUMBER_RE.search(final_lines[-1])
        if m:
            return (m.group(1), "final_number")

    return ("", "no_match")


# ---------------------------------------------------------------------------
# MCQ (MMLU, GPQA-Diamond letter form)
# ---------------------------------------------------------------------------

_MCQ_LINE_RE = re.compile(r"^[\(\s]*([A-J])[\)\s.:,]*$")
_MCQ_INLINE_RE = re.compile(r"\b(?:answer|choice)\s*(?:is|=|:)?\s*[\(\[]?([A-J])\b", re.IGNORECASE)
_MCQ_ANY_LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_mcq(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")

    final_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Stage 1: final line is just a letter (possibly wrapped)
    if final_lines:
        m = _MCQ_LINE_RE.match(final_lines[-1])
        if m:
            return (m.group(1).upper(), "letter_only")

    # Stage 2: "answer is X" pattern in last few lines
    tail = "\n".join(final_lines[-3:]) if final_lines else text
    m = _MCQ_INLINE_RE.search(tail)
    if m:
        return (m.group(1).upper(), "answer_is_X")

    # Stage 3: LAST standalone A-E in the text (most likely the final answer
    # if model emitted it as part of a sentence)
    matches = list(_MCQ_ANY_LETTER_RE.finditer(text))
    if matches:
        return (matches[-1].group(1).upper(), "any_letter_last")

    return ("", "no_match")


# ---------------------------------------------------------------------------
# Code (HumanEval, MBPP, LiveCodeBench)
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


def extract_code(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")

    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        # LAST python block — earlier blocks may be reasoning scratch
        return (blocks[-1].strip(), "python_block_last")

    # Fallback: first def in raw (untagged code)
    def_match = re.search(r"((?:def |class )\w.*)", text, re.DOTALL)
    if def_match:
        return (def_match.group(1).strip(), "untagged_def")

    return ("", "no_match")


# ---------------------------------------------------------------------------
# GSM8K (####  N format)
# ---------------------------------------------------------------------------

_GSM8K_HASH_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_TRAILING_INT_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*[.,!?]?\s*$")


def extract_gsm8k(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")
    # Stage 1: GSM8K canonical "#### N"
    matches = list(_GSM8K_HASH_RE.finditer(text))
    if matches:
        return (matches[-1].group(1).replace(",", ""), "gsm8k_hash")
    # Stage 2: \\boxed (some workers reformat)
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return (boxed[-1].group(1).strip().replace(",", ""), "boxed")
    # Stage 3: trailing integer on final non-empty line
    final_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if final_lines:
        m = _TRAILING_INT_RE.search(final_lines[-1])
        if m:
            return (m.group(1).replace(",", ""), "trailing_int")
    return ("", "no_match")


# ---------------------------------------------------------------------------
# TriviaQA / NQ — short-answer free-form
#
# The grader (graders.normalize_triviaqa_answer + alias matching) does the
# hard work; the extractor just returns the model's stated answer.
# ---------------------------------------------------------------------------

_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:answer|the\s+answer\s+is|final\s+answer)\s*[:=\-]?\s*",
    re.IGNORECASE,
)


def extract_short_answer(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")
    final_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not final_lines:
        return ("", "no_match")
    last = final_lines[-1]
    # Strip "Answer: ..." prefix if present
    stripped = _ANSWER_PREFIX_RE.sub("", last).strip().rstrip(".")
    if stripped:
        return (stripped, "final_line")
    # Last line was just the prefix; back off
    if len(final_lines) >= 2:
        return (final_lines[-2].strip().rstrip("."), "second_to_last")
    return ("", "no_match")


# ---------------------------------------------------------------------------
# BFCL — JSON tool-call extraction
#
# Workers emit a JSON object describing a function call; gold provides
# expected name + required args. We extract the LAST {...} JSON object that
# parses, prefer those wrapped in ```json``` blocks. Grader does AST-equality.
# ---------------------------------------------------------------------------

import json as _json

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL)
# Inline JSON (no fence) — match {...} that contains "name" or "function"
_INLINE_JSON_RE = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)


def extract_bfcl(raw: str) -> Tuple[str, str]:
    text = strip_reasoning(raw)
    if not text:
        return ("", "empty_after_strip")

    # Stage 1: ```json``` fenced block, last one
    blocks = _JSON_BLOCK_RE.findall(text)
    for b in reversed(blocks):
        try:
            _json.loads(b)
            return (b.strip(), "json_block")
        except _json.JSONDecodeError:
            continue

    # Stage 2: inline {...} that parses as JSON, prefer last
    candidates = _INLINE_JSON_RE.findall(text)
    for c in reversed(candidates):
        try:
            obj = _json.loads(c)
            if isinstance(obj, dict) and ("name" in obj or "function" in obj):
                return (c.strip(), "inline_json")
        except _json.JSONDecodeError:
            continue

    return ("", "no_match")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def extract(raw: str, dataset: str) -> Tuple[str, str]:
    """Top-level dispatch by dataset name."""
    ds = dataset.lower()
    if ds in ("math500", "math", "aime25", "aime", "mgsm", "gpqa-numeric"):
        return extract_math(raw)
    if ds in ("gsm8k",):
        return extract_gsm8k(raw)
    if ds in ("mmlu", "mmlu-pro", "mmlu_pro", "gpqa-diamond", "gpqa",
              "arc", "agieval", "quality", "longbench-v2", "longbench_v2",
              "bbh-mcq"):
        return extract_mcq(raw)
    if ds in ("humaneval", "mbpp", "livecodebench", "lcb"):
        return extract_code(raw)
    if ds in ("triviaqa", "nq", "nq-open", "natural-questions"):
        return extract_short_answer(raw)
    if ds in ("bfcl", "bfcl-v3", "function-calling"):
        return extract_bfcl(raw)
    if ds in ("mtbench",):
        # No parseable gold; reward computed by judge_fn upstream.
        return (strip_reasoning(raw), "passthrough_for_judge")
    if ds in ("bbh", "bbh-freeform"):
        # BBH has 23 sub-tasks; some MCQ, some free-form. Default to short answer.
        return extract_short_answer(raw)
    raise ValueError(f"Unknown dataset: {dataset!r}")
