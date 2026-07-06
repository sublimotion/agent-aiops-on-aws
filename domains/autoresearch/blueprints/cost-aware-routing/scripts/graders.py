"""Per-dataset answer correctness graders.

Adapted from rl-conductor/reward.py:check_math_answer + check_mcq_answer, plus
a new code grader that runs gold tests in subprocess. Each grader returns
bool: True iff predicted matches gold.

Math grading is deliberately strict: exact match after normalization, then
tuple-elementwise, then sympy structural. Anything that requires LLM-judge to
pass goes through `judge.py` separately and is logged with extraction_method='judge'.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _normalize_math_string(s: str) -> str:
    """Normalize LaTeX/math for equality compare. Lifted from rl-conductor."""
    if not isinstance(s, str):
        return ""
    out = s.strip()
    if out.startswith("$") and out.endswith("$"):
        out = out[1:-1]
    if out.startswith("\\(") and out.endswith("\\)"):
        out = out[2:-2]
    out = out.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    out = out.replace("\\left(", "(").replace("\\right)", ")")
    out = out.replace("\\left[", "[").replace("\\right]", "]")
    out = out.replace("\\left\\{", "{").replace("\\right\\}", "}")
    out = out.replace("\\left\\|", "|").replace("\\right\\|", "|")
    for spc in ("\\!", "\\,", "\\;", "\\:"):
        out = out.replace(spc, "")
    out = re.sub(r"\\q?quad\b", "", out)
    out = out.replace("\\cdot", "*").replace("\\times", "*")

    def _frac_repl(m):
        a, b = m.group(1), m.group(2)
        a_paren = a if re.fullmatch(r"\w+|\\\w+", a) else f"({a})"
        b_paren = b if re.fullmatch(r"\w+|\\\w+", b) else f"({b})"
        return f"{a_paren}/{b_paren}"
    for _ in range(3):
        new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _frac_repl, out)
        if new == out:
            break
        out = new

    out = out.replace("\\pi", "pi").replace("π", "pi").replace("Pi", "pi")
    out = re.sub(r"\s*(?:°|degrees?|deg)\s*$", "", out)
    out = out.rstrip(".,;:")
    return re.sub(r"\s+", "", out).lower()


def _split_tuple(s: str):
    s = s.strip()
    while len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        s = s[1:-1].strip()
    if "," not in s:
        return None
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def grade_math(predicted: str, gold: str) -> bool:
    """Strict math grader. LLM-judge fallback handled by caller, not here."""
    if not isinstance(predicted, str) or not isinstance(gold, str):
        return False

    p_norm = _normalize_math_string(predicted)
    g_norm = _normalize_math_string(gold)
    if not p_norm or not g_norm:
        return False
    if p_norm == g_norm:
        return True

    p_parts = _split_tuple(p_norm)
    g_parts = _split_tuple(g_norm)
    if p_parts and g_parts and len(p_parts) == len(g_parts):
        if all(_normalize_math_string(a) == _normalize_math_string(b)
               for a, b in zip(p_parts, g_parts)):
            return True

    # Sympy structural equivalence (best effort; many LaTeX exprs won't parse)
    try:
        from sympy import simplify, sympify
        def _latex_to_sympy(x: str) -> str:
            return re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", x)
        if simplify(sympify(_latex_to_sympy(p_norm)) - sympify(_latex_to_sympy(g_norm))) == 0:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# MCQ — supports A-J (MMLU-Pro has up to 10 options)
# ---------------------------------------------------------------------------

def grade_mcq(predicted: str, gold: str) -> bool:
    """Multiple choice (A-J)."""
    if not predicted or not gold:
        return False
    p = predicted.strip().upper()[:1]
    g = gold.strip().upper()[:1]
    if not p or not g:
        return False
    return p == g and "A" <= p <= "J"


# ---------------------------------------------------------------------------
# GSM8K — numeric equality after comma stripping
# ---------------------------------------------------------------------------

def _gsm8k_normalize(x: str) -> str:
    if not isinstance(x, str):
        return ""
    # GSM8K gold answers come in "...explanation #### 42" form; strip everything before ####
    if "####" in x:
        x = x.rsplit("####", 1)[1]
    return x.strip().replace(",", "").rstrip(".")


def grade_gsm8k(predicted: str, gold: str) -> bool:
    p = _gsm8k_normalize(predicted)
    g = _gsm8k_normalize(gold)
    if not p or not g:
        return False
    # Try numeric compare first (handles "42" vs "42.0")
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p == g


# ---------------------------------------------------------------------------
# TriviaQA / NQ — normalized exact match against gold + alias list
# ---------------------------------------------------------------------------

import string

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_short_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles, strip punct, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def grade_short_answer(predicted: str, gold) -> bool:
    """Match against gold, optionally with a list of aliases.

    `gold` can be:
        - str: single canonical answer
        - list[str]: canonical + aliases (TriviaQA-style)
        - dict: {'canonical': str, 'aliases': list[str]}
    """
    p = _normalize_short_answer(predicted)
    if not p:
        return False
    if isinstance(gold, str):
        candidates = [gold]
    elif isinstance(gold, list):
        candidates = gold
    elif isinstance(gold, dict):
        candidates = [gold.get("canonical", "")] + list(gold.get("aliases", []))
    else:
        return False
    for c in candidates:
        if not c:
            continue
        cn = _normalize_short_answer(c)
        if not cn:
            continue
        if p == cn:
            return True
        # Token-overlap fallback: predicted contains gold OR gold contains predicted
        # (TriviaQA grader behavior — handles "President George Washington" vs "Washington")
        if cn in p or p in cn:
            return True
    return False


# ---------------------------------------------------------------------------
# BFCL — function-calling AST equality
#
# Gold format (BFCL v3): {"name": "fn", "arguments": {...required args...}}
# Predicted parsed from extractors.extract_bfcl as a JSON string.
# Grader checks: function name matches AND all required args present with
# value-equal entries AND no hallucinated args (extras allowed only if BFCL
# rubric permits — strict mode here disallows them by default).
# ---------------------------------------------------------------------------

import json as _json


def grade_bfcl(predicted: str, gold, strict: bool = True) -> bool:
    """gold may be a JSON string or already a dict."""
    if not predicted or not gold:
        return False
    try:
        pred = _json.loads(predicted) if isinstance(predicted, str) else predicted
    except (_json.JSONDecodeError, TypeError):
        return False
    if isinstance(gold, str):
        try:
            gold = _json.loads(gold)
        except _json.JSONDecodeError:
            return False
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False

    # Function name (BFCL uses "name" or "function" inconsistently)
    p_name = pred.get("name") or pred.get("function") or pred.get("function_name")
    g_name = gold.get("name") or gold.get("function") or gold.get("function_name")
    if not p_name or p_name != g_name:
        return False

    p_args = pred.get("arguments") or pred.get("args") or {}
    g_args = gold.get("arguments") or gold.get("args") or {}
    if not isinstance(p_args, dict) or not isinstance(g_args, dict):
        return False

    # All required args present with matching values
    for k, v in g_args.items():
        if k not in p_args:
            return False
        # Loose value equality: stringify both sides (BFCL allows int/str variation)
        if str(p_args[k]).lower() != str(v).lower():
            return False

    if strict:
        # No hallucinated args
        extras = set(p_args.keys()) - set(g_args.keys())
        if extras:
            return False

    return True


# ---------------------------------------------------------------------------
# MTBench — judge-only stub
#
# MTBench has no parseable gold. Caller must pass a judge_fn at the reward
# layer; this grader is a no-op that always returns False (so the reward
# pipeline falls back to judge_fn). Kept as an explicit case to make the
# dispatch table complete.
# ---------------------------------------------------------------------------

def grade_mtbench(predicted, gold) -> bool:
    return False


# ---------------------------------------------------------------------------
# Code (HumanEval / MBPP)
#
# `predicted` is the Python source emitted by the worker (already extracted by
# extractors.extract_code); `gold` is the dataset record's `test` string —
# typically a sequence of `assert` statements that exercise the function.
# We concatenate predicted + gold + entry_point invocation and run in subprocess
# with a 10s timeout. Any AssertionError or other exception → fail.
# ---------------------------------------------------------------------------

def grade_code(predicted: str, gold_test: str, entry_point: str = "", timeout: int = 10) -> bool:
    if not predicted or not gold_test:
        return False
    # If entry_point is provided, append a `check(entry_point)` call (HumanEval style).
    runner = predicted + "\n\n" + gold_test
    if entry_point:
        runner += f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(runner)
            path = f.name
        proc = subprocess.run(
            ["python3", path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade(predicted: str, gold, dataset: str, **kwargs) -> bool:
    ds = dataset.lower()
    if ds in ("math500", "math", "aime25", "aime", "mgsm"):
        return grade_math(predicted, gold)
    if ds in ("gsm8k",):
        return grade_gsm8k(predicted, gold)
    if ds in ("mmlu", "mmlu-pro", "mmlu_pro", "gpqa-diamond", "gpqa",
              "arc", "agieval", "quality", "longbench-v2", "longbench_v2",
              "bbh-mcq"):
        return grade_mcq(predicted, gold)
    if ds in ("humaneval", "mbpp", "livecodebench", "lcb"):
        return grade_code(predicted, gold, entry_point=kwargs.get("entry_point", ""))
    if ds in ("triviaqa", "nq", "nq-open", "natural-questions", "bbh", "bbh-freeform"):
        return grade_short_answer(predicted, gold)
    if ds in ("bfcl", "bfcl-v3", "function-calling"):
        return grade_bfcl(predicted, gold, strict=kwargs.get("strict", True))
    if ds in ("mtbench",):
        return grade_mtbench(predicted, gold)
    raise ValueError(f"Unknown dataset: {dataset!r}")
