"""Ternary reward function for RL Conductor.

Reward scheme (from paper):
  0.0 — malformed output (can't parse subtasks/model_id/access_list)
  0.5 — parseable workflow, wrong final answer
  1.0 — correct final answer

Answer checking is domain-specific:
  - MATH: symbolic equivalence via sympy
  - LiveCodeBench: execution-based (run code, check output)
  - GPQA/MMLU: exact match on multiple choice
"""

import re
import ast
from typing import Optional


def parse_conductor_output(text: str) -> Optional[dict]:
    """Parse the Conductor's workflow step output.

    Expected format (Python lists):
        subtasks = ["solve X", "verify Y", ...]
        model_id = [0, 2, 1, ...]
        access_list = [[0], [0, 1], ...]

    Returns dict with keys {subtasks, model_id, access_list} or None if malformed.
    """
    try:
        subtasks_match = re.search(r'subtasks\s*=\s*(\[.*?\])', text, re.DOTALL)
        model_id_match = re.search(r'model_id\s*=\s*(\[.*?\])', text, re.DOTALL)

        if not all([subtasks_match, model_id_match]):
            return None

        # access_list has nested brackets — extract with balanced bracket matching
        access_start = re.search(r'access_list\s*=\s*\[', text)
        if not access_start:
            return None
        bracket_start = access_start.end() - 1
        depth = 0
        i = bracket_start
        while i < len(text):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            return None
        access_list_str = text[bracket_start:i+1]

        subtasks = ast.literal_eval(subtasks_match.group(1))
        model_id = ast.literal_eval(model_id_match.group(1))
        access_list = ast.literal_eval(access_list_str)

        if not (isinstance(subtasks, list) and isinstance(model_id, list) and isinstance(access_list, list)):
            return None
        if not (len(subtasks) == len(model_id) == len(access_list)):
            return None
        if len(subtasks) == 0:
            return None

        # Normalize access_list items to always be lists
        access_list = [a if isinstance(a, list) else [a] if isinstance(a, int) else [] for a in access_list]

        return {"subtasks": subtasks, "model_id": model_id, "access_list": access_list}

    except (SyntaxError, ValueError):
        return None


def check_math_answer(predicted: str, gold: str) -> bool:
    """Check math answer equivalence. Uses string normalization + sympy fallback."""
    pred_clean = predicted.strip().lower().replace(" ", "")
    gold_clean = gold.strip().lower().replace(" ", "")

    if pred_clean == gold_clean:
        return True

    # Extract boxed answer if present
    boxed = re.search(r'\\boxed\{(.*?)\}', predicted)
    if boxed:
        pred_clean = boxed.group(1).strip().lower().replace(" ", "")
        if pred_clean == gold_clean:
            return True

    # Sympy equivalence check
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex

        pred_expr = sympify(pred_clean)
        gold_expr = sympify(gold_clean)
        if simplify(pred_expr - gold_expr) == 0:
            return True
    except Exception:
        pass

    return False


def check_mcq_answer(predicted: str, gold: str) -> bool:
    """Check multiple choice answer (A/B/C/D/E)."""
    pred_letter = re.search(r'\b([A-E])\b', predicted.strip().upper())
    gold_letter = re.search(r'\b([A-E])\b', gold.strip().upper())
    if pred_letter and gold_letter:
        return pred_letter.group(1) == gold_letter.group(1)
    return predicted.strip().upper() == gold.strip().upper()


def compute_reward(
    conductor_output: str,
    final_answer: str,
    gold_answer: str,
    task_type: str = "math",
) -> float:
    """Compute ternary reward for a single rollout.

    Args:
        conductor_output: raw text from the Conductor model (workflow definition)
        final_answer: aggregated answer after workflow execution
        gold_answer: ground truth
        task_type: one of "math", "mcq", "code"

    Returns:
        0.0, 0.5, or 1.0
    """
    parsed = parse_conductor_output(conductor_output)
    if parsed is None:
        return 0.0

    if task_type == "math":
        correct = check_math_answer(final_answer, gold_answer)
    elif task_type == "mcq":
        correct = check_mcq_answer(final_answer, gold_answer)
    elif task_type == "code":
        # Code correctness checked externally (execution sandbox)
        correct = final_answer.strip() == gold_answer.strip()
    else:
        correct = final_answer.strip() == gold_answer.strip()

    return 1.0 if correct else 0.5
