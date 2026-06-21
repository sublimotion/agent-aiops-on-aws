#!/usr/bin/env python3
"""
FinanceBench machinery for the CONSENSUS-reward half of E_harness3 (cells E/F).

FinanceBench (patronus-ai/financebench, n=150 open-source) is free-text financial
QA over 10-K filings with human gold answers + an `evidence` field (annotator-
extracted source text). There is NO programmatic oracle — this is the point: the
reward is supplied by an LLM-JUDGE scoring the worker's free-text answer against
the reference (reasoning-equivalence, not string-match). That judge IS the
"verification asymmetry" the consensus regime puts on the table.

Design choices (grounded in carryover lessons):

* Scope worker context to the `evidence` text (the retrieval oracle), exactly as
  E_fin1 scoped FinQA context to `gold_inds`. Keeps inputs small (median ~1.5K
  chars) and models a realistic RAG setup; avoids parsing PDFs.
* Worker answers in ONE call ("think, then ANSWER: <final>"), parsed to a crisp
  final answer the judge scores. No multi-tool loop -> the E_harness2 multi-tool
  Bedrock bug cannot bite here.
* Judge does REASONING-EQUIVALENCE (accepts unit/format/rounding differences),
  because E_fin1 showed financial free-text answers have representation traps
  ("$22.57 billion" vs "22570" vs "22.57B"). The judge gate (judge_gate.py)
  validates the judge DISCRIMINATES (AUC), not merely that it is stable — the
  E_fin1 "engaged-but-not-discriminating" failure mode is the thing to catch.
"""
import json
import os
import re

import dbbench_common as C  # reuse converse(), pricing, cost_usd, REGION

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_finbench(path=None, limit=0):
    path = path or os.path.join(ROOT, "data", "financebench_open_source.jsonl")
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        ev = "\n\n".join(e["evidence_text"] for e in r.get("evidence", []) if e.get("evidence_text"))
        if not ev:
            continue
        out.append({
            "task_id": f"fb_{r['financebench_id']}",
            "question": r["question"],
            "gold": str(r["answer"]).strip(),
            "evidence": ev,
            "company": r.get("company"),
            "doc_name": r.get("doc_name"),
            "question_type": r.get("question_type"),
            "reasoning": r.get("question_reasoning"),
        })
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# Worker — free-text financial QA, evidence-scoped, JIT notes injectable
# ---------------------------------------------------------------------------
_WORKER_SYS = (
    "You are a careful financial analyst. Answer the question using ONLY the "
    "provided evidence excerpts from the company's SEC filing. Show brief "
    "reasoning, then give your final answer on its own last line in the form:\n"
    "ANSWER: <your final answer>\n"
    "Be precise about units (millions/billions, %, $) and include them in the answer."
)

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def _extract_answer(text):
    # take the LAST "ANSWER:" line's content (last in case the model restates)
    matches = list(re.finditer(r"ANSWER:\s*(.+)", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip().splitlines()[0].strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()[:200]


def answer_question(entry, model_key, jit_notes=""):
    sys_prompt = _WORKER_SYS
    if jit_notes:
        sys_prompt += "\n\nLearned guidance from earlier tasks:\n" + jit_notes
    user = (f"Evidence:\n{entry['evidence']}\n\n"
            f"Question: {entry['question']}\n\n"
            "Reason briefly, then end with 'ANSWER: <final answer>'.")
    resp = C.converse([{"role": "user", "content": [{"text": user}]}],
                      model_key=model_key, system=sys_prompt,
                      temperature=0.0, max_tokens=700)
    ans = _extract_answer(resp["text"])
    return {
        "answer": ans,
        "reasoning": resp["text"][:1500],
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "cost_usd": C.cost_usd(resp["input_tokens"], resp["output_tokens"], model_key),
    }


# ---------------------------------------------------------------------------
# LLM-judge — reasoning-equivalence reward (the consensus oracle)
# ---------------------------------------------------------------------------
_JUDGE_SYS = (
    "You are grading a candidate answer to a financial question against a reference "
    "answer. Judge REASONING-EQUIVALENCE: the candidate is CORRECT if it conveys the "
    "same factual answer as the reference, allowing differences in formatting, units "
    "expressed equivalently (e.g. '$1,577 million' == '$1.577 billion' == '1577'), "
    "rounding to comparable precision, and extra explanation. It is INCORRECT if the "
    "core value/conclusion differs, is missing, or contradicts the reference.\n"
    "Output ONLY a JSON object: "
    '{"verdict": "correct"|"incorrect", "confidence": 0.0-1.0, "reason": "<one sentence>"}.'
)


def judge_answer(question, gold, candidate, model_key="haiku", temperature=0.0):
    user = (f"Question: {question}\n\n"
            f"Reference answer: {gold}\n\n"
            f"Candidate answer: {candidate}\n\n"
            "Grade the candidate. Output the JSON object only.")
    resp = C.converse([{"role": "user", "content": [{"text": user}]}],
                      model_key=model_key, system=_JUDGE_SYS,
                      temperature=temperature, max_tokens=200)
    obj = _parse_judge(resp["text"])
    obj["input_tokens"] = resp["input_tokens"]
    obj["output_tokens"] = resp["output_tokens"]
    obj["cost_usd"] = C.cost_usd(resp["input_tokens"], resp["output_tokens"], model_key)
    return obj


def _parse_judge(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    verdict, conf, reason = "incorrect", 0.5, ""
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().lower()
            verdict = "correct" if v == "correct" else "incorrect"
            conf = float(obj.get("confidence", 0.5))
            reason = str(obj.get("reason", ""))[:200]
        except Exception:  # noqa: BLE001
            pass
    return {"verdict": verdict, "is_correct": verdict == "correct",
            "confidence": conf, "reason": reason}
