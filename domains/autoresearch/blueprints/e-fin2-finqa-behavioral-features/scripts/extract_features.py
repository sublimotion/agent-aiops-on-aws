#!/usr/bin/env python3
"""
E_fin2 — extract behavioral/process features from FinQA reasoning programs.

Data-only, local. Reuses E_fin1's generated-answer sample + exact-match labels.
No GPU, no API generation. Joins each E_fin1 answer (by id) back to the original
czyssrs/FinQA dev.json to recover qa.program / qa.steps.

Two feature families (kept separate so the report can interpret them honestly):

  GOLD-PROGRAM STRUCTURAL (prog_*) — extracted from the *task's* gold DSL
  derivation (qa.program / qa.steps). These describe the structure of the
  required reasoning, the financial analog of "how long/complex is the code
  trajectory". They are properties of the TASK, not the agent's behavior, so
  they are closer to a difficulty proxy than to the Phase-3 behavioral four.

  AGENT-TRAJECTORY BEHAVIORAL (beh_*) — extracted from the agent's actual
  generation (token usage, cost, reasoning verbosity, self-revision markers,
  abstention). These are the *direct* analog of the Phase-3 four
  (beh_total_cost_usd, beh_tokens_per_edit, beh_loop_count, svg_accepted):
  process signal from the agent's own run, not from the gold answer.

Writes results/features.csv with one row per E_fin1 example.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLUEPRINT = HERE.parent
RESULTS = BLUEPRINT / "results"
RESULTS.mkdir(exist_ok=True)

E_FIN1 = BLUEPRINT.parent / "e-fin1-finqa-skill-verifier"
ANSWERS = E_FIN1 / "results" / "agent_answers.jsonl"
VERIFIERS = E_FIN1 / "results" / "verifier_results.jsonl"
FINQA_DEV = Path("/tmp/FinQA/dataset/dev.json")

# DSL op vocabulary observed in the E_fin1 sample (audited): add, subtract,
# multiply, divide, table_average, table_sum. Regex pulls "op(" tokens.
_OP_RE = re.compile(r"([a-z_]+)\(")
# self-revision / reconsideration markers in free-text reasoning (loop_count analog)
_REVISION_MARKERS = [
    "however", "but wait", "actually", "reconsider", "recalculat", "re-calculat",
    "wait,", "correction", "instead", "on second thought", "not the same",
    "revis", "let me redo", "rethink", "i made an error", "scratch that",
]
_ABSTAIN_MARKERS = [
    "cannot be determined", "cannot be calculated", "not provided",
    "insufficient", "unable to determine", "not enough information",
]


def parse_res(res):
    """Parse a FinQA step 'res' string (e.g. '10.94%', '-492', '6.35%') to float."""
    if res is None:
        return None
    s = str(res).strip()
    is_pct = s.endswith("%")
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("%", "").replace("$", "").replace(",", "")
    s = s.replace("(", "").replace(")", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if neg:
        v = -abs(v)
    return v


def program_features(qa):
    """Structural features of the gold reasoning program (task-side)."""
    program = qa.get("program", "") or ""
    steps = qa.get("steps", []) or []

    ops = _OP_RE.findall(program)
    op_count = len(ops)
    op_diversity = len(set(ops))
    has_chain = 1 if "#" in program else 0           # multi-step chaining (#0, #1)
    uses_const = 1 if "const_" in program else 0
    uses_table_op = 1 if any(o.startswith("table_") for o in ops) else 0

    # Intermediate-value sanity (loop/thrash-detection analog):
    # parse each step's 'res', look for sign flips and magnitude blowups across
    # the derivation. Short clean derivations should show little of this.
    interm = [parse_res(s.get("res")) for s in steps]
    interm = [v for v in interm if v is not None]
    sign_flip = 0
    magnitude_blowup = 0.0
    max_abs_interm = 0.0
    if interm:
        signs = [(-1 if v < 0 else (1 if v > 0 else 0)) for v in interm]
        nz_signs = [s for s in signs if s != 0]
        sign_flip = 1 if len(set(nz_signs)) > 1 else 0
        abs_nz = [abs(v) for v in interm if abs(v) > 1e-9]
        max_abs_interm = max(abs_nz) if abs_nz else 0.0
        if len(abs_nz) >= 2:
            magnitude_blowup = max(abs_nz) / min(abs_nz)

    return {
        "prog_op_count": op_count,
        "prog_op_diversity": op_diversity,
        "prog_has_chain": has_chain,
        "prog_uses_const": uses_const,
        "prog_uses_table_op": uses_table_op,
        "prog_sign_flip": sign_flip,
        "prog_magnitude_blowup": round(magnitude_blowup, 4),
        "prog_max_abs_interm": round(max_abs_interm, 4),
    }


def behavioral_features(ans):
    """Agent-trajectory process features (the Phase-3 behavioral analog)."""
    reasoning = (ans.get("reasoning") or "")
    raw = (ans.get("agent_answer_raw") or "")
    low = reasoning.lower()
    raw_low = raw.lower()
    words = reasoning.split()
    n_words = len(words)
    out_tok = ans.get("output_tokens", 0) or 0
    in_tok = ans.get("input_tokens", 0) or 0

    revision_count = sum(low.count(m) for m in _REVISION_MARKERS)
    abstain = 1 if any(m in low or m in raw_low for m in _ABSTAIN_MARKERS) else 0
    # token density: output tokens per reasoning word (tokens_per_edit analog —
    # how much "thinking budget" was spent per unit of stated reasoning).
    tokens_per_word = (out_tok / n_words) if n_words > 0 else 0.0
    # number-mention count in reasoning: thrash proxy (more distinct figures
    # quoted -> more numeric juggling).
    num_mentions = len(re.findall(r"\d[\d,]*\.?\d*", reasoning))

    return {
        "beh_output_tokens": out_tok,
        "beh_input_tokens": in_tok,
        "beh_cost_usd": round(ans.get("cost_usd", 0.0) or 0.0, 6),
        "beh_reasoning_words": n_words,
        "beh_tokens_per_word": round(tokens_per_word, 4),
        "beh_revision_count": revision_count,
        "beh_num_mentions": num_mentions,
        "beh_abstain": abstain,
        "beh_latency_ms": ans.get("latency_ms", 0) or 0,
    }


def skill_features(vrow):
    """E_fin1 skill-verifier signals (for the head-to-head). Numeric encodings
    of the confirmatory (1-5 rating) and adversarial 4-call ensemble."""
    adv = vrow.get("adv_verdicts", []) or []
    return {
        "skill_conf_rating": vrow.get("conf_rating", 0) or 0,
        "skill_adv_lc_count": sum(1 for v in adv if v == "likely_correct"),
        "skill_adv_li_count": sum(1 for v in adv if v == "likely_incorrect"),
        "skill_adv_unc_count": sum(1 for v in adv if v == "uncertain"),
        "skill_adv_confident": 1 if vrow.get("adv_confident") else 0,
    }


def main():
    dev = {ex["id"]: ex for ex in json.load(open(FINQA_DEV))}
    answers = [json.loads(l) for l in open(ANSWERS)]
    verifiers = {json.loads(l)["id"]: json.loads(l)
                 for l in open(VERIFIERS)}

    rows = []
    missing = 0
    for a in answers:
        ex = dev.get(a["id"])
        if ex is None:
            missing += 1
            continue
        qa = ex["qa"]
        row = {"id": a["id"], "gold_pass": 1 if a.get("match") else 0}
        row.update(program_features(qa))
        row.update(behavioral_features(a))
        v = verifiers.get(a["id"])
        if v is not None:
            row.update(skill_features(v))
        rows.append(row)

    assert missing == 0, f"{missing} answers failed to join to FinQA dev.json"

    import csv
    cols = list(rows[0].keys())
    out = RESULTS / "features.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows x {len(cols)} cols to {out}")
    print(f"base rate (gold_pass): {sum(r['gold_pass'] for r in rows)}/{len(rows)}")
    print("columns:", cols)


if __name__ == "__main__":
    main()
