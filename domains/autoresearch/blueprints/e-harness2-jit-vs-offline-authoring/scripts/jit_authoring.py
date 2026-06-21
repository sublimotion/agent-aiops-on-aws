#!/usr/bin/env python3
"""
JIT harness authoring for L2 (self) and L3 (external) — the headline axis.

A JitStore holds short interventions authored at RUNTIME from observed failures
and applied to SUBSEQUENT tasks (the runtime analog of Life-Harness's offline
evolve-then-freeze: L1 evolves on a frozen train split; L2/L3 evolve continuously
on the eval stream). The ONLY difference between L2 and L3 is the AUTHOR:

  L2 self     — the worker model reflects on its OWN failed trajectory.
  L3 external — a SEPARATE verifier agent observes the worker's failed trajectory,
                classifies the primary failure type, and authors the intervention.

This tests the T5 prior (self-critique in generation HURTS, 54%->30%) applied to
harness AUTHORING rather than patch generation: prediction L3 > L2.

Intervention vocabulary is held FIXED across L2/L3 (contract | action-guard | skill),
so only authoring LOCUS varies. Each intervention is tagged with the task type it
applies to (or "all"); the store renders the most recent K relevant ones, capped.
"""
import json
import re

import dbbench_common as C

MAX_NOTES_RENDERED = 6
MAX_STATE_CHARS = 1800  # ≈ MAX_STATE_TOKENS cap on injected intervention state

INTERVENTION_TYPES = ["contract", "action-guard", "skill"]

_AUTHOR_INSTRUCTIONS = """You are improving the runtime HARNESS around a frozen SQL agent that just FAILED a DBBench task. You cannot retrain the model or change the database. You can only author a short, reusable INTERVENTION that will be shown to the agent on FUTURE similar tasks to prevent this failure class.

Choose exactly ONE intervention type:
- "contract": make an implicit task/DB constraint explicit (e.g. "all columns are TEXT; quote values").
- "action-guard": a rule that catches a specific malformed action (e.g. "wrap identifiers with spaces in backticks").
- "skill": a reusable procedure for a task pattern (e.g. "for 'total per X', GROUP BY X and submit only the counts, not the (label,count) pairs").

Rules:
- Keep the note under 30 words, imperative, GENERAL to the failure class (not this row's specific values).
- Set "applies_to" to the task type it should trigger on, or "all".
- Output ONLY a JSON object: {"failure_type": "...", "intervention_type": "...", "applies_to": "...", "note": "..."}.
"""


def _trajectory_digest(ep):
    lines = [f"Question: {ep['question']}",
             f"Task type: {ep['type']}",
             f"Gold answer: {ep['label']}",
             f"Agent committed (WRONG): {ep['committed']}",
             f"SQL attempts: {json.dumps(ep['sqls'][:6])}"]
    if ep["errors"]:
        lines.append(f"SQL errors seen: {json.dumps(ep['errors'][:3])}")
    lines.append(f"Finish reason: {ep['finish_reason']}")
    return "\n".join(lines)


def _parse_intervention(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not obj.get("note"):
        return None
    it = obj.get("intervention_type", "contract")
    if it not in INTERVENTION_TYPES:
        it = "contract"
    return {
        "failure_type": str(obj.get("failure_type", "unknown"))[:60],
        "intervention_type": it,
        "applies_to": str(obj.get("applies_to", "all"))[:30],
        "note": str(obj["note"])[:240],
    }


def author_self(ep, model_key):
    """L2: the WORKER reflects on its own failed trajectory."""
    prompt = (_AUTHOR_INSTRUCTIONS +
              "\nYou are the SAME agent that just failed. Reflect on your own failure:\n\n" +
              _trajectory_digest(ep))
    resp = C.converse([{"role": "user", "content": [{"text": prompt}]}],
                      model_key=model_key, temperature=0.0, max_tokens=400)
    iv = _parse_intervention(resp["text"])
    return iv, resp["input_tokens"], resp["output_tokens"]


def author_external(ep, verifier_key):
    """L3: a SEPARATE verifier agent observes the worker's failed trajectory."""
    prompt = (_AUTHOR_INSTRUCTIONS +
              "\nYou are an EXTERNAL verifier observing ANOTHER agent's failure. "
              "Diagnose what the agent did wrong and author an intervention to fix "
              "the failure CLASS for future tasks:\n\n" +
              _trajectory_digest(ep))
    resp = C.converse([{"role": "user", "content": [{"text": prompt}]}],
                      model_key=verifier_key, temperature=0.0, max_tokens=400)
    iv = _parse_intervention(resp["text"])
    return iv, resp["input_tokens"], resp["output_tokens"]


class JitStore:
    """Session-scoped intervention store, capped (carryover: cap L2/L3 state)."""

    def __init__(self):
        self.notes = []  # list of intervention dicts, in authoring order

    def add(self, iv):
        if iv:
            self.notes.append(iv)

    def render(self, entry):
        """Render interventions relevant to this entry's task type, most-recent
        first, deduped by note text, capped at MAX_NOTES_RENDERED / MAX_STATE_CHARS."""
        ttype = entry["type"][0]
        relevant = [n for n in self.notes
                    if n["applies_to"] in ("all", ttype)]
        seen, out, total = set(), [], 0
        for n in reversed(relevant):
            key = n["note"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            line = f"- [{n['intervention_type']}] {n['note']}"
            if total + len(line) > MAX_STATE_CHARS or len(out) >= MAX_NOTES_RENDERED:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out)

    def size_chars(self, entry):
        return len(self.render(entry))
