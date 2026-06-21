#!/usr/bin/env python3
"""
JIT harness authoring for the DBBench half of E_harness3 (cells A/B/C/D).

Generalizes E_harness2's jit_authoring along a SECOND axis — reward VISIBILITY —
while keeping E_harness2's first axis (authoring LOCUS: self vs external) intact:

  axis 1  LOCUS      : self (worker reflects)  | external (separate agent observes)
  axis 2  REWARD     : visible (sees pass/fail) | withheld (blind to pass/fail)

  cell A = self  + visible   (== E_harness2 L2, LOADED, not re-run here)
  cell C = ext   + visible   (== E_harness2 L3, LOADED, not re-run here)
  cell B = self  + withheld  (NEW)
  cell D = ext   + withheld  (NEW)

The reward-visible digest (A/C) is byte-for-byte E_harness2's: it states the gold
label and tags the committed answer "(WRONG)" — i.e. it hands the author the reward.
The WITHHELD digest (B/D) strips every success signal the spec enumerates and is the
subject of the Stage-0 leak-audit HARD GATE (`leak_audit.py`):

  * NO gold label.
  * committed answer shown NEUTRALLY ("Agent committed:"), never "(WRONG)".
  * NO is_correct / no pass/fail flag.
  * the author is invoked on EVERY task on a reward-INDEPENDENT schedule (see
    run_dbbench_cell.py) — so the *invocation itself* leaks nothing (the single
    biggest residual leak in E_harness2, where the author fired only on failures).

Under withholding the author must SELF-GATE: decide from the trajectory alone
whether a general intervention is warranted (else emit {"skip": true}). That
inference — "which trajectories reveal a likely problem" — is the whole point of
the withheld regime; it is NOT a leak (it is reasoning, not an oracle read).

Intervention vocabulary is held FIXED across all cells (contract | action-guard |
skill), so only LOCUS and REWARD-VISIBILITY vary.
"""
import json
import re

import dbbench_common as C

MAX_NOTES_RENDERED = 6
MAX_STATE_CHARS = 1800  # ≈ MAX_STATE_TOKENS cap on injected intervention state

INTERVENTION_TYPES = ["contract", "action-guard", "skill"]

_AUTHOR_INSTRUCTIONS = """You are improving the runtime HARNESS around a frozen SQL agent on a DBBench task. You cannot retrain the model or change the database. You can only author a short, reusable INTERVENTION that will be shown to the agent on FUTURE similar tasks to prevent a failure class.

Choose exactly ONE intervention type:
- "contract": make an implicit task/DB constraint explicit (e.g. "all columns are TEXT; quote values").
- "action-guard": a rule that catches a specific malformed action (e.g. "wrap identifiers with spaces in backticks").
- "skill": a reusable procedure for a task pattern (e.g. "for 'total per X', GROUP BY X and submit only the counts, not the (label,count) pairs").

Rules:
- Keep the note under 30 words, imperative, GENERAL to the failure class (not this row's specific values).
- Set "applies_to" to the task type it should trigger on, or "all".
- Output ONLY a JSON object: {"failure_type": "...", "intervention_type": "...", "applies_to": "...", "note": "..."}.
"""

_WITHHELD_SUFFIX = """
IMPORTANT: you are NOT told whether this task succeeded or failed, and you are NOT given the correct answer. You must judge from the trajectory alone whether it reveals a likely, GENERAL problem worth a reusable intervention. If the trajectory looks fine or any problem is too task-specific to generalize, output exactly {"skip": true} instead of an intervention.
"""


def _trajectory_digest(ep, reward_visible):
    """Build the trajectory summary handed to the author.

    reward_visible=True  -> E_harness2's digest (states gold label, tags WRONG).
    reward_visible=False -> withheld: no label, no WRONG tag, no pass/fail.
    """
    lines = [f"Question: {ep['question']}",
             f"Task type: {ep['type']}"]
    if reward_visible:
        lines.append(f"Gold answer: {ep['label']}")
        lines.append(f"Agent committed (WRONG): {ep['committed']}")
    else:
        lines.append(f"Agent committed: {ep['committed']}")
    lines.append(f"SQL attempts: {json.dumps(ep['sqls'][:6])}")
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
    if obj.get("skip") is True:
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


def _author(ep, model_key, reward_visible, role):
    """One authoring call. role in {"self","external"}."""
    instr = _AUTHOR_INSTRUCTIONS
    if not reward_visible:
        instr += _WITHHELD_SUFFIX
    if role == "self":
        framing = ("\nYou are the SAME agent that just produced this trajectory. "
                   "Reflect on it:\n\n")
    else:
        framing = ("\nYou are an EXTERNAL verifier observing ANOTHER agent's "
                   "trajectory. Diagnose what may have gone wrong and author an "
                   "intervention to fix the failure CLASS for future tasks:\n\n")
    prompt = instr + framing + _trajectory_digest(ep, reward_visible)
    resp = C.converse([{"role": "user", "content": [{"text": prompt}]}],
                      model_key=model_key, temperature=0.0, max_tokens=400)
    iv = _parse_intervention(resp["text"])
    return iv, resp["input_tokens"], resp["output_tokens"]


def author_self(ep, model_key, reward_visible=True):
    return _author(ep, model_key, reward_visible, "self")


def author_external(ep, verifier_key, reward_visible=True):
    return _author(ep, verifier_key, reward_visible, "external")


class JitStore:
    """Session-scoped intervention store, capped (carryover: cap L2/L3 state)."""

    def __init__(self):
        self.notes = []  # list of intervention dicts, in authoring order

    def add(self, iv):
        if iv:
            self.notes.append(iv)

    def render(self, entry):
        ttype = entry["type"][0]
        relevant = [n for n in self.notes if n["applies_to"] in ("all", ttype)]
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
