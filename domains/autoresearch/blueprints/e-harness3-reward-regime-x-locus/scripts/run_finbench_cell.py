#!/usr/bin/env python3
"""
Run one FinanceBench cell (E/F) for one worker model — the CONSENSUS-reward half.

  E = self     + consensus reward (LLM-judge)   (NEW)
  F = external + consensus reward (LLM-judge)    (NEW)

This is the structural twin of cells A/C (reward-VISIBLE), but the reward is the
LLM-judge's reasoning-equivalence verdict instead of a programmatic oracle. The
judge IS the verification asymmetry the consensus regime supplies — so the core
hypothesis predicts F > E by the LARGEST margin in the matrix.

Per task:
  1. worker answers (evidence-scoped, JIT notes injected) -> free-text answer.
  2. LLM-judge scores answer vs gold -> reward (correct/incorrect) = Pass@1 signal.
  3. on a judge-INCORRECT verdict, an intervention is authored from the trajectory
     (self = worker; external = separate verifier) and added to the JIT store for
     subsequent tasks. The author SEES the reward (gold + judge verdict) — this is
     the reward-VISIBLE consensus regime (mirrors A/C, not B/D).

Authoring digest is FinanceBench-shaped (question, gold, candidate, judge reason),
with the SAME self/external framing and FIXED intervention vocabulary as the
DBBench cells, so only the regime (consensus vs verifiable) and locus vary.
"""
import argparse
import json
import os
import re

import dbbench_common as C  # converse, cost
import finbench_common as F

MAX_NOTES = 6
MAX_STATE_CHARS = 1800

_AUTHOR_INSTR = """You are improving the runtime guidance around a frozen financial-QA agent on a FinanceBench task. You cannot retrain the model. You can only author a short, reusable INTERVENTION shown to the agent on FUTURE similar questions to prevent a failure class.

Choose exactly ONE intervention type:
- "contract": make an implicit answer-format/units requirement explicit (e.g. "report $ in millions with the unit").
- "action-guard": a rule catching a specific mistake (e.g. "use the value from the cash-flow statement, not the income statement").
- "skill": a reusable procedure for a question pattern (e.g. "for 'capex', read 'purchases of PP&E' from investing activities").

Rules:
- Under 30 words, imperative, GENERAL to the failure class (not this row's specific numbers).
- Set "applies_to" to the question_type it triggers on, or "all".
- Output ONLY a JSON object: {"failure_type":"...","intervention_type":"...","applies_to":"...","note":"..."}.
"""

INTERVENTION_TYPES = ["contract", "action-guard", "skill"]


def _digest(entry, worker_ans, judge):
    return "\n".join([
        f"Question: {entry['question']}",
        f"Question type: {entry['question_type']}",
        f"Gold answer: {entry['gold']}",
        f"Agent answered (judged INCORRECT): {worker_ans}",
        f"Judge reason: {judge.get('reason','')}",
    ])


def _parse(text):
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
    return {"failure_type": str(obj.get("failure_type", "unknown"))[:60],
            "intervention_type": it,
            "applies_to": str(obj.get("applies_to", "all"))[:30],
            "note": str(obj["note"])[:240]}


def author(entry, worker_ans, judge, model_key, locus):
    if locus == "self":
        framing = "\nYou are the SAME agent that just answered. Reflect on the failure:\n\n"
    else:
        framing = ("\nYou are an EXTERNAL reviewer observing ANOTHER agent's answer. "
                   "Diagnose the error and author an intervention for the failure CLASS:\n\n")
    prompt = _AUTHOR_INSTR + framing + _digest(entry, worker_ans, judge)
    resp = C.converse([{"role": "user", "content": [{"text": prompt}]}],
                      model_key=model_key, temperature=0.0, max_tokens=400)
    return _parse(resp["text"]), resp["input_tokens"], resp["output_tokens"]


class NoteStore:
    def __init__(self):
        self.notes = []

    def add(self, iv):
        if iv:
            self.notes.append(iv)

    def render(self, entry):
        qt = entry["question_type"]
        rel = [n for n in self.notes if n["applies_to"] in ("all", qt)]
        seen, out, total = set(), [], 0
        for n in reversed(rel):
            k = n["note"].strip().lower()
            if k in seen:
                continue
            seen.add(k)
            line = f"- [{n['intervention_type']}] {n['note']}"
            if total + len(line) > MAX_STATE_CHARS or len(out) >= MAX_NOTES:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out)

    def size_chars(self, entry):
        return len(self.render(entry))


CELLS = {"E": "self", "F": "external"}


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done[r["task_id"]] = r
            except Exception:  # noqa: BLE001
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--model", required=True, choices=["haiku", "sonnet"])
    ap.add_argument("--judge", default="haiku", choices=["haiku", "sonnet"])
    ap.add_argument("--verifier", default="haiku", choices=["haiku", "sonnet"],
                    help="external authoring model for cell F")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    locus = CELLS[args.cell]
    data = F.load_finbench(limit=args.limit)
    tag = f"{args.cell}_{args.model}"
    if locus == "external":
        tag += f"_v-{args.verifier}"
    out = args.out or f"{F.ROOT}/results/{tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = load_done(out)
    print(f"[{tag}] cell={args.cell} locus={locus} consensus-reward judge={args.judge} "
          f"{len(data)} tasks, {len(done)} done -> {out}")

    store = NoteStore()
    n_correct = n_run = n_authored = 0
    auth_in = auth_out = 0
    fp = open(out, "a")
    for i, entry in enumerate(data):
        tid = entry["task_id"]
        if tid in done:
            r = done[tid]
            if r.get("authored"):
                store.add(r["authored"])
                n_authored += 1
        else:
            notes = store.render(entry)
            w = F.answer_question(entry, args.model, jit_notes=notes)
            j = F.judge_answer(entry["question"], entry["gold"], w["answer"],
                               model_key=args.judge)
            r = {
                "task_id": tid, "cell": args.cell, "locus": locus,
                "regime": "consensus", "model": args.model,
                "question_type": entry["question_type"],
                "question": entry["question"], "gold": entry["gold"],
                "answer": w["answer"], "is_correct": bool(j["is_correct"]),
                "judge_verdict": j["verdict"], "judge_conf": j["confidence"],
                "judge_reason": j["reason"],
                "input_tokens": w["input_tokens"] + j["input_tokens"],
                "output_tokens": w["output_tokens"] + j["output_tokens"],
                "cost_usd": w["cost_usd"] + j["cost_usd"],
            }
            # author on judged-incorrect (reward-visible consensus regime)
            if not r["is_correct"]:
                akey = args.verifier if locus == "external" else args.model
                iv, ai, ao = author(entry, w["answer"], j, akey, locus)
                auth_in += ai
                auth_out += ao
                r["authored"] = iv
                r["auth_input_tokens"] = ai
                r["auth_output_tokens"] = ao
                if iv:
                    store.add(iv)
                    n_authored += 1
            else:
                r["authored"] = None
            r["jit_state_chars"] = store.size_chars(entry)
            r["jit_notes_total"] = len(store.notes)
            fp.write(json.dumps(r) + "\n")
            fp.flush()
        n_run += 1
        n_correct += int(r["is_correct"])
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            print(f"  [{tag}] {i+1}/{len(data)}  acc={n_correct/n_run:.3f}  "
                  f"authored={n_authored}  notes={len(store.notes)}")
    fp.close()

    total_cost = sum(json.loads(l).get("cost_usd", 0) for l in open(out))
    akey = args.verifier if locus == "external" else args.model
    print(f"[{tag}] DONE  Pass@1={n_correct}/{n_run}={n_correct/n_run:.3f}  "
          f"cost=${total_cost:.2f}  author_cost=${C.cost_usd(auth_in,auth_out,akey):.3f}  "
          f"interventions={n_authored}")


if __name__ == "__main__":
    main()
