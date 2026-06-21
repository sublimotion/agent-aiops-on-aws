#!/usr/bin/env python3
"""
DBBench agent loop for E_harness2 — one function runs an episode under any layer.

Protocol mirrors Life-Harness's DBBenchTask: the worker is given the SYSTEM_PROMPT
+ question, and two tools (execute_sql, commit_final_answer). It runs SQL against
our SQLite oracle, observes results, and commits a final answer, scored by the
vendored DBResultProcessor.compare_results against the gold label.

Layers:
  L0  bare         — system prompt + question + tools. No harness.
  L1  offline-frozen — VENDORED Life-Harness DBBenchHarnessRuntime, used frozen
                       (H2 SQL gate, H3 prompt/tool patch, H4 monitor, H5 skills),
                       driven through the same hooks task.py uses. No JIT authoring.
  L2  jit-self     — bare base + a session-scoped, JIT intervention store that the
                     WORKER itself appends to after a failed task (from its own
                     trajectory). Capped at MAX_STATE_TOKENS (carryover: cap state).
  L3  jit-external — same JIT store, but a SEPARATE verifier agent observes the
                     worker's failed trajectory, classifies the failure type, and
                     authors the intervention. Only the AUTHOR differs from L2.

L2/L3 authoring happens BETWEEN tasks (cross-task carryover within a model's run),
which is the runtime analog of Life-Harness's offline evolve-then-freeze: L1 evolves
on a train split and freezes; L2/L3 evolve continuously on the eval stream itself.
"""
import json
import re

import dbbench_common as C
import life_harness_dbbench as LH

SYSTEM_PROMPT = (
    "I will ask you a question, then you should help me operate a MySQL database "
    "with SQL to answer the question.\n"
    "You have to explain the problem and your solution to me and write down your thoughts.\n"
    "After thinking and explaining thoroughly, every round you can choose to operate "
    "or to answer with the two specific tools provided.\n"
    "If you should execute a SQL query, use the `execute_sql` function. Your SQL should be in one line.\n"
    "Every time you can only execute one SQL statement.\n"
    "If you are done operating, and you want to commit your final answer, then use the "
    "`commit_final_answer` function.\n"
    "DO NOT use this tool unless you are sure about your answer. I expect an accurate and correct answer.\n"
    "Your answer must be exactly the same as the correct answer.\n"
    "You should always use the tools provided to submit your answer.\n"
    "Your input will be raw MySQL response, you have to deal with it by yourself."
)

TOOLS = [
    {"toolSpec": {
        "name": "execute_sql",
        "description": "Execute one SQL statement against the database and return the raw result.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"sql": {"type": "string",
                                                        "description": "A single SQL statement, one line."}},
                                 "required": ["sql"]}}}},
    {"toolSpec": {
        "name": "commit_final_answer",
        "description": "Commit your final answer. Provide the answer value(s) as a list of strings.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"answers": {"type": "array", "items": {"type": "string"},
                                                            "description": "The final answer value(s)."}},
                                 "required": ["answers"]}}}},
]

MAX_ROUNDS = 12
MAX_STATE_TOKENS = 1500  # carryover: cap L2/L3 intervention state (≈ words*1.3)


# ---------------------------------------------------------------------------
# Tool-config builders (H3 patches tool descriptions for L1)
# ---------------------------------------------------------------------------
def _tools_for(layer, runtime=None):
    if layer == "L1" and runtime is not None:
        # H3: patch tool descriptions. Convert to the {function:{}} shape H3
        # expects, patch, then convert back to Bedrock toolSpec.
        fns = [{"function": {"name": t["toolSpec"]["name"],
                             "description": t["toolSpec"]["description"]}} for t in TOOLS]
        patched = LH.patch_dbbench_tool_descriptions(fns)
        out = []
        for t, p in zip(TOOLS, patched):
            tt = json.loads(json.dumps(t))
            tt["toolSpec"]["description"] = p["function"]["description"]
            out.append(tt)
        return out
    return TOOLS


def _system_for(layer, entry, runtime, jit_store):
    sys = SYSTEM_PROMPT
    schema_card = ""
    cold = []
    if layer == "L1" and runtime is not None:
        sys = LH.patch_dbbench_system_prompt(sys)
        cold = runtime.cold_start_skill_hints()
        if cold:
            sys += "\n\nProcedural skill hints:\n" + "\n".join(f"- {c['text']}" for c in cold)
        schema_card = runtime.schema_card()
    if layer in ("L2", "L3") and jit_store:
        notes = jit_store.render(entry)
        if notes:
            sys += "\n\nLearned interventions from earlier tasks:\n" + notes
    return sys, schema_card


def _user_prompt(entry, schema_card):
    up = ""
    if entry.get("add_description"):
        up += entry["add_description"] + "\n"
    if schema_card:
        up += schema_card + "\n"
    up += "Question: " + entry["description"] + "\n"
    return up


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(entry, model_key, layer, jit_store=None, verbose=False):
    """Run one DBBench episode. Returns a result dict with pass/fail, trajectory
    summary, tokens, cost, and (for L2/L3) the failure signal for authoring."""
    runtime = None
    if layer == "L1":
        cfg = LH.DBBenchHarnessConfig(enabled=True)
        runtime = LH.DBBenchHarnessRuntime(config=cfg)
        runtime.init_task(entry)

    sys_prompt, schema_card = _system_for(layer, entry, runtime, jit_store)
    tools = _tools_for(layer, runtime)
    conn, _ = C.build_sqlite(entry)

    messages = [{"role": "user", "content": [{"text": _user_prompt(entry, schema_card)}]}]
    in_tok = out_tok = 0
    sqls, errors = [], []
    committed_answers = None
    h2_blocks = 0
    finish_reason = "max_rounds"

    for rnd in range(MAX_ROUNDS):
        # H2 force-action consumption (L1)
        if runtime is not None and runtime.force_next_action is not None:
            fa = runtime.force_next_action
            runtime.force_next_action = None
            if fa["name"] == "commit_final_answer":
                committed_answers = (fa.get("arguments") or {}).get("answers", [])
                finish_reason = "h2_force_commit"
                break

        resp = C.converse(messages, model_key=model_key, system=sys_prompt,
                          tools=tools, temperature=0.0, max_tokens=900)
        in_tok += resp["input_tokens"]
        out_tok += resp["output_tokens"]
        messages.append({"role": "assistant", "content": resp["assistant_content"]})

        tool_uses = resp["tool_uses"]
        if not tool_uses:
            # L1 rescue parser
            rescued = None
            if runtime is not None:
                rescued = LH.rescue_tool_call_from_text(resp["text"])
            if rescued:
                tool_uses = [{"id": f"rescue_{rnd}", "name": rescued["name"],
                              "input": rescued.get("arguments", {})}]
            else:
                if runtime is not None:
                    runtime.note_text_only_turn()
                messages.append({"role": "user", "content": [{"text":
                    "No tool call found. Use execute_sql or commit_final_answer."}]})
                continue
        if runtime is not None:
            runtime.reset_text_only_streak()

        # Bedrock requires a toolResult for EVERY toolUse id in the assistant
        # turn. We act on the first tool_use and answer the rest with filler.
        tu = tool_uses[0]
        name, args = tu["name"], tu["input"]

        def _turn(main_result_text=None, extra_text=None):
            """Build a user-turn content list: main toolResult for tu + filler
            toolResults for any other tool_use ids, plus optional extra text."""
            content = []
            if main_result_text is not None:
                content.append({"toolResult": {"toolUseId": tu["id"],
                                                "content": [{"text": main_result_text}]}})
            for other in tool_uses[1:]:
                content.append({"toolResult": {"toolUseId": other["id"], "content": [{"text":
                    "Skipped: execute only ONE tool per turn. Reissue this call next turn if needed."}]}})
            if extra_text:
                content.append({"text": extra_text})
            return content

        if name == "commit_final_answer":
            answers = args.get("answers", [])
            if runtime is not None:
                gate = runtime.gate_commit(answers)
                if gate["action"] == "block" and h2_blocks < 1:
                    h2_blocks += 1
                    messages.append({"role": "user", "content": _turn(
                        "Harness: " + (gate.get("blocked_reason") or
                                       gate.get("recovery_prompt") or "reconsider your answer."))})
                    continue
                answers = gate.get("answers", answers)
            committed_answers = answers
            finish_reason = "committed"
            break

        elif name == "execute_sql":
            sql = args.get("sql", "") if isinstance(args, dict) else ""
            if not sql and isinstance(args, dict) and args:
                sql = list(args.values())[0]
            # H2 pre-validate (L1)
            if runtime is not None:
                chk = runtime.pre_validate_sql(sql)
                if chk["action"] == "block":
                    h2_blocks += 1
                    messages.append({"role": "user", "content": _turn(
                        f"Error: SQL blocked by harness ({chk['blocked_reason']}).")})
                    continue
                if chk["action"] == "force_commit":
                    committed_answers = (chk["force_args"] or {}).get("answers", [])
                    finish_reason = "h2_force_commit"
                    break
                sql = chk["sql"]
            result = C.run_sql(conn, sql)
            sqls.append(sql)
            if "error" in result.lower():
                errors.append(result)

            # L1 post-step monitor + step guidance
            nudge = None
            if runtime is not None:
                runtime.update_state_after_sql(sql, result)
                h4 = runtime.post_step_monitor(remaining_rounds=MAX_ROUNDS - rnd - 1)
                if h4.get("force_action"):
                    runtime.force_next_action = h4["force_action"]
                nudge = h4.get("recovery_prompt") or runtime.step_guidance(rnd, bool(h4.get("recovery_prompt")))
            messages.append({"role": "user", "content": _turn(result, extra_text=nudge)})
        else:
            messages.append({"role": "user", "content": _turn("Unknown tool.")})

    conn.close()

    # Score
    is_correct = False
    if committed_answers is not None:
        cmp = str(committed_answers) if len(committed_answers) != 1 else committed_answers[0]
        is_correct = C.compare_results(cmp, entry["label"], entry["type"][0])

    return {
        "task_id": entry.get("task_id"),
        "type": entry["type"][0],
        "layer": layer,
        "model": model_key,
        "is_correct": bool(is_correct),
        "committed": committed_answers,
        "label": entry["label"],
        "n_sql": len(sqls),
        "n_errors": len(errors),
        "h2_blocks": h2_blocks,
        "finish_reason": finish_reason,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": C.cost_usd(in_tok, out_tok, model_key),
        "sqls": sqls,
        "errors": errors[:3],
        "gold_sql": entry["sql"]["query"],
        "question": entry["description"],
    }
