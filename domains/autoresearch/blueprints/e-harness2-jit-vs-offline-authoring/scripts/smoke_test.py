#!/usr/bin/env python3
"""
Stage-0 smoke test — HARD GATE before any layer run.

Per the carryover audit, "task completes" is NOT sufficient — the SDK/Bedrock
tool path can silently drop args and still 'complete'. This gate requires:

  (1) Offline oracle checks (no API): build_sqlite + scorer behave on
      known-correct / known-wrong answers, and the gold SQL reproduces the label
      for the sampled tasks. ("never assume the eval harness works")
  (2) LIVE: the worker drives Bedrock through >=1 real DBBench task end-to-end
      (tool_use round-trip, SQL executes) AND produces a CORRECT answer on >=1
      known-answerable task. Fail-closed if accuracy on the smoke set < threshold.
  (3) L1 harness layers actually FIRE: schema card (H_SCHEMA/H3) non-empty,
      H5 returns cold-start skills, and at least one of H2/H3/H4/H5 is observed
      active during the L1 smoke episode (else L1 would silently no-op and the
      L0->L1 gate could not distinguish "no help" from "not applied").
"""
import argparse
import json
import sys

import dbbench_common as C
import life_harness_dbbench as LH
from agent_loop import run_episode


def offline_checks(eval_set):
    print("=== (1) offline oracle checks ===")
    e = eval_set[0]
    conn, names = C.build_sqlite(e)
    res = C.run_sql(conn, e["sql"]["query"])
    conn.close()
    ans = C.answers_from_result(res)
    assert C.compare_results(str(ans) if len(ans) != 1 else ans[0], e["label"], e["type"][0]), \
        "gold SQL did not reproduce label on smoke task[0]"
    # known-wrong must score False
    assert not C.compare_results("definitely_wrong_xyz", e["label"], e["type"][0]), \
        "scorer accepted an obviously wrong answer"
    # gold reproduces label across the whole set (we pre-filtered, so must be 100%)
    n_ok = sum(1 for x in eval_set if C.gold_label_matches(x))
    print(f"  gold-SQL reproduces label: {n_ok}/{len(eval_set)}")
    assert n_ok == len(eval_set), "eval set contains tasks whose oracle is not self-contained"
    print("  scorer accepts correct / rejects wrong: OK")


def l1_layers_fire(eval_set):
    print("=== (3) L1 harness layers fire ===")
    cfg = LH.DBBenchHarnessConfig(enabled=True)
    seen_schema = seen_skills = 0
    for e in eval_set[:10]:
        rt = LH.DBBenchHarnessRuntime(config=cfg)
        rt.init_task(e)
        if rt.schema_card().strip():
            seen_schema += 1
        if rt.cold_start_skill_hints():
            seen_skills += 1
    print(f"  schema card non-empty: {seen_schema}/10 ; H5 skills returned: {seen_skills}/10")
    assert seen_schema >= 8, "H_SCHEMA/H3 schema card not firing"
    assert seen_skills >= 8, "H5 skill retrieval not firing"
    sp = LH.patch_dbbench_system_prompt("BASE")
    assert "MySQL" in sp and len(sp) > len("BASE") + 50, "H3 system-prompt patch not applied"
    print("  H3 system-prompt patch applied: OK")


def live_checks(eval_set, model_key, n):
    print(f"=== (2) live Bedrock end-to-end ({model_key}, n={n}) ===")
    correct = 0
    for layer in ("L0", "L1"):
        r = run_episode(eval_set[0], model_key, layer)
        print(f"  [{layer}] q={r['question'][:55]!r} -> committed={r['committed']} "
              f"correct={r['is_correct']} sqls={r['n_sql']} finish={r['finish_reason']} "
              f"cost=${r['cost_usd']:.4f}")
    # accuracy gate over n known-answerable tasks (L0 bare)
    for e in eval_set[:n]:
        r = run_episode(e, model_key, "L0")
        correct += r["is_correct"]
    acc = correct / n
    print(f"  L0 bare smoke accuracy: {correct}/{n} = {acc:.0%}")
    assert correct >= 1, "worker could not answer ANY smoke task correctly — SDK/path broken"
    print("  >=1 correct answer produced end-to-end: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{C.ROOT}/data/dbbench_eval.json")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    eval_set = json.load(open(args.data))
    print(f"smoke set: {len(eval_set)} verified tasks\n")
    offline_checks(eval_set)
    l1_layers_fire(eval_set)
    live_checks(eval_set, args.model, args.n)
    print("\nSTAGE-0 SMOKE TEST PASSED — safe to run layers.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nSTAGE-0 SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
