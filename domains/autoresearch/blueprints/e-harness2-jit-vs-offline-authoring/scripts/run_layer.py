#!/usr/bin/env python3
"""
Run one layer for one worker model over the verified DBBench eval set.

Incremental checkpointing + resume (carryover audit): results append to
results/<layer>_<model>.jsonl after each task; a re-run skips completed task_ids.

L2/L3 process tasks IN ORDER (the JIT store accumulates across the eval stream).
After a FAILED task, an intervention is authored (self for L2, external for L3)
and added to the store for subsequent tasks. L0/L1 are order-independent.
"""
import argparse
import json
import os

import dbbench_common as C
from agent_loop import run_episode
from jit_authoring import JitStore, author_self, author_external


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
    ap.add_argument("--data", default=f"{C.ROOT}/data/dbbench_eval.json")
    ap.add_argument("--layer", required=True, choices=["L0", "L1", "L2", "L3"])
    ap.add_argument("--model", required=True, choices=["haiku", "sonnet"])
    ap.add_argument("--verifier", default="haiku", choices=["haiku", "sonnet"],
                    help="external authoring model for L3")
    ap.add_argument("--limit", type=int, default=0, help="cap tasks (0=all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eval_set = json.load(open(args.data))
    if args.limit:
        eval_set = eval_set[:args.limit]
    tag = f"{args.layer}_{args.model}"
    if args.layer == "L3":
        tag += f"_v-{args.verifier}"
    out = args.out or f"{C.ROOT}/results/{tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = load_done(out)
    print(f"[{tag}] {len(eval_set)} tasks, {len(done)} already done -> {out}")

    jit = JitStore()
    auth_in = auth_out = n_authored = 0
    fp = open(out, "a")
    n_correct = n_run = 0

    for i, entry in enumerate(eval_set):
        tid = entry["task_id"]
        if tid in done:
            r = done[tid]
            # Resume: rebuild the JIT store from already-authored interventions
            # so cross-task accumulation survives an interrupted run.
            if args.layer in ("L2", "L3") and r.get("authored"):
                jit.add(r["authored"])
                n_authored += 1
        else:
            r = run_episode(entry, args.model, args.layer, jit_store=jit)
            # L2/L3: author an intervention from a failed trajectory
            if args.layer in ("L2", "L3") and not r["is_correct"]:
                if args.layer == "L2":
                    iv, ai, ao = author_self(r, args.model)
                else:
                    iv, ai, ao = author_external(r, args.verifier)
                auth_in += ai
                auth_out += ao
                if iv:
                    jit.add(iv)
                    n_authored += 1
                r["authored"] = iv
                r["auth_input_tokens"] = ai
                r["auth_output_tokens"] = ao
            r["jit_state_chars"] = jit.size_chars(entry) if args.layer in ("L2", "L3") else 0
            r["jit_notes_total"] = len(jit.notes) if args.layer in ("L2", "L3") else 0
            fp.write(json.dumps(r) + "\n")
            fp.flush()
        n_run += 1
        n_correct += int(r["is_correct"])
        if (i + 1) % 10 == 0 or i == len(eval_set) - 1:
            print(f"  [{tag}] {i+1}/{len(eval_set)}  acc={n_correct/n_run:.3f}  "
                  f"authored={n_authored}  jit_notes={len(jit.notes)}")
    fp.close()

    # summary
    total_cost = sum(json.loads(l).get("cost_usd", 0) for l in open(out))
    auth_cost = C.cost_usd(auth_in, auth_out,
                           args.verifier if args.layer == "L3" else args.model)
    print(f"[{tag}] DONE  Pass@1={n_correct}/{n_run}={n_correct/n_run:.3f}  "
          f"episode_cost=${total_cost:.2f}  author_cost=${auth_cost:.3f}  "
          f"interventions={n_authored}")


if __name__ == "__main__":
    main()
