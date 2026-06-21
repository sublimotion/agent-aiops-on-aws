#!/usr/bin/env python3
"""
Run one DBBench cell (A/B/C/D) for one worker model over the verified n=120 set.

Cells (reward-regime × authoring-locus), DBBench half of the E_harness3 matrix:

  A = self + reward-VISIBLE    -> == E_harness2 L2 (LOADED, not run here)
  C = external + reward-VISIBLE -> == E_harness2 L3 (LOADED, not run here)
  B = self + reward-WITHHELD    (NEW; this runner)
  D = external + reward-WITHHELD (NEW; this runner)

The worker episode is IDENTICAL across all four cells (same L2-style base loop,
no L1 frozen harness, JIT store injected into the system prompt). Only the
AUTHORING step differs:

  * LOCUS    : self (worker model) vs external (separate verifier model).
  * VISIBILITY (the new axis): visible cells author exactly as E_harness2 (fire on
    FAILURE, see the gold label). For parity, this runner can also reproduce A/C —
    but the spec says LOAD A/C from E_harness2, so by default we only run B/D.

REWARD-LEAK HARD GATE — the design that makes B/D a clean ablation
(carryover-auditor P0-1: "the invocation itself is a perfect reward signal"):

  Visible (A/C):   author is invoked IFF the task FAILED  -> invocation leaks reward.
  Withheld (B/D):  author is invoked on a REWARD-INDEPENDENT schedule — EVERY task,
                   pass or fail — and the digest carries no label/WRONG/is_correct.
                   The author must SELF-GATE (emit {"skip":true} when it judges the
                   trajectory fine). Whether an intervention gets authored is then a
                   function of the AUTHOR'S inference, never of the hidden reward.

The worker still EXECUTES SQL to act and is still SCORED (we need is_correct to
report Pass@1) — but is_correct is computed AFTER authoring and never enters the
authoring path. See leak_audit.py for the empirical proof.
"""
import argparse
import json
import os

import dbbench_common as C
from agent_loop import run_episode
from jit_authoring import JitStore, author_self, author_external

CELLS = {
    # cell: (locus, reward_visible)
    "A": ("self", True),
    "B": ("self", False),
    "C": ("external", True),
    "D": ("external", False),
}


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
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--model", required=True, choices=["haiku", "sonnet"])
    ap.add_argument("--verifier", default="haiku", choices=["haiku", "sonnet"],
                    help="external authoring model for cells C/D")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    locus, reward_visible = CELLS[args.cell]
    eval_set = json.load(open(args.data))
    if args.limit:
        eval_set = eval_set[:args.limit]

    tag = f"{args.cell}_{args.model}"
    if locus == "external":
        tag += f"_v-{args.verifier}"
    out = args.out or f"{C.ROOT}/results/{tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = load_done(out)
    print(f"[{tag}] cell={args.cell} locus={locus} reward_visible={reward_visible} "
          f"{len(eval_set)} tasks, {len(done)} done -> {out}")

    jit = JitStore()
    auth_in = auth_out = n_authored = n_author_calls = 0
    fp = open(out, "a")
    n_correct = n_run = 0
    author_key = args.verifier if locus == "external" else args.model

    for i, entry in enumerate(eval_set):
        tid = entry["task_id"]
        if tid in done:
            r = done[tid]
            if r.get("authored"):
                jit.add(r["authored"])
                n_authored += 1
        else:
            r = run_episode(entry, args.model, "L2", jit_store=jit)  # L2-style base loop, JIT injected

            # ---- AUTHORING STEP (the only thing that differs across cells) ----
            invoke = False
            if reward_visible:
                # A/C parity: fire only on failure (E_harness2 behavior).
                invoke = not r["is_correct"]
            else:
                # B/D: reward-INDEPENDENT — author on EVERY task. The invocation
                # pattern carries no reward (carryover P0-1). is_correct is NOT
                # passed into the digest (jit_authoring strips it under withhold).
                invoke = True

            if invoke:
                n_author_calls += 1
                if locus == "self":
                    iv, ai, ao = author_self(r, args.model, reward_visible=reward_visible)
                else:
                    iv, ai, ao = author_external(r, args.verifier, reward_visible=reward_visible)
                auth_in += ai
                auth_out += ao
                r["authored"] = iv
                r["auth_input_tokens"] = ai
                r["auth_output_tokens"] = ao
                if iv:
                    jit.add(iv)
                    n_authored += 1
            else:
                r["authored"] = None

            r["cell"] = args.cell
            r["locus"] = locus
            r["reward_visible"] = reward_visible
            r["jit_state_chars"] = jit.size_chars(entry)
            r["jit_notes_total"] = len(jit.notes)
            fp.write(json.dumps(r) + "\n")
            fp.flush()
        n_run += 1
        n_correct += int(r["is_correct"])
        if (i + 1) % 10 == 0 or i == len(eval_set) - 1:
            print(f"  [{tag}] {i+1}/{len(eval_set)}  acc={n_correct/n_run:.3f}  "
                  f"author_calls={n_author_calls}  authored={n_authored}  "
                  f"jit_notes={len(jit.notes)}")
    fp.close()

    total_cost = sum(json.loads(l).get("cost_usd", 0) for l in open(out))
    auth_cost = C.cost_usd(auth_in, auth_out, author_key)
    print(f"[{tag}] DONE  Pass@1={n_correct}/{n_run}={n_correct/n_run:.3f}  "
          f"episode_cost=${total_cost:.2f}  author_cost=${auth_cost:.3f}  "
          f"author_calls={n_author_calls}  interventions={n_authored}")


if __name__ == "__main__":
    main()
