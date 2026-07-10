#!/usr/bin/env python3
"""Long-horizon FILE-state task generator — the drift-inducing substrate.

Synthesizes dict_sum-over-files (per arXiv:2509.09677's horizon mechanism, but
file-based so the trace-lineage detector applies):
  - K account files, each holding a running balance.
  - A derived `summary.md` holding the total of all balances (the coupled site).
  - N sequential operations the agent must apply IN ORDER, in one session.

Success requires: (a) every account ends at its true balance, and (b) summary
stays consistent with the accounts. Because each op depends on prior state, the
task COMPOUNDS — success ≈ p^H — so as N grows a capable model drifts (unlike
the saturated one-shot value-propagation task). The account↔summary coupling is
exactly what lineage.py value-drift catches when the agent updates accounts but
forgets the summary.

Deterministic: operations derived from --seed via a fixed LCG (no RNG import;
Math.random-style nondeterminism breaks reproducibility). Dials: --k (accounts),
--turns (horizon length).

Usage:
  gen_horizon_task.py --out DIR --seed 0 --k 4 --turns 25
Writes account files, summary.md, task.json (prompt + ground-truth final state).
"""
import argparse
import json
import os


def lcg(seed):
    """Deterministic pseudo-random stream (numerical recipes LCG)."""
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        yield x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="number of account files")
    ap.add_argument("--turns", type=int, default=25, help="horizon: number of sequential ops")
    ap.add_argument("--start", type=int, default=100, help="starting balance per account")
    args = ap.parse_args()

    keys = [f"acct_{i}" for i in range(args.k)]
    balances = {k: args.start for k in keys}

    rng = lcg(args.seed + 1)
    ops = []
    for _ in range(args.turns):
        k = keys[next(rng) % args.k]
        delta = (next(rng) % 41) - 20  # -20..+20
        balances[k] += delta
        ops.append({"account": k, "delta": delta})

    total = sum(balances.values())

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "accounts"), exist_ok=True)
    # Seed files at STARTING balances (agent must apply ops to reach final).
    for k in keys:
        with open(os.path.join(args.out, "accounts", f"{k}.txt"), "w") as f:
            f.write(f"{args.start}\n")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# Ledger summary\n\nTotal across all accounts: {args.start * args.k}\n")

    op_lines = "\n".join(
        f"{i+1}. {o['account']}: {'+' if o['delta']>=0 else ''}{o['delta']}"
        for i, o in enumerate(ops))
    prompt = (
        f"This repo has {args.k} account files under accounts/ (each holds a balance) "
        f"and summary.md (holds the total across all accounts). Apply the following "
        f"{args.turns} operations IN ORDER, updating the relevant account file after each. "
        f"After all operations, every account file must hold its correct final balance "
        f"AND summary.md must hold the correct total across all accounts. Keep them "
        f"consistent.\n\nOperations:\n{op_lines}"
    )

    manifest = {
        "task_type": "horizon_ledger", "seed": args.seed, "k": args.k,
        "turns": args.turns, "start": args.start,
        "final_balances": balances,      # ground truth (mechanical oracle)
        "final_total": total,
        "coupled_sites": [f"accounts/{k}.txt" for k in keys] + ["summary.md"],
        "prompt": prompt,
    }
    with open(os.path.join(args.out, "task.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"out": args.out, "k": args.k, "turns": args.turns,
                      "final_total": total}, indent=2))


if __name__ == "__main__":
    main()
