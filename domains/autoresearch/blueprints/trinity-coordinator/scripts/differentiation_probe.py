#!/usr/bin/env python3
"""Differentiation probe — the VIABILITY GATE for cost-aware routing.

Routing only earns its keep if the pool has a genuine cost/capability trade-off.
This probe runs each candidate model SOLO (best-static, single-turn solver, no
scaffold) on a set of LiveCodeBench problems, grades with the SAME executor the
harness uses, and reports:
  - per-(model, problem) pass matrix
  - per-model pass rate + REAL measured cost (Converse tokens × verified PRICES)
  - pairwise pass-correlation + per-model UNIQUE solves (capability variance)

Three outcomes decide the experiment (spec Phase 1.5 viability gate):
  1. models correlated / low capability variance → routing can't help; finding =
     "cheap models capability-equivalent → Opus premium wasted" (quantify $ saved)
  2. one cheap model dominates → "just use it"
  3. decorrelated + differently priced → genuine routing headroom → train

Run on the box (boto3 + Bedrock creds + datasets):
  PYTHONPATH=scripts:vendor/trinity-upstream python scripts/differentiation_probe.py \
      --n-problems 30 --out results/diff_probe.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bedrock_clients as bc
import cost_bedrock as cb

# Candidate pool to probe: current 7 + the reachable flagships (Gate 0.0). Each
# entry: friendly -> (modelId, reasoning_effort_or_None, no_temperature, region_hint).
# region_hint is informational; _query_converse rotates REGIONS itself.
CANDIDATES = {
    "claude-opus-4-8":        ("us.anthropic.claude-opus-4-8", None, True),
    "claude-sonnet-4-6":      ("us.anthropic.claude-sonnet-4-6", None, False),
    "claude-haiku-4-5":       ("us.anthropic.claude-haiku-4-5-20251001-v1:0", None, False),
    "nova-pro":               ("us.amazon.nova-pro-v1:0", None, False),
    "deepseek-r1":            ("us.deepseek.r1-v1:0", None, False),
    "deepseek-v3":            ("deepseek.v3-v1:0", None, False),
    "gemma-3-27b":            ("google.gemma-3-27b-it", None, False),
    "qwen3-32b":              ("qwen.qwen3-32b-v1:0", None, False),
    "qwen3-235b":             ("qwen.qwen3-235b-a22b-2507-v1:0", None, False),
    "qwen3-coder-480b":       ("qwen.qwen3-coder-480b-a35b-v1:0", None, False),
    "gpt-oss-120b":           ("openai.gpt-oss-120b-1:0", None, False),
}

# Register prices for the NEW flagships (cost_bedrock.PRICES has the original 7).
# USD / 1M tokens, on-demand snapshot 2026-06-26 (verify before publishing).
EXTRA_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "deepseek-v3":      (0.58, 1.68),
    "qwen3-235b":       (0.22, 0.88),
    "qwen3-coder-480b": (0.45, 1.80),
    "gpt-oss-120b":     (0.15, 0.60),
    "qwen3-32b":        (0.15, 0.62),   # CANDIDATES key is 'qwen3-32b'; PRICES table is
                                        # keyed -reasoning/-direct → priced $0 in run 1. Same rate.
}

SOLVER_PROMPT = (
    "Question:\n{q}\n\n"
    "Read inputs from stdin, solve, write the answer to stdout. "
    "Return ONLY the final Python code in <answer> </answer> tags, no markdown/backticks.\n\n"
    "{starter}"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="results/diff_probe.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bc.install(); cb.install()
    from cost_bedrock import Price, PRICES
    for nm, (i, o) in EXTRA_PRICES.items():
        PRICES[nm] = Price(i, o)

    # Load the SAME LiveCodeBench split the harness uses (seed-fixed for repro).
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", split="test",
                      version_tag="release_v1", trust_remote_code=True)
    import random
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[:args.n_problems])

    from fugu.tasks.livecodebench import extract_code, CodeGenerationProblem
    from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
    ex = DirectCodeExecutor()

    # Pre-build gradeable samples + prompts.
    problems = []
    for i in idx:
        row = ds[i]
        try:
            prob = CodeGenerationProblem(**row)
            sample = {"input_output": prob.get_evaluation_sample()["input_output"]}
        except Exception as e:
            continue
        q = row.get("question_content", "")
        starter = (row.get("starter_code") or "")
        prompt = SOLVER_PROMPT.format(q=q, starter=(f"Starter code:\n{starter}\n\n" if starter else ""))
        problems.append({"qid": row.get("question_id", str(i)), "prompt": prompt, "sample": sample})
    print(f"[probe] {len(problems)} gradeable problems, {len(CANDIDATES)} models")

    results = {}   # model -> {"passes":[0/1...], "cost":float, "errors":int}
    for nm, (mid, eff, no_temp) in CANDIDATES.items():
        passes, ferr = [], 0
        cb._TOKEN_OBS.clear()   # reset per-model token accounting for clean cost
        t0 = time.time()
        for p in problems:
            try:
                resp = bc._query_converse(mid, nm, 8, eff,
                    [{"role": "user", "content": p["prompt"]}],
                    max_tokens=args.max_tokens, temperature=0.0, no_temperature=no_temp)
                # Code-block isolation. Models comply with the <answer> tag
                # unevenly: DeepSeek-V3 ignores it and returns prose + a ```python
                # fence (scored 0.00 in the first run because extract_code only
                # strips fences, it doesn't ISOLATE the block from prose). Prefer
                # <answer>, else the last ```python fenced block, else raw.
                ans = resp
                if "<answer>" in resp and "</answer>" in resp:
                    ans = resp.split("<answer>", 1)[1].split("</answer>", 1)[0]
                elif "```" in resp:
                    import re as _re
                    blocks = _re.findall(r"```(?:python)?\s*(.*?)```", resp, _re.DOTALL)
                    if blocks:
                        ans = max(blocks, key=len)   # the substantive code block
                code = extract_code(ans)
                if not code or code.startswith("CONTENT_REMOVED"):
                    passes.append(0); continue
                # using_closed_models=False is REQUIRED: True skips the signal.alarm
                # (executor line 69) → an infinite-loop generated solution hangs the
                # probe forever (no timeout). False arms the per-test alarm. We run
                # single-threaded in the main thread, so signal.alarm works.
                res, _ = ex.check_correctness(p["sample"], code, timeout=6, using_closed_models=False)
                ok = isinstance(res, list) and res and all(r == True for r in res)
                passes.append(1 if ok else 0)
            except Exception:
                passes.append(0); ferr += 1
        # real measured cost for THIS model's calls
        obs = cb._TOKEN_OBS.get(nm, {"in": 0, "out": 0})
        pr = PRICES.get(nm) or Price(0, 0)
        cost = obs["in"] * pr.in_per_1m / 1e6 + obs["out"] * pr.out_per_1m / 1e6
        results[nm] = {"passes": passes, "pass_rate": sum(passes) / max(len(passes), 1),
                       "cost_total": cost, "cost_per_problem": cost / max(len(problems), 1),
                       "errors": ferr, "seconds": round(time.time() - t0, 1)}
        print(f"  {nm:18s} pass={results[nm]['pass_rate']:.2f} "
              f"${results[nm]['cost_per_problem']:.5f}/prob err={ferr} "
              f"({results[nm]['seconds']:.0f}s)")

    # Capability-variance analysis.
    import itertools
    names = list(results.keys())
    n = len(problems)
    print("\n=== capability variance ===")
    # per-problem: how many models solved it (0..M) — differentiation signal
    solved_by = [sum(results[m]["passes"][j] for m in names) for j in range(n)]
    from collections import Counter
    dist = Counter(solved_by)
    print("  problems solved by k models:", dict(sorted(dist.items())))
    none_solved = dist.get(0, 0); all_solved = dist.get(len(names), 0)
    print(f"  solved-by-none={none_solved}  solved-by-ALL={all_solved}  "
          f"differentiating(1..M-1)={n - none_solved - all_solved}")
    # pairwise pass correlation
    import numpy as np
    M = np.array([results[m]["passes"] for m in names], dtype=float)
    print("  mean pairwise pass-correlation:", end=" ")
    cors = []
    for a, b in itertools.combinations(range(len(names)), 2):
        if M[a].std() > 0 and M[b].std() > 0:
            cors.append(float(np.corrcoef(M[a], M[b])[0, 1]))
    print(f"{np.mean(cors):.3f}" if cors else "n/a")
    # unique solves per model
    print("  unique solves (only this model passed):")
    for i, m in enumerate(names):
        uniq = sum(1 for j in range(n) if M[i, j] == 1 and solved_by[j] == 1)
        print(f"    {m:18s} {uniq}")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"results": results, "solved_by_k": dict(dist),
                                "mean_pass_correlation": (float(np.mean(cors)) if cors else None),
                                "n_problems": n, "seed": args.seed}, indent=2))
    print(f"\n[probe] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
