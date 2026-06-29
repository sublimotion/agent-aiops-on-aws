#!/usr/bin/env python3
"""Clean standalone routing eval — bypasses trinity core.py entirely.

WHY: trinity core.py's scaffold has layered prompt/grading bugs (tag-only
extraction; solver prompt needs prior-Thinker context; routing-override quirks).
Each baseline attempt through it hit a new bug. This harness OWNS the turn loop,
prompts, and extraction, reusing only the three vendored primitives that work:
  - bedrock_clients._query_converse  (model calls, region routing, token capture)
  - cost_bedrock                     (real measured $/episode via PRICES)
  - DirectCodeExecutor.check_correctness  (the grading engine)
  - the probe's fenced-block extraction (prefer <answer>, else ```python, else raw)

Arms (all on the SAME fixed seed-42 problem set, apples-to-apples):
  static:<model>   — single-solve baseline (the bar; == differentiation-probe regime)
  cascade          — VERIFIER-GATED ESCALATION (the Trinity thesis, clean impl):
                     cheap solver → frontier verifier ACCEPT/REJECT → escalate to
                     the next-stronger model on REJECT, up the ladder. Grade the
                     final solution. This needs NO learned head — it tests whether
                     verifier-gated routing beats best-static on accuracy AND cost.
  oracle           — solved-if-ANY-model-solves (upper bound on routing headroom).

Report pass@1 + measured $/problem per arm. The learned CMA-ES head is NOT used
here (its training was corrupted by the now-fixed biased grader; a clean retrain
is a separate follow-up).

Run:
  PYTHONPATH=scripts:vendor/trinity-upstream python scripts/clean_router_eval.py \
      --n-problems 40 --out results/clean_router.json
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bedrock_clients as bc
import cost_bedrock as cb
from worker_pool_bedrock import by_friendly_name, reasoning_effort_for

# Escalation ladder for the cascade: cheapest-strong → ... → frontier. Ordered by
# (probe accuracy ascending cost). gpt-oss-120b is the cheapest strong solver;
# deepseek-v3 is the strongest open; opus is the frontier last resort.
CASCADE = ["gpt-oss-120b", "deepseek-v3", "claude-opus-4-8"]
VERIFIER = "claude-opus-4-8"   # frontier verifier judges each solve
BASELINES = ["deepseek-v3", "gpt-oss-120b", "qwen3-235b", "claude-opus-4-8"]

SOLVE_PROMPT = (
    "Solve this competitive-programming problem. Read input from stdin, write the "
    "answer to stdout.\n\nQuestion:\n{q}\n\n{starter}"
    "Return ONLY the final Python program in a ```python code block."
)
VERIFY_PROMPT = (
    "You are a strict code reviewer. Here is a competitive-programming problem and a "
    "candidate Python solution.\n\nProblem:\n{q}\n\nCandidate solution:\n```python\n{code}\n```\n\n"
    "Will this solution pass ALL test cases (correct algorithm, handles edge cases, "
    "reads stdin / writes stdout correctly)? Reply with EXACTLY one word first: "
    "ACCEPT or REJECT, then a one-line reason."
)


def extract_code(resp: str) -> str:
    """Probe-parity extraction: prefer <answer>, else longest ```python block, else raw."""
    if not resp:
        return ""
    if "<answer>" in resp and "</answer>" in resp:
        resp = resp.split("<answer>", 1)[1].split("</answer>", 1)[0]
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", resp, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return resp.strip()


def call(model_friendly: str, prompt: str, max_tokens: int = 8192) -> str:
    w = by_friendly_name[model_friendly]
    return bc._query_converse(
        w.model_id, w.friendly_name, w.concurrency, reasoning_effort_for(w),
        [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0,
        no_temperature=("no-temperature" in getattr(w, "api_quirks", ())),
        home_region=getattr(w, "region", None),
    )


def grade(ex, sample, code) -> bool:
    if not code or code.startswith("CONTENT_REMOVED"):
        return False
    try:
        res, _ = ex.check_correctness(sample, code, timeout=6, using_closed_models=False)
        return isinstance(res, list) and len(res) > 0 and all(r is True for r in res)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/clean_router.json")
    args = ap.parse_args()

    bc.install(); cb.install()
    from cost_bedrock import PRICES, Price
    # ensure all arms' models are priced
    for m in set(CASCADE + BASELINES + [VERIFIER]):
        assert m in PRICES, f"unpriced model {m}"

    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", split="test",
                      version_tag="release_v1", trust_remote_code=True)
    import random
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[:args.n_problems])

    from fugu.tasks.livecodebench import CodeGenerationProblem
    from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
    ex = DirectCodeExecutor()

    problems = []
    for i in idx:
        row = ds[i]
        try:
            sample = {"input_output": CodeGenerationProblem(**row).get_evaluation_sample()["input_output"]}
        except Exception:
            continue
        starter = row.get("starter_code") or ""
        prompt = SOLVE_PROMPT.format(q=row.get("question_content", ""),
                                     starter=(f"Starter code:\n{starter}\n\n" if starter else ""))
        problems.append({"prompt": prompt, "q": row.get("question_content", ""), "sample": sample})
    n = len(problems)
    print(f"[clean] {n} problems, seed {args.seed}")

    def price(model, in_t, out_t):
        p = PRICES.get(model) or Price(0, 0)
        return in_t * p.in_per_1m / 1e6 + out_t * p.out_per_1m / 1e6

    results = {}

    # --- baselines: single-solve per model -----------------------------------
    # Cache each model's solve (code, pass, cost) per problem so the cascade can
    # reuse them and we never double-charge / re-call for the same (model,problem).
    solve_cache = {}   # (model, pidx) -> {"code","pass","cost","resp"}

    def solve(model, pi):
        key = (model, pi)
        if key in solve_cache:
            return solve_cache[key]
        cb._TOKEN_OBS.clear()
        resp = call(model, problems[pi]["prompt"])
        obs = cb._TOKEN_OBS.get(model, {"in": 0, "out": 0})
        code = extract_code(resp)
        ok = grade(ex, problems[pi]["sample"], code)
        rec = {"code": code, "pass": ok, "cost": price(model, obs["in"], obs["out"]), "resp": resp}
        solve_cache[key] = rec
        return rec

    for model in BASELINES:
        passes, cost = 0, 0.0
        for pi in range(n):
            r = solve(model, pi)
            passes += int(r["pass"]); cost += r["cost"]
        results[f"static:{model}"] = {"pass_rate": passes / n, "cost_per_problem": cost / n,
                                      "passes": passes, "n": n}
        print(f"  static:{model:16s} pass={passes/n:.3f}  ${cost/n:.5f}/prob")

    # --- cascade: verifier-gated escalation -----------------------------------
    casc_pass, casc_cost = 0, 0.0
    ladder_usage = {m: 0 for m in CASCADE}
    for pi in range(n):
        ep_cost, solved, final_code = 0.0, False, ""
        for stage, model in enumerate(CASCADE):
            r = solve(model, pi)                 # reuse cached solve
            ep_cost += r["cost"]; ladder_usage[model] += 1
            final_code = r["code"]
            # verifier judges (skip verify on the last rung — nothing to escalate to)
            if stage == len(CASCADE) - 1:
                solved = r["pass"]; break
            cb._TOKEN_OBS.clear()
            vresp = call(VERIFIER, VERIFY_PROMPT.format(q=problems[pi]["q"], code=r["code"]), max_tokens=2048)
            vobs = cb._TOKEN_OBS.get(VERIFIER, {"in": 0, "out": 0})
            ep_cost += price(VERIFIER, vobs["in"], vobs["out"])
            accepted = vresp.strip().lower().startswith("accept") or \
                ("accept" in vresp.lower() and "reject" not in vresp.lower())
            if accepted:
                solved = r["pass"]; break       # trust the verifier, stop escalating
        casc_pass += int(solved); casc_cost += ep_cost
    results["cascade_verifier_gated"] = {"pass_rate": casc_pass / n, "cost_per_problem": casc_cost / n,
                                         "passes": casc_pass, "n": n, "ladder_usage": ladder_usage}
    print(f"  cascade (verifier-gated) pass={casc_pass/n:.3f}  ${casc_cost/n:.5f}/prob  ladder={ladder_usage}")

    # --- oracle: solved if ANY baseline model solved -------------------------
    oracle = sum(1 for pi in range(n) if any(solve_cache.get((m, pi), {}).get("pass") for m in BASELINES)) / n
    results["oracle_union"] = {"pass_rate": oracle, "n": n}
    print(f"  oracle (any model solves) pass={oracle:.3f}")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\n[clean] wrote {outp}")
    best_static = max((results[f"static:{m}"] for m in BASELINES), key=lambda r: r["pass_rate"])
    casc = results["cascade_verifier_gated"]
    print("\n=== VERDICT ===")
    print(f"  best-static : pass={best_static['pass_rate']:.3f} ${best_static['cost_per_problem']:.5f}/prob")
    print(f"  cascade     : pass={casc['pass_rate']:.3f} ${casc['cost_per_problem']:.5f}/prob")
    print(f"  oracle      : pass={oracle:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
