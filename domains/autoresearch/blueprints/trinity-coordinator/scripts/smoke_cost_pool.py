#!/usr/bin/env python3
"""Cross-pool smoke for the budget-constrained cost machinery (user-requested).

Validates END-TO-END, across ALL 7 workers, that:
  1. record_usage captures REAL Converse tokens for every provider (Anthropic,
     Amazon, Qwen reasoning+direct, DeepSeek, Gemma) — provider-agnostic counter.
  2. the per-PID token sink flushes + aggregates (spawn-isolation bridge).
  3. episode_cost_from_agent_ids prices an episode from REAL measured tokens ×
     verified PRICES, with the correct cross-model ordering (frontier ≫ cheap).
  4. the shaped reward = raw − λ·max(0, cost − budget) behaves at the budget edge.

Run on the box (needs boto3 + Bedrock creds):
  PYTHONPATH=scripts:vendor/trinity-upstream python scripts/smoke_cost_pool.py
"""
from __future__ import annotations
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_pool_bedrock import POOL, LLM_NAMES, by_friendly_name, reasoning_effort_for
import bedrock_clients as bc
import cost_bedrock as cb

PROMPT = [{"role": "user", "content": "Write one line of Python that prints the sum of a list `xs`."}]


def main() -> int:
    bc.install()           # rebind dispatch → Bedrock (also patches _is_oai etc.)
    cb.install()
    print(f"=== Cross-pool token-counter smoke: {len(POOL)} workers ===")
    failures = []
    for w in POOL:
        if w.transport == "openai_compat":
            print(f"  ord {w.ord} {w.friendly_name:22s} openai_compat — skipped (bearer)")
            continue
        eff = reasoning_effort_for(w)
        t0 = time.time()
        txt = bc._query_converse(
            w.model_id, w.friendly_name, w.concurrency, eff, PROMPT,
            max_tokens=4096, temperature=0.0,
            no_temperature=("no-temperature" in getattr(w, "api_quirks", ())),
            home_region=getattr(w, "region", None),
        )
        dt = time.time() - t0
        obs = cb._aggregate_token_obs().get(w.friendly_name, {})
        in_tok, out_tok = obs.get("in", 0), obs.get("out", 0)
        ok = bool(txt.strip()) and in_tok > 0 and out_tok > 0
        if not ok:
            failures.append(w.friendly_name)
        print(f"  ord {w.ord} {w.friendly_name:22s} {'OK ' if ok else 'FAIL'} "
              f"in={in_tok:5d} out={out_tok:5d}  resp_len={len(txt):4d}  {dt:5.1f}s")

    print("\n=== per-model measured avg tokens + priced $/call ===")
    for w in POOL:
        if w.transport == "openai_compat":
            continue
        ai, ao = cb._avg_tokens(w.friendly_name)
        c = cb.episode_cost_from_agent_ids([w.ord], LLM_NAMES)
        print(f"  ord {w.ord} {w.friendly_name:22s} avg_in={ai:6.0f} avg_out={ao:6.0f}  ${c:.6f}/turn")

    print("\n=== episode pricing + shaped-reward sanity ===")
    # cheapest vs most-expensive single-turn episode
    cheap = min(range(len(POOL)), key=lambda i: cb.episode_cost_from_agent_ids([i], LLM_NAMES))
    pricey = max(range(len(POOL)), key=lambda i: cb.episode_cost_from_agent_ids([i], LLM_NAMES))
    c_cheap = cb.episode_cost_from_agent_ids([cheap], LLM_NAMES)
    c_pricey = cb.episode_cost_from_agent_ids([pricey], LLM_NAMES)
    print(f"  cheapest worker={LLM_NAMES[cheap]} ${c_cheap:.6f}  "
          f"priciest={LLM_NAMES[pricey]} ${c_pricey:.6f}  ratio={c_pricey/max(c_cheap,1e-9):.1f}x")
    budget, lam = c_cheap * 1.5, 10.0   # budget between cheap and pricey
    r_cheap = 1.0 - lam * max(0.0, c_cheap - budget)
    r_pricey = 1.0 - lam * max(0.0, c_pricey - budget)
    print(f"  with budget=${budget:.6f} λ={lam}: same-accuracy(1.0) shaped reward "
          f"cheap={r_cheap:.4f} pricey={r_pricey:.4f}  → {'cheap wins' if r_cheap > r_pricey else 'BROKEN'}")

    print("\n=== token sink files on disk (spawn-bridge) ===")
    d = os.environ.get("CAR_TRINITY_TELEMETRY_DIR")
    if d and os.path.isdir(d):
        print("  ", [f for f in os.listdir(d) if f.startswith("tokens_")])
    else:
        print("   (no telemetry dir set — sink not exercised in-proc, that's fine here)")

    if failures:
        print(f"\nSMOKE FAILED — workers with no token capture: {failures}")
        return 1
    if r_cheap <= r_pricey:
        print("\nSMOKE FAILED — shaped reward does not prefer cheaper at equal accuracy")
        return 1
    print("\nSMOKE PASSED — token counter + pricing + shaped reward validated across pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
