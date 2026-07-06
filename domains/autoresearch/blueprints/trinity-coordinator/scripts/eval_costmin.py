#!/usr/bin/env python3
"""Held-out eval + baseline comparison for the cost-min trained router.

Loads the trained best_model.npy and runs CMATrainingLoop.run_test on the test
split under three routing policies (same split, apples-to-apples per the spec's
Pareto-dominance exit criterion):
  learned    — the evolved head (the system)
  static:0   — best-static single model (deepseek-v3 at ord 0)
  random     — uniform-random worker per turn

run_test reports test_score (held-out pass@1); we also pull mean episode cost
from the per-PID token sink so each arm has a (accuracy, cost) point. The
question: does the evolved router Pareto-dominate best-static (match/beat its
accuracy at lower cost)?

Run on the box:
  PYTHONPATH=scripts:. python scripts/eval_costmin.py --model-file logs/costmin8/models/best_model.npy --test-size 40
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_trainer(vendor: Path, test_size: int, max_turns: int = 5):
    # max_turns=5 → real multi-turn scaffold (learned router). max_turns=1 → a clean
    # single-solve baseline (used with static_solver:<ord> so the model just solves
    # once, no random-role dance — the honest best-static bar). run_test seeds its
    # task-id sampling with RandomState(seed=42), so EVERY arm built here hits the
    # IDENTICAL problem set (shared-split apples-to-apples).
    import run_trinity_agent as rta
    rta.install_bedrock_adaptation(vendor)
    cfgs, llm_names = rta.bedrock_agent_configs()
    from fugu.trainer import RouterInfrastructure
    from cma_train import CMATrainingLoop
    log_dir = (vendor / "logs" / "costmin8").resolve()
    infra = RouterInfrastructure(
        task="livecodebench", model_name="Qwen/Qwen3-0.6B", llm_names=llm_names,
        log_dir=str(log_dir), seed=42, temperature=0.1, max_tokens=8192, max_turns=max_turns,
        servers={n: None for n in llm_names}, ports={n: None for n in llm_names},
        num_workers=8, debug=False, worker_gpu_assignments=[0] * 8,
        test_ratio=0.2, valid_ratio=0.2, configure_splits=True, trinity=True,
    )
    trainer = CMATrainingLoop(
        infrastructure=infra, num_iters=1, test_interval=0, num_repeats=1,
        sigma0=0.03, seed=42, num_tests=test_size, test_size=test_size,
        servers={n: None for n in llm_names}, opt_layer_indices=[26], popsize_override=2,
        diversity_bonus_weight=0.15, cost_bonus_weight=0.0, turn_bonus_weight=0.1,
        role_bonus_weight=0.0, use_structured_router=False, closed_model_config=None,
        agent_configs=cfgs, use_consultant=False, use_verifier=True,
        trinity=True, last_token_predict=False,
    )
    return trainer, llm_names


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--test-size", type=int, default=40)
    ap.add_argument("--vendor-root", default=".")
    ap.add_argument("--out", default="results/costmin8_eval.json")
    # Default comparison: learned router (multi-turn) vs the two best single models
    # as clean single-solve baselines (static_solver → force Solver role + max_turns=1).
    # ord 0 = deepseek-v3 (probe-best 0.87), ord 3 = gpt-oss-120b (what the router
    # collapsed onto). All arms share run_test's seed-42 task ids.
    ap.add_argument("--policies", default="learned,static_solver:0,static_solver:3")
    args = ap.parse_args()

    vendor = Path(args.vendor_root).resolve()
    sol = np.load(args.model_file)
    print(f"[eval] loaded {args.model_file} dim={sol.shape}")

    import cost_bedrock, bedrock_clients, routing_policy
    results = {}
    for pol in args.policies.split(","):
        pol = pol.strip()
        os.environ["CAR_TRINITY_ROUTING_POLICY"] = pol
        cost_bedrock._TOKEN_OBS.clear()
        # Baselines (static_solver) are single-solve → max_turns=1; the learned/random
        # router gets the full multi-turn scaffold (max_turns=5).
        mt = 1 if pol.startswith("static_solver:") else 5
        trainer, llm_names = build_trainer(vendor, args.test_size, max_turns=mt)
        routing_policy.install()   # re-bind get_action wrapper for this policy
        trainer._ensure_pool()     # run_test asserts pool!=None (es.py:517); init it
        print(f"\n[eval] === policy={pol} (max_turns={mt}) ===")
        out = trainer.run_test(solution=sol)
        score = out.get("test_score") if out else None
        # mean episode cost from the token sink aggregate for this arm
        obs = cost_bedrock._aggregate_token_obs()
        total_cost = 0.0
        from cost_bedrock import PRICES, Price
        for m, d in obs.items():
            p = PRICES.get(m) or Price(0, 0)
            total_cost += d["in"] * p.in_per_1m / 1e6 + d["out"] * p.out_per_1m / 1e6
        n = args.test_size
        results[pol] = {"test_score": score, "total_cost_usd": total_cost,
                        "cost_per_problem": total_cost / max(n, 1),
                        "agent_distribution": (out or {}).get("agent_distribution", {})}
        print(f"[eval] {pol}: score={score}  $/prob={results[pol]['cost_per_problem']:.5f}")
        try:
            trainer._ensure_pool  # noqa
            from fugu.job_manager import get_job_manager
            get_job_manager().cleanup()
        except Exception:
            pass

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY (held-out) ===")
    for pol, r in results.items():
        print(f"  {pol:12s} pass@1={r['test_score']}  ${r['cost_per_problem']:.5f}/prob")
    print(f"[eval] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
