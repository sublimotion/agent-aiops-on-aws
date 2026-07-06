"""CMA-ES training loop for the Trinity coordinator.

WHY THIS EXISTS: the OpenReview Trinity submission is EVAL-ONLY. The vendored
`CMAEvolutionTrainer` (fugu/algorithms/es.py) ships `__init__`, `run_test`,
`_setup_svd_info`, and diagnostics — but NO training loop: no `cma.CMAEvolutionStrategy`
instantiation, no ask/tell, no `.train()`. References to `self.solver.result.xfavorite`
read a solver that's never created. The README points to an
`experiments/with_training/testing_standalone.py` that was not included.

This module implements the missing loop using ONLY the building blocks already
present in the vendored code (so the algorithm contract is preserved):
  - `cma` library (already imported in es.py)
  - `trainer.num_learnable_params` (19,456-dim vector = SVD scales + linear head)
  - `trainer.svd_weights_cpu`, `trainer.servers`, `trainer.agent_configs`
  - `job_manager.submit_training_job(...)` — the EXACT scoring primitive `run_test`
    uses; returns `(reward, n_turns, _, agent_ids, ...)` per episode.

It subclasses CMAEvolutionTrainer rather than editing it, honoring the spec's
"fixed files: es.py" contract.

CMA-ES contract (paper §: sep-CMA-ES, λ=32, mCMA=16, σ0=0.03):
  x0 = zeros(num_learnable_params)   # SVD scales init at 0 → multiplicative identity
  each iter:
    candidates = es.ask()            # λ solutions
    for each candidate: run mCMA training episodes (random train tasks) → mean reward
    es.tell(candidates, [-fitness])  # cma MINIMIZES, so negate reward
    every test_interval: run_test on validation → track best
"""
from __future__ import annotations

import os
import time
import numpy as np

try:
    import cma
except ImportError as e:  # pragma: no cover
    raise SystemExit("cma not installed (pip install cma==4.0.0)") from e

from fugu.algorithms.es import CMAEvolutionTrainer
from fugu.job_manager import get_job_manager
from fugu.utils import calculate_agent_stats


class CMATrainingLoop(CMAEvolutionTrainer):
    """Adds the missing CMA-ES optimization loop to the vendored eval-only trainer."""

    def _calculate_closed_source_costs(self, test_results) -> dict:
        """Missing from the eval-only submission (like the training loop itself):
        run_test (es.py:569) calls self._calculate_closed_source_costs(test_results)
        but no such method exists → AttributeError AFTER a full 300-task eval, which
        silently discarded every validation score. We track real Bedrock spend in
        cost_bedrock, so return the aggregated cost merged into token_stats. The
        per-test delta isn't separable from training spend here, so report the
        run-cumulative aggregate (informational; test_score is computed independently
        at es.py:578 and is the number we actually gate on)."""
        try:
            import cost_bedrock
            summary = cost_bedrock.aggregate_cost_summary()
            return {
                "closed_source_cost_usd": summary.get("total_cost", 0.0),
                "pricing_version": summary.get("pricing_version", ""),
            }
        except Exception:
            return {"closed_source_cost_usd": 0.0}

    def _ensure_pool(self) -> None:
        """Initialize the spawn Pool once, with the EXACT worker_config schema the
        eval harness builds (evaluate_trinity_livecodebench.py:541-569). Nobody in
        the vendored fugu calls job_manager.initialize() — that lived in the omitted
        training driver — so the loop must do it. Spawned workers re-install the
        Bedrock dispatch via sitecustomize.py (CAR_TRINITY_BEDROCK_PATCH=1)."""
        jm = get_job_manager()
        if jm.pool is not None:
            return
        if not getattr(self, "agent_configs", None):
            self.agent_configs = {}
        if not getattr(self, "servers", None):
            self.servers = getattr(self.infra, "servers", {}) or {}
        worker_config = {
            "router_model_name": self.infra.model_name,
            "llm_names": self.infra.llm_names,
            "debug": self.infra.debug,
            "debug_log_dir": self.infra.debug_log_dir,
            "task_name": self.infra.task,
            "max_tokens": self.infra.max_tokens,
            "temperature": self.infra.temperature,
            "max_turns": self.infra.max_turns,
            "ports": self.infra.ports,
            "servers": self.servers,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "test_split_enabled": True,
            "seed": self.infra.seed,
            "agent_configs": self.agent_configs,
            "use_consultant": self.use_consultant,
            "use_verifier": getattr(self, "use_verifier", False),
            "trinity": getattr(self, "trinity", False),
            "worker_gpu_assignments": getattr(self.infra, "worker_gpu_assignments", [0]),
            "closed_model_config": getattr(self, "closed_model_config", None),
            "using_closed_models": getattr(self, "closed_model_config", None) is not None,
            "max_samples": -1,
            "last_token_predict": getattr(self, "last_token_predict", False),
        }
        jm.cleanup()
        jm.initialize(self.infra.num_workers, worker_config)
        print(f"[cma_train] job pool initialized: {self.infra.num_workers} workers")

    def _score_candidate(self, solution: np.ndarray, n_repeats: int,
                         iteration_idx: int, eps_explore: float = 0.0) -> dict:
        """Run n_repeats training episodes for one candidate; return a dict with
        mean_reward + the diagnostics the Phase-0.5 gate needs.

        Trinity episode result tuple (fugu/trainer.py evaluate_episode_trinity):
          (reward, num_turns, obs_action, agent_ids, response, token_stats, role_ids)
        -1.0 reward = infra-failure sentinel. role_id: 0=Solver,1=Thinker,2=Verifier.
        Early-halt = a Verifier (role 2) ACCEPT before the turn budget (num_turns < max).
        """
        jm = get_job_manager()
        if jm.pool is None:
            self._ensure_pool()
        rng = np.random.RandomState(seed=self.seed + iteration_idx)
        futures = []
        for _ in range(n_repeats):
            tid = int(rng.randint(0, self.infra.train_dataset_size))
            fut = jm.submit_training_job(
                task_id=tid, split="train",
                flat_params=solution.astype(np.float32),
                svd_weights_cpu=self.svd_weights_cpu,
                iteration_idx=iteration_idx, eps_explore=eps_explore,
                servers_dict=self.servers,
                use_structured_router=self.use_structured_router,
                closed_model_config=getattr(self, "closed_model_config", None),
                agent_configs=self.agent_configs,
            )
            futures.append(fut)
        import cost_bedrock
        budget = float(getattr(self, "cost_budget_usd", 0.0) or 0.0)
        clam = float(getattr(self, "cost_lambda", 0.0) or 0.0)
        cost_aware = budget > 0.0 and clam > 0.0

        raw_rewards, shaped_rewards, ep_costs = [], [], []
        agent_ids, role_ids = [], []
        early_halts, dropped = 0, 0
        for fut in futures:
            try:
                r = fut.get(timeout=900)
                # -999.0 = infra-failure sentinel, -1.0 = other failure (eval harness
                # treats both as non-clean). Only count clean episodes toward fitness.
                if r and r[0] not in (-999.0, -1.0):
                    ep_agent_ids = list(r[3]) if (len(r) >= 4 and r[3]) else []
                    raw_rewards.append(r[0])
                    # Budget-constrained Pareto reward (user pivot): price THIS
                    # episode from its own routing trace × measured tokens × PRICES,
                    # then penalize overage above the per-episode budget. λ=0 or
                    # budget=0 → reward == raw pass-rate (upstream behavior).
                    if cost_aware and ep_agent_ids:
                        c = cost_bedrock.episode_cost_from_agent_ids(
                            ep_agent_ids, self.infra.llm_names)
                        ep_costs.append(c)
                        shaped_rewards.append(r[0] - clam * max(0.0, c - budget))
                    else:
                        shaped_rewards.append(r[0])
                    agent_ids.extend(ep_agent_ids)
                    if len(r) >= 7 and r[6]:
                        role_ids.extend(r[6])
                        n_turns = r[1] if len(r) >= 2 else self.infra.max_turns
                        if 2 in r[6] and n_turns < self.infra.max_turns:
                            early_halts += 1
                else:
                    dropped += 1
            except Exception as e:
                dropped += 1
                print(f"[cma_train] episode failed: {e}")
        # Fitness is the SHAPED reward (what CMA-ES optimizes); raw pass-rate and
        # mean episode cost are reported separately for the Pareto-frontier readout.
        mean_shaped = float(np.mean(shaped_rewards)) if shaped_rewards else 0.0
        mean_raw = float(np.mean(raw_rewards)) if raw_rewards else 0.0
        mean_cost = float(np.mean(ep_costs)) if ep_costs else 0.0
        return {"mean_reward": mean_shaped, "mean_raw_reward": mean_raw,
                "mean_episode_cost": mean_cost, "agent_ids": agent_ids,
                "role_ids": role_ids, "early_halts": early_halts, "dropped": dropped}

    def train(self, iteration_callback=None) -> int:
        """Run the CMA-ES optimization loop. Returns 0 on success.

        iteration_callback(iter_idx, info_dict) is called after each iter for
        checkpointing / gates / telemetry (run_trinity_agent supplies it).
        """
        self._ensure_pool()   # spawn the worker Pool before the first ask/tell
        n = self.num_learnable_params
        popsize = self.popsize_override or (4 + int(3 * np.log(n)))  # cma default if not set

        # RESUME (spot/restart safety for the multi-day full run): the CMA-ES state
        # (es object) + loop bookkeeping are pickled to es_state.pkl every iter. On
        # startup, if it exists, restore and continue from the next iter — so a spot
        # reclaim at hour 40 costs one iter, not the whole run. CMA state is tiny.
        import pickle
        state_path = os.path.join(self.infra.log_dir, "es_state.pkl")
        start_iter = 0
        es = None
        if os.path.exists(state_path):
            try:
                with open(state_path, "rb") as f:
                    st = pickle.load(f)
                es = st["es"]
                start_iter = st["next_iter"]
                self.best_score = st.get("best_score", self.best_score)
                self.best_iter = st.get("best_iter", self.best_iter)
                self.log_data = st.get("log_data", self.log_data)
                print(f"[cma_train] RESUMED from {state_path}: continuing at iter "
                      f"{start_iter} (best_score={self.best_score:.4f})")
            except Exception as e:
                print(f"[cma_train] resume failed ({e}); starting fresh")
                es = None
        if es is None:
            x0 = np.zeros(n, dtype=np.float64)   # SVD scale deltas init 0 = identity
            es = cma.CMAEvolutionStrategy(
                x0, self.sigma0,
                {"popsize": popsize, "seed": self.seed, "maxiter": self.num_iters,
                 "verbose": -1},
            )
        self.solver = es   # so run_test() can fall back to es.result.xfavorite
        print(f"[cma_train] CMA-ES start: dim={n} popsize={popsize} "
              f"sigma0={self.sigma0} iters={self.num_iters} repeats={self.num_repeats} "
              f"start_iter={start_iter}")

        role_names = {0: "solver", 1: "thinker", 2: "verifier"}
        # Pool cleanup is automatic: job_manager.initialize() registers
        # atexit.register(self.cleanup), so the spawn Pool is torn down on any exit
        # (incl. the cost-cap/gate SystemExit raised inside iteration_callback).
        for it in range(start_iter, self.num_iters):
            t0 = time.time()
            candidates = es.ask()
            fitnesses = []
            iter_agent_ids, iter_role_ids = [], []
            iter_early_halts, iter_dropped = 0, 0
            cand_results = []
            for cand in candidates:
                res = self._score_candidate(
                    np.asarray(cand), self.num_repeats, iteration_idx=it)
                fitnesses.append(-res["mean_reward"])   # cma MINIMIZES → negate (shaped) reward
                cand_results.append(res)
                iter_agent_ids.extend(res["agent_ids"])
                iter_role_ids.extend(res["role_ids"])
                iter_early_halts += res["early_halts"]
                iter_dropped += res["dropped"]
            es.tell(candidates, fitnesses)

            best_mean_r = -min(fitnesses)
            # Pareto readout: the best candidate's RAW pass-rate and mean episode
            # cost (the shaped fitness is what's optimized, but the frontier is
            # raw-accuracy vs cost). best fitness index → its raw/cost.
            _best_idx = int(np.argmin(fitnesses))
            best_raw = cand_results[_best_idx].get("mean_raw_reward", best_mean_r)
            best_cost = cand_results[_best_idx].get("mean_episode_cost", 0.0)
            iter_mean_cost = float(np.mean([r.get("mean_episode_cost", 0.0)
                                            for r in cand_results])) if cand_results else 0.0
            xbest = np.asarray(es.result.xbest if es.result.xbest is not None else es.ask(1)[0])
            if best_mean_r > self.best_score:
                self.best_score = best_mean_r
                self.best_solution = xbest
                self.best_iter = it
                try:
                    np.save(self.best_model_path, xbest)
                except Exception as e:
                    print(f"[cma_train] best-model save failed: {e}")

            # Emit the EXACT keys check_phase05_gates reads:
            #   agent_distribution (name→count), role_usage (solver/thinker/verifier),
            #   verifier_early_halts, throttle.dropped.
            agent_stats, _ = calculate_agent_stats(iter_agent_ids, self.infra.llm_names)
            agent_distribution = agent_stats.get("agent_distribution", {})
            role_usage = {role_names[k]: 0 for k in role_names}
            for rid in iter_role_ids:
                if rid in role_names:
                    role_usage[role_names[rid]] += 1
            info = {
                "iter": it,
                "mean_reward_best_candidate": best_mean_r,   # SHAPED (optimized) reward
                "best_raw_passrate": best_raw,               # Pareto axis: raw accuracy
                "best_episode_cost_usd": best_cost,          # Pareto axis: $/episode
                "iter_mean_episode_cost_usd": iter_mean_cost,
                "cost_budget_usd": float(getattr(self, "cost_budget_usd", 0.0) or 0.0),
                "cost_lambda": float(getattr(self, "cost_lambda", 0.0) or 0.0),
                "best_score_so_far": self.best_score,
                "best_iter": self.best_iter,
                "agent_distribution": agent_distribution,
                "role_usage": role_usage,
                "verifier_early_halts": iter_early_halts,
                "iter_dropped_episodes": iter_dropped,
                "popsize": popsize,
                "iter_seconds": round(time.time() - t0, 1),
            }
            print(f"[cma_train] iter {it}: shaped={best_mean_r:.4f} "
                  f"raw_pass={best_raw:.4f} cost=${best_cost:.4f}/ep "
                  f"best_so_far={self.best_score:.4f} roles={role_usage} "
                  f"halts={iter_early_halts} dropped={iter_dropped} "
                  f"({info['iter_seconds']:.0f}s)")

            # Validation at interval (uses the proven run_test path)
            if self.test_interval and (it % self.test_interval == 0) and it > 0:
                try:
                    val = self.run_test(solution=xbest)
                    if val:
                        info["validation_score"] = val.get("test_score")
                except Exception as e:
                    print(f"[cma_train] validation failed: {e}")

            # Persist to es_log.json so _collect_smoke_stats (final gate) AND the
            # per-iter S3 sync see the same per-iter entries the vendored eval path
            # would. self.log_data[0] is the config entry written in __init__.
            info["type"] = "train"
            try:
                self.log_data.append(info)
                import json as _json
                with open(self.log_file, "w") as f:
                    _json.dump(self.log_data, f, indent=2)
            except Exception as e:
                print(f"[cma_train] es_log write failed: {e}")

            # Persist CMA-ES resume state (atomic) BEFORE the callback — the cost-cap
            # SystemExit fires inside the callback, and we want the next-iter state
            # durable even then so a capped/reclaimed run resumes cleanly.
            try:
                tmp = state_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump({"es": es, "next_iter": it + 1,
                                 "best_score": self.best_score, "best_iter": self.best_iter,
                                 "log_data": self.log_data}, f)
                os.replace(tmp, state_path)
            except Exception as e:
                print(f"[cma_train] es_state write failed: {e}")

            if iteration_callback is not None:
                iteration_callback(it, info)

        print(f"[cma_train] done. best_score={self.best_score:.4f} at iter {self.best_iter}")
        return 0
