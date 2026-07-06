"""Scaffold-matched routing-policy baselines for the de-confounded comparison.

WHY: Trinity bundles TWO things — (1) evolved ROUTING (the head picks which model)
and (2) the Thinker→Solver→Verifier multi-turn SCAFFOLD (verifier early-halt etc.).
Our own verifier-reward experiment already proved the scaffold alone lifts outcomes
a lot, independent of routing. So "Trinity beats best-static-1shot" is confounded:
it can't separate "routing helps" from "scaffold helps". To make the headline claim
("evolved coordination beats X") defensible, every baseline must run on the SAME
3-role multi-turn harness and vary ONLY the routing policy:

  learned       — Trinity's evolved head (the actual system; no override)
  static:<ord>  — always route to worker <ord>, all 3 roles, same loop (isolates
                  the scaffold's contribution from routing)
  random        — uniform-random worker per turn, same loop (does LEARNED routing
                  beat ANY routing?)

MECHANISM (no vendored-code edit): fugu/core.py:step_trinity samples agent_id from
logits[:-3] and role_id from logits[-3:] via softmax. We monkeypatch
EvaluationManager.get_action to post-process its returned logit vector: rewrite the
AGENT logits (first L) to force the policy, while leaving the ROLE logits (last 3)
exactly as the head produced them — so the role/scaffold behavior is preserved and
ONLY routing changes. Selected via env CAR_TRINITY_ROUTING_POLICY (inherited by
spawned workers; installed in sitecustomize alongside the Bedrock patch).
"""
from __future__ import annotations

import os
import numpy as np

_BIG = 1e4   # one-hot logit magnitude: softmax → ~prob 1 on the forced index


def _policy() -> str:
    return os.environ.get("CAR_TRINITY_ROUTING_POLICY", "learned").strip().lower()


def _num_workers() -> int:
    # Pool length defines the agent-logit width; import lazily (worker_pool is light).
    from worker_pool_bedrock import POOL
    return len(POOL)


def _rewrite_agent_logits(logits: np.ndarray) -> np.ndarray:
    """Apply the routing policy to the AGENT portion (logits[:-3]); leave ROLE
    logits (logits[-3:]) untouched. Returns a new array (does not mutate)."""
    pol = _policy()
    if pol == "learned" or logits is None:
        return logits
    arr = np.asarray(logits, dtype=np.float64).copy()
    if arr.ndim != 1 or arr.shape[0] < 4:
        return logits  # unexpected shape; don't interfere
    L = arr.shape[0] - 3
    agent = arr[:L]

    if pol.startswith("static_solver:"):
        # Honest best-static baseline: force this worker AND the Solver role every
        # turn → effectively single-model, single-role solving. WHY: plain `static:`
        # forces only the agent and leaves ROLE logits = the (untrained/random) head,
        # so the model gets jerked through a random Thinker/Verifier multi-turn dance
        # it isn't suited for — which tanked deepseek-v3 from 0.87 (solo) to 0.27.
        # This variant pins role 0 (Solver) so the baseline is "model solves it",
        # the true best-single bar for the Pareto comparison.
        try:
            ord_i = int(pol.split(":", 1)[1])
        except ValueError:
            return logits
        ord_i = max(0, min(L - 1, ord_i))
        agent[:] = 0.0
        agent[ord_i] = _BIG
        role = arr[-3:]
        role[:] = 0.0
        role[0] = _BIG               # force Solver role → single-turn solve
        arr[-3:] = role
    elif pol.startswith("static:"):
        try:
            ord_i = int(pol.split(":", 1)[1])
        except ValueError:
            return logits
        ord_i = max(0, min(L - 1, ord_i))
        agent[:] = 0.0
        agent[ord_i] = _BIG          # softmax → forces this worker every turn (role left to head)
    elif pol == "random":
        # Uniform over workers: equal logits → step_trinity's softmax samples
        # uniformly. Per-turn randomness comes from core's np.random.choice.
        agent[:] = 0.0
    else:
        return logits                # unknown policy → no-op (learned)

    arr[:L] = agent
    return arr


def install() -> None:
    """Wrap EvaluationManager.get_action so its logits pass through the policy.
    Idempotent. No-op when policy == 'learned' (the wrapper still installs but
    returns logits unchanged, so the production path is untouched)."""
    import fugu.trainer as T

    EM = T.EvaluationManager
    if getattr(EM, "_car_routing_wrapped", False):
        return
    orig = EM.get_action.__func__ if hasattr(EM.get_action, "__func__") else EM.get_action

    def _wrapped(*args, **kwargs):
        logits = orig(*args, **kwargs)
        try:
            return _rewrite_agent_logits(logits)
        except Exception:
            return logits   # never let baseline rewriting break an episode

    EM.get_action = staticmethod(_wrapped)
    EM._car_routing_wrapped = True
    pol = _policy()
    if pol != "learned":
        print(f"[routing_policy] ACTIVE: {pol} (agent logits forced; role logits preserved)")


if __name__ == "__main__":
    # Self-test the logit rewrite (no GPU/Bedrock needed).
    import worker_pool_bedrock as wp
    L = len(wp.POOL)
    base = np.arange(L + 3, dtype=float)  # distinct values so we can see preservation

    os.environ["CAR_TRINITY_ROUTING_POLICY"] = "static:3"
    out = _rewrite_agent_logits(base)
    assert int(np.argmax(out[:L])) == 3, out[:L]
    assert np.allclose(out[-3:], base[-3:]), "role logits must be preserved"
    print("static:3 OK — argmax agent =", int(np.argmax(out[:L])), "| roles preserved")

    os.environ["CAR_TRINITY_ROUTING_POLICY"] = "random"
    out = _rewrite_agent_logits(base)
    assert np.allclose(out[:L], out[:L][0]), "random → uniform agent logits"
    assert np.allclose(out[-3:], base[-3:]), "role logits must be preserved"
    print("random OK — agent logits uniform | roles preserved")

    os.environ["CAR_TRINITY_ROUTING_POLICY"] = "learned"
    out = _rewrite_agent_logits(base)
    assert np.allclose(out, base), "learned → untouched"
    print("learned OK — logits untouched")
