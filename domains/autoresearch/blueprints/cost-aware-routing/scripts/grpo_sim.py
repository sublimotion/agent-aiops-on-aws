"""CPU-only GRPO simulator for the cost-aware-routing reward landscape.

Strips the trainer down to its essential dynamics:

  - "Router" = a per-source categorical distribution over the 9 ords.
    There are 3 sources (math500, aime25, wildchat); the router has 3
    independent softmax distributions, one per source.
  - At each iteration:
      1. Sample N questions (a mix of sources).
      2. For each question, sample R rollouts from the appropriate
         per-source policy (independent samples).
      3. Look up the (worker, question) reward from the actual measured
         baselines (always_x_math500, always_x_aime25, always_x_wildchat).
      4. Compute within-question advantage normalization.
      5. Update the per-source softmax logits with vanilla policy gradient
         (-adv * grad_logit_at_chosen_ord) + entropy regularization.
  - Track per-source pick distribution evolution and compare to the
    oracle (math500->Opus, aime25->Opus, wildchat->Qwen-Coder for alpha=1.0).

This is 50 lines of NumPy. If it converges, GRPO + reward function are
sound; the 7B-trainer collapse is a sample-efficiency / advantage-noise
issue. If it collapses too, the reward function itself is the problem
and we need a redesign.

Run:
  python3 grpo_sim.py --alpha 1.0 --iters 2000
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np


def load_per_question_rewards(math_path: str, aime_path: str, wc_path: str) -> dict:
    """Returns: source -> qid -> ord -> {is_correct, cost_usd}."""
    out: dict[str, dict[str, dict[int, dict]]] = {}
    for path, source, key in [
        (math_path, "math500", "is_correct"),
        (aime_path, "aime25", "is_correct"),
        (wc_path, "wildchat", "acceptable"),
    ]:
        raw = json.load(open(path))
        out[source] = {}
        for r in raw["rollouts"]:
            qid = r.get("id") or r.get("question", "")[:60]
            out[source].setdefault(qid, {})[r["ord"]] = {
                "is_correct": bool(r[key]),
                "cost_usd": r["cost_usd"],
            }
    return out


# Cost normalization anchors (must match worker_pool.py).
_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cost_normalized(cost_usd: float) -> float:
    return max(0.0, min(1.0, (cost_usd - _MIN_REF) / (_MAX_REF - _MIN_REF)))


def reward_fn(is_correct: bool, cost_usd: float, alpha: float, floor: float = -1.0) -> float:
    if not is_correct:
        return 0.0
    cn = cost_normalized(cost_usd)
    return max(1.0 - alpha * cn, floor)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json")
    ap.add_argument("--aime", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json")
    ap.add_argument("--wildchat", default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--questions-per-iter", type=int, default=4)
    ap.add_argument("--rollouts-per-q", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--entropy-bonus", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--per-source", action="store_true",
                    help="Use a separate policy per source (math/aime/wildchat). "
                         "If false, one shared policy across all sources — should "
                         "collapse to cheap-tier (matches the 7B trainer behavior).")
    ap.add_argument("--per-difficulty", action="store_true",
                    help="Use a 2-way policy split: 'easy' (math500+wildchat) "
                         "vs 'hard' (aime25). Tests whether a 2-class difficulty "
                         "head gets most of the per-source benefit. Mutually "
                         "exclusive with --per-source.")
    ap.add_argument("--stratified-batch", action="store_true",
                    help="Each GRPO iter samples questions from a SINGLE "
                         "difficulty class (all easy or all hard). Tests "
                         "whether shared-policy GRPO can implicitly learn "
                         "per-difficulty routing if the within-batch advantage "
                         "signal isn't muddled by mixing classes.")
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="With --per-difficulty: prob of mis-routing a "
                         "question to the WRONG difficulty head (symmetric). "
                         "Models a noisy difficulty classifier.")
    ap.add_argument("--label-noise-easy-to-hard", type=float, default=0.0,
                    help="Asymmetric noise: prob of mis-classifying an easy "
                         "question as hard. Independent of --label-noise.")
    ap.add_argument("--label-noise-hard-to-easy", type=float, default=0.0,
                    help="Asymmetric noise: prob of mis-classifying a hard "
                         "question as easy. Independent of --label-noise.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"alpha={args.alpha} iters={args.iters} per_source={args.per_source}")

    rewards_by_q = load_per_question_rewards(args.math, args.aime, args.wildchat)
    sources = list(rewards_by_q.keys())
    print(f"Loaded baselines: {{src: n_questions}} = ", {s: len(rewards_by_q[s]) for s in sources})

    # Initialize policies based on the head topology.
    # Each entry maps a source name -> the policy array it should look up.
    # In per-source: each source has its own array.
    # In per-difficulty: math500/wildchat share an "easy" array, aime25 has "hard".
    # In shared: all sources share one array.
    if args.per_source:
        policies = {s: np.zeros(9) for s in sources}
        topology = "per_source"
    elif args.per_difficulty:
        easy = np.zeros(9)
        hard = np.zeros(9)
        policies = {"math500": easy, "wildchat": easy, "aime25": hard}
        topology = "per_difficulty"
    else:
        shared = np.zeros(9)
        policies = {s: shared for s in sources}
        topology = "shared"
    print(f"topology: {topology}")

    # Mix of questions: every iter picks q_per_iter questions uniformly
    # across the 3 sources.
    qid_per_source = {s: list(rewards_by_q[s].keys()) for s in sources}

    # History
    pick_history: list[dict] = []  # per-iter pick counts
    reward_history: list[float] = []
    accuracy_history: list[float] = []

    for it in range(args.iters):
        iter_picks: collections.Counter = collections.Counter()
        iter_rewards = []
        iter_correct = 0
        iter_total = 0

        # If stratified-batch: pick one difficulty class for the entire iter.
        # Easy = math500 or wildchat; Hard = aime25.
        if args.stratified_batch:
            iter_class = rng.choice(["easy", "hard"])
            if iter_class == "easy":
                allowed = [s for s in sources if s in ("math500", "wildchat")]
            else:
                allowed = ["aime25"]
        else:
            allowed = sources

        for q_choice in range(args.questions_per_iter):
            # Sample a source then a qid within it. Weight by question count
            # so the distribution matches the data mix.
            src_weights = np.array([len(qid_per_source[s]) for s in allowed], dtype=float)
            src_weights /= src_weights.sum()
            src = rng.choice(allowed, p=src_weights)
            qid = rng.choice(qid_per_source[src])
            q_table = rewards_by_q[src][qid]  # ord -> {is_correct, cost_usd}

            # Sample R rollouts from the per-source policy.
            # With --label-noise + --per-difficulty: flip the difficulty assignment
            # with the given probability (models a noisy classifier).
            effective_src = src
            if args.per_difficulty and args.label_noise > 0 and rng.random() < args.label_noise:
                # Flip easy<->hard
                if src in ("math500", "wildchat"):
                    effective_src = "aime25"  # routes through 'hard' head
                else:
                    effective_src = "math500"  # routes through 'easy' head
            policy = policies[effective_src]
            probs = softmax(policy)
            ord_choices = rng.choice(9, size=args.rollouts_per_q, p=probs)

            rewards = np.zeros(args.rollouts_per_q)
            for i, ord_ in enumerate(ord_choices):
                if ord_ not in q_table:
                    rewards[i] = 0.0  # missing data: treat as wrong
                    continue
                rewards[i] = reward_fn(
                    q_table[ord_]["is_correct"],
                    q_table[ord_]["cost_usd"],
                    args.alpha,
                )
                iter_correct += int(q_table[ord_]["is_correct"])
                iter_total += 1

            # Always log the reward for measurement, even if std=0 (no gradient).
            for ord_ in ord_choices:
                iter_picks[int(ord_)] += 1
            iter_rewards.append(rewards.mean())

            # GRPO advantage
            mean_r = rewards.mean()
            std_r = rewards.std()
            if std_r < 1e-8:
                continue  # no signal — but reward already logged above
            advantages = (rewards - mean_r) / std_r

            # Policy gradient update on the per-source logits.
            # gradient of log p(ord) wrt logit[ord_] = (1 - p[ord_]) at chosen, -p[k] otherwise
            # We accumulate scaled advantages.
            grad = np.zeros(9)
            for ord_, adv in zip(ord_choices, advantages):
                grad_one = -probs.copy()
                grad_one[ord_] += 1.0
                grad += adv * grad_one
            grad /= args.rollouts_per_q

            # Entropy regularization (encourages exploration)
            ent_grad = -probs * (np.log(probs + 1e-12) + 1.0)
            grad += args.entropy_bonus * ent_grad

            # SGD step. With label-noise + per-difficulty, the gradient updates
            # the head we ROUTED through (effective_src), not the true source —
            # this models a classifier that confidently mis-labels.
            policies[effective_src] += args.lr * grad

        if iter_rewards:
            mean_r_iter = sum(iter_rewards) / len(iter_rewards)
        else:
            mean_r_iter = 0.0
        reward_history.append(mean_r_iter)
        accuracy_history.append(iter_correct / max(iter_total, 1))
        pick_history.append(dict(iter_picks))

        if it % max(args.iters // 20, 1) == 0:
            top_per_src = {}
            for s in sources:
                p = softmax(policies[s])
                top_ord = int(np.argmax(p))
                top_per_src[s] = (top_ord, round(float(p[top_ord]), 2))
            print(f"  iter {it:>5}  reward={mean_r_iter:+.3f} acc={accuracy_history[-1]:.1%} top: {top_per_src}")

    # Final per-source distributions
    print("\n=== FINAL PER-SOURCE POLICY ===")
    names = ['gemma','gpt-oss','qwen3-32b','qwen-coder','mistral','deepseek','haiku','sonnet','opus']
    for s in sources:
        p = softmax(policies[s])
        ranked = sorted(enumerate(p), key=lambda x: -x[1])
        print(f"\n  {s} (top picks):")
        for ord_, prob in ranked[:5]:
            print(f"    ord_{ord_} {names[ord_]:14s} {prob*100:>5.1f}%")

    # Compare to oracle picks
    print("\n=== ORACLE (alpha={}) PICKS ===".format(args.alpha))
    print("  math500: ord_8 (Opus) — 94% acc, $0.00410/q")
    print("  aime25:  ord_8 (Opus) — 70% acc, $0.00941/q")
    print("  wildchat: ord_3 (Qwen-Coder-480B) — 84% accept, $0.00075/q")

    # Final summary
    final_n = min(50, args.iters // 4)
    final_reward = sum(reward_history[-final_n:]) / final_n
    final_acc = sum(accuracy_history[-final_n:]) / final_n
    print(f"\nFinal {final_n}-iter mean reward: {final_reward:+.3f}")
    print(f"Final {final_n}-iter accuracy:    {final_acc:.1%}")

    # Save
    out = {
        "config": vars(args),
        "final_policies": {s: list(softmax(policies[s])) for s in sources},
        "reward_history_last100": reward_history[-100:],
        "accuracy_history_last100": accuracy_history[-100:],
    }
    out_path = Path(f"domains/autoresearch/blueprints/cost-aware-routing/results/runs/grpo_sim_alpha{args.alpha}_per_source_{args.per_source}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
