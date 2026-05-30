"""GRPO trainer for cost-aware-routing Phase 1.

Adapted from rl-conductor Phase 1.6 trainer (vendored at
vendor/rl-conductor-phase1/scripts/train_conductor.py). Key changes:

  Output format:    multi-step workflow (subtasks/model_id/access_list)
                    -> single-pick "PICK ord_N"
  Reward function:  ternary 0/0.5/1
                    -> max(1 - alpha * cost_normalized, -1) | 0
                       (using Haiku-as-judge via cost_reward.score_rollouts)
  Worker pool:      old WorkerPool(phase=1) over 8 ords
                    -> our worker_pool.POOL over 9 Bedrock-only ords
  Prompt:           rl-conductor system prompt + question
                    -> build_metadata_prompt() + render_few_shot_block() + question
  Training data:    rl-conductor train.jsonl (MATH only)
                    -> mix of MATH500 + AIME25 (train split) + WildChat 300
  Iter-0 gate:      none
                    -> 256-rollout histogram diagnostic before iter 0; abort
                       on degenerate distribution (any worker >25% or <2%).

Run:
  python3 train_cost_aware_router.py --alpha 1.0 --num-iters 50 \\
      --output-dir /opt/dlami/nvme/cost-aware-routing/runs/alpha1.0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local modules (assumed on sys.path; trainer is normally invoked from
# the cost-aware-routing/scripts/ directory or with PYTHONPATH set).
from cost_reward import Rollout, score_rollouts
from few_shot import render_few_shot_block
from worker_pool import POOL, build_metadata_prompt, invoke_worker, per_call_cost_usd

# === Defaults (overridable via argparse) ===
DEFAULT_MODEL = "/opt/dlami/nvme/models/Qwen2.5-7B"
DEFAULT_DATA = "/opt/dlami/nvme/cost-aware-routing/data/train.jsonl"
DEFAULT_OUTPUT = "/opt/dlami/nvme/cost-aware-routing/runs/alpha1.0"
DEFAULT_S3_BUCKET = "agent-aiops-checkpoints"
DEFAULT_S3_PREFIX = "cost-aware-routing"

# Pull this constant from cost_reward to keep the floor consistent
REWARD_FLOOR = -1.0


PICK_RE = re.compile(r"PICK\s+ord[_\s]*(\d)", re.IGNORECASE)


def parse_pick(text: str) -> int | None:
    """Extract the worker_id from the router's response.

    Expected format: "PICK ord_N" (any case) on the first non-empty line.
    Returns int 0..8 or None if unparseable / out of range.
    """
    m = PICK_RE.search(text)
    if not m:
        return None
    ord_ = int(m.group(1))
    if 0 <= ord_ < len(POOL):
        return ord_
    return None


FEW_SHOT_MODE = "balanced"  # one of: 'balanced', 'format_only', 'none'


def build_router_prompt(question: str) -> str:
    """Build the system+few-shot prompt and append the live question.

    FEW_SHOT_MODE controls the few-shot block:
      'balanced'    — original 9 type-coded examples (math→DeepSeek, code→Qwen-Coder, ...)
                      Anchors the model's prior to those (q_type, worker) pairs.
                      See results/runs/prompt_variants.json — this drove the
                      24% DeepSeek bias at iter 0.
      'format_only' — 9 examples all on the SAME generic question, picks rotated
                      0..8. Teaches output format without prescribing routing.
      'none'        — metadata header only. Hard-fails iter-0 gate (model
                      collapses to ord_1 42%) — most workers go to 0%.
    """
    head = build_metadata_prompt()
    if FEW_SHOT_MODE == "balanced":
        return head + "\n\n" + render_few_shot_block() + f"\nQuestion: {question}\nAnswer:"
    if FEW_SHOT_MODE == "format_only":
        from few_shot_format_only import render_format_only_block
        return head + "\n\n" + render_format_only_block() + f"\nQuestion: {question}\nAnswer:"
    if FEW_SHOT_MODE == "none":
        return head + f"\n\nQuestion: {question}\nAnswer:"
    raise ValueError(f"Unknown FEW_SHOT_MODE: {FEW_SHOT_MODE}")


def load_training_data(path: str) -> list[dict]:
    """Load training records. Each must have {id, question, source} and
    optionally {gold} (math/aime) or empty (wildchat = LLM-judge scored).
    """
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "source" not in r:
                # Default to math500 if not tagged
                r["source"] = "math500"
            out.append(r)
    return out


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, output in enumerate(outputs):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            all_outputs.append(generated)
    return all_outputs


def execute_rollouts(
    generated_texts: list[str],
    questions: list[dict],
    rollouts_per_q: int,
    alpha: float,
) -> list[dict]:
    """For each generated routing decision, invoke the picked worker on
    Bedrock, judge correctness, compute the cost-aware reward.

    Returns one dict per rollout with keys:
      response, picked_ord, worker_text, in_tok, out_tok, cost_usd,
      is_correct, reward, judge_raw.
    """
    import concurrent.futures as cf
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
    )

    # Phase A: parse all picks first, then invoke workers concurrently.
    rollout_data: list[dict | None] = [None] * len(generated_texts)
    pick_failures = 0
    invoke_jobs: list[tuple[int, int, dict]] = []  # (rollout_idx, picked_ord, q)
    for idx, gen_text in enumerate(generated_texts):
        q_idx = idx // rollouts_per_q
        q = questions[q_idx]
        picked = parse_pick(gen_text)
        if picked is None:
            pick_failures += 1
            rollout_data[idx] = {
                "q": q,
                "gen_text": gen_text,
                "picked_ord": None,
                "worker_text": "",
                "in_tok": 0,
                "out_tok": 0,
                "is_correct": False,
                "cost_usd": 0.0,
                "reward": 0.0,
                "judge_raw": "PARSE_FAIL",
            }
        else:
            rollout_data[idx] = {
                "q": q,
                "gen_text": gen_text,
                "picked_ord": picked,
            }
            invoke_jobs.append((idx, picked, q))

    # Concurrent worker invocations (16-way parallel).
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futures = {
            ex.submit(invoke_worker, bedrock, p, q["question"], 1024, 0.7): i
            for (i, p, q) in invoke_jobs
        }
        for fut in cf.as_completed(futures):
            i = futures[fut]
            r = fut.result()
            rollout_data[i].update({
                "worker_text": r["text"] if r["error"] is None else "",
                "in_tok": r["input_tokens"],
                "out_tok": r["output_tokens"],
                "worker_error": r["error"],
            })

    # Phase B: judge with Haiku via cost_reward.score_rollouts.
    # Build Rollout objects only for parseable picks; failures get reward 0.
    judgeable = [
        Rollout(
            question=rd["q"]["question"],
            gold=rd["q"].get("gold", ""),  # wildchat has no gold; cost_reward
                                           # uses a wildchat path elsewhere if needed
            worker_ord=rd["picked_ord"],
            worker_response=rd["worker_text"],
            worker_input_tokens=rd["in_tok"],
            worker_output_tokens=rd["out_tok"],
        )
        for rd in rollout_data
        if rd["picked_ord"] is not None
    ]

    # NOTE: cost_reward.score_rollouts uses the math judge prompt. For
    # wildchat questions we should use the wildchat_judge prompt, but
    # for the smoke run we keep one judge for simplicity. Phase 1 will
    # branch on q["source"].
    judged = score_rollouts(judgeable, alpha=alpha, workers=16)

    # Stitch judgements back into rollout_data
    j_iter = iter(judged)
    for rd in rollout_data:
        if rd["picked_ord"] is None:
            continue
        res = next(j_iter)
        rd["is_correct"] = res.is_correct
        rd["cost_usd"] = res.cost_usd
        rd["reward"] = res.reward
        rd["judge_raw"] = res.judge_raw

    return rollout_data


def _find_pick_token_position(
    prompt: str, gen_text: str, tokenizer
) -> tuple[int | None, int | None]:
    """Locate the token index of the digit N in the live PICK line.

    The few-shot block in the prompt contains many `PICK ord_K` lines;
    the routing decision we want to train on is in `gen_text` only.
    Returns (digit_token_idx_in_full, expected_digit_token_id).
    """
    m = PICK_RE.search(gen_text)  # search only the generated portion
    if not m:
        return None, None
    digit_char_pos_in_gen = m.start(1)
    # Char index in the concatenated prompt+gen_text:
    digit_char_pos = len(prompt) + digit_char_pos_in_gen
    full_text = prompt + gen_text
    prefix = full_text[:digit_char_pos]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    digit_position = len(prefix_ids)
    expected_id = tokenizer(m.group(1), add_special_tokens=False)["input_ids"]
    if not expected_id:
        return None, None
    return digit_position, expected_id[0]


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: list[str],
    rollout_data: list[dict],
    rollouts_per_q: int,
    device: torch.device,
    max_contributing: int = 4,
) -> torch.Tensor:
    """GRPO loss with within-question advantage normalization.

    Critical change vs rl-conductor's loss: cross-entropy is computed
    *only* at the worker_id digit token, not over the full response.
    The router's decision is one token; everything else is free-form
    justification text whose log-prob is irrelevant to the routing
    objective.
    """
    n_contributing = 0
    accumulated_loss = 0.0
    n_questions = len(rollout_data) // rollouts_per_q

    for q_idx in range(n_questions):
        start = q_idx * rollouts_per_q
        end = start + rollouts_per_q
        group = rollout_data[start:end]
        prompt = prompts[start]

        rewards = torch.tensor([r["reward"] for r in group], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std()
        if std_r < 1e-6:
            continue
        advantages = (rewards - mean_r) / std_r

        for r, adv in zip(group, advantages):
            if abs(adv.item()) < 0.01:
                continue
            if r["picked_ord"] is None:
                continue  # parse-fail rollouts contribute no gradient

            digit_pos, expected_id = _find_pick_token_position(
                prompt, r["gen_text"], tokenizer
            )
            if digit_pos is None:
                continue

            full_text = prompt + r["gen_text"]
            encoded = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=4096
            ).to(device)
            input_ids = encoded["input_ids"]
            seq_len = input_ids.shape[1]

            # Skip if truncation killed the digit position.
            if digit_pos >= seq_len:
                continue

            out = model(**encoded)  # logits only; no labels
            # logits at position digit_pos-1 predict input_ids at digit_pos.
            if digit_pos == 0:
                continue
            digit_logits = out.logits[0, digit_pos - 1, :]  # (vocab,)
            target = torch.tensor([expected_id], device=device)
            ce = torch.nn.functional.cross_entropy(
                digit_logits.unsqueeze(0), target, reduction="mean"
            )
            scaled_loss = (-adv.to(device) * ce) / 4
            scaled_loss.backward()
            accumulated_loss += scaled_loss.item()
            n_contributing += 1
            del out, encoded, scaled_loss, digit_logits
            torch.cuda.empty_cache()
            if n_contributing >= max_contributing:
                break
        if n_contributing >= max_contributing:
            break

    return torch.tensor(accumulated_loss * 4 if n_contributing > 0 else 0.0)


def iter0_histogram_gate(
    model,
    tokenizer,
    train_data: list[dict],
    n_rollouts: int = 256,
    gen_batch: int = 16,
    rng: random.Random | None = None,
) -> dict:
    """Pre-training diagnostic: sample n_rollouts questions and route each.

    Pass criterion: every ord picked between 5-20% (target 11%±6pp).
    Hard-fail if any ord is below 2% OR Opus (ord_8) above 25%.

    Returns {pick_counts, parse_failures, status: 'pass'|'soft_fail'|'hard_fail'}.
    """
    rng = rng or random.Random(17)
    questions = [rng.choice(train_data) for _ in range(n_rollouts)]
    prompts = [build_router_prompt(q["question"]) for q in questions]

    print(f"\n=== Iter-0 histogram gate (n={n_rollouts}, single rollout per Q) ===")
    print("Generating...")
    t0 = time.time()
    model.eval()
    texts = batch_generate(
        model, tokenizer, prompts,
        batch_size=gen_batch, max_new_tokens=64, temperature=0.7,
    )
    gen_time = time.time() - t0

    picks: collections.Counter = collections.Counter()
    parse_fail = 0
    for t in texts:
        p = parse_pick(t)
        if p is None:
            parse_fail += 1
        else:
            picks[p] += 1

    print(f"Generated in {gen_time:.0f}s. Pick distribution:")
    print(f"  ord  worker             count   pct")
    for ord_ in range(len(POOL)):
        cnt = picks.get(ord_, 0)
        pct = 100 * cnt / n_rollouts
        flag = ""
        if pct < 2.0:
            flag = "  ⚠️ <2% (hard-fail floor)"
        elif pct < 5.0:
            flag = "  (below 5% target)"
        elif pct > 25.0 and ord_ == 8:
            flag = "  ⚠️ Opus >25% (hard-fail brand bias)"
        elif pct > 20.0:
            flag = "  (above 20% target)"
        print(f"  ord_{ord_}  {POOL[ord_].name:18s} {cnt:>5d}  {pct:>5.1f}%{flag}")
    if parse_fail:
        print(f"  parse_failures: {parse_fail} ({100*parse_fail/n_rollouts:.1f}%)")

    # Decide pass/fail
    status = "pass"
    for ord_ in range(len(POOL)):
        pct = 100 * picks.get(ord_, 0) / n_rollouts
        if pct < 2.0:
            status = "hard_fail"
            break
        if pct < 5.0 or pct > 20.0:
            status = "soft_fail"
    if picks.get(8, 0) / n_rollouts > 0.25:
        status = "hard_fail"
    if parse_fail / n_rollouts > 0.10:
        status = "hard_fail"

    print(f"\nGate status: {status.upper()}")
    return {
        "pick_counts": dict(picks),
        "parse_failures": parse_fail,
        "n_rollouts": n_rollouts,
        "status": status,
    }


def save_checkpoint(model, tokenizer, optimizer, iteration: int, output_dir: Path,
                    s3_bucket: str | None, s3_prefix: str, alpha: float):
    ckpt_path = output_dir / f"iter-{iteration:04d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    if s3_bucket:
        try:
            s3 = boto3.client("s3")
            for f in ckpt_path.rglob("*"):
                if f.is_file():
                    key = f"{s3_prefix}/alpha{alpha}/iter-{iteration:04d}/{f.relative_to(ckpt_path)}"
                    s3.upload_file(str(f), s3_bucket, key)
        except Exception as e:
            print(f"  S3 upload failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--num-iters", type=int, default=200)
    ap.add_argument("--questions-per-batch", type=int, default=4)
    ap.add_argument("--rollouts-per-question", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-gen-tokens", type=int, default=128)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    ap.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-iter0-gate", action="store_true",
                    help="Skip the iter-0 histogram diagnostic; for resume runs.")
    ap.add_argument("--save-final", action="store_true",
                    help="Save final checkpoint regardless of checkpoint-every. "
                         "Off by default for smoke runs.")
    ap.add_argument("--few-shot-mode", choices=["balanced", "format_only", "none"],
                    default="balanced",
                    help="balanced: original type-coded 9-shot (anchors routing). "
                         "format_only: 9 examples on one generic question with rotated "
                         "picks (teaches format only). none: metadata only (typically "
                         "iter-0 hard-fails).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    global FEW_SHOT_MODE
    FEW_SHOT_MODE = args.few_shot_mode
    print(f"few-shot mode: {FEW_SHOT_MODE}")

    print("=== Cost-aware-routing GRPO trainer ===")
    print(f"alpha={args.alpha} num_iters={args.num_iters}")
    print(f"batch: {args.questions_per_batch}q × {args.rollouts_per_question}r "
          f"= {args.questions_per_batch * args.rollouts_per_question} rollouts/iter")
    print(f"output: {output_dir}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
    )
    model.gradient_checkpointing_enable()

    print(f"\nLoading training data from {args.data}...")
    train_data = load_training_data(args.data)
    print(f"  {len(train_data)} questions")
    src_counts: collections.Counter = collections.Counter(r["source"] for r in train_data)
    print(f"  by source: {dict(src_counts)}")

    # Iter-0 gate
    if not args.skip_iter0_gate:
        gate = iter0_histogram_gate(model, tokenizer, train_data, n_rollouts=256,
                                    gen_batch=args.gen_batch, rng=rng)
        gate_path = output_dir / "iter0_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2))
        if gate["status"] == "hard_fail":
            print("\n⚠️  Iter-0 histogram gate HARD-FAILED. Aborting before training compute.")
            print("   Adjust the few-shot prompt and re-run.")
            sys.exit(2)
        if gate["status"] == "soft_fail":
            print("\n⚠️  Iter-0 histogram gate soft-failed (some ords outside 5-20% band). "
                  "Continuing — GRPO is robust to imperfect priors but watch the histogram drift.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters)

    log_file = open(log_path, "a")

    # Training loop
    device = torch.device("cuda:0")
    for iteration in range(args.num_iters):
        iter_start = time.time()
        questions = rng.sample(
            train_data, min(args.questions_per_batch, len(train_data))
        )
        prompts = []
        for q in questions:
            p = build_router_prompt(q["question"])
            prompts.extend([p] * args.rollouts_per_question)

        gen_start = time.time()
        model.eval()
        gen_texts = batch_generate(
            model, tokenizer, prompts,
            batch_size=args.gen_batch,
            max_new_tokens=args.max_gen_tokens,
            temperature=0.7,
        )
        gen_time = time.time() - gen_start

        exec_start = time.time()
        rollout_data = execute_rollouts(
            gen_texts, questions, args.rollouts_per_question, alpha=args.alpha
        )
        exec_time = time.time() - exec_start

        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            model, tokenizer, prompts, rollout_data,
            args.rollouts_per_question, device,
        )
        if loss.item() != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_time = time.time() - train_start

        # Stats
        rewards = [r["reward"] for r in rollout_data]
        n_correct = sum(1 for r in rollout_data if r["is_correct"])
        n_parse_fail = sum(1 for r in rollout_data if r["picked_ord"] is None)
        avg_cost = sum(r["cost_usd"] for r in rollout_data) / len(rollout_data)
        worker_picks: collections.Counter = collections.Counter(
            r["picked_ord"] for r in rollout_data if r["picked_ord"] is not None
        )

        stats = {
            "iteration": iteration,
            "alpha": args.alpha,
            "loss": loss.item() if hasattr(loss, "item") else 0.0,
            "mean_reward": sum(rewards) / len(rewards),
            "reward_std": (
                sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)
            ) ** 0.5,
            "correct_rate": n_correct / len(rollout_data),
            "parse_fail_rate": n_parse_fail / len(rollout_data),
            "avg_cost_usd": round(avg_cost, 6),
            "worker_picks": {str(k): v for k, v in worker_picks.items()},
            "lr": scheduler.get_last_lr()[0],
            "gen_time_s": round(gen_time, 1),
            "exec_time_s": round(exec_time, 1),
            "train_time_s": round(train_time, 1),
            "wall_time_s": round(time.time() - iter_start, 1),
        }
        log_file.write(json.dumps(stats) + "\n")
        log_file.flush()

        print(
            f"[Iter {iteration:03d}] loss={stats['loss']:+.4f} "
            f"reward={stats['mean_reward']:+.3f}±{stats['reward_std']:.3f} "
            f"correct={stats['correct_rate']:.0%} "
            f"parse_fail={stats['parse_fail_rate']:.0%} "
            f"$/q={stats['avg_cost_usd']:.5f} "
            f"[gen={gen_time:.0f}s exec={exec_time:.0f}s train={train_time:.0f}s "
            f"total={stats['wall_time_s']:.0f}s]"
        )
        if worker_picks:
            usage = " ".join(f"{POOL[k].name}:{v}" for k, v in sorted(worker_picks.items()))
            print(f"           workers: {usage}")

        if (iteration + 1) % args.checkpoint_every == 0:
            save_checkpoint(model, tokenizer, optimizer, iteration, output_dir,
                            args.s3_bucket, args.s3_prefix, args.alpha)

    if args.save_final:
        save_checkpoint(model, tokenizer, optimizer, args.num_iters - 1, output_dir,
                        args.s3_bucket, args.s3_prefix, args.alpha)
    log_file.close()
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
