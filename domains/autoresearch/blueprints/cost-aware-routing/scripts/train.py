"""GRPO trainer for cost-aware routing — Phase 1 single-pick.

Adapted from rl-conductor/train_conductor_paper.py. Differences:
  - Phase 1 router emits a single ord (not a workflow) — no execute_rollouts step
  - Cost-aware reward (scripts/reward.py) replaces ternary 0/0.5/1
  - Neutral-codes worker prompt (scripts/metadata_prompt.py)
  - CheckpointManager (scripts/checkpoint.py) handles RNG + S3 + resume
  - Reference model loaded on CPU (saves ~16GB GPU); KL only during warmup-after-resume

Usage:
    python -m scripts.train \\
        --alpha 1.0 \\
        --pool configs/pool.yaml \\
        --train-data data/train.jsonl \\
        --s3-prefix s3://agent-aiops-research/cost-aware-routing \\
        [--resume-from-iter N]    # optional; if omitted, finds latest in S3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make scripts importable as a package even when invoked via -m scripts.train
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.checkpoint import (
    CheckpointManager, TrainConfig, TrainState,
    capture_rng_state, restore_rng_state, s3_pull,
)
from scripts.cost import CostModel
from scripts.metadata_prompt import (
    build_router_messages, render_for_generation, cards_from_pool,
)
from scripts.reward import compute_reward, parse_router_output
from scripts.worker_proxy import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate_vllm(vllm_engine, tokenizer, prompts: list[str], max_new_tokens: int,
                        temperature: float = 1.0, top_p: float = 1.0) -> list[str]:
    """vLLM-based batched generation. Returns decoded continuation per prompt."""
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=None,   # let vLLM use its own RNG; we capture rollouts in raw form
    )
    outs = vllm_engine.generate(prompts, sp, use_tqdm=False)
    # Return the .text of the first completion per prompt
    return [o.outputs[0].text for o in outs]


def sync_weights_hf_to_vllm(hf_model, vllm_engine):
    """Push the latest HF policy weights into vLLM's internal model.
    Uses vLLM 0.21's collective_rpc('update_weights') API."""
    state = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
    # Convert to (name, dtype, shape, data) tuples for vLLM
    weights = [(name, tensor) for name, tensor in state.items()]
    # vLLM 0.21: collective_rpc broadcasts to all TP workers
    vllm_engine.collective_rpc("update_weights", args=(weights,))


@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
                   batch_size: int = 32) -> list[str]:
    """HF-based batched router generation. Returns one decoded string per prompt.
    With bnb.AdamW8bit cutting AdamW state 64GB→16GB, bs=32 fits comfortably."""
    all_out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=4096).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, seq in enumerate(out):
            prompt_len = inputs["attention_mask"][j].sum().item()
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            all_out.append(text)
    return all_out


# ---------------------------------------------------------------------------
# Worker call (async over Bedrock pool)
# ---------------------------------------------------------------------------

async def call_workers_for_rollouts(pool: WorkerPool, items: list[dict]) -> list[dict]:
    """Call the Bedrock worker selected by each rollout. Returns list with
    `worker_text`, `worker_input_tok`, `worker_output_tok`, `worker_error`
    appended. Items missing a valid ord (format_fail) are passed through with
    empty worker fields."""
    async def _one(item):
        ord_ = item.get("worker_ord")
        if ord_ is None:
            return {**item, "worker_text": "", "worker_input_tok": 0,
                    "worker_output_tok": 0, "worker_error": "format_fail"}
        result = await pool.call(ord_, item["question"])
        return {
            **item,
            "worker_text": result.text,
            "worker_input_tok": result.input_tok,
            "worker_output_tok": result.output_tok,
            "worker_latency_ms": result.latency_ms,
            "worker_error": result.error or "",
        }
    return await asyncio.gather(*[_one(it) for it in items])


# ---------------------------------------------------------------------------
# GRPO loss (group-relative advantage, no KL except during warmup)
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    model, ref_model, tokenizer, prompts: list[str], rollouts: list[dict],
    questions: list[dict], rollouts_per_q: int, kl_coef: float,
    micro_batch: int = 8,
) -> torch.Tensor:
    """Returns scalar loss tensor. Group-relative advantage normalization per
    question. Optional KL term against frozen reference (CPU) when kl_coef>0
    (used only during the first 5 iters post-resume)."""
    device = next(model.parameters()).device
    rewards = torch.tensor([r["reward"] for r in rollouts], dtype=torch.float32)

    # Group-relative advantage: subtract per-question mean, divide by per-question std.
    rewards_g = rewards.view(len(questions), rollouts_per_q)
    means = rewards_g.mean(dim=1, keepdim=True)
    stds = rewards_g.std(dim=1, keepdim=True).clamp(min=1e-6)
    advs = (rewards_g - means) / stds
    advs = advs.flatten().to(device)

    total_loss = torch.zeros((), device=device)
    n_micro = 0

    for i in range(0, len(rollouts), micro_batch):
        batch_prompts = prompts[i:i + micro_batch]
        batch_outputs = [r["router_text"] for r in rollouts[i:i + micro_batch]]
        batch_advs = advs[i:i + micro_batch]

        full = [p + o for p, o in zip(batch_prompts, batch_outputs)]
        enc = tokenizer(full, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        input_ids = enc.input_ids
        attn = enc.attention_mask

        # Build a mask that is 1 only on the GENERATED tokens (after the prompt).
        prompt_lens = []
        for p in batch_prompts:
            pl = tokenizer(p, return_tensors="pt", truncation=True, max_length=4096
                           ).input_ids.shape[1]
            prompt_lens.append(pl)
        gen_mask = torch.zeros_like(attn)
        for j, pl in enumerate(prompt_lens):
            gen_mask[j, pl:] = attn[j, pl:]

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        # log p of token at position t given prefix [0..t-1] = logits[:, t-1] for token at t
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_lp = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        gen_mask_shift = gen_mask[:, 1:]
        seq_lp = (token_lp * gen_mask_shift).sum(dim=1) / gen_mask_shift.sum(dim=1).clamp(min=1)

        pg_loss = -(batch_advs * seq_lp).mean()

        kl_term = torch.zeros((), device=device)
        if kl_coef > 0:
            # KL against frozen reference (CPU). Compute one micro at a time to keep memory bounded.
            with torch.no_grad():
                ref_logits = ref_model(
                    input_ids=input_ids.cpu(), attention_mask=attn.cpu()
                ).logits.to(device)
            ref_lp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_token_lp = ref_lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            kl = (token_lp - ref_token_lp) * gen_mask_shift
            kl_term = kl.sum() / gen_mask_shift.sum().clamp(min=1)

        loss = pg_loss + kl_coef * kl_term
        loss.backward()
        total_loss = total_loss + loss.detach()
        n_micro += 1

    return total_loss / max(n_micro, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--pool", default="configs/pool.yaml")
    p.add_argument("--train-data", default="data/train.jsonl")
    p.add_argument("--s3-prefix", required=True,
                   help="e.g. s3://agent-aiops-research/cost-aware-routing")
    p.add_argument("--local-dir", default="/mnt/nvme/cost-aware-routing/runs")
    p.add_argument("--resume-from-iter", type=int, default=None,
                   help="iter idx to resume from; if omitted, find latest in S3")
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--pool-seed", type=int, default=17)
    args = p.parse_args()

    cfg = TrainConfig(
        alpha=args.alpha, max_iters=args.max_iters,
        seed=args.seed, pool_seed=args.pool_seed,
    )
    s3_alpha_prefix = f"{args.s3_prefix.rstrip('/')}/checkpoints/alpha{args.alpha}"
    local_alpha_dir = Path(args.local_dir) / f"alpha{args.alpha}"

    ckpt = CheckpointManager(local_alpha_dir, s3_alpha_prefix, cfg)

    # ---- Find resume target ----
    resume_iter = args.resume_from_iter
    if resume_iter is None:
        resume_iter = CheckpointManager.find_latest_iter(s3_alpha_prefix)
        if resume_iter is not None:
            log.info("found latest checkpoint in S3: iter-%d", resume_iter)

    # ---- Load tokenizer + model ----
    log.info("loading base model %s", cfg.base_model)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if resume_iter is not None:
        train_state, rng_state = ckpt.restore_from_s3(resume_iter)
        # CRITICAL: restore RNG BEFORE any random sampling
        restore_rng_state(rng_state)
        random.seed(train_state.seed)  # belt-and-suspenders for any module that re-seeds
        model_path = local_alpha_dir / f"iter-{resume_iter}" / "model"
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map={"": 0},
        )
        start_iter = train_state.iter_idx + 1
        log.info("resumed from iter-%d, resuming at iter %d", resume_iter, start_iter)
    else:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, dtype=torch.bfloat16, device_map={"": 0},
        )
        train_state = TrainState(
            iter_idx=-1, samples_seen=0, alpha=cfg.alpha,
            seed=cfg.seed, pool_seed=cfg.pool_seed,
            config_hash=cfg.hash(),
        )
        start_iter = 0
    model.gradient_checkpointing_enable()

    # vLLM is gated as a future optimization (see batch_generate_vllm).
    # For Phase 1a we use HF generate with batch_size=64 — adequate on a single
    # H100 80GB for an 8B model. Iter time projects to ~3-4 min after this bump.
    vllm_engine = None

    # Reference model on CPU, only used during KL-warmup-after-resume
    log.info("loading reference model on CPU for KL warmup")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="cpu",
    )
    ref_model.eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)

    # Optimizer + scheduler
    # 8-bit AdamW (bitsandbytes) — cuts AdamW state from 64GB (fp32 m+v for 8B) to ~16GB.
    # Required to fit Qwen3-8B + grads + activations + KV cache on a single H100 80GB.
    # Standard for production GRPO trainers (TRL, verl); stochastic rounding compensates
    # for 8-bit precision in m/v moments.
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=0.0)
        log.info("optimizer: bnb.AdamW8bit (8-bit moments; ~16GB state)")
    except ImportError:
        log.warning("bitsandbytes unavailable; falling back to torch.optim.AdamW (~64GB state)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    if resume_iter is not None:
        opt_path = local_alpha_dir / f"iter-{resume_iter}" / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cuda:0", weights_only=False))
            log.info("restored optimizer state from %s", opt_path)

    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = max(1, int(0.03 * cfg.max_iters))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, cfg.max_iters)
    if resume_iter is not None:
        sched_path = local_alpha_dir / f"iter-{resume_iter}" / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=False))

    # ---- Pool + cost model + data ----
    cost_model = CostModel.from_yaml(args.pool)
    pool = WorkerPool(args.pool, seed=cfg.pool_seed)
    cards = cards_from_pool(pool)
    pool_mapping = {f"ord_{o}": {"code": cards[o].code, "bedrock_id": w.bedrock_id}
                    for o, w in pool.workers.items()}
    log.info("pool mapping: %s", json.dumps(pool_mapping, indent=2))

    train_data = [json.loads(l) for l in Path(args.train_data).read_text().splitlines() if l.strip()]
    log.info("loaded %d train items", len(train_data))

    # ---- Training loop ----
    QUESTIONS_PER_BATCH = cfg.batch_size
    ROLLOUTS_PER_Q = cfg.rollouts_per_question

    for it in range(start_iter, cfg.max_iters):
        t_iter = time.monotonic()
        questions = random.sample(train_data, min(QUESTIONS_PER_BATCH, len(train_data)))

        # Build prompts
        prompts = []
        for q in questions:
            msgs = build_router_messages(q["question"], cards)
            rendered = render_for_generation(msgs, tok)
            prompts.extend([rendered] * ROLLOUTS_PER_Q)

        # Generate router outputs via HF (vLLM deferred; see batch_generate_vllm)
        model.eval()
        t_gen = time.monotonic()
        router_outputs = batch_generate(model, tok, prompts, max_new_tokens=cfg.max_router_tokens)
        gen_s = time.monotonic() - t_gen
        # Free the KV cache from generation before backward — required to fit AdamW
        # and activations on a single H100 80GB at this batch size.
        torch.cuda.empty_cache()

        # Parse + dispatch worker calls
        items_for_workers = []
        for q_idx, q in enumerate(questions):
            for r_idx in range(ROLLOUTS_PER_Q):
                idx = q_idx * ROLLOUTS_PER_Q + r_idx
                ord_ = parse_router_output(router_outputs[idx])
                items_for_workers.append({
                    "iter": it, "q_idx": q_idx, "rollout_idx": r_idx,
                    "question": q["question"], "gold": q["answer"],
                    "dataset": q.get("dataset", "math500"),
                    "router_text": router_outputs[idx],
                    "worker_ord": ord_,
                })

        t_workers = time.monotonic()
        rollouts_with_workers = asyncio.run(call_workers_for_rollouts(pool, items_for_workers))
        worker_s = time.monotonic() - t_workers

        # Compute reward per rollout
        rollouts = []
        for item in rollouts_with_workers:
            br = compute_reward(
                router_output=item["router_text"],
                worker_response=item["worker_text"],
                gold_answer=item["gold"],
                dataset=item["dataset"],
                cost_model=cost_model,
                alpha=cfg.alpha,
                actual_input_tok=item["worker_input_tok"],
                actual_output_tok=item["worker_output_tok"],
            )
            rec = {**item, "reward": br.reward, "is_correct": br.is_correct,
                   "format_valid": br.format_valid,
                   "extracted_answer": br.extracted_answer,
                   "extraction_method": br.extraction_method,
                   "cost_dollars": br.cost_dollars,
                   "cost_norm_log": br.cost_norm_log}
            rollouts.append(rec)
        ckpt.append_rollouts(rollouts)

        # GRPO update
        # KL warmup: kl_coef>0 only for first 5 iters AFTER a resume
        kl_coef = (cfg.kl_warmup_coef
                   if (resume_iter is not None and it - start_iter < cfg.kl_warmup_iters)
                   else cfg.kl_coef)
        # LR warmup: ÷10 for first 5 iters after resume, then back to scheduler value
        if resume_iter is not None and it - start_iter < cfg.lr_warmup_iters_post_resume:
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * cfg.lr_warmup_post_resume

        model.train()
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            model, ref_model, tok, prompts, rollouts, questions, ROLLOUTS_PER_Q,
            kl_coef=kl_coef,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Restore LR after warmup
        if resume_iter is not None and it - start_iter == cfg.lr_warmup_iters_post_resume - 1:
            for g in optimizer.param_groups:
                g["lr"] = scheduler.get_last_lr()[0]

        # Metrics
        correct = sum(r["is_correct"] for r in rollouts)
        format_fail = sum(1 for r in rollouts if not r["format_valid"])
        worker_hist = {}
        for r in rollouts:
            o = r["worker_ord"]
            if o is not None:
                worker_hist[o] = worker_hist.get(o, 0) + 1
        mean_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
        mean_cost = sum(r["cost_dollars"] for r in rollouts) / len(rollouts)
        # Histogram entropy in nats — Gate 0.4 monitor
        n_total = sum(worker_hist.values())
        ent = 0.0
        if n_total > 0:
            import math
            for v in worker_hist.values():
                p_ = v / n_total
                if p_ > 0:
                    ent -= p_ * math.log(p_)

        metric_row = {
            "iter": it,
            "loss": float(loss.item()),
            "lr": float(scheduler.get_last_lr()[0]),
            "kl_coef": float(kl_coef),
            "correct_rate": correct / len(rollouts),
            "format_fail_rate": format_fail / len(rollouts),
            "mean_reward": mean_reward,
            "mean_cost_$": round(mean_cost, 5),
            "worker_hist": worker_hist,
            "histogram_entropy": round(ent, 3),
            "gen_s": round(gen_s, 1),
            "worker_s": round(worker_s, 1),
            "iter_s": round(time.monotonic() - t_iter, 1),
        }
        ckpt.append_metric(it, **metric_row)
        log.info("iter %d  correct=%.3f  fmt_fail=%.3f  reward=%.3f  $=%.5f  ent=%.2f  loss=%.4f  %.0fs",
                 it, metric_row["correct_rate"], metric_row["format_fail_rate"],
                 mean_reward, mean_cost, ent, loss.item(), metric_row["iter_s"])

        # Update train state for ckpt
        train_state.iter_idx = it
        train_state.samples_seen += len(rollouts)
        train_state.last_lr = scheduler.get_last_lr()[0]
        train_state.rollout_count += len(rollouts)

        # Sync small files (rollouts/metrics) every 5 iters
        ckpt.maybe_sync(it)

        # Halt conditions (Gate 0.4 / 0.6)
        if metric_row["format_fail_rate"] > 0.05 and it > 5:
            log.warning("[HALT] format_fail_rate > 5%% — chat-template bug? entropy=%.2f", ent)
            ckpt.full_checkpoint(it, model, optimizer, scheduler, train_state, pool_mapping)
            break
        if it > 0 and ent < 0.5:
            log.warning("[HALT] worker-histogram entropy collapsed to %.2f — mode collapse", ent)
            ckpt.full_checkpoint(it, model, optimizer, scheduler, train_state, pool_mapping)
            break

        # Full checkpoint (model+opt+rng+state) every 25 iters
        ckpt.maybe_full_checkpoint(it, model, optimizer, scheduler, train_state, pool_mapping)

    # Final checkpoint
    ckpt.full_checkpoint(cfg.max_iters - 1, model, optimizer, scheduler, train_state, pool_mapping)
    log.info("training complete at iter %d", train_state.iter_idx)
    log.info("worker stats: %s", json.dumps(pool.stats_summary(), indent=2))


if __name__ == "__main__":
    main()
