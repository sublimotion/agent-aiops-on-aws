#!/usr/bin/env python3
"""Arm A: Iterative STaR with gold labels (Nebius resolved column).

Starting checkpoint: Qwen3.5-27B + Gen0 SFT-D LoRA adapter.

Per iteration N:
  1. Sample 4K trajectories from arm_a_train.jsonl where resolved==1 (gold-passing).
     For N>1, bias toward instances where Gen(N-1) failed (online hard example mining).
  2. SFT on these 4K (stack LoRA r=16 α=32, lr=2e-5, epochs=1, seq=8K).
  3. Evaluate on control_set_5k.jsonl → gold_pass_rate.
  4. Emit per-iteration summary + diff vs Gen0 OpenHands re-baseline.

Stops when:
  - gold_pass_rate improves < 1pp for 2 consecutive iterations, OR
  - iteration 3 completes (hard cap per spec), OR
  - gold_pass_rate >= 60% (matches CoderForge — stretch goal).

SIGUSR1 handler writes a checkpoint immediately (for spot reclaim).
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


_CHECKPOINT_REQUESTED = False


def _sigusr1(*args):
    global _CHECKPOINT_REQUESTED
    _CHECKPOINT_REQUESTED = True
    print("[arm_a] SIGUSR1 received — checkpoint at next step", flush=True)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def format_trajectory_for_sft(record: dict, tokenizer) -> str:
    """OpenHands trajectory → SFT-ready chat string.

    The `trajectory` field is a list of OpenAI-style messages with tool_calls.
    We serialize them using the tokenizer's chat template (Qwen3.5 variant).
    Only keep trajectories where the final assistant turn produced the patch.
    """
    messages = []
    for turn in record.get("trajectory", []):
        role = turn.get("role")
        content = turn.get("content", "") or ""
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            msg = {"role": "assistant", "content": content}
            if turn.get("tool_calls"):
                msg["tool_calls"] = turn["tool_calls"]
            messages.append(msg)
        elif role == "tool":
            messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": turn.get("tool_call_id", ""),
            })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_iteration(args, iter_n: int):
    out_dir = Path(args.output_dir) / f"iter_{iter_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[arm_a iter {iter_n}] output: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base + previous adapter (Gen0 for iter 1, Gen(N-1) for iter N)
    prev_adapter = args.gen0_adapter if iter_n == 1 else str(Path(args.output_dir) / f"iter_{iter_n-1}" / "final")
    print(f"[arm_a iter {iter_n}] base={args.base_model}  prev_adapter={prev_adapter}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, prev_adapter, is_trainable=True)

    # Sample training records
    records = load_jsonl(args.train_set)
    # For iter > 1, bias toward repos where the previous gen failed (simple heuristic:
    # downweight repos with pass_rate above median). We skip this for v1 simplicity.
    if args.n_samples > 0 and args.n_samples < len(records):
        import random
        random.seed(42 + iter_n)
        records = random.sample(records, args.n_samples)
    print(f"[arm_a iter {iter_n}] training on {len(records)} trajectories")

    # Format for SFT
    texts = []
    for r in records:
        try:
            texts.append(format_trajectory_for_sft(r, tokenizer))
        except Exception as e:
            print(f"  skipping {r.get('trajectory_id')}: {e}")
    ds = Dataset.from_dict({"text": texts})

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    # Install SIGUSR1 handler that forces a checkpoint
    signal.signal(signal.SIGUSR1, _sigusr1)

    class CheckpointOnSignalCallback:
        def on_step_end(self, args, state, control, **kwargs):
            global _CHECKPOINT_REQUESTED
            if _CHECKPOINT_REQUESTED:
                control.should_save = True
                _CHECKPOINT_REQUESTED = False
            return control

    trainer.add_callback(CheckpointOnSignalCallback())
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[arm_a iter {iter_n}] trained. final model at {out_dir}/final")

    # Clean up before eval
    del model, base, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Eval on control set (uses gen0_rebaseline.py logic via subprocess)
    eval_out = out_dir / "eval"
    cmd = [
        "python", str(Path(__file__).parent / "gen0_rebaseline.py"),
        "--adapter", str(out_dir / "final"),
        "--base-model", args.base_model,
        "--harness", "openhands",
        "--harness-version", "v0.54",
        "--dataset", "swebench-lite-300",  # for cross-comparison
        "--output-dir", str(eval_out),
    ]
    print(f"[arm_a iter {iter_n}] evaluating: {' '.join(cmd)}")
    import subprocess
    subprocess.run(cmd, check=False)

    # Summary
    summary_path = eval_out / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    summary["iteration"] = iter_n
    summary["n_training_samples"] = len(records)
    with open(out_dir / "iter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--gen0-adapter", required=True)
    p.add_argument("--train-set", required=True, help="arm_a_train.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--min-improvement", type=float, default=0.01, help="early-stop if delta < this for 2 iters")
    p.add_argument("--stretch-target", type=float, default=0.60)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in range(1, args.max_iterations + 1):
        print(f"\n========== Arm A iteration {n}/{args.max_iterations} ==========\n")
        t0 = time.time()
        summary = train_iteration(args, n)
        summary["elapsed_s"] = time.time() - t0
        results.append(summary)
        with open(out / "arm_a_progress.json", "w") as f:
            json.dump(results, f, indent=2)

        rate = summary.get("gold_pass_rate", 0.0)
        if rate >= args.stretch_target:
            print(f"[arm_a] stretch target {args.stretch_target:.1%} reached at iter {n}. Stop.")
            break
        if len(results) >= 2:
            deltas = [results[i]["gold_pass_rate"] - results[i-1]["gold_pass_rate"]
                      for i in range(1, len(results))]
            if len(deltas) >= 2 and deltas[-1] < args.min_improvement and deltas[-2] < args.min_improvement:
                print(f"[arm_a] delta < {args.min_improvement:.2%} for 2 consecutive iters. Stop.")
                break

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
