#!/usr/bin/env python3
"""
Phase 2: SFT training on filtered CoderForge trajectories.

Trains QLoRA on Qwen3.5-32B using accepted trajectories from Phase 1.
Syncs checkpoints to S3 periodically (spot instance protection).

Usage:
  python3 train_sft.py --config A --model Qwen/Qwen3.5-Coder-32B-Instruct
  python3 train_sft.py --config D --model Qwen/Qwen3.5-Coder-32B-Instruct  # gold baseline
  python3 train_sft.py --config B --resume-from s3://bucket/path/checkpoint-500

Configs:
  1/none  — all data (no filter)
  A       — RF raw p>0.5
  B       — Cascade calibrated p>0.5
  C       — Cascade calibrated p>0.7
  D       — Gold labels only (reward=1)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

S3_BUCKET = "s3://agent-aiops-artifacts"
S3_PREFIX = "rejection-sampling-sft"
WORK_DIR = Path("/mnt/nvme/rejection-sampling-sft")
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"


def sync_to_s3(local_path, s3_path, description=""):
    """Sync a local directory or file to S3."""
    cmd = f"aws s3 sync {local_path} {s3_path} --quiet"
    if not Path(local_path).is_dir():
        cmd = f"aws s3 cp {local_path} {s3_path} --quiet"
    print(f"  [S3 sync] {description}: {local_path} → {s3_path}")
    try:
        subprocess.run(cmd, shell=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        print(f"  [S3 sync] WARNING: sync timed out for {description}")


def sync_from_s3(s3_path, local_path):
    """Download from S3 to local."""
    cmd = f"aws s3 sync {s3_path} {local_path} --quiet"
    subprocess.run(cmd, shell=True, timeout=300, check=False)


class S3CheckpointCallback:
    """Syncs checkpoints to S3 on each save."""

    def __init__(self, run_name, sync_every_steps=500):
        self.run_name = run_name
        self.sync_every_steps = sync_every_steps
        self.s3_base = f"{S3_BUCKET}/{S3_PREFIX}/checkpoints/{run_name}"

    def on_save(self, output_dir):
        sync_to_s3(output_dir, f"{self.s3_base}/{Path(output_dir).name}", "checkpoint")

    def on_train_end(self, output_dir):
        sync_to_s3(output_dir, f"{self.s3_base}/final", "final model")


def prepare_sft_data(scores_path, config, messages_dir):
    """Load scores, filter by config, return list of trajectory row indices."""
    all_scores = []
    with open(scores_path) as f:
        for line in f:
            all_scores.append(json.loads(line))

    config_filters = {
        "none": lambda r: True,  # all data
        "a": lambda r: r["rf_prob"] > 0.5,
        "b": lambda r: r["calibrated_prob"] > 0.5,
        "c": lambda r: r["calibrated_prob"] > 0.7,
        "d": lambda r: r["gold_label"] == 1,
    }

    pred_fn = config_filters.get(config.lower())
    if pred_fn is None:
        raise ValueError(f"Unknown config: {config}. Use: none, A, B, C, D")

    accepted = [s for s in all_scores if pred_fn(s)]
    print(f"Config {config.upper()}: {len(accepted)}/{len(all_scores)} accepted "
          f"({len(accepted)/len(all_scores)*100:.1f}%)")
    return accepted


def main():
    parser = argparse.ArgumentParser(description="SFT training on filtered data")
    parser.add_argument("--config", type=str, required=True,
                        help="Filter config: none, A, B, C, D")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-Coder-32B-Instruct")
    parser.add_argument("--scores-path", type=str, default=None,
                        help="Path to scores JSONL from Phase 1")
    parser.add_argument("--split", type=str, default="SWE_Rebench")
    parser.add_argument("--n-traces", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (for quick test runs)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--s3-sync", action="store_true", default=True,
                        help="Sync checkpoints to S3 (default: True)")
    parser.add_argument("--no-s3-sync", action="store_false", dest="s3_sync")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint (local path or s3://...)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare data only, don't train")
    args = parser.parse_args()

    run_name = f"run_{args.config.lower()}_{args.model.split('/')[-1]}"
    output_dir = MODELS_DIR / run_name

    print("=" * 60)
    print(f"SFT TRAINING: Config {args.config.upper()}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Output: {output_dir}")
    print(f"S3 sync: {args.s3_sync}")

    # Find scores file
    scores_path = args.scores_path
    if scores_path is None:
        scores_path = DATA_DIR / f"scores_{args.split}_{args.n_traces}.jsonl"
        if not Path(scores_path).exists():
            # Try to download from S3
            s3_scores = f"{S3_BUCKET}/{S3_PREFIX}/data/scores_{args.split}_{args.n_traces}.jsonl"
            print(f"Scores not found locally, trying S3: {s3_scores}")
            os.makedirs(DATA_DIR, exist_ok=True)
            subprocess.run(f"aws s3 cp {s3_scores} {scores_path} --quiet",
                         shell=True, check=False)

    if not Path(scores_path).exists():
        print(f"ERROR: Scores file not found: {scores_path}")
        print("Run filter_trajectories.py first.")
        sys.exit(1)

    # Filter data
    accepted = prepare_sft_data(scores_path, args.config, DATA_DIR)
    accepted_indices = {s["row_idx"] for s in accepted}

    if args.max_samples and len(accepted_indices) > args.max_samples:
        import random
        random.seed(42)
        accepted_indices = set(random.sample(sorted(accepted_indices), args.max_samples))
        print(f"Capped to {args.max_samples} samples")

    # Sync filter stats to S3
    if args.s3_sync:
        sync_to_s3(str(scores_path), f"{S3_BUCKET}/{S3_PREFIX}/data/{Path(scores_path).name}",
                   "scores")

    if args.dry_run:
        print("\nDry run — skipping training.")
        return

    # Resume from checkpoint if specified
    resume_checkpoint = args.resume_from
    if resume_checkpoint and resume_checkpoint.startswith("s3://"):
        local_ckpt = output_dir / "resumed_checkpoint"
        print(f"Downloading checkpoint from {resume_checkpoint}...")
        sync_from_s3(resume_checkpoint, str(local_ckpt))
        resume_checkpoint = str(local_ckpt)

    # =========================================================
    # Load model with QLoRA + Flash Attention 2
    # =========================================================
    print("\nLoading model...")
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =========================================================
    # Prepare dataset
    # =========================================================
    print("\nPreparing training dataset...")
    from datasets import load_dataset as hf_load

    ds = hf_load(
        "togethercomputer/CoderForge-Preview",
        "trajectories",
        split=args.split,
        streaming=True,
    )

    # Convert to chat format and filter
    def format_for_sft(row, idx):
        """Convert CoderForge row to chat-formatted text for SFT."""
        if idx not in accepted_indices:
            return None

        messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

        # Convert to standard chat format the tokenizer expects
        chat_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            # Flatten tool_calls into content for SFT
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and role == "assistant":
                tc_text = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    tc_text.append(f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>")
                content = (content + "\n" if content else "") + "\n".join(tc_text)

            if content.strip():
                chat_messages.append({"role": role, "content": content.strip()})

        if not chat_messages:
            return None

        try:
            text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return text
        except Exception:
            return None

    # Stream and collect formatted examples
    print("Streaming and formatting accepted trajectories...")
    formatted_texts = []
    n_seen = 0
    for idx, row in enumerate(ds):
        if idx > max(accepted_indices, default=0) + 100:
            break  # past all accepted indices
        text = format_for_sft(row, idx)
        if text is not None:
            # Truncate to max_seq_length tokens (approximate)
            if len(text) > args.max_seq_length * 5:  # rough 5 chars/token
                text = text[:args.max_seq_length * 5]
            formatted_texts.append({"text": text})
            n_seen += 1
            if n_seen % 1000 == 0:
                print(f"  Formatted {n_seen} / {len(accepted_indices)} accepted")

    print(f"  Total formatted: {len(formatted_texts)}")

    # Convert to HF dataset
    from datasets import Dataset
    train_ds = Dataset.from_list(formatted_texts)

    # Save formatted data to S3
    if args.s3_sync:
        formatted_path = DATA_DIR / f"formatted_{args.config.lower()}.jsonl"
        with open(formatted_path, "w") as f:
            for item in formatted_texts[:10]:  # save first 10 as sample
                f.write(json.dumps(item) + "\n")
        sync_to_s3(str(formatted_path),
                   f"{S3_BUCKET}/{S3_PREFIX}/data/{formatted_path.name}", "sample data")

    # =========================================================
    # Train
    # =========================================================
    print(f"\nStarting SFT training ({len(train_ds)} samples)...")
    os.makedirs(output_dir, exist_ok=True)

    s3_cb = S3CheckpointCallback(run_name) if args.s3_sync else None

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # Custom save callback for S3 sync
    if s3_cb:
        from transformers import TrainerCallback

        class S3SyncCallback(TrainerCallback):
            def on_save(self, _args, state, control, **kwargs):
                ckpt_dir = output_dir / f"checkpoint-{state.global_step}"
                s3_cb.on_save(str(ckpt_dir))

            def on_train_end(self, _args, state, control, **kwargs):
                s3_cb.on_train_end(str(output_dir / "final"))

        trainer.add_callback(S3SyncCallback())

    t0 = time.time()
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    elapsed = time.time() - t0

    # Save final adapter
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.s3_sync:
        sync_to_s3(str(final_dir), f"{S3_BUCKET}/{S3_PREFIX}/checkpoints/{run_name}/final",
                   "final adapter")

    # Save training results
    train_results = {
        "run_name": run_name,
        "config": args.config.upper(),
        "model": args.model,
        "n_samples": len(train_ds),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "elapsed_s": round(elapsed, 1),
        "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
        "output_dir": str(final_dir),
    }

    results_path = RESULTS_DIR / f"training_{args.config.lower()}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(train_results, f, indent=2)

    if args.s3_sync:
        sync_to_s3(str(results_path),
                   f"{S3_BUCKET}/{S3_PREFIX}/results/{results_path.name}", "training results")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Config: {args.config.upper()}")
    print(f"  Samples: {len(train_ds)}")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Final loss: {train_results['final_loss']}")
    print(f"  Adapter: {final_dir}")
    print(f"  S3: {S3_BUCKET}/{S3_PREFIX}/checkpoints/{run_name}/final")


if __name__ == "__main__":
    main()
