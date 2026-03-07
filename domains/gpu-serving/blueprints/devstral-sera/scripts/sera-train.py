#!/usr/bin/env python3
"""SERA LoRA Training for Devstral Small 2.

Single-GPU training with LoRA on FP8 base weights.
Uses pre-tokenized datasets (run sera-pretokenize.py first).

Note: Multi-GPU via DeepSpeed/NCCL is blocked by an NCCL kernel bug on
Blackwell (sm_120) PCIe-only topology (NCCL 2.25.1 / CUDA 12.8).
Single-GPU with gradient accumulation achieves the same effective batch size.

Usage:
    python3 sera-pretokenize.py  # run once
    python3 sera-train.py        # single GPU, no accelerate needed
"""

import os
import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

# --- Configuration ---
MODEL_PATH = "/mnt/nvme/models/devstral-small-2-bf16"
TOKENIZED_DIR = "/mnt/nvme/sera-data/mixed/tokenized"
OUTPUT_DIR = "/mnt/nvme/sera-checkpoints/mixed-run-001"


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading pre-tokenized datasets...")
    train_ds = load_from_disk(os.path.join(TOKENIZED_DIR, "train"))
    val_ds = load_from_disk(os.path.join(TOKENIZED_DIR, "val"))
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val:   {len(val_ds)} examples")
    print(f"  Columns: {train_ds.column_names}")

    print("Configuring LoRA (on FP8 base)...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Single GPU: batch=1, grad_accum=64 for effective batch=64
    # 180K examples / 64 = 2,819 steps per epoch
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=16384,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    print(f"Loading model from {MODEL_PATH} (FP8 weights + LoRA)...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("Creating SFTTrainer (single GPU, pre-tokenized)...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"Starting training: {len(train_ds)} examples, ~{len(train_ds) // 64} steps")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"Done! Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
