#!/usr/bin/env python3
"""Pre-tokenize the mixed SERA dataset using the chat template.

Run this ONCE (single process) before launching multi-GPU training.
Saves Arrow datasets to disk so SFTTrainer skips the main_process_first()
barrier that causes NCCL timeouts with large datasets.

Usage:
    python3 sera-pretokenize.py
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "/mnt/nvme/models/devstral-small-2-bf16"
TRAIN_DATA = "/mnt/nvme/sera-data/mixed/train_norm.jsonl"
VAL_DATA = "/mnt/nvme/sera-data/mixed/val_norm.jsonl"
OUTPUT_DIR = "/mnt/nvme/sera-data/mixed/tokenized"
MAX_LENGTH = 16384


def tokenize_fn(examples, tokenizer):
    """Apply chat template and tokenize."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_ds = load_dataset("json", data_files=TRAIN_DATA, split="train")
    val_ds = load_dataset("json", data_files=VAL_DATA, split="train")
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val:   {len(val_ds)} examples")

    print(f"Tokenizing train set (num_proc=16, max_length={MAX_LENGTH})...")
    train_tok = train_ds.map(
        tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=100,
        num_proc=16,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    print(f"  Train tokenized: {len(train_tok)} examples")

    print(f"Tokenizing val set...")
    val_tok = val_ds.map(
        tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=100,
        num_proc=4,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )
    print(f"  Val tokenized: {len(val_tok)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train")
    val_path = os.path.join(OUTPUT_DIR, "val")

    print(f"Saving to {OUTPUT_DIR}...")
    train_tok.save_to_disk(train_path)
    val_tok.save_to_disk(val_path)

    print(f"Done!")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"\nNow run: accelerate launch --num_processes 4 --mixed_precision bf16 sera-train.py")


if __name__ == "__main__":
    main()
