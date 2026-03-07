#!/usr/bin/env python3
"""Mix AI2 SERA + CoderForge datasets for training.

Reads both JSONL files, shuffles them together, and writes a combined
train/val split. No format normalization — model learns both tool schemas.

Usage:
    python3 sera-mix-datasets.py \
        --sera /mnt/nvme/sera-data/ai2-sera/train_trl.jsonl \
        --coderforge /mnt/nvme/sera-data/coderforge/train_trl.jsonl \
        --output-dir /mnt/nvme/sera-data/mixed \
        --val-ratio 0.02 \
        --max-messages 200 \
        --seed 42
"""

import argparse
import json
import os
import random


def count_messages(line_bytes: bytes) -> int:
    """Fast message count without full JSON parse when possible."""
    obj = json.loads(line_bytes)
    msgs = obj.get("messages", [])
    return len(msgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sera", required=True, help="AI2 SERA JSONL path")
    parser.add_argument("--coderforge", required=True, help="CoderForge JSONL path")
    parser.add_argument("--sera-val", default=None, help="AI2 SERA validation JSONL (optional)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-messages", type=int, default=200,
                        help="Skip examples with more messages than this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    all_lines = []

    # Load SERA
    print(f"Loading SERA from {args.sera}...")
    sera_count = 0
    sera_skipped = 0
    with open(args.sera, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_msgs = count_messages(line)
            if n_msgs > args.max_messages:
                sera_skipped += 1
                continue
            all_lines.append(("sera", line))
            sera_count += 1
            if sera_count % 10000 == 0:
                print(f"  SERA: {sera_count} loaded...")
    print(f"  SERA: {sera_count} loaded, {sera_skipped} skipped (>{args.max_messages} msgs)")

    # Load CoderForge
    print(f"Loading CoderForge from {args.coderforge}...")
    cf_count = 0
    cf_skipped = 0
    with open(args.coderforge, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_msgs = count_messages(line)
            if n_msgs > args.max_messages:
                cf_skipped += 1
                continue
            all_lines.append(("coderforge", line))
            cf_count += 1
            if cf_count % 10000 == 0:
                print(f"  CoderForge: {cf_count} loaded...")
    print(f"  CoderForge: {cf_count} loaded, {cf_skipped} skipped (>{args.max_messages} msgs)")

    total = len(all_lines)
    print(f"\nTotal examples: {total} (SERA: {sera_count}, CoderForge: {cf_count})")
    print(f"Mix ratio: SERA {sera_count/total*100:.1f}%, CoderForge {cf_count/total*100:.1f}%")

    # Shuffle
    print("Shuffling...")
    random.shuffle(all_lines)

    # Split
    val_size = int(total * args.val_ratio)
    train_size = total - val_size
    train_lines = all_lines[:train_size]
    val_lines = all_lines[train_size:]

    # Count sources in each split
    train_sera = sum(1 for s, _ in train_lines if s == "sera")
    train_cf = sum(1 for s, _ in train_lines if s == "coderforge")
    val_sera = sum(1 for s, _ in val_lines if s == "sera")
    val_cf = sum(1 for s, _ in val_lines if s == "coderforge")

    print(f"\nTrain: {train_size} (SERA: {train_sera}, CoderForge: {train_cf})")
    print(f"Val:   {val_size} (SERA: {val_sera}, CoderForge: {val_cf})")

    # Write
    train_path = os.path.join(args.output_dir, "train_trl.jsonl")
    val_path = os.path.join(args.output_dir, "val_trl.jsonl")

    print(f"\nWriting {train_path}...")
    with open(train_path, "wb") as f:
        for _, line in train_lines:
            f.write(line + b"\n")

    print(f"Writing {val_path}...")
    with open(val_path, "wb") as f:
        for _, line in val_lines:
            f.write(line + b"\n")

    # Stats
    train_mb = os.path.getsize(train_path) / 1e6
    val_mb = os.path.getsize(val_path) / 1e6
    print(f"\nDone!")
    print(f"  Train: {train_path} ({train_mb:.0f} MB, {train_size} examples)")
    print(f"  Val:   {val_path} ({val_mb:.0f} MB, {val_size} examples)")


if __name__ == "__main__":
    main()
