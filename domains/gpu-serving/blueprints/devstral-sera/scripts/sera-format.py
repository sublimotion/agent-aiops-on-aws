#!/usr/bin/env python3
"""
Convert SERA SVG training data to formats compatible with trl and axolotl.

Input: train.jsonl from sera-datagen.py (chat-format with tool calls)
Output:
  - train_trl.jsonl: trl SFTTrainer compatible format
  - train_axolotl.jsonl: axolotl chat_template format
  - train_stats.json: dataset statistics

Usage:
  python3 sera-format.py --input /mnt/nvme/sera-data/train.jsonl \
      --output-dir /mnt/nvme/sera-data/formatted \
      --max-seq-length 8192
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """Rough token count estimate (4 chars per token for English code)."""
    return len(text) // 4


def clean_messages(messages: list[dict], max_seq_length: int = 8192) -> list[dict]:
    """
    Clean and truncate messages to fit within max_seq_length.

    - Truncate long tool results
    - Remove empty messages
    - Ensure proper alternation
    """
    cleaned = []
    total_tokens = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        # Truncate long tool responses
        if role == "tool" and content and len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"

        # Estimate tokens
        msg_tokens = estimate_token_count(content or "")
        if msg.get("tool_calls"):
            msg_tokens += estimate_token_count(json.dumps(msg["tool_calls"]))

        # Skip if would exceed limit (leave room for response)
        if total_tokens + msg_tokens > max_seq_length * 4:  # tokens * 4 chars
            break

        entry = {"role": role}
        if content is not None:
            entry["content"] = content
        if msg.get("tool_calls"):
            entry["content"] = None
            entry["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]

        cleaned.append(entry)
        total_tokens += msg_tokens

    return cleaned


def format_for_trl(example: dict, max_seq_length: int) -> dict:
    """Format for trl SFTTrainer (OpenAI chat format)."""
    messages = clean_messages(example["messages"], max_seq_length)
    return {
        "messages": messages,
        "instance_id": example.get("instance_id", ""),
    }


def format_for_axolotl(example: dict, max_seq_length: int) -> dict:
    """Format for axolotl (conversation format with tool support)."""
    messages = clean_messages(example["messages"], max_seq_length)

    # Axolotl expects conversations in its own format
    conversations = []
    for msg in messages:
        entry = {"from": msg["role"], "value": msg.get("content", "")}
        if msg.get("tool_calls"):
            entry["value"] = json.dumps(msg["tool_calls"])
            entry["tool_calls"] = True
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]
        conversations.append(entry)

    return {
        "conversations": conversations,
        "instance_id": example.get("instance_id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Format SERA training data")
    parser.add_argument("--input", required=True, help="Input train.jsonl from sera-datagen.py")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length in tokens")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw training data
    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    log.info(f"Loaded {len(examples)} training examples")

    if not examples:
        log.error("No training examples found")
        sys.exit(1)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(examples)
    val_count = max(1, int(len(examples) * args.val_split))
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]

    log.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Format and write
    stats = {
        "total_examples": len(examples),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "max_seq_length": args.max_seq_length,
        "message_counts": [],
        "turn_counts": [],
    }

    for split_name, split_data in [("train", train_examples), ("val", val_examples)]:
        trl_path = os.path.join(args.output_dir, f"{split_name}_trl.jsonl")
        axolotl_path = os.path.join(args.output_dir, f"{split_name}_axolotl.jsonl")

        with open(trl_path, "w") as f_trl, open(axolotl_path, "w") as f_axo:
            for ex in split_data:
                trl_ex = format_for_trl(ex, args.max_seq_length)
                axo_ex = format_for_axolotl(ex, args.max_seq_length)
                f_trl.write(json.dumps(trl_ex) + "\n")
                f_axo.write(json.dumps(axo_ex) + "\n")

                stats["message_counts"].append(len(trl_ex["messages"]))
                assistant_turns = sum(
                    1 for m in trl_ex["messages"] if m["role"] == "assistant"
                )
                stats["turn_counts"].append(assistant_turns)

    # Compute distribution stats
    if stats["message_counts"]:
        mc = stats["message_counts"]
        tc = stats["turn_counts"]
        stats["avg_messages"] = round(sum(mc) / len(mc), 1)
        stats["avg_turns"] = round(sum(tc) / len(tc), 1)
        stats["max_messages"] = max(mc)
        stats["min_messages"] = min(mc)
    stats.pop("message_counts")
    stats.pop("turn_counts")

    with open(os.path.join(args.output_dir, "format_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nFormatted {len(examples)} examples:")
    print(f"  Train: {len(train_examples)} → {args.output_dir}/train_trl.jsonl, train_axolotl.jsonl")
    print(f"  Val:   {len(val_examples)} → {args.output_dir}/val_trl.jsonl, val_axolotl.jsonl")
    print(f"  Stats: {args.output_dir}/format_stats.json")
    print(f"  Avg messages/example: {stats.get('avg_messages', 'n/a')}")
    print(f"  Avg assistant turns:  {stats.get('avg_turns', 'n/a')}")


if __name__ == "__main__":
    main()
