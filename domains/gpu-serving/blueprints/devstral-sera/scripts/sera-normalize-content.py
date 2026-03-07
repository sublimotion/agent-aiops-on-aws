#!/usr/bin/env python3
"""Normalize message content fields to strings in mixed JSONL datasets.

Some datasets use list-of-dicts content (e.g., [{"type": "text", "text": "..."}])
while others use plain strings. This normalizes everything to strings for
compatible loading with HuggingFace datasets.

Usage:
    python3 sera-normalize-content.py \
        --input /mnt/nvme/sera-data/mixed/train_trl.jsonl \
        --output /mnt/nvme/sera-data/mixed/train_norm.jsonl
"""

import argparse
import json
import sys


def normalize_content(content):
    """Convert list-of-dicts content to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def normalize_messages(messages):
    """Normalize all messages in a conversation."""
    for msg in messages:
        if "content" in msg:
            msg["content"] = normalize_content(msg["content"])
        # Normalize tool_calls function arguments to string if needed
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict) and "function" in tc:
                    args = tc["function"].get("arguments")
                    if isinstance(args, dict):
                        tc["function"]["arguments"] = json.dumps(args)
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = 0
    normalized = 0
    with open(args.input, "rb") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])

            # Check if any content is not a string
            had_list = any(
                isinstance(m.get("content"), list) for m in msgs
            )
            if had_list:
                normalized += 1

            obj["messages"] = normalize_messages(msgs)
            fout.write(json.dumps(obj) + "\n")
            count += 1
            if count % 50000 == 0:
                print(f"  Processed {count} ({normalized} normalized)...")

    print(f"Done: {count} total, {normalized} had list content -> normalized")


if __name__ == "__main__":
    main()
