#!/usr/bin/env python3
"""
Diff preprocessor: strip cosmetic-only hunks, keep functional changes.

Addresses:
  - FM-001: Reformatting noise hiding functional fixes
  - Truncation: 47% of sonnet diffs exceed 12K chars, 3/6 passing patches truncated

Strategy:
  1. Parse diff into hunks
  2. Classify each hunk as cosmetic or functional
  3. Return only functional hunks (or all if classification is uncertain)
  4. Apply smart truncation: cosmetic hunks first, then long functional hunks

A hunk is cosmetic if ALL changes are:
  - Whitespace-only (indentation, trailing spaces)
  - Quote style changes (single ↔ double)
  - Import reordering (same imports, different order)
  - Line wrapping (same content, different line breaks)
  - Comment-only changes

Usage:
  python3 preprocess_diff.py input.diff                    # Print functional-only diff
  python3 preprocess_diff.py input.diff --max-chars 50000  # With size limit
  python3 preprocess_diff.py input.diff --stats             # Print hunk statistics
"""

import argparse
import re
import sys
from pathlib import Path


def parse_hunks(diff_text: str) -> list[dict]:
    """Parse a unified diff into file-level groups of hunks."""
    hunks = []
    current_file = None
    current_header = []
    current_lines = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            # Save previous hunk group
            if current_file and current_lines:
                hunks.append({
                    "file": current_file,
                    "header": "\n".join(current_header),
                    "lines": current_lines[:],
                    "raw": "\n".join(current_header + current_lines),
                })
            current_file = _extract_filename(line)
            current_header = [line]
            current_lines = []
        elif line.startswith("---") or line.startswith("+++") or line.startswith("index "):
            current_header.append(line)
        elif line.startswith("@@"):
            # New hunk within same file — split
            if current_lines:
                hunks.append({
                    "file": current_file,
                    "header": "\n".join(current_header),
                    "lines": current_lines[:],
                    "raw": "\n".join(current_header + current_lines),
                })
                current_lines = []
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Save last hunk
    if current_file and current_lines:
        hunks.append({
            "file": current_file,
            "header": "\n".join(current_header),
            "lines": current_lines[:],
            "raw": "\n".join(current_header + current_lines),
        })

    return hunks


def _extract_filename(diff_line: str) -> str:
    """Extract filename from 'diff --git a/path b/path'."""
    match = re.search(r"diff --git a/(.*?) b/", diff_line)
    return match.group(1) if match else "unknown"


def classify_hunk(hunk: dict) -> str:
    """Classify a hunk as 'cosmetic' or 'functional'.

    Conservative: only labels as cosmetic if ALL changes are whitespace/quote/import.
    Returns 'functional' when uncertain.
    """
    added = []
    removed = []

    for line in hunk["lines"]:
        if line.startswith("@@"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])

    if not added and not removed:
        return "cosmetic"  # Context-only hunk

    # Normalize and compare
    norm_added = [_normalize(l) for l in added]
    norm_removed = [_normalize(l) for l in removed]

    # If normalized versions are identical, it's cosmetic
    if sorted(norm_added) == sorted(norm_removed):
        return "cosmetic"

    # Check if only quote style changed
    quote_added = [_normalize_quotes(l) for l in added]
    quote_removed = [_normalize_quotes(l) for l in removed]
    if sorted(quote_added) == sorted(quote_removed):
        return "cosmetic"

    return "functional"


def _normalize(line: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(line.split())


def _normalize_quotes(line: str) -> str:
    """Normalize both whitespace and quote style."""
    normalized = _normalize(line)
    # Replace all single quotes with double quotes for comparison
    return normalized.replace("'", '"')


def preprocess(diff_text: str, max_chars: int = 0) -> dict:
    """Preprocess a diff: classify hunks, strip cosmetic ones, apply smart truncation.

    Returns dict with:
      - functional_diff: str (only functional hunks)
      - full_diff: str (original)
      - stats: dict with hunk counts and sizes
    """
    hunks = parse_hunks(diff_text)

    functional_hunks = []
    cosmetic_hunks = []

    for hunk in hunks:
        cls = classify_hunk(hunk)
        hunk["classification"] = cls
        if cls == "functional":
            functional_hunks.append(hunk)
        else:
            cosmetic_hunks.append(hunk)

    # Build functional-only diff
    seen_headers = set()
    functional_parts = []
    for hunk in functional_hunks:
        if hunk["header"] not in seen_headers:
            functional_parts.append(hunk["header"])
            seen_headers.add(hunk["header"])
        functional_parts.append("\n".join(hunk["lines"]))

    functional_diff = "\n".join(functional_parts)

    # Smart truncation: if still too long, truncate largest hunks
    if max_chars > 0 and len(functional_diff) > max_chars:
        # Sort by size descending, truncate largest first
        functional_hunks.sort(key=lambda h: len("\n".join(h["lines"])), reverse=True)
        while len(functional_diff) > max_chars and functional_hunks:
            largest = functional_hunks[0]
            lines = largest["lines"]
            # Keep first 20 and last 10 lines of the largest hunk
            if len(lines) > 40:
                largest["lines"] = lines[:20] + ["... (truncated) ..."] + lines[-10:]
                # Rebuild
                seen_headers = set()
                functional_parts = []
                for hunk in functional_hunks:
                    if hunk["header"] not in seen_headers:
                        functional_parts.append(hunk["header"])
                        seen_headers.add(hunk["header"])
                    functional_parts.append("\n".join(hunk["lines"]))
                functional_diff = "\n".join(functional_parts)
            else:
                break

    stats = {
        "total_hunks": len(hunks),
        "functional_hunks": len(functional_hunks),
        "cosmetic_hunks": len(cosmetic_hunks),
        "original_chars": len(diff_text),
        "functional_chars": len(functional_diff),
        "reduction_pct": round(100 * (1 - len(functional_diff) / max(len(diff_text), 1)), 1),
    }

    return {
        "functional_diff": functional_diff,
        "full_diff": diff_text,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess diff to strip cosmetic changes")
    parser.add_argument("diff_file", help="Path to diff file")
    parser.add_argument("--max-chars", type=int, default=0, help="Max output chars (0=unlimited)")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    args = parser.parse_args()

    diff_text = Path(args.diff_file).read_text()
    result = preprocess(diff_text, max_chars=args.max_chars)

    if args.stats:
        import json
        print(json.dumps(result["stats"], indent=2))
    else:
        print(result["functional_diff"])


if __name__ == "__main__":
    main()
