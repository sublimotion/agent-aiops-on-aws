#!/usr/bin/env python3
"""
E_norm: AST-level patch normalization for cross-model verifier transfer.

Normalizes code patches by parsing through Python AST and regenerating
canonical source, then computing a clean diff. This removes scaffold-specific
formatting artifacts (whitespace, comment style, import ordering, parenthesization)
that may cause v009 precision to drop across scaffolds.

Inspired by:
- Agentless AST-normalized majority voting (arXiv:2407.01489)
- Shopify Engineering Python DSL transpiler (+22pp from representation change)

Usage:
    # Normalize all diffs in a directory
    python normalize_patches.py --input-dir results/diffs/opencode_sonnet --output-dir results/diffs_normalized/opencode_sonnet

    # Normalize a single diff
    python normalize_patches.py --input results/diffs/opencode_sonnet/django__django-10924.diff --output results/diffs_normalized/opencode_sonnet/django__django-10924.diff

    # Run scaffold discriminability test
    python normalize_patches.py --discriminability --dirs results/diffs/opencode_sonnet results/diffs/qwen35_opencode results/diffs/devstral_sera_verifier_loop

    # Stats only (no output files)
    python normalize_patches.py --input-dir results/diffs/opencode_sonnet --stats
"""

import argparse
import ast
import io
import os
import re
import sys
import json
import tokenize
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional


def strip_comments_and_docstrings(source: str) -> str:
    """Remove comments and docstrings from Python source.

    Uses the tokenizer approach from Agentless (postprocess_data.py).
    Preserves regular strings, only strips comments and module/class/function docstrings.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source  # If tokenization fails, return as-is

    result_tokens = []
    prev_token_type = None

    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            continue
        # Heuristic for docstrings: STRING token immediately after NEWLINE/INDENT
        # at module/class/function level
        if tok.type == tokenize.STRING and prev_token_type in (
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.NL,
            None,
        ):
            # Check if this looks like a docstring (triple-quoted)
            if tok.string.startswith('"""') or tok.string.startswith("'''"):
                continue
        result_tokens.append(tok)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev_token_type = tok.type
        elif tok.type in (tokenize.NEWLINE, tokenize.NL):
            prev_token_type = tok.type

    return tokenize.untokenize(result_tokens)


def normalize_python_source(source: str) -> Optional[str]:
    """Parse Python source through AST and regenerate canonical form.

    Returns None if the source cannot be parsed (not valid Python).
    """
    try:
        tree = ast.parse(source)
        return ast.unparse(tree)
    except SyntaxError:
        return None


def parse_unified_diff(diff_text: str) -> list[dict]:
    """Parse a unified diff into a list of file-level hunks.

    Returns list of dicts with keys:
        - old_file: str
        - new_file: str
        - hunks: list of (old_lines, new_lines) tuples
        - header: str (the --- / +++ lines)
    """
    files = []
    current_file = None
    current_old = []
    current_new = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if current_file:
                if current_old or current_new:
                    current_file["hunks"].append(
                        ("".join(current_old), "".join(current_new))
                    )
                files.append(current_file)
            current_file = {"old_file": "", "new_file": "", "hunks": [], "header": line}
            current_old = []
            current_new = []
        elif line.startswith("--- "):
            if current_file:
                current_file["old_file"] = line[4:].strip()
                current_file["header"] += line
        elif line.startswith("+++ "):
            if current_file:
                current_file["new_file"] = line[4:].strip()
                current_file["header"] += line
        elif line.startswith("@@"):
            if current_old or current_new:
                if current_file:
                    current_file["hunks"].append(
                        ("".join(current_old), "".join(current_new))
                    )
            current_old = []
            current_new = []
        elif line.startswith("-"):
            current_old.append(line[1:])
        elif line.startswith("+"):
            current_new.append(line[1:])
        elif line.startswith(" "):
            current_old.append(line[1:])
            current_new.append(line[1:])

    if current_file:
        if current_old or current_new:
            current_file["hunks"].append(("".join(current_old), "".join(current_new)))
        files.append(current_file)

    return files


def normalize_diff(diff_text: str) -> tuple[str, dict]:
    """Normalize a unified diff by AST-parsing changed Python code.

    Returns (normalized_diff, stats) where stats contains:
        - files_total: int
        - files_python: int
        - files_normalized: int
        - files_failed: int (Python files that couldn't be parsed)
        - original_size: int
        - normalized_size: int
    """
    stats = {
        "files_total": 0,
        "files_python": 0,
        "files_normalized": 0,
        "files_failed": 0,
        "original_size": len(diff_text),
        "normalized_size": 0,
    }

    files = parse_unified_diff(diff_text)
    stats["files_total"] = len(files)

    if not files:
        stats["normalized_size"] = len(diff_text)
        return diff_text, stats

    normalized_parts = []

    for file_info in files:
        is_python = file_info["new_file"].endswith(".py") or file_info["old_file"].endswith(".py")

        if not is_python:
            # Pass through non-Python files unchanged
            normalized_parts.append(_reconstruct_file_diff(file_info))
            continue

        stats["files_python"] += 1

        # Collect all old and new code for this file
        all_old = ""
        all_new = ""
        for old_code, new_code in file_info["hunks"]:
            all_old += old_code
            all_new += new_code

        # Normalize through AST
        norm_old = _normalize_code_block(all_old)
        norm_new = _normalize_code_block(all_new)

        if norm_old is None or norm_new is None:
            # AST parse failed — pass through unchanged
            stats["files_failed"] += 1
            normalized_parts.append(_reconstruct_file_diff(file_info))
            continue

        stats["files_normalized"] += 1

        # Generate canonical diff between normalized versions
        canonical = _generate_canonical_diff(
            norm_old, norm_new, file_info["old_file"], file_info["new_file"]
        )
        if canonical:
            normalized_parts.append(canonical)

    result = "\n".join(normalized_parts)
    stats["normalized_size"] = len(result)
    return result, stats


def _normalize_code_block(code: str) -> Optional[str]:
    """Normalize a code block: strip comments/docstrings, then AST round-trip."""
    if not code.strip():
        return ""

    # Step 1: Strip comments and docstrings
    stripped = strip_comments_and_docstrings(code)

    # Step 2: AST round-trip
    normalized = normalize_python_source(stripped)
    if normalized is not None:
        return normalized

    # Fallback: try without comment stripping (in case stripping broke syntax)
    normalized = normalize_python_source(code)
    if normalized is not None:
        return normalized

    return None


def _reconstruct_file_diff(file_info: dict) -> str:
    """Reconstruct a unified diff section from parsed file info."""
    parts = [file_info["header"].rstrip()]
    for old_code, new_code in file_info["hunks"]:
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        parts.append(f"@@ -{1},{len(old_lines)} +{1},{len(new_lines)} @@")
        for line in old_lines:
            parts.append(f"-{line.rstrip()}")
        for line in new_lines:
            parts.append(f"+{line.rstrip()}")
    return "\n".join(parts)


def _generate_canonical_diff(
    old_code: str, new_code: str, old_file: str, new_file: str
) -> Optional[str]:
    """Generate a canonical diff between two normalized code strings using git."""
    if old_code == new_code:
        return None  # No functional difference after normalization

    with tempfile.TemporaryDirectory() as tmpdir:
        old_path = os.path.join(tmpdir, "old.py")
        new_path = os.path.join(tmpdir, "new.py")

        with open(old_path, "w") as f:
            f.write(old_code)
        with open(new_path, "w") as f:
            f.write(new_code)

        try:
            result = subprocess.run(
                ["diff", "-u", old_path, new_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return None  # Files are identical
            diff_output = result.stdout
            # Replace temp paths with original file paths
            diff_output = diff_output.replace(old_path, f"a/{old_file}")
            diff_output = diff_output.replace(new_path, f"b/{new_file}")
            return diff_output
        except Exception:
            return None


def compute_discriminability(diff_dirs: list[str]) -> dict:
    """Test whether a simple classifier can identify which scaffold produced a patch.

    Extracts surface-level features from diffs and tests if they're scaffold-specific.
    Run before and after normalization to measure whether normalization removes scaffold signal.
    """
    from collections import Counter

    features_by_scaffold = defaultdict(list)

    for diff_dir in diff_dirs:
        scaffold_name = Path(diff_dir).name
        diff_files = sorted(Path(diff_dir).glob("*.diff"))

        for diff_file in diff_files:
            diff_text = diff_file.read_text()
            features = _extract_surface_features(diff_text)
            features["instance"] = diff_file.stem
            features_by_scaffold[scaffold_name].append(features)

    # Compute per-scaffold feature distributions
    results = {"scaffolds": {}, "discriminability_features": []}

    for scaffold, feature_list in features_by_scaffold.items():
        if not feature_list:
            continue
        results["scaffolds"][scaffold] = {
            "n": len(feature_list),
            "median_diff_size": sorted(f["diff_size"] for f in feature_list)[
                len(feature_list) // 2
            ],
            "median_hunks": sorted(f["hunk_count"] for f in feature_list)[
                len(feature_list) // 2
            ],
            "median_files_changed": sorted(f["files_changed"] for f in feature_list)[
                len(feature_list) // 2
            ],
            "has_comments_pct": sum(1 for f in feature_list if f["has_comments"])
            / len(feature_list),
            "has_docstrings_pct": sum(1 for f in feature_list if f["has_docstrings"])
            / len(feature_list),
            "avg_context_ratio": sum(f["context_ratio"] for f in feature_list)
            / len(feature_list),
        }

    # Simple discriminability test: which features differ most across scaffolds
    all_features = ["diff_size", "hunk_count", "files_changed", "context_ratio"]
    for feat in all_features:
        scaffold_means = {}
        for scaffold, feature_list in features_by_scaffold.items():
            vals = [f[feat] for f in feature_list]
            scaffold_means[scaffold] = sum(vals) / len(vals) if vals else 0

        if len(scaffold_means) >= 2:
            vals = list(scaffold_means.values())
            max_ratio = max(vals) / min(vals) if min(vals) > 0 else float("inf")
            results["discriminability_features"].append(
                {"feature": feat, "scaffold_means": scaffold_means, "max_ratio": max_ratio}
            )

    # Sort by discriminability (highest ratio = most scaffold-specific)
    results["discriminability_features"].sort(key=lambda x: x["max_ratio"], reverse=True)

    return results


def _extract_surface_features(diff_text: str) -> dict:
    """Extract surface-level features from a diff for discriminability testing."""
    lines = diff_text.splitlines()
    return {
        "diff_size": len(diff_text),
        "line_count": len(lines),
        "hunk_count": sum(1 for l in lines if l.startswith("@@")),
        "files_changed": sum(1 for l in lines if l.startswith("diff --git")),
        "added_lines": sum(1 for l in lines if l.startswith("+") and not l.startswith("+++")),
        "removed_lines": sum(1 for l in lines if l.startswith("-") and not l.startswith("---")),
        "context_ratio": sum(1 for l in lines if l.startswith(" ")) / max(len(lines), 1),
        "has_comments": any("#" in l for l in lines if l.startswith("+")),
        "has_docstrings": any('"""' in l or "'''" in l for l in lines if l.startswith("+")),
    }


def process_directory(input_dir: str, output_dir: str, stats_only: bool = False) -> dict:
    """Normalize all diffs in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None

    if output_path and not stats_only:
        output_path.mkdir(parents=True, exist_ok=True)

    aggregate_stats = {
        "total_diffs": 0,
        "normalized_diffs": 0,
        "failed_diffs": 0,
        "total_original_size": 0,
        "total_normalized_size": 0,
        "per_file": [],
    }

    for diff_file in sorted(input_path.glob("*.diff")):
        diff_text = diff_file.read_text()
        normalized, stats = normalize_diff(diff_text)

        aggregate_stats["total_diffs"] += 1
        aggregate_stats["total_original_size"] += stats["original_size"]
        aggregate_stats["total_normalized_size"] += stats["normalized_size"]

        if stats["files_normalized"] > 0:
            aggregate_stats["normalized_diffs"] += 1
        if stats["files_failed"] > 0:
            aggregate_stats["failed_diffs"] += 1

        stats["file"] = diff_file.name
        aggregate_stats["per_file"].append(stats)

        if output_path and not stats_only:
            out_file = output_path / diff_file.name
            out_file.write_text(normalized)

    # Summary
    if aggregate_stats["total_original_size"] > 0:
        aggregate_stats["compression_ratio"] = (
            aggregate_stats["total_normalized_size"]
            / aggregate_stats["total_original_size"]
        )
    else:
        aggregate_stats["compression_ratio"] = 1.0

    return aggregate_stats


def main():
    parser = argparse.ArgumentParser(
        description="E_norm: AST patch normalization for cross-model verifier transfer"
    )
    parser.add_argument("--input", help="Single diff file to normalize")
    parser.add_argument("--output", help="Output path for normalized diff")
    parser.add_argument("--input-dir", help="Directory of diffs to normalize")
    parser.add_argument("--output-dir", help="Output directory for normalized diffs")
    parser.add_argument("--stats", action="store_true", help="Print stats only, no output files")
    parser.add_argument(
        "--discriminability",
        action="store_true",
        help="Run scaffold discriminability test",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Diff directories for discriminability test",
    )

    args = parser.parse_args()

    if args.discriminability:
        if not args.dirs:
            parser.error("--discriminability requires --dirs")
        results = compute_discriminability(args.dirs)
        print(json.dumps(results, indent=2))
        return

    if args.input:
        diff_text = Path(args.input).read_text()
        normalized, stats = normalize_diff(diff_text)
        print(json.dumps(stats, indent=2), file=sys.stderr)
        if args.output:
            Path(args.output).write_text(normalized)
        else:
            print(normalized)
        return

    if args.input_dir:
        stats = process_directory(args.input_dir, args.output_dir, stats_only=args.stats)
        print(json.dumps({k: v for k, v in stats.items() if k != "per_file"}, indent=2))
        if args.stats:
            print("\nPer-file breakdown:")
            for f in stats["per_file"]:
                status = "OK" if f["files_normalized"] > 0 else "SKIP" if f["files_failed"] > 0 else "PASS"
                print(f"  {status} {f['file']}: {f['original_size']}→{f['normalized_size']} chars, {f['files_python']} py files")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
