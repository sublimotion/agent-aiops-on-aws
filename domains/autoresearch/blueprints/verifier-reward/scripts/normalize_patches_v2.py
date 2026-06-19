#!/usr/bin/env python3
"""
E_norm v2: Full AST normalization using repo checkouts.

Unlike v1 (which tried to parse diff hunks as standalone code and failed),
this version:
1. Clones the repo at base_commit (using Docker volume cache from gold_eval.py)
2. Records the original source files touched by the patch
3. Applies the patch
4. Records the modified source files
5. AST-normalizes both versions
6. Generates a canonical diff

This is the Agentless approach (ast.parse → ast.unparse → canonical diff)
applied with full source context, solving the iter 31 blocker:
"Full AST normalization needs original source files."

Usage:
    # Normalize all diffs for a scaffold (uses Docker repo cache)
    python3 normalize_patches_v2.py --model sonnet --output-dir results/diffs_normalized/opencode_sonnet

    # Normalize cross-model diffs
    python3 normalize_patches_v2.py --diff-dir results/diffs/qwen35_opencode --output-dir results/diffs_normalized/qwen35_opencode

    # Single issue (debug)
    python3 normalize_patches_v2.py --model sonnet --issue django__django-10924 --verbose

    # Discriminability comparison (before/after normalization)
    python3 normalize_patches_v2.py --discriminability \
        results/diffs/opencode_sonnet \
        results/diffs/qwen35_opencode \
        results/diffs/devstral_sera_verifier_loop

    # Skip Docker, use local repo clones (if you have them)
    python3 normalize_patches_v2.py --model sonnet --local-repos /tmp/swebench-repos
"""

import argparse
import ast
import io
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBSET_SEED = 42
SUBSET_SIZE = 50
DOCKER_IMAGE = "python:3.11-bookworm"
REPO_CACHE_VOL = "swebench-repo-cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class Issue:
    instance_id: str
    repo: str
    base_commit: str


def load_subset() -> dict[str, Issue]:
    """Load the same 50-issue subset used by gold_eval.py."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    all_issues = []
    for row in ds:
        all_issues.append(Issue(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
        ))

    rng = random.Random(SUBSET_SEED)
    by_repo = {}
    for issue in all_issues:
        by_repo.setdefault(issue.repo, []).append(issue)

    subset = {}
    for repo_issues in by_repo.values():
        rng.shuffle(repo_issues)

    flat = []
    for repo_issues in sorted(by_repo.values(), key=lambda x: -len(x)):
        flat.extend(repo_issues)

    rng.shuffle(flat)
    for issue in flat[:SUBSET_SIZE]:
        subset[issue.instance_id] = issue

    return subset


def parse_diff_files(diff_text: str) -> list[str]:
    """Extract list of Python files modified by a diff."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # Extract b/path from "diff --git a/foo.py b/foo.py"
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                if path.endswith(".py"):
                    files.append(path)
    return list(set(files))


def strip_comments_and_docstrings(source: str) -> str:
    """Remove comments and docstrings from Python source."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source

    result_tokens = []
    prev_token_type = None

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_token_type in (
            tokenize.NEWLINE, tokenize.INDENT, tokenize.NL, None,
        ):
            if tok.string.startswith('"""') or tok.string.startswith("'''"):
                continue
        result_tokens.append(tok)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev_token_type = tok.type
        elif tok.type in (tokenize.NEWLINE, tokenize.NL):
            prev_token_type = tok.type

    return tokenize.untokenize(result_tokens)


def ast_normalize(source: str) -> str | None:
    """AST round-trip: parse → unparse. Returns None if unparseable."""
    try:
        tree = ast.parse(source)
        return ast.unparse(tree)
    except SyntaxError:
        return None


def normalize_file(source: str) -> str | None:
    """Full normalization pipeline for a Python source file."""
    stripped = strip_comments_and_docstrings(source)
    normalized = ast_normalize(stripped)
    if normalized is not None:
        return normalized
    # Fallback: try without comment stripping
    return ast_normalize(source)


def normalize_via_docker(issue: Issue, diff_path: str, output_path: str, verbose: bool = False) -> dict:
    """
    Clone repo in Docker, get before/after source, AST-normalize, produce canonical diff.

    Uses the same Docker volume cache as gold_eval.py for repo clones.
    """
    diff_text = Path(diff_path).read_text()
    py_files = parse_diff_files(diff_text)

    if not py_files:
        # No Python files in diff — copy through unchanged
        Path(output_path).write_text(diff_text)
        return {"status": "no_python_files", "files": 0, "normalized": 0}

    repo_cache_name = issue.repo.replace("/", "__")

    # Docker script: checkout base, cat original files, apply patch, cat modified files
    file_list = " ".join(py_files)
    cat_commands_before = "\n".join(
        f'echo "===FILE_BEFORE:{f}==="; cat "{f}" 2>/dev/null || echo "===FILE_MISSING==="; echo "===FILE_END==="'
        for f in py_files
    )
    cat_commands_after = "\n".join(
        f'echo "===FILE_AFTER:{f}==="; cat "{f}" 2>/dev/null || echo "===FILE_MISSING==="; echo "===FILE_END==="'
        for f in py_files
    )

    script = f"""#!/bin/bash
CACHE_DIR="/repo-cache/{repo_cache_name}"
if [ -d "$CACHE_DIR/.git" ]; then
    cp -a "$CACHE_DIR" /workspace
    cd /workspace
    git fetch origin 2>/dev/null
else
    git clone https://github.com/{issue.repo}.git /workspace 2>&1 | tail -1
    cd /workspace
    cp -a /workspace "$CACHE_DIR" 2>/dev/null || true
fi
git checkout -f {issue.base_commit} 2>&1

# Capture BEFORE files
{cat_commands_before}

# Apply agent patch
{{ cat /mnt/agent.patch; printf "\\n"; }} > /tmp/agent.patch
patch -p1 --no-backup-if-mismatch < /tmp/agent.patch 2>&1

# Capture AFTER files
{cat_commands_after}
"""

    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "host",
                "--memory", "4g",
                "--cpus", "2",
                "-v", f"{REPO_CACHE_VOL}:/repo-cache",
                "-v", f"{os.path.abspath(diff_path)}:/mnt/agent.patch:ro",
                DOCKER_IMAGE,
                "bash", "-c", script,
            ],
            capture_output=True, text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        Path(output_path).write_text(diff_text)
        return {"status": "timeout", "files": len(py_files), "normalized": 0}

    output = proc.stdout

    # Parse before/after file contents
    before_files = _parse_file_sections(output, "FILE_BEFORE")
    after_files = _parse_file_sections(output, "FILE_AFTER")

    if verbose:
        log.info(f"  Before files: {list(before_files.keys())}")
        log.info(f"  After files: {list(after_files.keys())}")

    # AST-normalize each file and generate canonical diff
    normalized_diff_parts = []
    stats = {"files": len(py_files), "normalized": 0, "failed": 0, "unchanged": 0}

    for f in py_files:
        before = before_files.get(f)
        after = after_files.get(f)

        if before is None or after is None:
            stats["failed"] += 1
            continue

        norm_before = normalize_file(before)
        norm_after = normalize_file(after)

        if norm_before is None or norm_after is None:
            stats["failed"] += 1
            # Include raw diff for files that can't be normalized
            continue

        if norm_before == norm_after:
            stats["unchanged"] += 1
            continue

        stats["normalized"] += 1

        # Generate canonical diff
        canonical = _diff_strings(norm_before, norm_after, f"a/{f}", f"b/{f}")
        if canonical:
            normalized_diff_parts.append(canonical)

    if normalized_diff_parts:
        result_diff = "\n".join(normalized_diff_parts) + "\n"
    else:
        # Fallback to original if normalization produced nothing
        result_diff = diff_text

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(result_diff)

    stats["status"] = "ok"
    stats["original_size"] = len(diff_text)
    stats["normalized_size"] = len(result_diff)
    stats["compression"] = round(len(result_diff) / max(len(diff_text), 1), 3)
    return stats


def normalize_via_local(issue: Issue, diff_path: str, output_path: str,
                        repos_dir: str, verbose: bool = False) -> dict:
    """Same as Docker version but using local repo clones."""
    diff_text = Path(diff_path).read_text()
    py_files = parse_diff_files(diff_text)

    if not py_files:
        Path(output_path).write_text(diff_text)
        return {"status": "no_python_files", "files": 0, "normalized": 0}

    repo_name = issue.repo.replace("/", "__")
    repo_path = Path(repos_dir) / repo_name

    # Clone if needed
    if not repo_path.exists():
        log.info(f"  Cloning {issue.repo}...")
        subprocess.run(
            ["git", "clone", f"https://github.com/{issue.repo}.git", str(repo_path)],
            capture_output=True, timeout=300,
        )

    # Checkout base commit
    subprocess.run(
        ["git", "checkout", "-f", issue.base_commit],
        cwd=repo_path, capture_output=True, timeout=30,
    )

    # Read BEFORE files
    before_files = {}
    for f in py_files:
        fp = repo_path / f
        if fp.exists():
            before_files[f] = fp.read_text(errors="replace")

    # Apply patch
    subprocess.run(
        ["git", "apply", "--allow-empty", os.path.abspath(diff_path)],
        cwd=repo_path, capture_output=True, timeout=30,
    )

    # Read AFTER files
    after_files = {}
    for f in py_files:
        fp = repo_path / f
        if fp.exists():
            after_files[f] = fp.read_text(errors="replace")

    # Reset repo for next use
    subprocess.run(["git", "checkout", "-f", "."], cwd=repo_path, capture_output=True)

    # Normalize and diff
    normalized_diff_parts = []
    stats = {"files": len(py_files), "normalized": 0, "failed": 0, "unchanged": 0}

    for f in py_files:
        before = before_files.get(f)
        after = after_files.get(f)

        if before is None or after is None:
            stats["failed"] += 1
            continue

        norm_before = normalize_file(before)
        norm_after = normalize_file(after)

        if norm_before is None or norm_after is None:
            stats["failed"] += 1
            continue

        if norm_before == norm_after:
            stats["unchanged"] += 1
            continue

        stats["normalized"] += 1
        canonical = _diff_strings(norm_before, norm_after, f"a/{f}", f"b/{f}")
        if canonical:
            normalized_diff_parts.append(canonical)

    if normalized_diff_parts:
        result_diff = "\n".join(normalized_diff_parts) + "\n"
    else:
        result_diff = diff_text

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(result_diff)

    stats["status"] = "ok"
    stats["original_size"] = len(diff_text)
    stats["normalized_size"] = len(result_diff)
    stats["compression"] = round(len(result_diff) / max(len(diff_text), 1), 3)
    return stats


def _parse_file_sections(output: str, prefix: str) -> dict[str, str]:
    """Parse ===FILE_BEFORE:path=== ... ===FILE_END=== sections from Docker output."""
    files = {}
    marker = f"==={prefix}:"
    for section in output.split(marker)[1:]:
        if "===" not in section:
            continue
        filename = section.split("===")[0]
        content_start = section.index("===") + 3
        if "===FILE_END===" in section:
            content = section[content_start:section.index("===FILE_END===")]
            if "===FILE_MISSING===" not in content:
                files[filename] = content
    return files


def _diff_strings(old: str, new: str, old_label: str, new_label: str) -> str | None:
    """Generate unified diff between two strings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f_old:
        f_old.write(old)
        f_old_name = f_old.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f_new:
        f_new.write(new)
        f_new_name = f_new.name

    try:
        result = subprocess.run(
            ["diff", "-u", f_old_name, f_new_name],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return None
        diff_out = result.stdout
        diff_out = diff_out.replace(f_old_name, old_label).replace(f_new_name, new_label)
        return diff_out
    finally:
        os.unlink(f_old_name)
        os.unlink(f_new_name)


def compute_discriminability(diff_dirs: list[str]) -> dict:
    """Compare surface features across scaffold directories."""
    features_by_scaffold = defaultdict(list)

    for diff_dir in diff_dirs:
        scaffold_name = Path(diff_dir).name
        for diff_file in sorted(Path(diff_dir).glob("*.diff")):
            diff_text = diff_file.read_text()
            lines = diff_text.splitlines()
            features_by_scaffold[scaffold_name].append({
                "diff_size": len(diff_text),
                "line_count": len(lines),
                "hunk_count": sum(1 for l in lines if l.startswith("@@")),
                "files_changed": sum(1 for l in lines if l.startswith("diff --git")),
                "added_lines": sum(1 for l in lines if l.startswith("+") and not l.startswith("+++")),
                "removed_lines": sum(1 for l in lines if l.startswith("-") and not l.startswith("---")),
            })

    results = {"scaffolds": {}}
    for scaffold, feats in features_by_scaffold.items():
        if not feats:
            continue
        n = len(feats)
        results["scaffolds"][scaffold] = {
            "n": n,
            "median_diff_size": sorted(f["diff_size"] for f in feats)[n // 2],
            "median_hunks": sorted(f["hunk_count"] for f in feats)[n // 2],
            "mean_files_changed": round(sum(f["files_changed"] for f in feats) / n, 1),
            "mean_added_lines": round(sum(f["added_lines"] for f in feats) / n, 1),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="E_norm v2: Full AST normalization with repo checkouts")
    parser.add_argument("--model", help="Model name (haiku/sonnet/opus) — uses results/diffs/opencode_{model}/")
    parser.add_argument("--diff-dir", help="Custom diff directory (for non-standard scaffolds)")
    parser.add_argument("--output-dir", help="Output directory for normalized diffs")
    parser.add_argument("--issue", help="Single issue ID to process")
    parser.add_argument("--local-repos", help="Local repos directory (skip Docker)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--discriminability", nargs="+", metavar="DIR",
                        help="Run discriminability test on these diff directories")

    args = parser.parse_args()

    if args.discriminability:
        results = compute_discriminability(args.discriminability)
        print(json.dumps(results, indent=2))
        return

    # Determine input/output dirs
    if args.diff_dir:
        input_dir = Path(args.diff_dir)
    elif args.model:
        input_dir = RESULTS_DIR / "diffs" / f"opencode_{args.model}"
    else:
        parser.error("Specify --model or --diff-dir")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / "diffs_normalized" / input_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load issue metadata
    issues = load_subset()

    # Process diffs
    diff_files = sorted(input_dir.glob("*.diff"))
    if args.issue:
        diff_files = [f for f in diff_files if args.issue in f.name]

    log.info(f"Processing {len(diff_files)} diffs from {input_dir}")
    log.info(f"Output to {output_dir}")

    all_stats = []
    for diff_file in diff_files:
        instance_id = diff_file.stem
        issue = issues.get(instance_id)

        if not issue:
            log.warning(f"  SKIP {instance_id}: not in subset")
            continue

        log.info(f"  {instance_id}...")
        output_path = str(output_dir / diff_file.name)

        if args.local_repos:
            stats = normalize_via_local(issue, str(diff_file), output_path,
                                        args.local_repos, args.verbose)
        else:
            stats = normalize_via_docker(issue, str(diff_file), output_path, args.verbose)

        stats["instance_id"] = instance_id
        all_stats.append(stats)
        log.info(f"    {stats['status']}: {stats.get('normalized', 0)}/{stats.get('files', 0)} files, "
                 f"compression={stats.get('compression', 'N/A')}")

    # Summary
    print("\n=== E_norm Summary ===")
    total = len(all_stats)
    ok = sum(1 for s in all_stats if s["status"] == "ok")
    normalized_files = sum(s.get("normalized", 0) for s in all_stats)
    failed_files = sum(s.get("failed", 0) for s in all_stats)
    unchanged_files = sum(s.get("unchanged", 0) for s in all_stats)
    total_orig = sum(s.get("original_size", 0) for s in all_stats)
    total_norm = sum(s.get("normalized_size", 0) for s in all_stats)

    print(f"Diffs processed: {total} ({ok} successful)")
    print(f"Files: {normalized_files} normalized, {unchanged_files} unchanged (cosmetic-only), {failed_files} failed")
    if total_orig > 0:
        print(f"Size: {total_orig:,} → {total_norm:,} chars ({total_norm/total_orig:.1%})")

    # Save stats
    stats_path = output_dir / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
