#!/usr/bin/env python3
"""
Context-augmented verifier prototype.

For small surgical diffs (Devstral SERA style), the verifier lacks surrounding
code context to reason adversarially. This script:
  1. Parses diff hunk headers for file paths + line ranges
  2. Fetches N lines of surrounding context from the repo at base_commit
  3. Augments the verifier prompt with file context
  4. Runs v001∩v009 ensemble
  5. Compares precision/recall vs baseline (no context)

Uses Docker to checkout repos at base_commit (same infra as gold_eval.py).

Usage:
  python3 verify_with_context.py --context-lines 30
  python3 verify_with_context.py --context-lines 50 --limit 5
  python3 verify_with_context.py --issue pallets__flask-4045
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

SCRIPT_DIR = Path(__file__).resolve().parent
BLUEPRINT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BLUEPRINT_DIR / "results"
DIFFS_DIR = RESULTS_DIR / "diffs" / "devstral_sera_verifier_loop"
VERSIONS_DIR = BLUEPRINT_DIR / "skills" / "patch-verifier" / "versions"
V001_RUBRIC = VERSIONS_DIR / "v001_baseline.md"
V009_RUBRIC = VERSIONS_DIR / "v009_adversarial.md"

DOCKER_IMAGE = "python:3.11-bookworm"
REPO_CACHE_VOL = "swebench-repo-cache"
CONTEXT_CACHE_DIR = RESULTS_DIR / "context_cache"

V009_RUNS = 3
V009_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Step 1: Parse diff hunk headers
# ---------------------------------------------------------------------------

def parse_diff_hunks(diff_text: str) -> list[dict]:
    """Extract file paths and line ranges from a unified diff."""
    hunks = []
    current_file = None

    for line in diff_text.split("\n"):
        # Track current file
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m and current_file:
                old_start = int(m.group(1))
                old_count = int(m.group(2) or 1)
                hunks.append({
                    "file": current_file,
                    "start": old_start,
                    "count": old_count,
                })

    return hunks


# ---------------------------------------------------------------------------
# Step 2: Fetch surrounding context from repo at base_commit
# ---------------------------------------------------------------------------

def fetch_file_context(repo: str, base_commit: str, file_path: str,
                       start: int, count: int, context_lines: int) -> str:
    """Fetch file content around a hunk from the repo at base_commit using Docker."""
    # Calculate line range with context
    ctx_start = max(1, start - context_lines)
    ctx_end = start + count + context_lines

    cache_key = f"{repo.replace('/', '__')}_{base_commit[:8]}_{file_path.replace('/', '_')}"
    cache_file = CONTEXT_CACHE_DIR / f"{cache_key}.txt"

    # Check cache for full file
    if cache_file.exists():
        lines = cache_file.read_text().split("\n")
        selected = lines[max(0, ctx_start - 1):ctx_end]
        return _format_context(file_path, ctx_start, selected)

    # Fetch via Docker (reuse gold_eval repo cache)
    repo_cache_name = repo.replace("/", "__")
    script = f"""
CACHE_DIR="/repo-cache/{repo_cache_name}"
if [ -d "$CACHE_DIR/.git" ]; then
    cd "$CACHE_DIR"
    git checkout -f {base_commit} -- {file_path} 2>/dev/null
    cat {file_path} 2>/dev/null
else
    git clone --depth 1 https://github.com/{repo}.git /tmp/repo 2>/dev/null
    cd /tmp/repo
    git fetch --depth 50 origin {base_commit} 2>/dev/null
    git checkout -f {base_commit} -- {file_path} 2>/dev/null
    cat {file_path} 2>/dev/null
fi
"""
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{REPO_CACHE_VOL}:/repo-cache",
             DOCKER_IMAGE, "bash", "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(proc.stdout)
            lines = proc.stdout.split("\n")
            selected = lines[max(0, ctx_start - 1):ctx_end]
            return _format_context(file_path, ctx_start, selected)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return ""


def _format_context(file_path: str, start_line: int, lines: list[str]) -> str:
    """Format context with line numbers."""
    numbered = []
    for i, line in enumerate(lines):
        numbered.append(f"{start_line + i:4d} | {line}")
    return f"### {file_path} (lines {start_line}-{start_line + len(lines) - 1})\n```\n" + "\n".join(numbered) + "\n```"


def get_context_for_diff(diff_text: str, repo: str, base_commit: str,
                         context_lines: int) -> str:
    """Get surrounding code context for all hunks in a diff."""
    hunks = parse_diff_hunks(diff_text)
    if not hunks:
        return ""

    # Deduplicate by file + merge overlapping ranges
    by_file = {}
    for h in hunks:
        key = h["file"]
        if key not in by_file:
            by_file[key] = []
        by_file[key].append(h)

    contexts = []
    for file_path, file_hunks in by_file.items():
        # Skip test files
        if "test" in file_path.lower():
            continue
        for h in file_hunks:
            ctx = fetch_file_context(
                repo, base_commit, file_path,
                h["start"], h["count"], context_lines,
            )
            if ctx:
                contexts.append(ctx)

    return "\n\n".join(contexts)


# ---------------------------------------------------------------------------
# Step 3: Verifier with context-augmented prompt
# ---------------------------------------------------------------------------

def call_haiku(prompt: str, temperature: float = 0.0) -> dict:
    """Call Claude Haiku via Bedrock."""
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.monotonic()
    response = client.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]
    usage = result.get("usage", {})
    cost = (usage.get("input_tokens", 0) * 0.80 + usage.get("output_tokens", 0) * 4.00) / 1_000_000
    return {"text": text, "cost_usd": cost, "latency_ms": latency_ms}


def parse_json_output(text: str) -> dict:
    """Extract JSON from model response."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def verify_single(rubric_path: Path, problem: str, diff: str,
                   context: str, temperature: float) -> dict:
    """Run a single verification with context augmentation."""
    rubric = rubric_path.read_text()
    MAX_DIFF = 100000

    # Build augmented prompt
    context_section = ""
    if context:
        context_section = f"""
## Surrounding Code Context

The following shows the source code around each changed region, BEFORE the patch was applied.
Use this to understand what the code does and whether the patch changes are correct.

{context}

"""

    prompt = f"""{rubric}

## Problem Statement

{problem[:8000]}

{context_section}## Proposed Patch

```diff
{diff[:MAX_DIFF]}
```

Now evaluate this patch according to the rubric above. Respond with ONLY the JSON object."""

    try:
        resp = call_haiku(prompt, temperature)
        parsed = parse_json_output(resp["text"])
        return {
            "parsed": parsed,
            "cost_usd": resp["cost_usd"],
            "latency_ms": resp["latency_ms"],
            "error": None if parsed else "parse_failed",
            "raw_text": resp["text"][:300],
        }
    except Exception as e:
        return {"parsed": None, "error": str(e)[:500], "cost_usd": 0}


def run_ensemble_with_context(problem: str, diff: str, context: str) -> dict:
    """Run v001∩v009(2+/3) ensemble with context augmentation."""
    calls = [
        ("v001", V001_RUBRIC, 0.0),
        ("v009_r1", V009_RUBRIC, 0.3),
        ("v009_r2", V009_RUBRIC, 0.3),
        ("v009_r3", V009_RUBRIC, 0.3),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(verify_single, rub, problem, diff, context, temp): name
            for name, rub, temp in calls
        }
        for f in as_completed(futures):
            name = futures[f]
            results[name] = f.result()

    v001 = results["v001"]
    v001_verdict = (v001.get("parsed") or {}).get("verdict", "")
    v001_pass = v001_verdict == "likely_correct"
    v001_score = (v001.get("parsed") or {}).get("overall_score")
    v001_reasoning = (v001.get("parsed") or {}).get("reasoning", "")[:200]

    v009_names = [n for n in results if n.startswith("v009")]
    v009_lc = sum(
        1 for n in v009_names
        if (results[n].get("parsed") or {}).get("verdict") == "likely_correct"
    )
    v009_pass = v009_lc >= V009_THRESHOLD
    ensemble_pass = v001_pass and v009_pass

    total_cost = sum(r.get("cost_usd", 0) for r in results.values())
    errors = [f"{n}: {r['error']}" for n, r in results.items() if r.get("error")]

    return {
        "ensemble_pass": ensemble_pass,
        "v001_verdict": v001_verdict,
        "v001_score": v001_score,
        "v001_reasoning": v001_reasoning,
        "v009_lc_count": v009_lc,
        "v009_verdicts": {
            n: (results[n].get("parsed") or {}).get("verdict", "error")
            for n in sorted(v009_names)
        },
        "total_cost_usd": round(total_cost, 6),
        "wall_latency_ms": max(r.get("latency_ms", 0) for r in results.values()),
        "errors": errors if errors else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Context-augmented verifier prototype")
    parser.add_argument("--context-lines", type=int, default=30,
                        help="Lines of context around each hunk (default: 30)")
    parser.add_argument("--limit", type=int, help="Limit to first N patches")
    parser.add_argument("--issue", type=str, help="Run a single issue")
    parser.add_argument("--no-context", action="store_true",
                        help="Run without context (baseline comparison)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load SWE-bench metadata
    print("Loading SWE-bench data...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    issues = {row["instance_id"]: row for row in ds}
    problems = {row["instance_id"]: row["problem_statement"] for row in ds}

    # Load gold labels
    gold = {}
    gold_file = RESULTS_DIR / "gold_devstral_sera_vloop_opencode.jsonl"
    with open(gold_file) as f:
        for line in f:
            row = json.loads(line)
            gold[row["instance_id"]] = row["passed"]

    # Find diffs
    diff_files = sorted(DIFFS_DIR.glob("*.diff"))
    if args.issue:
        diff_files = [d for d in diff_files if d.stem == args.issue]
    if args.limit:
        diff_files = diff_files[:args.limit]

    ctx_label = f"ctx{args.context_lines}" if not args.no_context else "no_ctx"
    output_file = RESULTS_DIR / f"verify_context_{ctx_label}.jsonl"

    # Resume
    completed = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                completed.add(json.loads(line)["instance_id"])
        print(f"Resuming: {len(completed)} done")

    print(f"Diffs: {len(diff_files)}, Gold passes: {sum(gold.values())}, Context: {ctx_label}")
    print(f"Output: {output_file}\n")

    tp = fp = fn = tn = 0
    total_cost = 0.0

    for idx, diff_file in enumerate(diff_files):
        iid = diff_file.stem
        if iid in completed:
            continue
        if iid not in gold:
            continue

        diff_text = diff_file.read_text()
        problem = problems.get(iid, "")
        gold_pass = gold[iid]
        issue_meta = issues.get(iid, {})

        # Fetch context
        context = ""
        if not args.no_context:
            repo = issue_meta.get("repo", "")
            base_commit = issue_meta.get("base_commit", "")
            if repo and base_commit:
                context = get_context_for_diff(
                    diff_text, repo, base_commit, args.context_lines
                )

        ctx_size = len(context)
        print(f"[{idx+1}/{len(diff_files)}] {iid} (gold={'PASS' if gold_pass else 'FAIL'}, ctx={ctx_size} chars)...", end=" ", flush=True)

        result = run_ensemble_with_context(problem, diff_text, context)
        predicted = result["ensemble_pass"]

        if predicted and gold_pass:
            tp += 1; label = "TP"
        elif predicted and not gold_pass:
            fp += 1; label = "FP"
        elif not predicted and gold_pass:
            fn += 1; label = "FN"
        else:
            tn += 1; label = "TN"

        total_cost += result["total_cost_usd"]
        print(f"{label} | v001={result['v001_verdict']} v009={result['v009_lc_count']}/3 | ${result['total_cost_usd']:.4f}")

        row = {
            "instance_id": iid,
            "gold_pass": gold_pass,
            "predicted_pass": predicted,
            "context_chars": ctx_size,
            **result,
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Context-augmented verifier ({ctx_label})")
    print(f"{'='*60}")
    print(f"Total: {total} | TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision: {prec:.2f} | Recall: {rec:.2f} | F0.5: {f05:.2f}")
    print(f"Cost: ${total_cost:.4f}")

    # Save summary
    summary = {
        "context_lines": args.context_lines if not args.no_context else 0,
        "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "total_cost_usd": round(total_cost, 4),
    }
    summary_file = RESULTS_DIR / f"verify_context_{ctx_label}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
