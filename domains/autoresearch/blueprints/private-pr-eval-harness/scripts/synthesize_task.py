#!/usr/bin/env python3
"""Phase 1 — Gate 2: turn candidate PRs into leak-stripped tasks.

Invariant (spec Gate 2): the task prompt MUST NOT contain (a) the diff/patch,
(b) the PR's statement of WHY the fix is correct, or (c) the held-out test
function names. It MAY contain the observable problem (bug report / feature ask).

Strategy: derive the prompt from the linked ISSUE where one exists (issues state
the problem, not the fix), else from a conservatively-stripped PR title + a
generic instruction. We do NOT pass the PR body through verbatim — bodies often
say "fixed by doing X", which leaks the solution. Test function names are scrubbed.

Output: results/tasks.jsonl — {instance_id, repo, base_commit, prompt,
held_out_tests, source_files, pr_number, merge_commit, net_lines, n_files}.
base_commit is the merge commit's FIRST parent (the tree the PR branched from).

Usage:
  python3 synthesize_task.py --in results/candidates.jsonl --out results/tasks.jsonl
"""
import argparse, json, re, subprocess, pathlib, sys

# Issue bodies describe SYMPTOMS — keep them. Only redact lines that explicitly
# state the SOLUTION (a fix narrative) or embed the patch. Conservative: drop a
# line only if it clearly announces how the change was made, not merely mentions
# a verb. Code fences (```...```) are dropped wholesale (often the fix diff).
SOLUTION_LINE_RE = re.compile(
    r"^\s*(the fix|the solution|to fix|to resolve|i (fixed|resolved|changed|added)|"
    r"this (pr|patch|commit) (fix|resolv|add|chang|implement)|"
    r"(fixed|resolved) (by|in) )", re.I)
TEST_FN_RE = re.compile(r"\btest_[a-zA-Z0-9_]+")


def first_parent(repo, merge_commit):
    """merge_commit^ = the base tree the PR was written against."""
    if not merge_commit:
        return None
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/commits/{merge_commit}",
         "-q", ".parents[0].sha"], capture_output=True, text=True)
    return out.stdout.strip() or None


HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# closing keyword followed by either #N or the full issues/N URL (pydantic uses URLs)
CLOSES_RE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+"
    r"(?:https?://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)|#(\d+))", re.I)


def linked_issue_body(repo, pr_number, pr_body):
    """Find 'Fixes #N' / 'Fixes <issues-url>' in the PR body (HTML comments stripped
    first so template boilerplate can't false-match), pull that issue's TITLE+body.
    Issues describe the problem, not the fix — the ideal leak-free prompt source."""
    body = HTML_COMMENT_RE.sub("", pr_body or "")
    m = CLOSES_RE.search(body)
    if not m:
        return None
    num = m.group(1) or m.group(2)
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/issues/{num}", "-q", "{t: .title, b: .body}"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    body = (d.get("b") or "")[:2000]
    return f"{d.get('t','')}\n\n{body}".strip()


def strip_leak(text):
    """Drop code fences + explicit solution-narrative lines; scrub test-fn names.
    Keeps symptom description intact (that's the whole point of using the issue)."""
    text = HTML_COMMENT_RE.sub("", text)
    kept, in_fence = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue  # code blocks often contain the fix / stack traces w/ paths
        if SOLUTION_LINE_RE.match(line):
            continue
        kept.append(TEST_FN_RE.sub("<test>", line))
    return "\n".join(kept).strip()


def build_prompt(repo, title, problem_text):
    # title minus leaky verbs, used only as a one-line hint
    hint = TEST_FN_RE.sub("<test>", title)
    body = f"\n\nReported problem:\n{problem_text}" if problem_text else ""
    return (
        f"You are working in the `{repo}` codebase. There is a bug or missing "
        f"behavior described below. Investigate the code, find the root cause, and "
        f"implement a fix. Do not just make tests pass — fix the underlying issue.\n\n"
        f"Summary: {hint}{body}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/candidates.jsonl")
    ap.add_argument("--out", default="results/tasks.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    if args.limit:
        rows = rows[:args.limit]

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tasks, used_issue, no_base = 0, 0, 0
    with outp.open("w") as fh:
        for r in rows:
            base = first_parent(r["repo"], r.get("merge_commit"))
            if not base:
                no_base += 1
                continue
            issue = linked_issue_body(r["repo"], r["pr_number"], r.get("body", ""))
            if issue:
                used_issue += 1
                problem = strip_leak(issue)
            else:
                problem = ""  # title-only prompt; harder, but leak-free
            task = {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "pr_number": r["pr_number"],
                "base_commit": base,
                "merge_commit": r["merge_commit"],
                "prompt": build_prompt(r["repo"], r["title"], problem),
                "held_out_tests": r["test_files"],
                "source_files": r["source_files"],
                "net_lines": r["net_lines"],
                "n_files": r["n_files"],
                "complexity_tier": r.get("complexity_tier"),
                "merged_at": r.get("merged_at"),
                "prompt_source": "issue" if issue else "title_only",
            }
            fh.write(json.dumps(task) + "\n")
            tasks += 1

    print(f"[synthesize] tasks={tasks} from_issue={used_issue} "
          f"title_only={tasks - used_issue} dropped_no_base={no_base}")
    print(f"[synthesize] wrote {outp}")
    print("[synthesize] NOTE Gate-2 requires hand-review of >=20 prompts before scoring.")


if __name__ == "__main__":
    main()
