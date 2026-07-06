#!/usr/bin/env python3
"""Mechanical grep oracle for the trace-lineage experiment.

Given a task directory (post-agent-run) with its task.json manifest, scores
consistency-completion with ZERO LLM judgment: at every coupled site, did the
NEW token replace the OLD?

  consistency_completion = sites where NEW present AND OLD absent  / K
  acted_on_stale          = any site still containing OLD          (bool)

This is the primary metric of the experiment. It is a string check by
construction, so it cannot inherit the verifier-reward semantic-mismatch recall
ceiling (that failure came from an LLM judging "completeness"; this counts
substrings).

Usage:
  grade.py --task /path/to/taskdir [--json out.json]
Exit code 0 always (grading is not pass/fail plumbing); the JSON carries the verdict.
"""
import argparse
import json
import os


def read(path):
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def grade(task_dir):
    with open(os.path.join(task_dir, "task.json")) as f:
        m = json.load(f)
    old, new = m["old_token"], m["new_token"]
    sites = m["coupled_sites"]

    per_site = []
    updated = 0
    stale = 0
    missing = 0
    for rel in sites:
        content = read(os.path.join(task_dir, rel))
        if content is None:
            per_site.append({"site": rel, "status": "missing"})
            missing += 1
            continue
        has_new = new in content
        has_old = old in content
        if has_new and not has_old:
            status = "updated"
            updated += 1
        elif has_old:
            status = "stale"        # OLD still present → not propagated
            stale += 1
        else:
            status = "absent"       # neither token — file mangled/deleted content
        per_site.append({"site": rel, "status": status,
                         "has_new": has_new, "has_old": has_old})

    k = len(sites)
    return {
        "name": m["name"], "seed": m["seed"], "k": k,
        "old": old, "new": new,
        "consistency_completion": round(updated / k, 4) if k else 0.0,
        "updated_sites": updated,
        "stale_sites": stale,          # OLD still present (the drift we care about)
        "missing_sites": missing,
        "acted_on_stale": stale > 0,   # primary correctness-failure flag
        "per_site": per_site,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task directory (with task.json)")
    ap.add_argument("--json", default=None, help="write verdict JSON here")
    args = ap.parse_args()
    verdict = grade(args.task)
    out = json.dumps(verdict, indent=2)
    if args.json:
        with open(args.json, "w") as f:
            f.write(out)
    print(out)


if __name__ == "__main__":
    main()
