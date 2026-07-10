#!/usr/bin/env python3
"""Phase 1 — mine merged PRs from a target repo, filter to eval-worthy tasks.

A PR is eval-worthy iff: merged, human-authored (no bots), touches at least one
test file AND at least one non-test source file (a self-contained behavior change
with a test that can serve as held-out ground truth), and is not pure docs/CI.

Output: results/candidates.jsonl — one line per surviving PR with the metadata
Phase-1 synthesize/tag steps need. Uses `gh` (already authed); no API token juggling.

Usage:
  python3 mine_prs.py --repo pydantic/pydantic --limit 200 --out results/candidates.jsonl
"""
import argparse, json, subprocess, sys, pathlib

BOT_MARKERS = ("dependabot", "[bot]", "pre-commit-ci", "github-actions")
# a PR must touch >=1 of these (test signal) and >=1 non-test .py (source change)
TEST_HINTS = ("test", "tests/")
DOC_CI_ONLY_DIRS = ("docs/", ".github/", "Makefile", "mkdocs")
# a self-contained behavior change must NOT touch build/CI/env plumbing — those
# PRs need special harness setup and aren't isolatable (mining artifact #12636).
NON_SELF_CONTAINED = ("pyproject.toml", ".github/", "requirements", "uv.lock",
                      "setup.py", "setup.cfg", "tox.ini", "Makefile", "Dockerfile")
# mypy-plugin tests need a mypy run, not plain pytest — out of scope for the harness
HARNESS_UNSUPPORTED_TESTS = ("mypy",)
# V1-era files that don't exist in current pydantic (stale PRs against old branches)
STALE_SOURCE_HINTS = ("pydantic/generics.py",)


def gh_json(args):
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"gh failed: {' '.join(args)}\n{out.stderr[:500]}")
    return json.loads(out.stdout)


def _files_via_api(repo, number):
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{number}/files", "--paginate"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return []
    # --paginate concatenates JSON arrays; normalize by parsing line-agnostic
    txt = out.stdout.strip()
    files = []
    for chunk in txt.replace("][", "]\x00[").split("\x00"):
        try:
            files.extend(json.loads(chunk))
        except json.JSONDecodeError:
            pass
    return [{"path": f["filename"], "add": f.get("additions", 0),
             "del": f.get("deletions", 0)} for f in files]


def is_test(path):
    p = path.lower()
    return p.endswith(".py") and any(h in p for h in TEST_HINTS)


def is_source(path):
    p = path.lower()
    return p.endswith(".py") and not any(h in p for h in TEST_HINTS) \
        and not any(p.startswith(d) for d in DOC_CI_ONLY_DIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--merged-after", default=None,
                    help="ISO date floor, e.g. 2025-07-01 — drop PRs merged before this "
                         "to reduce training-data contamination (spec Known Limitations).")
    ap.add_argument("--out", default="results/candidates.jsonl")
    args = ap.parse_args()

    prs = gh_json(["pr", "list", "--repo", args.repo, "--state", "merged",
                   "--limit", str(args.limit),
                   "--json", "number,title,mergedAt,mergeCommit,author,baseRefName,body"])

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    kept, seen, rej = [], 0, {"bot": 0, "stale": 0, "no_test": 0, "no_source": 0,
                              "not_self_contained": 0, "unsupported_test": 0}
    for pr in prs:
        seen += 1
        if args.merged_after and (pr.get("mergedAt") or "") < args.merged_after:
            rej["stale"] += 1
            continue
        login = (pr.get("author") or {}).get("login", "")
        if any(m in login.lower() for m in BOT_MARKERS):
            rej["bot"] += 1
            continue
        files = _files_via_api(args.repo, pr["number"])
        paths = [f["path"] for f in files]
        # reject PRs that touch build/CI/env plumbing or stale V1 files — not isolatable
        if any(any(m in p for m in NON_SELF_CONTAINED) for p in paths) or \
           any(any(s in p for s in STALE_SOURCE_HINTS) for p in paths):
            rej["not_self_contained"] += 1
            continue
        tests = [f for f in files if is_test(f["path"])]
        srcs = [f for f in files if is_source(f["path"])]
        # drop PRs whose ONLY tests are harness-unsupported (e.g. mypy-plugin tests)
        tests = [f for f in tests
                 if not any(u in f["path"] for u in HARNESS_UNSUPPORTED_TESTS)]
        if not tests:
            rej["no_test"] += 1
            continue
        if not srcs:
            rej["no_source"] += 1
            continue
        rec = {
            "instance_id": f"{args.repo.replace('/', '__')}-{pr['number']}",
            "repo": args.repo,
            "pr_number": pr["number"],
            "title": pr["title"],
            "merged_at": pr["mergedAt"],
            "merge_commit": (pr.get("mergeCommit") or {}).get("oid"),
            "base_ref": pr["baseRefName"],
            "author": login,
            "body": pr.get("body") or "",
            "test_files": [f["path"] for f in tests],
            "source_files": [f["path"] for f in srcs],
            "net_lines": sum(f["add"] + f["del"] for f in files),
            "n_files": len(files),
            "n_test_files": len(tests),
        }
        kept.append(rec)

    with outp.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    print(f"[mine_prs] scanned={seen} kept={len(kept)} rejected: "
          f"stale={rej['stale']} bot={rej['bot']} "
          f"not_self_contained={rej['not_self_contained']} "
          f"unsupported_test={rej['unsupported_test']} "
          f"no_test={rej['no_test']} no_source={rej['no_source']}")
    print(f"[mine_prs] wrote {outp}")


if __name__ == "__main__":
    main()
